import * as fs from "fs";
import * as path from "path";

import {
    loadAnswerRejectDataset,
    type AnswerRejectBehavior,
    type AnswerRejectCase,
} from "./answer_reject_dataset.ts";
import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    resolvePipelinePresetByName,
    executeSearchPipeline,
    type PipelineBehavior,
    type PipelinePreset,
} from "../src/worker/search_pipeline.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";
import {
    buildAnswerRejectResultFileName,
    resolveNamedDatasetProfile,
} from "./result_naming.ts";
import { updateCurrentResultRegistry } from "./result_registry.ts";

type CaseReport = {
    id: string;
    query: string;
    expected_behavior: AnswerRejectBehavior;
    db_absent_evidence_class?: AnswerRejectCase["db_absent_evidence_class"];
    predicted_behavior: PipelineBehavior;
    retrieval_behavior: PipelineBehavior;
    behavior_correct: boolean;
    false_reject: boolean;
    unsafe_answer: boolean;
    expected_otid: string | null;
    top_result_otid: string | null;
    answer_doc_hit: boolean | null;
    pair_id: string | null;
    pair_role: "positive" | "negative" | null;
    rejection_reason: string | null;
    retrieval_reject_score: number | null;
    final_reject_score: number | null;
    retrieval_reject_tier: string | null;
    final_reject_tier: string | null;
    evidence_top_role_tags: string[];
    weak_match_count: number;
    match_count: number;
    candidate_count: number;
    query_intent: {
        years: number[];
        topicIds: string[];
        intentIds: string[];
        preferLatest: boolean;
        preferLatestStrong: boolean;
        signals: {
            hasExplicitTopicOrIntent: boolean;
            hasExplicitYear: boolean;
            hasHistoricalHint: boolean;
            hasStrongDetailAnchor: boolean;
            hasEntryLikeAnchor: boolean;
            hasResultState: boolean;
            hasLatestPolicyState: boolean;
            hasGenericNextStep: boolean;
            tokenCount?: number;
        };
    };
    top_matches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
        top_role_tags: string[];
    }>;
    top_weak_matches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
        top_role_tags: string[];
    }>;
};

type Summary = {
    total: number;
    byExpectedBehavior: Record<AnswerRejectBehavior, number>;
    byPredictedBehavior: Record<PipelineBehavior, number>;
    binaryBehaviorAccuracy: number;
    answerHitRate: number;
    rejectRecall: number;
    falseRejectRate: number;
    unsafeAnswerRate: number;
    answerDocHitRate: number;
    pairCount: number;
    pairConsistency: number | null;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetName: string;
    datasetAlias?: string;
    datasetDisplayName?: string;
    total: number;
    config: {
        pipelineVersion: string;
        preset: PipelinePreset;
        note: string;
    };
    summary: Summary;
    caseReports: CaseReport[];
};

const DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_ANSWER_REJECT_DATASET_FILE ||
        CURRENT_EVAL_DATASET_FILES.answerRejectCurrent,
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;
const DEFAULT_REPORT_NOTE =
    "当前报告默认对齐前端 runtime preset，数据集固定为 80-case frozen AnswerReject 主线。";
const REPORT_NOTE = process.env.SUASK_ANSWER_REJECT_NOTE || DEFAULT_REPORT_NOTE;
const PIPELINE_PRESET_NAME =
    process.env.SUASK_PIPELINE_PRESET ||
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name;
const EVAL_PRESET = resolvePipelinePresetByName(PIPELINE_PRESET_NAME);

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function resolveTopResultOtid(
    predictedBehavior: PipelineBehavior,
    pipelineResult: Awaited<ReturnType<typeof executeSearchPipeline>>,
): string | null {
    if (predictedBehavior !== "answer") {
        return null;
    }
    return (
        pipelineResult.results[0]?.otid ||
        pipelineResult.searchOutput.matches[0]?.otid ||
        null
    );
}

function buildPairConsistency(caseReports: CaseReport[]): {
    pairCount: number;
    pairConsistency: number | null;
} {
    const pairMap = new Map<
        string,
        { positive?: CaseReport; negative?: CaseReport }
    >();

    caseReports.forEach((item) => {
        if (!item.pair_id || !item.pair_role) {
            return;
        }
        const current = pairMap.get(item.pair_id) || {};
        current[item.pair_role] = item;
        pairMap.set(item.pair_id, current);
    });

    const completePairs = Array.from(pairMap.values()).filter(
        (item) => item.positive && item.negative,
    );
    if (completePairs.length === 0) {
        return {
            pairCount: 0,
            pairConsistency: null,
        };
    }

    const consistentCount = completePairs.filter(
        (item) =>
            item.positive?.predicted_behavior === "answer" &&
            item.negative?.predicted_behavior === "reject",
    ).length;

    return {
        pairCount: completePairs.length,
        pairConsistency: safeRate(consistentCount, completePairs.length),
    };
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const byExpectedBehavior: Record<AnswerRejectBehavior, number> = {
        answer: 0,
        reject: 0,
    };
    const byPredictedBehavior: Record<PipelineBehavior, number> = {
        answer: 0,
        reject: 0,
    };

    let binaryCorrect = 0;
    let answerTotal = 0;
    let answerHit = 0;
    let rejectTotal = 0;
    let rejectHit = 0;
    let falseReject = 0;
    let unsafeAnswer = 0;
    let answerDocEligible = 0;
    let answerDocHit = 0;

    caseReports.forEach((item) => {
        byExpectedBehavior[item.expected_behavior] += 1;
        byPredictedBehavior[item.predicted_behavior] += 1;

        if (item.behavior_correct) {
            binaryCorrect += 1;
        }
        if (item.expected_behavior === "answer") {
            answerTotal += 1;
            if (item.predicted_behavior === "answer") {
                answerHit += 1;
            }
            if (item.false_reject) {
                falseReject += 1;
            }
            if (item.answer_doc_hit !== null) {
                answerDocEligible += 1;
                if (item.answer_doc_hit) {
                    answerDocHit += 1;
                }
            }
        }
        if (item.expected_behavior === "reject") {
            rejectTotal += 1;
            if (item.predicted_behavior === "reject") {
                rejectHit += 1;
            }
            if (item.unsafe_answer) {
                unsafeAnswer += 1;
            }
        }
    });

    const pairStats = buildPairConsistency(caseReports);

    return {
        total: caseReports.length,
        byExpectedBehavior,
        byPredictedBehavior,
        binaryBehaviorAccuracy: safeRate(binaryCorrect, caseReports.length),
        answerHitRate: safeRate(answerHit, answerTotal),
        rejectRecall: safeRate(rejectHit, rejectTotal),
        falseRejectRate: safeRate(falseReject, answerTotal),
        unsafeAnswerRate: safeRate(unsafeAnswer, rejectTotal),
        answerDocHitRate: safeRate(answerDocHit, answerDocEligible),
        pairCount: pairStats.pairCount,
        pairConsistency: pairStats.pairConsistency,
    };
}

async function main() {
    const datasetName = path.basename(DATASET_FILE, path.extname(DATASET_FILE));
    const datasetProfile = resolveNamedDatasetProfile(datasetName);
    const { cases: testCases, datasetNote } = loadAnswerRejectDataset(DATASET_FILE);

    console.log(`Loading answer_reject dataset: ${DATASET_FILE}`);
    console.log(`Loaded ${testCases.length} cases.`);

    const engine = await loadFrontendEvalEngine();
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        testCases.map((item) => item.query),
        engine.dimensions,
    );

    const caseReports: CaseReport[] = [];

    for (let index = 0; index < testCases.length; index += 1) {
        const testCase = testCases[index];
        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            engine.vocabMap,
            engine.topicPartitionIndex,
            EVAL_PRESET,
        );
        const pipelineResult = await executeSearchPipeline({
            query: testCase.query,
            queryVector: queryVectors[index],
            queryContext,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: engine.bm25Stats,
            extractor: engine.extractor,
            documentLoader,
            termMaps,
            preset: EVAL_PRESET,
        });

        const predictedBehavior = pipelineResult.finalDecision.behavior;
        const topResultOtid = resolveTopResultOtid(
            predictedBehavior,
            pipelineResult,
        );
        const answerDocHit =
            testCase.expected_behavior === "answer" && testCase.expected_otid
                ? predictedBehavior === "answer" &&
                  topResultOtid === testCase.expected_otid
                : null;

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            expected_behavior: testCase.expected_behavior,
            db_absent_evidence_class: testCase.db_absent_evidence_class,
            predicted_behavior: predictedBehavior,
            retrieval_behavior: pipelineResult.retrievalDecision.behavior,
            behavior_correct:
                predictedBehavior === testCase.expected_behavior,
            false_reject:
                testCase.expected_behavior === "answer" &&
                predictedBehavior === "reject",
            unsafe_answer:
                testCase.expected_behavior === "reject" &&
                predictedBehavior === "answer",
            expected_otid: testCase.expected_otid || null,
            top_result_otid: topResultOtid,
            answer_doc_hit: answerDocHit,
            pair_id: testCase.pair_id || null,
            pair_role: testCase.pair_role || null,
            rejection_reason:
                pipelineResult.finalDecision.rejectionReason ||
                pipelineResult.rejection?.reason ||
                null,
            retrieval_reject_score:
                pipelineResult.retrievalDecision.rejectScore ?? null,
            final_reject_score:
                pipelineResult.finalDecision.rejectScore ?? null,
            retrieval_reject_tier:
                pipelineResult.retrievalDecision.rejectTier ?? null,
            final_reject_tier:
                pipelineResult.finalDecision.rejectTier ?? null,
            evidence_top_role_tags:
                pipelineResult.searchOutput.diagnostics?.evidenceSignals
                    ?.topRoleTags || [],
            weak_match_count: pipelineResult.trace.weakMatchCount,
            match_count: pipelineResult.trace.matchCount,
            candidate_count: pipelineResult.trace.candidateCount,
            query_intent: {
                years: queryContext.queryIntent.years,
                topicIds: queryContext.queryIntent.topicIds,
                intentIds: queryContext.queryIntent.intentIds,
                preferLatest: queryContext.queryIntent.preferLatest,
                preferLatestStrong:
                    queryContext.queryIntent.preferLatestStrong,
                signals: {
                    hasExplicitTopicOrIntent:
                        queryContext.queryIntent.signals
                            .hasExplicitTopicOrIntent,
                    hasExplicitYear:
                        queryContext.queryIntent.signals.hasExplicitYear,
                    hasHistoricalHint:
                        queryContext.queryIntent.signals.hasHistoricalHint,
                    hasStrongDetailAnchor:
                        queryContext.queryIntent.signals
                            .hasStrongDetailAnchor,
                    hasEntryLikeAnchor:
                        queryContext.queryIntent.signals.hasEntryLikeAnchor,
                    hasResultState:
                        queryContext.queryIntent.signals.hasResultState,
                    hasLatestPolicyState:
                        queryContext.queryIntent.signals
                            .hasLatestPolicyState,
                    hasGenericNextStep:
                        queryContext.queryIntent.signals
                            .hasGenericNextStep,
                    tokenCount:
                        queryContext.queryIntent.signals.tokenCount,
                },
            },
            top_matches: pipelineResult.searchOutput.matches
                .slice(0, 3)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                    top_role_tags: Array.from(
                        new Set(
                            (match.kp_candidates || []).flatMap(
                                (candidate) => candidate.kp_role_tags || [],
                            ),
                        ),
                    ),
                })),
            top_weak_matches: pipelineResult.searchOutput.weakMatches
                .slice(0, 3)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                    top_role_tags: Array.from(
                        new Set(
                            (match.kp_candidates || []).flatMap(
                                (candidate) => candidate.kp_role_tags || [],
                            ),
                        ),
                    ),
                })),
        });
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetFile: DATASET_FILE,
        datasetName,
        datasetAlias: datasetProfile.alias,
        datasetDisplayName: datasetProfile.displayName,
        total: caseReports.length,
        config: {
            pipelineVersion: EVAL_PRESET.name,
            preset: EVAL_PRESET,
            note: datasetNote ? `${REPORT_NOTE} ${datasetNote}` : REPORT_NOTE,
        },
        summary: buildSummary(caseReports),
        caseReports,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        buildAnswerRejectResultFileName(datasetName, Date.now()),
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    updateCurrentResultRegistry({
        datasetName,
        datasetAlias: datasetProfile.alias,
        datasetDisplayName: datasetProfile.displayName,
        datasetFile: DATASET_FILE,
        outputPath,
        sourceScript: "eval_answer_reject.ts",
        note: "当前稳定入口使用 80-case frozen AnswerReject 主线。",
    });

    console.log(`Saved report to ${outputPath}`);
    console.log(
        [
            `binaryBehaviorAccuracy=${(report.summary.binaryBehaviorAccuracy * 100).toFixed(2)}%`,
            `answerHitRate=${(report.summary.answerHitRate * 100).toFixed(2)}%`,
            `rejectRecall=${(report.summary.rejectRecall * 100).toFixed(2)}%`,
            `falseRejectRate=${(report.summary.falseRejectRate * 100).toFixed(2)}%`,
            `unsafeAnswerRate=${(report.summary.unsafeAnswerRate * 100).toFixed(2)}%`,
            `answerDocHitRate=${(report.summary.answerDocHitRate * 100).toFixed(2)}%`,
            report.summary.pairConsistency === null
                ? "pairConsistency=NA"
                : `pairConsistency=${(report.summary.pairConsistency * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

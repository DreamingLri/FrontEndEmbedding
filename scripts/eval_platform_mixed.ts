import * as fs from "fs";
import * as path from "path";

import {
    CANONICAL_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    clonePipelinePreset,
    executeSearchPipeline,
    resolvePipelinePresetByName,
    type PipelineBehavior,
    type PipelinePreset,
} from "../src/worker/search_pipeline.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";
import { updateCurrentResultRegistry } from "./result_registry.ts";

type ScenarioFamily =
    | "direct_answer_simple"
    | "direct_answer_complex"
    | "route_or_clarify"
    | "latest_within_topic";

type ExpectedBehavior =
    | "direct_answer"
    | "clarify"
    | "route_to_entry"
    | "reject";

type PlatformMixedCase = {
    id: string;
    query: string;
    query_type?: string;
    scenario_family: ScenarioFamily;
    expected_behavior: ExpectedBehavior;
    expected_otid?: string;
    expected_entry_topic?: string;
    expected_support_kpids?: string[];
    expected_latest_otid?: string;
    topic_cluster?: string;
    challenge_tags?: string[];
    realism_weight?: number;
    source_family?: string;
    source_file?: string;
    source_item_ref?: string;
    notes?: string;
};

type CaseReport = {
    id: string;
    query: string;
    query_type?: string;
    scenario_family: ScenarioFamily;
    expected_behavior: ExpectedBehavior;
    predicted_behavior: PipelineBehavior;
    retrieval_behavior: PipelineBehavior;
    behavior_correct: boolean;
    success: boolean;
    rejection_reason: string | null;
    expected_otid?: string;
    expected_latest_otid?: string;
    expected_entry_topic?: string;
    predicted_entry_topic?: string;
    retrieval_doc_rank: number | null;
    rendered_doc_rank: number | null;
    doc_hit_at_1: boolean;
    doc_hit_at_3: boolean;
    doc_hit_at_5: boolean;
    retrieval_doc_hit_at_1: boolean;
    unsafe_direct_answer: boolean;
    candidate_count: number;
    weak_match_count: number;
    match_count: number;
    fetched_document_count: number;
    initial_top_confidence: number | null;
    final_top_confidence: number | null;
    rescue_attempted: boolean;
    rescue_accepted: boolean;
    rescue_succeeded: boolean;
    rescue_reason: string | null;
    retrieval_top1_top2_gap: number | null;
    retrieval_dominant_topic_ratio: number | null;
    query_has_explicit_topic_or_intent: boolean;
    query_has_strong_detail_anchor: boolean;
    query_has_explicit_year: boolean;
    top_matches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
    top_weak_matches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
    top_results: Array<{
        rank: number;
        otid: string;
        displayScore: number;
        snippetScore?: number;
        best_kpid?: string;
    }>;
    notes?: string;
};

type RateSummary = {
    total: number;
    success: number;
    rate: number;
};

type DocHitSummary = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    hitAt1Rate: number;
    hitAt3Rate: number;
    hitAt5Rate: number;
};

type ScenarioFamilySummary = {
    total: number;
    successRate: number;
    successCount: number;
    behaviorAccuracy: number;
    directAnswerRate?: number;
    clarifyRate?: number;
    routeToEntryRate?: number;
    rejectRate?: number;
    unsafeDirectAnswerRate?: number;
    docHitAt1Rate?: number;
    docHitAt3Rate?: number;
    docHitAt5Rate?: number;
    retrievalDocHitAt1Rate?: number;
};

type Summary = {
    total: number;
    overallBehaviorAccuracy: number;
    platformTaskSuccess: number;
    behaviorOnlyAccuracy: number;
    directAnswerSuccessRate: RateSummary;
    latestTopicSuccessRate: RateSummary;
    clarifySuccessRate: RateSummary;
    routeToEntrySuccessRate: RateSummary;
    clarifyOrRouteSuccessRate: RateSummary;
    routeFamilyBehaviorAccuracy: RateSummary;
    rejectHitRate: RateSummary;
    unsafeDirectAnswerRate: RateSummary;
    directAnswerRescueAttemptRate: RateSummary;
    directAnswerRescueSuccessRate: RateSummary;
    directAnswerDocHits: DocHitSummary;
    retrievalStageDocHits: DocHitSummary;
    complexDirectSubsetDocHits: DocHitSummary;
    byScenarioFamily: Record<ScenarioFamily, ScenarioFamilySummary>;
    byPredictedBehavior: Record<PipelineBehavior, number>;
    failureBreakdown: Record<string, number>;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetName: string;
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
    process.env.SUASK_PLATFORM_MIXED_DATASET_FILE ||
        CURRENT_EVAL_DATASET_FILES.platformMixedDailyV12,
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;
const DEFAULT_REPORT_NOTE =
    "当前报告直接调用统一 full pipeline，默认数据集已切到 mixed daily v1.2 reviewed。";
const RESULTS_PREFIX =
    process.env.SUASK_PLATFORM_MIXED_RESULTS_PREFIX || "platform_mixed";
const REPORT_NOTE =
    process.env.SUASK_PLATFORM_MIXED_NOTE || DEFAULT_REPORT_NOTE;
const PIPELINE_PRESET_NAME =
    process.env.SUASK_PIPELINE_PRESET || CANONICAL_PIPELINE_PRESET.name;

function parseThresholdOverride(): number | null {
    const raw = process.env.SUASK_PLATFORM_MIXED_REJECT_THRESHOLD;
    if (!raw) {
        return null;
    }

    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0 || value >= 1) {
        throw new Error(
            `Invalid SUASK_PLATFORM_MIXED_REJECT_THRESHOLD: ${raw}`,
        );
    }

    return value;
}

const REJECT_THRESHOLD_OVERRIDE = parseThresholdOverride();
const BASE_PRESET = resolvePipelinePresetByName(PIPELINE_PRESET_NAME);
const EVAL_PRESET: PipelinePreset =
    REJECT_THRESHOLD_OVERRIDE === null
        ? BASE_PRESET
        : {
              ...clonePipelinePreset(BASE_PRESET),
              name: `${BASE_PRESET.name}_rt${REJECT_THRESHOLD_OVERRIDE.toFixed(2)}`,
              display: {
                  ...BASE_PRESET.display,
                  rejectThreshold: REJECT_THRESHOLD_OVERRIDE,
              },
          };

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function getRankByOtid(
    otids: string[],
    expectedOtid?: string,
): number | null {
    if (!expectedOtid) {
        return null;
    }

    const index = otids.findIndex((otid) => otid === expectedOtid);
    return index >= 0 ? index + 1 : null;
}

function buildDocHitSummary(
    caseReports: CaseReport[],
    rankField: "rendered_doc_rank" | "retrieval_doc_rank",
): DocHitSummary {
    const total = caseReports.length;
    const ranks = caseReports.map((item) => item[rankField]);
    const hitAt1 = ranks.filter((rank) => rank === 1).length;
    const hitAt3 = ranks.filter((rank) => rank !== null && rank <= 3).length;
    const hitAt5 = ranks.filter((rank) => rank !== null && rank <= 5).length;

    return {
        total,
        hitAt1,
        hitAt3,
        hitAt5,
        hitAt1Rate: safeRate(hitAt1, total),
        hitAt3Rate: safeRate(hitAt3, total),
        hitAt5Rate: safeRate(hitAt5, total),
    };
}

function buildRateSummary(success: number, total: number): RateSummary {
    return {
        total,
        success,
        rate: safeRate(success, total),
    };
}

function buildScenarioFamilySummary(
    scenarioFamily: ScenarioFamily,
    caseReports: CaseReport[],
): ScenarioFamilySummary {
    const total = caseReports.length;
    const successCount = caseReports.filter((item) => item.success).length;
    const behaviorCorrectCount = caseReports.filter(
        (item) => item.behavior_correct,
    ).length;

    const base: ScenarioFamilySummary = {
        total,
        successCount,
        successRate: safeRate(successCount, total),
        behaviorAccuracy: safeRate(behaviorCorrectCount, total),
        directAnswerRate: safeRate(
            caseReports.filter((item) => item.predicted_behavior === "direct_answer")
                .length,
            total,
        ),
        clarifyRate: safeRate(
            caseReports.filter((item) => item.predicted_behavior === "clarify").length,
            total,
        ),
        routeToEntryRate: safeRate(
            caseReports.filter((item) => item.predicted_behavior === "route_to_entry")
                .length,
            total,
        ),
        rejectRate: safeRate(
            caseReports.filter((item) => item.predicted_behavior === "reject").length,
            total,
        ),
    };

    if (scenarioFamily === "route_or_clarify") {
        return {
            ...base,
            unsafeDirectAnswerRate: safeRate(
                caseReports.filter((item) => item.unsafe_direct_answer).length,
                total,
            ),
        };
    }

    return {
        ...base,
        docHitAt1Rate: safeRate(
            caseReports.filter((item) => item.doc_hit_at_1).length,
            total,
        ),
        docHitAt3Rate: safeRate(
            caseReports.filter((item) => item.doc_hit_at_3).length,
            total,
        ),
        docHitAt5Rate: safeRate(
            caseReports.filter((item) => item.doc_hit_at_5).length,
            total,
        ),
        retrievalDocHitAt1Rate: safeRate(
            caseReports.filter((item) => item.retrieval_doc_hit_at_1).length,
            total,
        ),
    };
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const total = caseReports.length;
    const overallSuccess = caseReports.filter((item) => item.success).length;
    const behaviorCorrect = caseReports.filter((item) => item.behavior_correct).length;

    const directAnswerFamilies = caseReports.filter(
        (item) =>
            item.scenario_family === "direct_answer_simple" ||
            item.scenario_family === "direct_answer_complex",
    );
    const latestFamily = caseReports.filter(
        (item) => item.scenario_family === "latest_within_topic",
    );
    const routeFamily = caseReports.filter(
        (item) => item.scenario_family === "route_or_clarify",
    );
    const clarifyCases = routeFamily.filter(
        (item) => item.expected_behavior === "clarify",
    );
    const routeCases = routeFamily.filter(
        (item) => item.expected_behavior === "route_to_entry",
    );
    const routeClarifyOrRouteCases = routeFamily.filter(
        (item) => item.expected_behavior !== "reject",
    );
    const routeRejectCases = routeFamily.filter(
        (item) => item.expected_behavior === "reject",
    );
    const docTargetCases = caseReports.filter(
        (item) => item.scenario_family !== "route_or_clarify",
    );
    const complexDirectCases = caseReports.filter(
        (item) => item.scenario_family === "direct_answer_complex",
    );

    const byScenarioFamily = {
        direct_answer_simple: buildScenarioFamilySummary(
            "direct_answer_simple",
            caseReports.filter(
                (item) => item.scenario_family === "direct_answer_simple",
            ),
        ),
        direct_answer_complex: buildScenarioFamilySummary(
            "direct_answer_complex",
            complexDirectCases,
        ),
        route_or_clarify: buildScenarioFamilySummary(
            "route_or_clarify",
            routeFamily,
        ),
        latest_within_topic: buildScenarioFamilySummary(
            "latest_within_topic",
            latestFamily,
        ),
    } as Record<ScenarioFamily, ScenarioFamilySummary>;

    const byPredictedBehavior: Record<PipelineBehavior, number> = {
        direct_answer: 0,
        clarify: 0,
        route_to_entry: 0,
        reject: 0,
    };
    const failureBreakdown: Record<string, number> = {
        direct_answer_wrong_doc: 0,
        direct_answer_rejected: 0,
        direct_answer_misrouted: 0,
        route_expected_clarify_but_routed: 0,
        route_expected_route_but_clarified: 0,
        route_expected_nonreject_but_rejected: 0,
        route_expected_nonreject_but_direct_answer: 0,
        route_expected_reject_but_clarified: 0,
        route_expected_reject_but_routed: 0,
        route_expected_reject_but_direct_answer: 0,
    };

    caseReports.forEach((item) => {
        byPredictedBehavior[item.predicted_behavior] += 1;
        if (item.success) {
            return;
        }

        if (item.scenario_family === "route_or_clarify") {
            if (item.expected_behavior === "reject") {
                if (item.predicted_behavior === "clarify") {
                    failureBreakdown.route_expected_reject_but_clarified += 1;
                } else if (item.predicted_behavior === "route_to_entry") {
                    failureBreakdown.route_expected_reject_but_routed += 1;
                } else if (item.predicted_behavior === "direct_answer") {
                    failureBreakdown.route_expected_reject_but_direct_answer += 1;
                }
                return;
            }

            if (item.predicted_behavior === "reject") {
                failureBreakdown.route_expected_nonreject_but_rejected += 1;
                return;
            }

            if (item.predicted_behavior === "direct_answer") {
                failureBreakdown.route_expected_nonreject_but_direct_answer += 1;
                return;
            }

            if (
                item.expected_behavior === "clarify" &&
                item.predicted_behavior === "route_to_entry"
            ) {
                failureBreakdown.route_expected_clarify_but_routed += 1;
            }
            if (
                item.expected_behavior === "route_to_entry" &&
                item.predicted_behavior === "clarify"
            ) {
                failureBreakdown.route_expected_route_but_clarified += 1;
            }
            return;
        }

        if (item.predicted_behavior === "direct_answer") {
            failureBreakdown.direct_answer_wrong_doc += 1;
        } else if (item.predicted_behavior === "reject") {
            failureBreakdown.direct_answer_rejected += 1;
        } else {
            failureBreakdown.direct_answer_misrouted += 1;
        }
    });

    return {
        total,
        overallBehaviorAccuracy: safeRate(overallSuccess, total),
        platformTaskSuccess: safeRate(overallSuccess, total),
        behaviorOnlyAccuracy: safeRate(behaviorCorrect, total),
        directAnswerSuccessRate: buildRateSummary(
            directAnswerFamilies.filter((item) => item.success).length,
            directAnswerFamilies.length,
        ),
        latestTopicSuccessRate: buildRateSummary(
            latestFamily.filter((item) => item.success).length,
            latestFamily.length,
        ),
        clarifySuccessRate: buildRateSummary(
            clarifyCases.filter((item) => item.success).length,
            clarifyCases.length,
        ),
        routeToEntrySuccessRate: buildRateSummary(
            routeCases.filter((item) => item.success).length,
            routeCases.length,
        ),
        clarifyOrRouteSuccessRate: buildRateSummary(
            routeClarifyOrRouteCases.filter((item) => item.success).length,
            routeClarifyOrRouteCases.length,
        ),
        routeFamilyBehaviorAccuracy: buildRateSummary(
            routeFamily.filter((item) => item.success).length,
            routeFamily.length,
        ),
        rejectHitRate: buildRateSummary(
            routeRejectCases.filter((item) => item.success).length,
            routeRejectCases.length,
        ),
        unsafeDirectAnswerRate: buildRateSummary(
            routeFamily.filter((item) => item.unsafe_direct_answer).length,
            routeFamily.length,
        ),
        directAnswerRescueAttemptRate: buildRateSummary(
            docTargetCases.filter((item) => item.rescue_attempted).length,
            docTargetCases.length,
        ),
        directAnswerRescueSuccessRate: buildRateSummary(
            docTargetCases.filter((item) => item.rescue_succeeded).length,
            docTargetCases.length,
        ),
        directAnswerDocHits: buildDocHitSummary(
            docTargetCases,
            "rendered_doc_rank",
        ),
        retrievalStageDocHits: buildDocHitSummary(
            docTargetCases,
            "retrieval_doc_rank",
        ),
        complexDirectSubsetDocHits: buildDocHitSummary(
            complexDirectCases,
            "rendered_doc_rank",
        ),
        byScenarioFamily,
        byPredictedBehavior,
        failureBreakdown,
    };
}

async function main() {
    const datasetName = path.basename(DATASET_FILE, path.extname(DATASET_FILE));
    const testCases = JSON.parse(
        fs.readFileSync(DATASET_FILE, "utf-8"),
    ) as PlatformMixedCase[];

    console.log(`Loading platform-mixed dataset: ${DATASET_FILE}`);
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
        const expectedDocOtid =
            testCase.scenario_family === "latest_within_topic"
                ? testCase.expected_latest_otid || testCase.expected_otid
                : testCase.expected_otid;
        const retrievalDocRank = getRankByOtid(
            pipelineResult.searchOutput.matches.map((item) => item.otid),
            expectedDocOtid,
        );
        const renderedDocRank = getRankByOtid(
            pipelineResult.results
                .map((item) => item.otid || item.id || "")
                .filter((item): item is string => Boolean(item)),
            expectedDocOtid,
        );

        const behaviorCorrect =
            testCase.scenario_family === "route_or_clarify"
                ? predictedBehavior === testCase.expected_behavior
                : predictedBehavior === "direct_answer";
        const success =
            testCase.scenario_family === "route_or_clarify"
                ? behaviorCorrect
                : behaviorCorrect && renderedDocRank === 1;

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            query_type: testCase.query_type,
            scenario_family: testCase.scenario_family,
            expected_behavior: testCase.expected_behavior,
            predicted_behavior: predictedBehavior,
            retrieval_behavior: pipelineResult.retrievalDecision.behavior,
            behavior_correct: behaviorCorrect,
            success,
            rejection_reason:
                pipelineResult.finalDecision.rejectionReason ||
                pipelineResult.rejection?.reason ||
                null,
            expected_otid: testCase.expected_otid,
            expected_latest_otid: testCase.expected_latest_otid,
            expected_entry_topic: testCase.expected_entry_topic,
            predicted_entry_topic: pipelineResult.finalDecision.entryTopic,
            retrieval_doc_rank: retrievalDocRank,
            rendered_doc_rank: renderedDocRank,
            doc_hit_at_1: renderedDocRank === 1,
            doc_hit_at_3: renderedDocRank !== null && renderedDocRank <= 3,
            doc_hit_at_5: renderedDocRank !== null && renderedDocRank <= 5,
            retrieval_doc_hit_at_1: retrievalDocRank === 1,
            unsafe_direct_answer:
                testCase.scenario_family === "route_or_clarify" &&
                predictedBehavior === "direct_answer",
            candidate_count: pipelineResult.trace.candidateCount,
            weak_match_count: pipelineResult.trace.weakMatchCount,
            match_count: pipelineResult.trace.matchCount,
            fetched_document_count: pipelineResult.trace.fetchedDocumentCount,
            initial_top_confidence: pipelineResult.trace.initialTopConfidence ?? null,
            final_top_confidence: pipelineResult.trace.topConfidence ?? null,
            rescue_attempted:
                pipelineResult.trace.directAnswerRescue?.attempted ?? false,
            rescue_accepted:
                pipelineResult.trace.directAnswerRescue?.accepted ?? false,
            rescue_succeeded:
                pipelineResult.trace.directAnswerRescue?.succeeded ?? false,
            rescue_reason:
                pipelineResult.trace.directAnswerRescue?.reason ?? null,
            retrieval_top1_top2_gap:
                pipelineResult.trace.retrievalSignals?.top1Top2Gap ?? null,
            retrieval_dominant_topic_ratio:
                pipelineResult.trace.retrievalSignals?.dominantTopicRatio ?? null,
            query_has_explicit_topic_or_intent:
                pipelineResult.trace.querySignals?.hasExplicitTopicOrIntent ?? false,
            query_has_strong_detail_anchor:
                pipelineResult.trace.querySignals?.hasStrongDetailAnchor ?? false,
            query_has_explicit_year:
                pipelineResult.trace.querySignals?.hasExplicitYear ?? false,
            top_matches: pipelineResult.searchOutput.matches
                .slice(0, 5)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                })),
            top_weak_matches: pipelineResult.searchOutput.weakMatches
                .slice(0, 5)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                })),
            top_results: pipelineResult.results.slice(0, 5).map((result, rank) => ({
                rank: rank + 1,
                otid: result.otid || result.id || "",
                displayScore:
                    result.displayScore ??
                    result.confidenceScore ??
                    result.coarseScore ??
                    result.score ??
                    0,
                snippetScore: result.snippetScore,
                best_kpid: result.best_kpid,
            })),
            notes: testCase.notes,
        });
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetFile: DATASET_FILE,
        datasetName,
        total: caseReports.length,
        config: {
            pipelineVersion: EVAL_PRESET.name,
            preset: EVAL_PRESET,
            note: REPORT_NOTE,
        },
        summary: buildSummary(caseReports),
        caseReports,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        `${RESULTS_PREFIX}_${datasetName}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    updateCurrentResultRegistry({
        datasetName,
        datasetFile: DATASET_FILE,
        outputPath,
        sourceScript: "eval_platform_mixed.ts",
        note: "当前稳定入口会跟随 mixed 主线与 `kb_absent_v2` 边界线一起更新。",
    });

    console.log(`Saved report to ${outputPath}`);
    console.log(
        [
            `overallBehaviorAccuracy=${(report.summary.overallBehaviorAccuracy * 100).toFixed(2)}%`,
            `platformTaskSuccess=${(report.summary.platformTaskSuccess * 100).toFixed(2)}%`,
            `behaviorOnlyAccuracy=${(report.summary.behaviorOnlyAccuracy * 100).toFixed(2)}%`,
            `directAnswerSuccessRate=${(report.summary.directAnswerSuccessRate.rate * 100).toFixed(2)}%`,
            `latestTopicSuccessRate=${(report.summary.latestTopicSuccessRate.rate * 100).toFixed(2)}%`,
            `clarifySuccessRate=${(report.summary.clarifySuccessRate.rate * 100).toFixed(2)}%`,
            `routeToEntrySuccessRate=${(report.summary.routeToEntrySuccessRate.rate * 100).toFixed(2)}%`,
            `rejectHitRate=${(report.summary.rejectHitRate.rate * 100).toFixed(2)}%`,
            `unsafeDirectAnswerRate=${(report.summary.unsafeDirectAnswerRate.rate * 100).toFixed(2)}%`,
            `directAnswerRescueAttemptRate=${(report.summary.directAnswerRescueAttemptRate.rate * 100).toFixed(2)}%`,
            `directAnswerRescueSuccessRate=${(report.summary.directAnswerRescueSuccessRate.rate * 100).toFixed(2)}%`,
            `directAnswerDocHit@1=${(report.summary.directAnswerDocHits.hitAt1Rate * 100).toFixed(2)}%`,
            `retrievalDocHit@1=${(report.summary.retrievalStageDocHits.hitAt1Rate * 100).toFixed(2)}%`,
            `complexDirectHit@1=${(report.summary.complexDirectSubsetDocHits.hitAt1Rate * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

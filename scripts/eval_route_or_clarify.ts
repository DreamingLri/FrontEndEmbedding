import * as fs from "fs";
import * as path from "path";

import {
    CANONICAL_PIPELINE_PRESET,
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
import { updateCurrentResultRegistry } from "./result_registry.ts";

type ExpectedAction = "clarify" | "route_to_entry" | "reject";

type RouteCase = {
    id: string;
    query: string;
    query_type?: string;
    ambiguity_level?: string;
    expected_action: ExpectedAction;
    entry_topic?: string;
    theme_family?: string;
    challenge_tags?: string[];
    notes?: string;
};

type CaseReport = {
    id: string;
    query: string;
    expected_action: ExpectedAction;
    expected_binary_behavior: PipelineBehavior;
    predicted_behavior: PipelineBehavior;
    retrieval_behavior: PipelineBehavior;
    behavior_correct: boolean;
    unsafe_answer: boolean;
    rejection_reason: string | null;
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
    }>;
    top_weak_matches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
};

type Summary = {
    total: number;
    byExpectedAction: Record<ExpectedAction, number>;
    byPredictedBehavior: Record<PipelineBehavior, number>;
    binaryBehaviorAccuracy: number;
    nonRejectAnswerHitRate: number;
    rejectHitRate: number;
    unsafeAnswerRate: number;
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
    process.env.SUASK_ROUTE_DATASET_FILE ||
        CURRENT_EVAL_DATASET_FILES.routeOrClarifyV2Holdout,
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;
const DEFAULT_REPORT_NOTE =
    "当前报告直接调用统一 full pipeline，默认数据集已切到 route_or_clarify_v2_holdout_reviewed；如需开发回归，请显式传入 route_or_clarify_v2_dev_reviewed。";
const REPORT_NOTE = process.env.SUASK_ROUTE_NOTE || DEFAULT_REPORT_NOTE;
const PIPELINE_PRESET_NAME =
    process.env.SUASK_PIPELINE_PRESET || CANONICAL_PIPELINE_PRESET.name;
const EVAL_PRESET = resolvePipelinePresetByName(PIPELINE_PRESET_NAME);

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function toExpectedBinaryBehavior(
    action: ExpectedAction,
): PipelineBehavior {
    return action === "reject" ? "reject" : "answer";
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const byExpectedAction: Record<ExpectedAction, number> = {
        clarify: 0,
        route_to_entry: 0,
        reject: 0,
    };
    const byPredictedBehavior: Record<PipelineBehavior, number> = {
        answer: 0,
        reject: 0,
    };

    let binaryCorrect = 0;
    let nonRejectTotal = 0;
    let nonRejectHit = 0;
    let rejectTotal = 0;
    let rejectHit = 0;
    let unsafeAnswer = 0;

    caseReports.forEach((item) => {
        byExpectedAction[item.expected_action] += 1;
        byPredictedBehavior[item.predicted_behavior] += 1;
        if (item.behavior_correct) {
            binaryCorrect += 1;
        }
        if (item.expected_binary_behavior === "answer") {
            nonRejectTotal += 1;
            if (item.predicted_behavior === "answer") {
                nonRejectHit += 1;
            }
        }
        if (item.expected_action === "reject") {
            rejectTotal += 1;
            if (item.predicted_behavior === "reject") {
                rejectHit += 1;
            }
        }
        if (item.unsafe_answer) {
            unsafeAnswer += 1;
        }
    });

    return {
        total: caseReports.length,
        byExpectedAction,
        byPredictedBehavior,
        binaryBehaviorAccuracy: safeRate(binaryCorrect, caseReports.length),
        nonRejectAnswerHitRate: safeRate(nonRejectHit, nonRejectTotal),
        rejectHitRate: safeRate(rejectHit, rejectTotal),
        unsafeAnswerRate: safeRate(unsafeAnswer, caseReports.length),
    };
}

async function main() {
    const datasetName = path.basename(DATASET_FILE, path.extname(DATASET_FILE));
    const testCases = JSON.parse(
        fs.readFileSync(DATASET_FILE, "utf-8"),
    ) as RouteCase[];

    console.log(`Loading route-or-clarify dataset: ${DATASET_FILE}`);
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
        const expectedBinaryBehavior = toExpectedBinaryBehavior(
            testCase.expected_action,
        );

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            expected_action: testCase.expected_action,
            expected_binary_behavior: expectedBinaryBehavior,
            predicted_behavior: predictedBehavior,
            retrieval_behavior: pipelineResult.retrievalDecision.behavior,
            behavior_correct: predictedBehavior === expectedBinaryBehavior,
            unsafe_answer:
                expectedBinaryBehavior === "reject" &&
                predictedBehavior === "answer",
            rejection_reason:
                pipelineResult.finalDecision.rejectionReason ||
                pipelineResult.rejection?.reason ||
                null,
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
                })),
            top_weak_matches: pipelineResult.searchOutput.weakMatches
                .slice(0, 3)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                })),
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
        `route_or_clarify_${datasetName}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    updateCurrentResultRegistry({
        datasetName,
        datasetFile: DATASET_FILE,
        outputPath,
        sourceScript: "eval_route_or_clarify.ts",
        note: "当前稳定入口默认保留 `v2_dev` 与 `v2_holdout` 两条边界回归线。",
    });

    console.log(`Saved report to ${outputPath}`);
    console.log(
        [
            `binaryBehaviorAccuracy=${(report.summary.binaryBehaviorAccuracy * 100).toFixed(2)}%`,
            `nonRejectAnswerHitRate=${(report.summary.nonRejectAnswerHitRate * 100).toFixed(2)}%`,
            `rejectHitRate=${(report.summary.rejectHitRate * 100).toFixed(2)}%`,
            `unsafeAnswerRate=${(report.summary.unsafeAnswerRate * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

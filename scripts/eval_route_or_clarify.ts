import * as fs from "fs";
import * as path from "path";

import {
    CANONICAL_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    executeSearchPipeline,
    type PipelineBehavior,
} from "../src/worker/search_pipeline.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";

type ExpectedAction = "clarify" | "route_to_entry" | "reject";
type CoarseBehavior = "clarify_or_route" | "reject" | "direct_answer";

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
    expected_coarse_behavior: Exclude<CoarseBehavior, "direct_answer">;
    predicted_behavior: PipelineBehavior;
    predicted_coarse_behavior: CoarseBehavior;
    retrieval_behavior: PipelineBehavior;
    predicted_entry_topic?: string;
    expected_entry_topic?: string;
    behavior_correct: boolean;
    coarse_behavior_correct: boolean;
    entry_topic_correct: boolean | null;
    unsafe_direct_answer: boolean;
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
    coarseBehaviorAccuracy: number;
    exactBehaviorAccuracy: number;
    clarifyHitRate: number;
    routeHitRate: number;
    routeEntryTopicAccuracy: number;
    rejectHitRate: number;
    unsafeDirectAnswerRate: number;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetName: string;
    total: number;
    config: {
        pipelineVersion: string;
        preset: typeof CANONICAL_PIPELINE_PRESET;
        note: string;
    };
    summary: Summary;
    caseReports: CaseReport[];
};

const DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_ROUTE_DATASET_FILE ||
        "../Backend/test/test_dataset_route_or_clarify/test_dataset_route_or_clarify_v1_seed.json",
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function toExpectedCoarseBehavior(
    action: ExpectedAction,
): Exclude<CoarseBehavior, "direct_answer"> {
    return action === "reject" ? "reject" : "clarify_or_route";
}

function toCoarseBehavior(behavior: PipelineBehavior): CoarseBehavior {
    if (behavior === "reject") {
        return "reject";
    }
    if (behavior === "clarify" || behavior === "route_to_entry") {
        return "clarify_or_route";
    }
    return "direct_answer";
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const byExpectedAction: Record<ExpectedAction, number> = {
        clarify: 0,
        route_to_entry: 0,
        reject: 0,
    };
    const byPredictedBehavior: Record<PipelineBehavior, number> = {
        direct_answer: 0,
        clarify: 0,
        route_to_entry: 0,
        reject: 0,
    };

    let coarseCorrect = 0;
    let exactCorrect = 0;
    let clarifyTotal = 0;
    let clarifyHit = 0;
    let routeTotal = 0;
    let routeHit = 0;
    let routeEntryTopicCorrect = 0;
    let rejectTotal = 0;
    let rejectHit = 0;
    let unsafeDirectAnswer = 0;

    caseReports.forEach((item) => {
        byExpectedAction[item.expected_action] += 1;
        byPredictedBehavior[item.predicted_behavior] += 1;
        if (item.coarse_behavior_correct) {
            coarseCorrect += 1;
        }
        if (item.behavior_correct) {
            exactCorrect += 1;
        }
        if (item.expected_action === "clarify") {
            clarifyTotal += 1;
            if (item.predicted_behavior === "clarify") {
                clarifyHit += 1;
            }
        }
        if (item.expected_action === "route_to_entry") {
            routeTotal += 1;
            if (item.predicted_behavior === "route_to_entry") {
                routeHit += 1;
            }
            if (item.entry_topic_correct) {
                routeEntryTopicCorrect += 1;
            }
        }
        if (item.expected_action === "reject") {
            rejectTotal += 1;
            if (item.predicted_behavior === "reject") {
                rejectHit += 1;
            }
        }
        if (item.unsafe_direct_answer) {
            unsafeDirectAnswer += 1;
        }
    });

    return {
        total: caseReports.length,
        byExpectedAction,
        byPredictedBehavior,
        coarseBehaviorAccuracy: safeRate(coarseCorrect, caseReports.length),
        exactBehaviorAccuracy: safeRate(exactCorrect, caseReports.length),
        clarifyHitRate: safeRate(clarifyHit, clarifyTotal),
        routeHitRate: safeRate(routeHit, routeTotal),
        routeEntryTopicAccuracy: safeRate(routeEntryTopicCorrect, routeTotal),
        rejectHitRate: safeRate(rejectHit, rejectTotal),
        unsafeDirectAnswerRate: safeRate(unsafeDirectAnswer, caseReports.length),
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
            preset: CANONICAL_PIPELINE_PRESET,
        });

        const predictedBehavior = pipelineResult.finalDecision.behavior;
        const expectedCoarseBehavior = toExpectedCoarseBehavior(
            testCase.expected_action,
        );
        const predictedCoarseBehavior = toCoarseBehavior(predictedBehavior);

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            expected_action: testCase.expected_action,
            expected_coarse_behavior: expectedCoarseBehavior,
            predicted_behavior: predictedBehavior,
            predicted_coarse_behavior: predictedCoarseBehavior,
            retrieval_behavior: pipelineResult.retrievalDecision.behavior,
            predicted_entry_topic: pipelineResult.finalDecision.entryTopic,
            expected_entry_topic: testCase.entry_topic,
            behavior_correct: predictedBehavior === testCase.expected_action,
            coarse_behavior_correct:
                predictedCoarseBehavior === expectedCoarseBehavior,
            entry_topic_correct:
                testCase.expected_action === "route_to_entry"
                    ? predictedBehavior === "route_to_entry" &&
                      (pipelineResult.finalDecision.entryTopic || "") ===
                          (testCase.entry_topic || "")
                    : null,
            unsafe_direct_answer: predictedBehavior === "direct_answer",
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
            pipelineVersion: CANONICAL_PIPELINE_PRESET.name,
            preset: CANONICAL_PIPELINE_PRESET,
            note: "当前报告直接调用统一 full pipeline，区分 clarify / route_to_entry / reject 三类行为。",
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

    console.log(`Saved report to ${outputPath}`);
    console.log(
        [
            `coarseBehaviorAccuracy=${(report.summary.coarseBehaviorAccuracy * 100).toFixed(2)}%`,
            `exactBehaviorAccuracy=${(report.summary.exactBehaviorAccuracy * 100).toFixed(2)}%`,
            `clarifyHitRate=${(report.summary.clarifyHitRate * 100).toFixed(2)}%`,
            `routeHitRate=${(report.summary.routeHitRate * 100).toFixed(2)}%`,
            `routeEntryTopicAccuracy=${(report.summary.routeEntryTopicAccuracy * 100).toFixed(2)}%`,
            `rejectHitRate=${(report.summary.rejectHitRate * 100).toFixed(2)}%`,
            `unsafeDirectAnswerRate=${(report.summary.unsafeDirectAnswerRate * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

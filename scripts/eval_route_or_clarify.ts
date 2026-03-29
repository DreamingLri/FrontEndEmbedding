import * as fs from "fs";
import * as path from "path";

import {
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import { getCandidateIndicesForQuery } from "../src/worker/topic_partition.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";

type ExpectedAction = "clarify" | "route_to_entry" | "reject";
type ExpectedBehavior = "clarify_or_route" | "reject";
type PredictedBehavior = "clarify_or_route" | "reject" | "direct_answer";

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
    expected_behavior: ExpectedBehavior;
    predicted_behavior: PredictedBehavior;
    rejection_reason: string | null;
    behavior_correct: boolean;
    unsafe_direct_answer: boolean;
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
    byPredictedBehavior: Record<PredictedBehavior, number>;
    coarseBehaviorAccuracy: number;
    clarifyOrRouteHitRate: number;
    rejectHitRate: number;
    unsafeDirectAnswerRate: number;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetName: string;
    total: number;
    summary: Summary;
    note: string;
    caseReports: CaseReport[];
};

const DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_ROUTE_DATASET_FILE ||
        "../Backend/test/test_dataset_route_or_clarify/test_dataset_route_or_clarify_v1_seed.json",
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
}

function toExpectedBehavior(action: ExpectedAction): ExpectedBehavior {
    return action === "reject" ? "reject" : "clarify_or_route";
}

function toPredictedBehavior(rejectionReason: string | null): PredictedBehavior {
    if (rejectionReason === "weak_anchor_needs_clarification") {
        return "clarify_or_route";
    }
    if (rejectionReason) {
        return "reject";
    }
    return "direct_answer";
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const byExpectedAction: Record<ExpectedAction, number> = {
        clarify: 0,
        route_to_entry: 0,
        reject: 0,
    };
    const byPredictedBehavior: Record<PredictedBehavior, number> = {
        clarify_or_route: 0,
        reject: 0,
        direct_answer: 0,
    };

    let behaviorCorrect = 0;
    let clarifyOrRouteTotal = 0;
    let clarifyOrRouteHit = 0;
    let rejectTotal = 0;
    let rejectHit = 0;
    let unsafeDirectAnswer = 0;

    caseReports.forEach((item) => {
        byExpectedAction[item.expected_action] += 1;
        byPredictedBehavior[item.predicted_behavior] += 1;
        if (item.behavior_correct) behaviorCorrect += 1;
        if (item.expected_behavior === "clarify_or_route") {
            clarifyOrRouteTotal += 1;
            if (item.predicted_behavior === "clarify_or_route") {
                clarifyOrRouteHit += 1;
            }
        }
        if (item.expected_behavior === "reject") {
            rejectTotal += 1;
            if (item.predicted_behavior === "reject") {
                rejectHit += 1;
            }
        }
        if (item.unsafe_direct_answer) {
            unsafeDirectAnswer += 1;
        }
    });

    const total = caseReports.length || 1;
    return {
        total: caseReports.length,
        byExpectedAction,
        byPredictedBehavior,
        coarseBehaviorAccuracy: behaviorCorrect / total,
        clarifyOrRouteHitRate:
            clarifyOrRouteTotal > 0 ? clarifyOrRouteHit / clarifyOrRouteTotal : 0,
        rejectHitRate: rejectTotal > 0 ? rejectHit / rejectTotal : 0,
        unsafeDirectAnswerRate: unsafeDirectAnswer / total,
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
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        testCases.map((item) => item.query),
        engine.dimensions,
    );

    const caseReports: CaseReport[] = [];
    for (let index = 0; index < testCases.length; index++) {
        const testCase = testCases[index];
        const queryIntent = parseQueryIntent(testCase.query);
        const candidateIndices =
            getCandidateIndicesForQuery(queryIntent, engine.topicPartitionIndex);
        const queryWords = dedupe(fmmTokenize(testCase.query, engine.vocabMap));
        const querySparse = getQuerySparse(queryWords, engine.vocabMap);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => engine.vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        const result = searchAndRank({
            queryVector: queryVectors[index],
            querySparse,
            queryYearWordIds,
            queryIntent,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: engine.bm25Stats,
            candidateIndices,
        });

        const rejectionReason = result.rejection?.reason || null;
        const expectedBehavior = toExpectedBehavior(testCase.expected_action);
        const predictedBehavior = toPredictedBehavior(rejectionReason);

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            expected_action: testCase.expected_action,
            expected_behavior: expectedBehavior,
            predicted_behavior: predictedBehavior,
            rejection_reason: rejectionReason,
            behavior_correct: expectedBehavior === predictedBehavior,
            unsafe_direct_answer: predictedBehavior === "direct_answer",
            weak_match_count: result.weakMatches.length,
            match_count: result.matches.length,
            candidate_count: candidateIndices?.length ?? engine.metadataList.length,
            query_intent: {
                years: queryIntent.years,
                topicIds: queryIntent.topicIds,
                intentIds: queryIntent.intentIds,
                preferLatest: queryIntent.preferLatest,
                preferLatestStrong: queryIntent.preferLatestStrong,
                signals: {
                    hasExplicitTopicOrIntent:
                        queryIntent.signals.hasExplicitTopicOrIntent,
                    hasExplicitYear: queryIntent.signals.hasExplicitYear,
                    hasHistoricalHint: queryIntent.signals.hasHistoricalHint,
                    hasStrongDetailAnchor:
                        queryIntent.signals.hasStrongDetailAnchor,
                    hasEntryLikeAnchor:
                        queryIntent.signals.hasEntryLikeAnchor,
                    hasResultState: queryIntent.signals.hasResultState,
                    hasLatestPolicyState:
                        queryIntent.signals.hasLatestPolicyState,
                    hasGenericNextStep:
                        queryIntent.signals.hasGenericNextStep,
                    tokenCount: queryIntent.signals.tokenCount,
                },
            },
            top_matches: result.matches.slice(0, 3).map((match, rank) => ({
                rank: rank + 1,
                otid: match.otid,
                score: match.score,
                best_kpid: match.best_kpid,
            })),
            top_weak_matches: result.weakMatches.slice(0, 3).map((match, rank) => ({
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
        summary: buildSummary(caseReports),
        note: "当前系统只支持 coarse-level 的 clarify_or_route，不区分 clarify 与 route_to_entry 的细粒度动作。",
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
            `clarifyOrRouteHitRate=${(report.summary.clarifyOrRouteHitRate * 100).toFixed(2)}%`,
            `rejectHitRate=${(report.summary.rejectHitRate * 100).toFixed(2)}%`,
            `unsafeDirectAnswerRate=${(report.summary.unsafeDirectAnswerRate * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

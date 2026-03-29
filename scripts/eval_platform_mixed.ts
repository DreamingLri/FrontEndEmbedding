import * as fs from "fs";
import * as path from "path";

import {
    DIRECT_ANSWER_EVIDENCE_TERMS,
    DEFAULT_WEIGHTS,
    getQuerySparse,
    parseQueryIntent,
    QUERY_SCOPE_SPECIFICITY_TERMS,
    searchAndRank,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import { getCandidateIndicesForQuery } from "../src/worker/topic_partition.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";

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

type PredictedBehavior = "direct_answer" | "clarify_or_route" | "reject";

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
    predicted_behavior: PredictedBehavior;
    behavior_correct: boolean;
    success: boolean;
    rejection_reason: string | null;
    expected_otid?: string;
    expected_latest_otid?: string;
    expected_entry_topic?: string;
    doc_rank: number | null;
    doc_hit_at_1: boolean;
    doc_hit_at_3: boolean;
    doc_hit_at_5: boolean;
    unsafe_direct_answer: boolean;
    candidate_count: number;
    weak_match_count: number;
    match_count: number;
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
    directAnswerRate?: number;
    rejectRate?: number;
    clarifyOrRouteRate?: number;
    unsafeDirectAnswerRate?: number;
    docHitAt1Rate?: number;
    docHitAt3Rate?: number;
    docHitAt5Rate?: number;
};

type Summary = {
    total: number;
    overallBehaviorAccuracy: number;
    platformTaskSuccess: number;
    directAnswerSuccessRate: RateSummary;
    latestTopicSuccessRate: RateSummary;
    clarifyOrRouteSuccessRate: RateSummary;
    routeFamilyBehaviorAccuracy: RateSummary;
    rejectHitRate: RateSummary;
    unsafeDirectAnswerRate: RateSummary;
    directAnswerDocHits: DocHitSummary;
    complexDirectSubsetDocHits: DocHitSummary;
    byScenarioFamily: Record<ScenarioFamily, ScenarioFamilySummary>;
    byPredictedBehavior: Record<PredictedBehavior, number>;
    failureBreakdown: Record<string, number>;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetName: string;
    total: number;
    config: {
        defaultWeights: typeof DEFAULT_WEIGHTS;
        note: string;
    };
    summary: Summary;
    caseReports: CaseReport[];
};

const DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_PLATFORM_MIXED_DATASET_FILE ||
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_mixed_daily_v1_reviewed.json",
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
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

function toExpectedRouteBehavior(
    expectedBehavior: ExpectedBehavior,
): PredictedBehavior {
    return expectedBehavior === "reject" ? "reject" : "clarify_or_route";
}

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function buildDocHitSummary(caseReports: CaseReport[]): DocHitSummary {
    const total = caseReports.length;
    const hitAt1 = caseReports.filter((item) => item.doc_hit_at_1).length;
    const hitAt3 = caseReports.filter((item) => item.doc_hit_at_3).length;
    const hitAt5 = caseReports.filter((item) => item.doc_hit_at_5).length;

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
    const predictedDirectAnswer = caseReports.filter(
        (item) => item.predicted_behavior === "direct_answer",
    ).length;
    const predictedReject = caseReports.filter(
        (item) => item.predicted_behavior === "reject",
    ).length;
    const predictedClarifyOrRoute = caseReports.filter(
        (item) => item.predicted_behavior === "clarify_or_route",
    ).length;

    const base: ScenarioFamilySummary = {
        total,
        successCount,
        successRate: safeRate(successCount, total),
        directAnswerRate: safeRate(predictedDirectAnswer, total),
        rejectRate: safeRate(predictedReject, total),
        clarifyOrRouteRate: safeRate(predictedClarifyOrRoute, total),
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
    };
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const total = caseReports.length;
    const overallSuccess = caseReports.filter((item) => item.success).length;

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

    const byPredictedBehavior: Record<PredictedBehavior, number> = {
        direct_answer: 0,
        clarify_or_route: 0,
        reject: 0,
    };
    const failureBreakdown: Record<string, number> = {
        direct_answer_wrong_doc: 0,
        direct_answer_rejected: 0,
        route_expected_clarify_or_route_but_rejected: 0,
        route_expected_clarify_or_route_but_direct_answer: 0,
        route_expected_reject_but_clarify_or_route: 0,
        route_expected_reject_but_direct_answer: 0,
    };

    caseReports.forEach((item) => {
        byPredictedBehavior[item.predicted_behavior] += 1;
        if (item.success) {
            return;
        }

        if (item.scenario_family === "route_or_clarify") {
            if (item.expected_behavior === "reject") {
                if (item.predicted_behavior === "clarify_or_route") {
                    failureBreakdown.route_expected_reject_but_clarify_or_route += 1;
                } else if (item.predicted_behavior === "direct_answer") {
                    failureBreakdown.route_expected_reject_but_direct_answer += 1;
                }
            } else if (item.predicted_behavior === "reject") {
                failureBreakdown.route_expected_clarify_or_route_but_rejected += 1;
            } else if (item.predicted_behavior === "direct_answer") {
                failureBreakdown.route_expected_clarify_or_route_but_direct_answer += 1;
            }
            return;
        }

        if (item.predicted_behavior === "direct_answer") {
            failureBreakdown.direct_answer_wrong_doc += 1;
        } else {
            failureBreakdown.direct_answer_rejected += 1;
        }
    });

    return {
        total,
        overallBehaviorAccuracy: safeRate(overallSuccess, total),
        platformTaskSuccess: safeRate(overallSuccess, total),
        directAnswerSuccessRate: buildRateSummary(
            directAnswerFamilies.filter((item) => item.success).length,
            directAnswerFamilies.length,
        ),
        latestTopicSuccessRate: buildRateSummary(
            latestFamily.filter((item) => item.success).length,
            latestFamily.length,
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
        directAnswerDocHits: buildDocHitSummary(docTargetCases),
        complexDirectSubsetDocHits: buildDocHitSummary(complexDirectCases),
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
    const scopeSpecificityWordIdToTerm = new Map<number, string>();
    QUERY_SCOPE_SPECIFICITY_TERMS.forEach((term) => {
        const wordId = engine.vocabMap.get(term);
        if (wordId !== undefined) {
            scopeSpecificityWordIdToTerm.set(wordId, term);
        }
    });
    const directAnswerEvidenceWordIdToTerm = new Map<number, string>();
    DIRECT_ANSWER_EVIDENCE_TERMS.forEach((term) => {
        const wordId = engine.vocabMap.get(term);
        if (wordId !== undefined) {
            directAnswerEvidenceWordIdToTerm.set(wordId, term);
        }
    });
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        testCases.map((item) => item.query),
        engine.dimensions,
    );

    const caseReports: CaseReport[] = [];

    for (let index = 0; index < testCases.length; index++) {
        const testCase = testCases[index];
        const queryIntent = parseQueryIntent(testCase.query);
        const candidateIndices = getCandidateIndicesForQuery(
            queryIntent,
            engine.topicPartitionIndex,
        );
        const queryWords = dedupe(fmmTokenize(testCase.query, engine.vocabMap));
        const querySparse = getQuerySparse(queryWords, engine.vocabMap);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => engine.vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        const result = searchAndRank({
            queryVector: queryVectors[index],
            querySparse,
            queryWords,
            queryYearWordIds,
            queryIntent,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: engine.bm25Stats,
            candidateIndices,
            scopeSpecificityWordIdToTerm,
            directAnswerEvidenceWordIdToTerm,
        });

        const rejectionReason = result.rejection?.reason || null;
        const predictedBehavior = toPredictedBehavior(rejectionReason);
        const expectedDocOtid =
            testCase.scenario_family === "latest_within_topic"
                ? testCase.expected_latest_otid || testCase.expected_otid
                : testCase.expected_otid;
        const docRank =
            expectedDocOtid
                ? result.matches.findIndex((match) => match.otid === expectedDocOtid) + 1
                : 0;
        const normalizedDocRank = docRank > 0 ? docRank : null;
        const docHitAt1 = normalizedDocRank === 1;
        const docHitAt3 =
            normalizedDocRank !== null && normalizedDocRank <= 3;
        const docHitAt5 =
            normalizedDocRank !== null && normalizedDocRank <= 5;

        const behaviorCorrect =
            testCase.scenario_family === "route_or_clarify"
                ? predictedBehavior ===
                  toExpectedRouteBehavior(testCase.expected_behavior)
                : predictedBehavior === "direct_answer";
        const success =
            testCase.scenario_family === "route_or_clarify"
                ? behaviorCorrect
                : behaviorCorrect && docHitAt1;

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            query_type: testCase.query_type,
            scenario_family: testCase.scenario_family,
            expected_behavior: testCase.expected_behavior,
            predicted_behavior: predictedBehavior,
            behavior_correct: behaviorCorrect,
            success,
            rejection_reason: rejectionReason,
            expected_otid: testCase.expected_otid,
            expected_latest_otid: testCase.expected_latest_otid,
            expected_entry_topic: testCase.expected_entry_topic,
            doc_rank: normalizedDocRank,
            doc_hit_at_1: docHitAt1,
            doc_hit_at_3: docHitAt3,
            doc_hit_at_5: docHitAt5,
            unsafe_direct_answer:
                testCase.scenario_family === "route_or_clarify" &&
                predictedBehavior === "direct_answer",
            candidate_count:
                candidateIndices?.length ?? engine.metadataList.length,
            weak_match_count: result.weakMatches.length,
            match_count: result.matches.length,
            top_matches: result.matches.slice(0, 5).map((match, rank) => ({
                rank: rank + 1,
                otid: match.otid,
                score: match.score,
                best_kpid: match.best_kpid,
            })),
            top_weak_matches: result.weakMatches
                .slice(0, 5)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
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
            defaultWeights: DEFAULT_WEIGHTS,
            note: "当前报告评测的是现有前端默认检索与拒答机制；route_or_clarify 仍按 coarse-level 的 clarify_or_route vs reject 口径统计。",
        },
        summary: buildSummary(caseReports),
        caseReports,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        `platform_mixed_${datasetName}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Saved report to ${outputPath}`);
    console.log(
        [
            `overallBehaviorAccuracy=${(report.summary.overallBehaviorAccuracy * 100).toFixed(2)}%`,
            `platformTaskSuccess=${(report.summary.platformTaskSuccess * 100).toFixed(2)}%`,
            `directAnswerSuccessRate=${(report.summary.directAnswerSuccessRate.rate * 100).toFixed(2)}%`,
            `latestTopicSuccessRate=${(report.summary.latestTopicSuccessRate.rate * 100).toFixed(2)}%`,
            `clarifyOrRouteSuccessRate=${(report.summary.clarifyOrRouteSuccessRate.rate * 100).toFixed(2)}%`,
            `rejectHitRate=${(report.summary.rejectHitRate.rate * 100).toFixed(2)}%`,
            `unsafeDirectAnswerRate=${(report.summary.unsafeDirectAnswerRate.rate * 100).toFixed(2)}%`,
            `directAnswerDocHit@1=${(report.summary.directAnswerDocHits.hitAt1Rate * 100).toFixed(2)}%`,
            `directAnswerDocHit@3=${(report.summary.directAnswerDocHits.hitAt3Rate * 100).toFixed(2)}%`,
            `complexDirectHit@1=${(report.summary.complexDirectSubsetDocHits.hitAt1Rate * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

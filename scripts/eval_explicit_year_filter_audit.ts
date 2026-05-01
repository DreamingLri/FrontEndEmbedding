import * as fs from "fs";
import * as path from "path";

import {
    searchAndRank,
    BM25_B,
    BM25_K1,
    getQuerySparse,
    resolveDocOtid,
    resolveMetadataTopicIds,
    RRF_K,
    RRF_RANK_LIMIT,
    dotProduct,
    selectTopLocalIndices,
    type Metadata,
    type SearchResult,
    type BM25Stats,
} from "../src/worker/vector_engine.ts";
import {
    createAggregatedDocScores,
    mergeAggregatedDocMetadata,
    applyScoreToAggregatedDocScores,
    type AggregatedDocScores,
} from "../src/worker/aggregated_doc_scores.ts";
import {
    buildSearchPipelineQueryContext,
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
} from "../src/worker/search_pipeline.ts";
import {
    createQueryIntentContext,
    getDocQuerySignals,
    shouldSkipForExplicitYear,
} from "../src/worker/vector_engine/search_context.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import {
    ACTIVE_MAIN_DB_VERSION,
    loadDataset,
    type EvalDatasetCase,
    type OtidEvalMode,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { resolveBackendArticlesFile } from "./kb_version_paths.ts";

type DatasetDefinition = {
    key: string;
    label: string;
    file: string;
};

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

type QueryPreparedState = {
    otidMap: Record<string, AggregatedDocScores>;
    yearHitMap: Map<string, boolean>;
    queryWords: string[];
    queryYearWordIds: number[];
};

type QueryEvaluationSummary = {
    noFilterRank: number | null;
    withFilterRank: number | null;
    goldReachableBeforeFilter: boolean;
    goldRetainedAfterFilter: boolean;
    goldRemovedByFilter: boolean;
};

type QueryFilterAudit = {
    caseId: string;
    dataset: string;
    query: string;
    queryYears: number[];
    expectedOtid: string;
    expectedTitle: string;
    expectedTargetYear?: number;
    expectedPublishYear?: number;
    expectedLexicalYearMatch: boolean;
    expectedStructuredYearMatch: boolean;
    expectedPublishYearMatch: boolean;
    expectedSuspiciousStructuredYear: boolean;
    expectedSkip: boolean;
    expectedKeepReason: "structured" | "publish" | "suspicious_lexical" | "none";
    docsBeforeFilter: number;
    docsAfterFilter: number;
    docsRemoved: number;
    top1NoFilter?: {
        otid: string;
        title: string;
        score: number;
        removedByFilter: boolean;
        targetYear?: number;
        publishYear?: number;
    };
    top1WithFilter?: {
        otid: string;
        title: string;
        score: number;
        targetYear?: number;
        publishYear?: number;
    };
    ranking: QueryEvaluationSummary;
};

type DatasetAuditSummary = {
    dataset: string;
    totalCases: number;
    explicitYearQueries: number;
    explicitYearRate: number;
    avgDocsBeforeFilter: number;
    avgDocsAfterFilter: number;
    avgDocsRemoved: number;
    queriesWithAnyRemoval: number;
    hitAt1NoFilter: number;
    hitAt1WithFilter: number;
    mrrNoFilter: number;
    mrrWithFilter: number;
    goldReachableBeforeFilter: number;
    goldRetainedAfterFilter: number;
    goldRemovedByFilter: number;
    goldMissingBeforeFilter: number;
    expectedPublishFallbackKeeps: number;
    expectedSuspiciousLexicalKeeps: number;
    top1YearConflictRemoved: number;
    top1RescuedToCorrect: number;
};

type AuditExample = {
    dataset: string;
    caseId: string;
    query: string;
    expectedTitle: string;
    expectedOtid: string;
    expectedTargetYear?: number;
    expectedPublishYear?: number;
    noFilterRank: number | null;
    withFilterRank: number | null;
    top1NoFilterTitle?: string;
    top1WithFilterTitle?: string;
};

type AuditReport = {
    generatedAt: string;
    mainDbVersion: string;
    pipelinePresetName: string;
    datasetFiles: Record<string, string>;
    datasets: DatasetAuditSummary[];
    combined: DatasetAuditSummary;
    publishYearFallbackExamples: AuditExample[];
    suspiciousStructuredYearExamples: AuditExample[];
    top1RescueExamples: AuditExample[];
    goldRemovedExamples: AuditExample[];
};

type ArticleRecord = {
    otid?: string;
    ot_title?: string;
    publish_time?: string;
    link?: string;
};

const DATASETS: DatasetDefinition[] = [
    {
        key: "main",
        label: "Main",
        file: CURRENT_EVAL_DATASET_FILES.granularityMain120,
    },
    {
        key: "in_domain",
        label: "InDomain",
        file: CURRENT_EVAL_DATASET_FILES.granularityInDomainGeneralization100,
    },
    {
        key: "ext_ood",
        label: "ExtOOD",
        file: CURRENT_EVAL_DATASET_FILES.granularityExtOod985Aligned100,
    },
];

const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function roundMetric(value: number): number {
    return Number.parseFloat(value.toFixed(4));
}

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
}

function parsePublishYearString(value?: string): number | undefined {
    if (!value) return undefined;
    const match = value.match(/(20\d{2})/);
    return match ? Number.parseInt(match[1] || "", 10) : undefined;
}

function loadArticleMap(): Map<string, ArticleRecord> {
    const absolutePath = path.resolve(process.cwd(), resolveBackendArticlesFile());
    const raw = JSON.parse(fs.readFileSync(absolutePath, "utf-8")) as ArticleRecord[];
    return new Map(
        raw
            .filter((item) => item.otid)
            .map((item) => [item.otid as string, item]),
    );
}

function getTitleForOtid(
    otid: string,
    articleMap: Map<string, ArticleRecord>,
    fallback?: string,
): string {
    return articleMap.get(otid)?.ot_title || fallback || otid;
}

function normalizeOtidEvalTarget(testCase: EvalDatasetCase): NormalizedOtidEvalTarget {
    const mode = testCase.otid_eval_mode || "single_expected";
    if (mode === "acceptable_otids") {
        return {
            mode,
            acceptableOtids: dedupe([
                testCase.expected_otid,
                ...(testCase.acceptable_otids || []),
            ]),
            requiredOtidGroups: [],
            minGroupsToCover: 1,
        };
    }
    if (mode === "required_otid_groups") {
        const requiredOtidGroups = (testCase.required_otid_groups || [])
            .map((group) => dedupe(group.filter(Boolean)))
            .filter((group) => group.length > 0);
        return {
            mode,
            acceptableOtids: [],
            requiredOtidGroups,
            minGroupsToCover: Math.max(
                1,
                Math.min(
                    testCase.min_otid_groups_to_cover || requiredOtidGroups.length || 1,
                    requiredOtidGroups.length || 1,
                ),
            ),
        };
    }
    return {
        mode: "single_expected",
        acceptableOtids: [testCase.expected_otid],
        requiredOtidGroups: [],
        minGroupsToCover: 1,
    };
}

function getRankForTarget(
    ranking: readonly SearchResult[],
    testCase: EvalDatasetCase,
): number | null {
    const target = normalizeOtidEvalTarget(testCase);
    if (target.mode !== "required_otid_groups") {
        const accepted = new Set(target.acceptableOtids);
        const index = ranking.findIndex((item) => accepted.has(item.otid));
        return index >= 0 ? index + 1 : null;
    }

    const covered = new Set<number>();
    for (let index = 0; index < ranking.length; index += 1) {
        const otid = ranking[index]?.otid;
        if (!otid) continue;
        target.requiredOtidGroups.forEach((group, groupIndex) => {
            if (!covered.has(groupIndex) && group.includes(otid)) {
                covered.add(groupIndex);
            }
        });
        if (covered.size >= target.minGroupsToCover) {
            return index + 1;
        }
    }
    return null;
}

function buildPreparedState(params: {
    metadataList: Metadata[];
    vectorMatrix: Int8Array;
    dimensions: number;
    queryVector: Float32Array;
    querySparse: Record<number, number>;
    queryYearWordIds: number[];
    bm25Stats: BM25Stats;
    candidateIndices?: number[];
}): QueryPreparedState {
    const {
        metadataList,
        vectorMatrix,
        dimensions,
        queryVector,
        querySparse,
        queryYearWordIds,
        bm25Stats,
        candidateIndices,
    } = params;

    const activeCandidateIndices =
        candidateIndices && candidateIndices.length > 0 ? candidateIndices : undefined;
    const candidateCount = activeCandidateIndices
        ? activeCandidateIndices.length
        : metadataList.length;
    const denseScores = new Float32Array(candidateCount);
    const sparseScores = new Float32Array(candidateCount);
    const yearHitMap = new Map<string, boolean>();
    const queryYearWordIdSet =
        queryYearWordIds.length > 0 ? new Set(queryYearWordIds) : undefined;

    for (let localIndex = 0; localIndex < candidateCount; localIndex += 1) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[localIndex]
            : localIndex;
        const meta = metadataList[metaIndex];

        let dense = dotProduct(
            queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null) {
            dense *= meta.scale;
        }
        denseScores[localIndex] = dense;

        let sparse = 0;
        if (meta.sparse && meta.sparse.length > 0) {
            const dl = bm25Stats.docLengths[metaIndex];
            const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);
            for (let sparseIndex = 0; sparseIndex < meta.sparse.length; sparseIndex += 2) {
                const wordId = meta.sparse[sparseIndex] as number;
                const tf = meta.sparse[sparseIndex + 1] as number;

                if (queryYearWordIdSet?.has(wordId)) {
                    yearHitMap.set(resolveDocOtid(meta), true);
                }
                if (querySparse[wordId]) {
                    const qWeight = querySparse[wordId] || 1;
                    const idf = bm25Stats.idfMap.get(wordId) || 0;
                    const numerator = tf * (BM25_K1 + 1);
                    const denominator =
                        tf +
                        BM25_K1 *
                            (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
                    sparse += qWeight * idf * (numerator / denominator);
                }
            }
        }
        sparseScores[localIndex] = sparse;
    }

    const rrfRankLimit = Math.min(RRF_RANK_LIMIT, candidateCount);
    const denseTopLocalIndices = selectTopLocalIndices(denseScores, rrfRankLimit);
    const rrfScores = new Map<Metadata, number>();

    for (let rank = 0; rank < denseTopLocalIndices.length; rank += 1) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[denseTopLocalIndices[rank] as number]
            : (denseTopLocalIndices[rank] as number);
        const meta = metadataList[metaIndex];
        rrfScores.set(meta, (1 / (rank + RRF_K)) * 100);
    }

    const sparseTopLocalIndices = selectTopLocalIndices(
        sparseScores,
        rrfRankLimit,
        { minimumScoreExclusive: 0 },
    );
    for (let rank = 0; rank < sparseTopLocalIndices.length; rank += 1) {
        const localIndex = sparseTopLocalIndices[rank] as number;
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[localIndex]
            : localIndex;
        const meta = metadataList[metaIndex];
        const current = rrfScores.get(meta) || 0;
        rrfScores.set(meta, current + (1 / (rank + RRF_K)) * 120);
    }

    const topHybrid = Array.from(rrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.topHybridLimit);

    const otidMap: Record<string, AggregatedDocScores> = {};
    for (const [meta, score] of topHybrid) {
        const otid = resolveDocOtid(meta);
        const topicIds = resolveMetadataTopicIds(meta);
        if (!otidMap[otid]) {
            otidMap[otid] = createAggregatedDocScores(meta, topicIds);
        }
        mergeAggregatedDocMetadata(otidMap[otid], meta, topicIds);
        applyScoreToAggregatedDocScores(otidMap[otid], meta, score);
    }

    return {
        otidMap,
        yearHitMap,
        queryWords: [],
        queryYearWordIds,
    };
}

function buildCombinedSummary(
    datasets: DatasetAuditSummary[],
): DatasetAuditSummary {
    const totalCases = datasets.reduce((sum, item) => sum + item.totalCases, 0);
    const explicitYearQueries = datasets.reduce(
        (sum, item) => sum + item.explicitYearQueries,
        0,
    );
    const combined: DatasetAuditSummary = {
        dataset: "Combined",
        totalCases,
        explicitYearQueries,
        explicitYearRate: roundMetric(safeRate(explicitYearQueries, totalCases)),
        avgDocsBeforeFilter: 0,
        avgDocsAfterFilter: 0,
        avgDocsRemoved: 0,
        queriesWithAnyRemoval: datasets.reduce(
            (sum, item) => sum + item.queriesWithAnyRemoval,
            0,
        ),
        hitAt1NoFilter: 0,
        hitAt1WithFilter: 0,
        mrrNoFilter: 0,
        mrrWithFilter: 0,
        goldReachableBeforeFilter: datasets.reduce(
            (sum, item) => sum + item.goldReachableBeforeFilter,
            0,
        ),
        goldRetainedAfterFilter: datasets.reduce(
            (sum, item) => sum + item.goldRetainedAfterFilter,
            0,
        ),
        goldRemovedByFilter: datasets.reduce(
            (sum, item) => sum + item.goldRemovedByFilter,
            0,
        ),
        goldMissingBeforeFilter: datasets.reduce(
            (sum, item) => sum + item.goldMissingBeforeFilter,
            0,
        ),
        expectedPublishFallbackKeeps: datasets.reduce(
            (sum, item) => sum + item.expectedPublishFallbackKeeps,
            0,
        ),
        expectedSuspiciousLexicalKeeps: datasets.reduce(
            (sum, item) => sum + item.expectedSuspiciousLexicalKeeps,
            0,
        ),
        top1YearConflictRemoved: datasets.reduce(
            (sum, item) => sum + item.top1YearConflictRemoved,
            0,
        ),
        top1RescuedToCorrect: datasets.reduce(
            (sum, item) => sum + item.top1RescuedToCorrect,
            0,
        ),
    };

    if (explicitYearQueries > 0) {
        const weightedDocsBefore = datasets.reduce(
            (sum, item) => sum + item.avgDocsBeforeFilter * item.explicitYearQueries,
            0,
        );
        const weightedDocsAfter = datasets.reduce(
            (sum, item) => sum + item.avgDocsAfterFilter * item.explicitYearQueries,
            0,
        );
        const weightedDocsRemoved = datasets.reduce(
            (sum, item) => sum + item.avgDocsRemoved * item.explicitYearQueries,
            0,
        );
        combined.avgDocsBeforeFilter = roundMetric(
            weightedDocsBefore / explicitYearQueries,
        );
        combined.avgDocsAfterFilter = roundMetric(
            weightedDocsAfter / explicitYearQueries,
        );
        combined.avgDocsRemoved = roundMetric(
            weightedDocsRemoved / explicitYearQueries,
        );

        const weightedHit1NoFilter = datasets.reduce(
            (sum, item) => sum + item.hitAt1NoFilter * item.explicitYearQueries,
            0,
        );
        const weightedHit1WithFilter = datasets.reduce(
            (sum, item) => sum + item.hitAt1WithFilter * item.explicitYearQueries,
            0,
        );
        const weightedMrrNoFilter = datasets.reduce(
            (sum, item) => sum + item.mrrNoFilter * item.explicitYearQueries,
            0,
        );
        const weightedMrrWithFilter = datasets.reduce(
            (sum, item) => sum + item.mrrWithFilter * item.explicitYearQueries,
            0,
        );
        combined.hitAt1NoFilter = roundMetric(
            weightedHit1NoFilter / explicitYearQueries,
        );
        combined.hitAt1WithFilter = roundMetric(
            weightedHit1WithFilter / explicitYearQueries,
        );
        combined.mrrNoFilter = roundMetric(weightedMrrNoFilter / explicitYearQueries);
        combined.mrrWithFilter = roundMetric(
            weightedMrrWithFilter / explicitYearQueries,
        );
    }

    return combined;
}

async function main() {
    fs.mkdirSync(RESULTS_DIR, { recursive: true });

    const engine = await loadFrontendEvalEngine();
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    void termMaps;
    const articleMap = loadArticleMap();
    const datasetFiles: Record<string, string> = {};
    const datasetSummaries: DatasetAuditSummary[] = [];
    const publishYearFallbackExamples: AuditExample[] = [];
    const suspiciousStructuredYearExamples: AuditExample[] = [];
    const top1RescueExamples: AuditExample[] = [];
    const goldRemovedExamples: AuditExample[] = [];

    for (const dataset of DATASETS) {
        const cases = loadDataset(dataset.file, { datasetLabel: dataset.label });
        datasetFiles[dataset.label] = dataset.file;
        const queryVectors = await embedFrontendQueries(
            engine.extractor,
            cases.map((item) => item.query),
            engine.dimensions,
        );

        const audits: QueryFilterAudit[] = [];

        for (let index = 0; index < cases.length; index += 1) {
            const testCase = cases[index]!;
            const queryVector = queryVectors[index]!;
            const queryContext = buildSearchPipelineQueryContext(
                testCase.query,
                engine.vocabMap,
                engine.topicPartitionIndex,
                FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
            );
            const queryIntent = queryContext.queryIntent;
            if ((queryIntent.years || []).length === 0) {
                continue;
            }

            const queryWords =
                queryContext.queryWords.length > 0
                    ? queryContext.queryWords
                    : dedupe(fmmTokenize(testCase.query, engine.vocabMap));
            const querySparse = getQuerySparse(queryWords, engine.vocabMap);
            const preparedState = buildPreparedState({
                metadataList: engine.metadataList,
                vectorMatrix: engine.vectorMatrix,
                dimensions: engine.dimensions,
                queryVector,
                querySparse,
                queryYearWordIds: queryContext.queryYearWordIds,
                bm25Stats: engine.bm25Stats,
                candidateIndices: queryContext.candidateIndices,
            });

            const intentContext = createQueryIntentContext(queryIntent, queryWords);
            const expectedScores = preparedState.otidMap[testCase.expected_otid];
            const expectedSignals = expectedScores
                ? getDocQuerySignals(
                      testCase.expected_otid,
                      expectedScores,
                      intentContext,
                      preparedState.yearHitMap,
                  )
                : undefined;
            const expectedSkip = expectedScores
                ? shouldSkipForExplicitYear(
                      expectedScores,
                      intentContext,
                      expectedSignals!,
                  )
                : false;
            const expectedKeepReason: QueryFilterAudit["expectedKeepReason"] =
                !expectedScores || expectedSkip
                    ? "none"
                    : expectedSignals?.hasStructuredYearMatch
                      ? "structured"
                      : expectedSignals?.hasPublishYearMatch
                        ? "publish"
                        : expectedSignals?.hasLexicalYearMatch &&
                            expectedSignals?.hasSuspiciousStructuredYear
                          ? "suspicious_lexical"
                          : "none";

            const removedOtids = Object.entries(preparedState.otidMap)
                .filter(([otid, scores]) => {
                    const signals = getDocQuerySignals(
                        otid,
                        scores,
                        intentContext,
                        preparedState.yearHitMap,
                    );
                    return shouldSkipForExplicitYear(scores, intentContext, signals);
                })
                .map(([otid]) => otid);
            const removedOtidSet = new Set(removedOtids);
            const docsBeforeFilter = Object.keys(preparedState.otidMap).length;
            const docsAfterFilter = docsBeforeFilter - removedOtids.length;

            const noFilterRanking = searchAndRank({
                queryVector,
                querySparse: queryContext.querySparse,
                queryWords: queryContext.queryWords,
                queryYearWordIds: queryContext.queryYearWordIds,
                queryIntent: queryContext.queryIntent,
                metadata: engine.metadataList,
                vectorMatrix: engine.vectorMatrix,
                dimensions: engine.dimensions,
                currentTimestamp: Date.now() / 1000,
                bm25Stats: engine.bm25Stats,
                candidateIndices: queryContext.candidateIndices,
                scopeSpecificityWordIdToTerm:
                    termMaps.scopeSpecificityWordIdToTerm,
                weights: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.weights,
                topHybridLimit:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.topHybridLimit,
                kpAggregationMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpAggregationMode,
                kpTopN: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTopN,
                kpTailWeight:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTailWeight,
                lexicalBonusMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.lexicalBonusMode,
                enableLexicalBonusBoost:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.enableLexicalBonusBoost,
                kpRoleRerankMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpRoleRerankMode,
                kpRoleDocWeight:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpRoleDocWeight,
                qConfusionMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionMode,
                qConfusionWeight:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionWeight,
                enableExplicitYearFilter: false,
                queryPlan: queryContext.queryPlan,
                enableQueryPlanner:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.enableQueryPlanner,
                minimalMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.minimalMode,
            });
            const withFilterRanking = searchAndRank({
                queryVector,
                querySparse: queryContext.querySparse,
                queryWords: queryContext.queryWords,
                queryYearWordIds: queryContext.queryYearWordIds,
                queryIntent: queryContext.queryIntent,
                metadata: engine.metadataList,
                vectorMatrix: engine.vectorMatrix,
                dimensions: engine.dimensions,
                currentTimestamp: Date.now() / 1000,
                bm25Stats: engine.bm25Stats,
                candidateIndices: queryContext.candidateIndices,
                scopeSpecificityWordIdToTerm:
                    termMaps.scopeSpecificityWordIdToTerm,
                weights: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.weights,
                topHybridLimit:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.topHybridLimit,
                kpAggregationMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpAggregationMode,
                kpTopN: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTopN,
                kpTailWeight:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTailWeight,
                lexicalBonusMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.lexicalBonusMode,
                enableLexicalBonusBoost:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.enableLexicalBonusBoost,
                kpRoleRerankMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpRoleRerankMode,
                kpRoleDocWeight:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpRoleDocWeight,
                qConfusionMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionMode,
                qConfusionWeight:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionWeight,
                enableExplicitYearFilter: true,
                queryPlan: queryContext.queryPlan,
                enableQueryPlanner:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.enableQueryPlanner,
                minimalMode:
                    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.minimalMode,
            });

            const noFilterRank = getRankForTarget(noFilterRanking.matches, testCase);
            const withFilterRank = getRankForTarget(withFilterRanking.matches, testCase);
            const ranking: QueryEvaluationSummary = {
                noFilterRank,
                withFilterRank,
                goldReachableBeforeFilter: noFilterRank !== null,
                goldRetainedAfterFilter: withFilterRank !== null,
                goldRemovedByFilter: noFilterRank !== null && withFilterRank === null,
            };

            const top1NoFilter = noFilterRanking.matches[0];
            const top1WithFilter = withFilterRanking.matches[0];
            const top1NoFilterScores = top1NoFilter
                ? preparedState.otidMap[top1NoFilter.otid]
                : undefined;
            const top1WithFilterScores = top1WithFilter
                ? preparedState.otidMap[top1WithFilter.otid]
                : undefined;

            audits.push({
                caseId: testCase.id || `${dataset.key}_${index + 1}`,
                dataset: dataset.label,
                query: testCase.query,
                queryYears: queryIntent.years || [],
                expectedOtid: testCase.expected_otid,
                expectedTitle: getTitleForOtid(
                    testCase.expected_otid,
                    articleMap,
                    (testCase as { ot_title?: string }).ot_title,
                ),
                expectedTargetYear: expectedScores?.target_year,
                expectedPublishYear:
                    expectedSignals?.docPublishYear ||
                    parsePublishYearString(
                        articleMap.get(testCase.expected_otid)?.publish_time,
                    ),
                expectedLexicalYearMatch:
                    expectedSignals?.hasLexicalYearMatch || false,
                expectedStructuredYearMatch:
                    expectedSignals?.hasStructuredYearMatch || false,
                expectedPublishYearMatch:
                    expectedSignals?.hasPublishYearMatch || false,
                expectedSuspiciousStructuredYear:
                    expectedSignals?.hasSuspiciousStructuredYear || false,
                expectedSkip,
                expectedKeepReason,
                docsBeforeFilter,
                docsAfterFilter,
                docsRemoved: removedOtids.length,
                top1NoFilter: top1NoFilter
                    ? {
                          otid: top1NoFilter.otid,
                          title: getTitleForOtid(top1NoFilter.otid, articleMap),
                          score: roundMetric(top1NoFilter.score),
                          removedByFilter: removedOtidSet.has(top1NoFilter.otid),
                          targetYear: top1NoFilterScores?.target_year,
                          publishYear: top1NoFilterScores
                              ? getDocQuerySignals(
                                    top1NoFilter.otid,
                                    top1NoFilterScores,
                                    intentContext,
                                    preparedState.yearHitMap,
                                ).docPublishYear
                              : undefined,
                      }
                    : undefined,
                top1WithFilter: top1WithFilter
                    ? {
                          otid: top1WithFilter.otid,
                          title: getTitleForOtid(top1WithFilter.otid, articleMap),
                          score: roundMetric(top1WithFilter.score),
                          targetYear: top1WithFilterScores?.target_year,
                          publishYear: top1WithFilterScores
                              ? getDocQuerySignals(
                                    top1WithFilter.otid,
                                    top1WithFilterScores,
                                    intentContext,
                                    preparedState.yearHitMap,
                                ).docPublishYear
                              : undefined,
                      }
                    : undefined,
                ranking,
            });
        }

        const explicitYearQueries = audits.length;
        const hitAt1NoFilterCount = audits.filter(
            (item) => item.ranking.noFilterRank === 1,
        ).length;
        const hitAt1WithFilterCount = audits.filter(
            (item) => item.ranking.withFilterRank === 1,
        ).length;
        const mrrNoFilter = audits.reduce(
            (sum, item) => sum + 1 / (item.ranking.noFilterRank || Number.POSITIVE_INFINITY),
            0,
        );
        const mrrWithFilter = audits.reduce(
            (sum, item) =>
                sum + 1 / (item.ranking.withFilterRank || Number.POSITIVE_INFINITY),
            0,
        );

        const summary: DatasetAuditSummary = {
            dataset: dataset.label,
            totalCases: cases.length,
            explicitYearQueries,
            explicitYearRate: roundMetric(safeRate(explicitYearQueries, cases.length)),
            avgDocsBeforeFilter: roundMetric(
                audits.reduce((sum, item) => sum + item.docsBeforeFilter, 0) /
                    Math.max(explicitYearQueries, 1),
            ),
            avgDocsAfterFilter: roundMetric(
                audits.reduce((sum, item) => sum + item.docsAfterFilter, 0) /
                    Math.max(explicitYearQueries, 1),
            ),
            avgDocsRemoved: roundMetric(
                audits.reduce((sum, item) => sum + item.docsRemoved, 0) /
                    Math.max(explicitYearQueries, 1),
            ),
            queriesWithAnyRemoval: audits.filter((item) => item.docsRemoved > 0).length,
            hitAt1NoFilter: roundMetric(safeRate(hitAt1NoFilterCount, explicitYearQueries)),
            hitAt1WithFilter: roundMetric(
                safeRate(hitAt1WithFilterCount, explicitYearQueries),
            ),
            mrrNoFilter: roundMetric(mrrNoFilter / Math.max(explicitYearQueries, 1)),
            mrrWithFilter: roundMetric(mrrWithFilter / Math.max(explicitYearQueries, 1)),
            goldReachableBeforeFilter: audits.filter(
                (item) => item.ranking.goldReachableBeforeFilter,
            ).length,
            goldRetainedAfterFilter: audits.filter(
                (item) => item.ranking.goldRetainedAfterFilter,
            ).length,
            goldRemovedByFilter: audits.filter(
                (item) => item.ranking.goldRemovedByFilter,
            ).length,
            goldMissingBeforeFilter: audits.filter(
                (item) => !item.ranking.goldReachableBeforeFilter,
            ).length,
            expectedPublishFallbackKeeps: audits.filter(
                (item) => item.expectedKeepReason === "publish",
            ).length,
            expectedSuspiciousLexicalKeeps: audits.filter(
                (item) => item.expectedKeepReason === "suspicious_lexical",
            ).length,
            top1YearConflictRemoved: audits.filter(
                (item) => item.top1NoFilter?.removedByFilter,
            ).length,
            top1RescuedToCorrect: audits.filter(
                (item) =>
                    item.top1NoFilter?.removedByFilter === true &&
                    item.ranking.noFilterRank !== 1 &&
                    item.ranking.withFilterRank === 1,
            ).length,
        };

        datasetSummaries.push(summary);

        for (const audit of audits) {
            const example: AuditExample = {
                dataset: audit.dataset,
                caseId: audit.caseId,
                query: audit.query,
                expectedTitle: audit.expectedTitle,
                expectedOtid: audit.expectedOtid,
                expectedTargetYear: audit.expectedTargetYear,
                expectedPublishYear: audit.expectedPublishYear,
                noFilterRank: audit.ranking.noFilterRank,
                withFilterRank: audit.ranking.withFilterRank,
                top1NoFilterTitle: audit.top1NoFilter?.title,
                top1WithFilterTitle: audit.top1WithFilter?.title,
            };
            if (audit.expectedKeepReason === "publish") {
                publishYearFallbackExamples.push(example);
            }
            if (audit.expectedKeepReason === "suspicious_lexical") {
                suspiciousStructuredYearExamples.push(example);
            }
            if (
                audit.top1NoFilter?.removedByFilter &&
                audit.ranking.noFilterRank !== 1 &&
                audit.ranking.withFilterRank === 1
            ) {
                top1RescueExamples.push(example);
            }
            if (audit.ranking.goldRemovedByFilter) {
                goldRemovedExamples.push(example);
            }
        }
    }

    const report: AuditReport = {
        generatedAt: new Date().toISOString(),
        mainDbVersion: ACTIVE_MAIN_DB_VERSION,
        pipelinePresetName: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name,
        datasetFiles,
        datasets: datasetSummaries,
        combined: buildCombinedSummary(datasetSummaries),
        publishYearFallbackExamples: publishYearFallbackExamples.slice(0, 8),
        suspiciousStructuredYearExamples:
            suspiciousStructuredYearExamples.slice(0, 8),
        top1RescueExamples: top1RescueExamples.slice(0, 8),
        goldRemovedExamples: goldRemovedExamples.slice(0, 8),
    };

    const outputPath = path.join(
        RESULTS_DIR,
        `explicit_year_filter_audit_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Saved explicit year filter audit to ${outputPath}`);
    datasetSummaries.forEach((item) => {
        console.log(
            [
                `${item.dataset}: explicitYear=${item.explicitYearQueries}/${item.totalCases}`,
                `Hit@1 ${item.hitAt1NoFilter.toFixed(4)} -> ${item.hitAt1WithFilter.toFixed(4)}`,
                `MRR ${item.mrrNoFilter.toFixed(4)} -> ${item.mrrWithFilter.toFixed(4)}`,
                `goldRemoved=${item.goldRemovedByFilter}`,
                `publishFallback=${item.expectedPublishFallbackKeeps}`,
                `suspiciousLexical=${item.expectedSuspiciousLexicalKeeps}`,
            ].join(" | "),
        );
    });
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

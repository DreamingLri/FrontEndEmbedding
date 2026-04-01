import * as fs from "fs";
import * as path from "path";

import {
    RRF_K,
    buildBM25Stats,
    dotProduct,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type BM25Stats,
    type Metadata,
} from "../src/worker/vector_engine.ts";
import { getCandidateIndicesForQuery } from "../src/worker/topic_partition.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import {
    FRONTEND_MODEL_NAME,
    loadDataset,
    resolveGranularityDatasetTarget,
    type EvalDatasetCase,
    type GranularityDatasetTargetKey,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";

type DatasetCase = EvalDatasetCase & {
    id?: string;
};

type DatasetTargetKey = GranularityDatasetTargetKey;

type DatasetTarget = {
    key: DatasetTargetKey;
    label: string;
    datasetFile: string;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryIntent: ReturnType<typeof parseQueryIntent>;
    queryYearWordIds: number[];
};

type RankedDoc = {
    otid: string;
    score: number;
    best_kpid?: string;
};

type PerCaseResult = {
    id?: string;
    query: string;
    expected_otid: string;
    rank: number | null;
    hitAt1: boolean;
    reciprocalRank: number;
    top1Otid?: string;
    topMatches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
};

type MetricSummary = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type MetricInterval = {
    lower: number;
    upper: number;
};

type ModelFamily =
    | "sparse"
    | "dense"
    | "hybrid"
    | "passage"
    | "metadata_filter"
    | "structured";

type ModelReport = {
    label: string;
    family: ModelFamily;
    description: string;
    metrics: MetricSummary;
    hitAt1Interval95: MetricInterval;
    mrrInterval95: MetricInterval;
    perCase: PerCaseResult[];
};

type PairwiseMetricReport = {
    observedDelta: number;
    interval95: MetricInterval;
    randomizationPValue: number;
};

type PairwiseComparison = {
    challenger: string;
    baseline: string;
    hitAt1: PairwiseMetricReport;
    mrr: PairwiseMetricReport;
};

type DatasetReport = {
    datasetKey: DatasetTargetKey;
    datasetLabel: string;
    datasetFile: string;
    caseCount: number;
    bestStandardBaseline: string;
    models: ModelReport[];
    pairwiseComparisons: PairwiseComparison[];
};

type Report = {
    generatedAt: string;
    embeddingModel: string;
    bootstrapIterations: number;
    randomizationIterations: number;
    datasets: DatasetReport[];
};

type CorpusView = {
    otMetadata: Metadata[];
    kpMetadata: Metadata[];
    otBm25Stats: BM25Stats;
    kpBm25Stats: BM25Stats;
    otIndices: number[];
    kpIndices: number[];
};

type ScoredMeta = {
    meta: Metadata;
    score: number;
};

const BM25_K1 = 1.2;
const BM25_B = 0.4;
const CURRENT_TIMESTAMP = 0;
const TOP_MATCH_COUNT = Number.parseInt(
    process.env.SUASK_STANDARD_BASELINE_TOP_MATCHES || "",
    10,
);
const BOOTSTRAP_ITERATIONS = Number.parseInt(
    process.env.SUASK_BOOTSTRAP_ITERATIONS || "",
    10,
);
const RANDOMIZATION_ITERATIONS = Number.parseInt(
    process.env.SUASK_RANDOMIZATION_ITERATIONS || "",
    10,
);
const RNG_SEED = Number.parseInt(process.env.SUASK_RANDOM_SEED || "", 10);

const SAFE_TOP_MATCH_COUNT =
    Number.isFinite(TOP_MATCH_COUNT) && TOP_MATCH_COUNT > 0 ? TOP_MATCH_COUNT : 5;
const SAFE_BOOTSTRAP_ITERATIONS =
    Number.isFinite(BOOTSTRAP_ITERATIONS) && BOOTSTRAP_ITERATIONS >= 1000
        ? BOOTSTRAP_ITERATIONS
        : 10000;
const SAFE_RANDOMIZATION_ITERATIONS =
    Number.isFinite(RANDOMIZATION_ITERATIONS) && RANDOMIZATION_ITERATIONS >= 1000
        ? RANDOMIZATION_ITERATIONS
        : 10000;
const SAFE_RANDOM_SEED =
    Number.isFinite(RNG_SEED) && RNG_SEED > 0 ? RNG_SEED : 20260331;

function resolveDatasetTarget(key: DatasetTargetKey): DatasetTarget | null {
    try {
        const target = resolveGranularityDatasetTarget(key);
        return {
            key,
            label: target.label,
            datasetFile: target.datasetFile,
        };
    } catch {
        return null;
    }
}

const AVAILABLE_DATASET_TARGETS = (
    [
        resolveDatasetTarget("main_bench_120"),
        resolveDatasetTarget("in_domain_holdout_50"),
        resolveDatasetTarget("external_ood_holdout_30"),
    ] as Array<DatasetTarget | null>
).filter((item): item is DatasetTarget => Boolean(item));

const DATASET_TARGETS = Object.fromEntries(
    AVAILABLE_DATASET_TARGETS.map((item) => [item.key, item]),
) as Partial<Record<DatasetTargetKey, DatasetTarget>>;

const DATASET_TARGET_ORDER = parseDatasetTargets(
    process.env.SUASK_STANDARD_BASELINE_DATASETS ||
        AVAILABLE_DATASET_TARGETS.map((item) => item.key).join(","),
);

const STRUCTURED_KP_OT_WEIGHTS = {
    Q: 0,
    KP: 0.28571428571428575,
    OT: 0.7142857142857143,
};

const STRUCTURED_Q_KP_OT_WEIGHTS = {
    Q: 0.3333333333333333,
    KP: 0.13333333333333333,
    OT: 0.5333333333333333,
};

let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let dimensions = 768;
let corpusView: CorpusView | null = null;
let allMetadataBm25Stats: BM25Stats | null = null;
let topicPartitionIndex: Awaited<
    ReturnType<typeof loadFrontendEvalEngine>
>["topicPartitionIndex"] | null = null;
let extractor: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["extractor"] | null =
    null;

function parseDatasetTargets(raw: string): DatasetTarget[] {
    const requested = raw
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean) as DatasetTargetKey[];

    const resolved = requested
        .map((key) => DATASET_TARGETS[key])
        .filter((item): item is DatasetTarget => Boolean(item));

    return resolved.length > 0 ? resolved : AVAILABLE_DATASET_TARGETS;
}

function hashString(input: string): number {
    let hash = 2166136261;
    for (let index = 0; index < input.length; index++) {
        hash ^= input.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function createSeededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state += 0x6d2b79f5;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function percentile(values: number[], q: number): number {
    if (values.length === 0) {
        return 0;
    }
    if (values.length === 1) {
        return values[0];
    }

    const position = (values.length - 1) * q;
    const lowerIndex = Math.floor(position);
    const upperIndex = Math.ceil(position);
    if (lowerIndex === upperIndex) {
        return values[lowerIndex];
    }
    const weight = position - lowerIndex;
    return (
        values[lowerIndex] * (1 - weight) + values[upperIndex] * weight
    );
}

function sortNumeric(values: number[]): number[] {
    return values.sort((a, b) => a - b);
}

function buildMetricSummary(perCase: readonly PerCaseResult[]): MetricSummary {
    const total = perCase.length || 1;
    const hitAt1Count = perCase.filter((item) => item.hitAt1).length;
    const hitAt3Count = perCase.filter(
        (item) => item.rank !== null && item.rank <= 3,
    ).length;
    const hitAt5Count = perCase.filter(
        (item) => item.rank !== null && item.rank <= 5,
    ).length;
    const reciprocalRankSum = perCase.reduce(
        (acc, item) => acc + item.reciprocalRank,
        0,
    );

    return {
        total: perCase.length,
        hitAt1: (hitAt1Count / total) * 100,
        hitAt3: (hitAt3Count / total) * 100,
        hitAt5: (hitAt5Count / total) * 100,
        mrr: reciprocalRankSum / total,
    };
}

function bootstrapMetricInterval(
    values: readonly number[],
    iterations: number,
    seedKey: string,
    scale = 1,
): MetricInterval {
    if (values.length === 0) {
        return { lower: 0, upper: 0 };
    }

    const random = createSeededRandom(SAFE_RANDOM_SEED ^ hashString(seedKey));
    const samples: number[] = [];

    for (let iteration = 0; iteration < iterations; iteration++) {
        let sum = 0;
        for (let index = 0; index < values.length; index++) {
            const pick = Math.floor(random() * values.length);
            sum += values[pick];
        }
        samples.push((sum / values.length) * scale);
    }

    const sorted = sortNumeric(samples);
    return {
        lower: percentile(sorted, 0.025),
        upper: percentile(sorted, 0.975),
    };
}

function pairedBootstrapDelta(
    challengerValues: readonly number[],
    baselineValues: readonly number[],
    iterations: number,
    seedKey: string,
    scale = 1,
): PairwiseMetricReport {
    if (challengerValues.length !== baselineValues.length) {
        throw new Error("Paired bootstrap requires equal-length arrays.");
    }

    const observedDelta =
        ((mean(challengerValues) - mean(baselineValues)) * scale) || 0;
    const random = createSeededRandom(SAFE_RANDOM_SEED ^ hashString(seedKey));
    const samples: number[] = [];
    const n = challengerValues.length;

    for (let iteration = 0; iteration < iterations; iteration++) {
        let challengerSum = 0;
        let baselineSum = 0;
        for (let index = 0; index < n; index++) {
            const pick = Math.floor(random() * n);
            challengerSum += challengerValues[pick];
            baselineSum += baselineValues[pick];
        }
        samples.push(((challengerSum - baselineSum) / n) * scale);
    }

    const sorted = sortNumeric(samples);
    return {
        observedDelta,
        interval95: {
            lower: percentile(sorted, 0.025),
            upper: percentile(sorted, 0.975),
        },
        randomizationPValue: approximateRandomizationPValue(
            challengerValues,
            baselineValues,
            observedDelta / scale,
            iterations,
            `${seedKey}:randomization`,
        ),
    };
}

function approximateRandomizationPValue(
    challengerValues: readonly number[],
    baselineValues: readonly number[],
    observedDeltaUnscaled: number,
    iterations: number,
    seedKey: string,
): number {
    if (challengerValues.length !== baselineValues.length) {
        throw new Error("Approximate randomization requires equal-length arrays.");
    }

    const random = createSeededRandom(SAFE_RANDOM_SEED ^ hashString(seedKey));
    const n = challengerValues.length;
    let exceedCount = 0;

    for (let iteration = 0; iteration < iterations; iteration++) {
        let challengerSum = 0;
        let baselineSum = 0;
        for (let index = 0; index < n; index++) {
            if (random() < 0.5) {
                challengerSum += challengerValues[index];
                baselineSum += baselineValues[index];
            } else {
                challengerSum += baselineValues[index];
                baselineSum += challengerValues[index];
            }
        }

        const shuffledDelta = challengerSum / n - baselineSum / n;
        if (Math.abs(shuffledDelta) >= Math.abs(observedDeltaUnscaled)) {
            exceedCount += 1;
        }
    }

    return (exceedCount + 1) / (iterations + 1);
}

function mean(values: readonly number[]): number {
    if (values.length === 0) {
        return 0;
    }
    return values.reduce((acc, item) => acc + item, 0) / values.length;
}

function compareMetrics(a: MetricSummary, b: MetricSummary): number {
    if (a.hitAt1 !== b.hitAt1) {
        return b.hitAt1 - a.hitAt1;
    }
    if (a.mrr !== b.mrr) {
        return b.mrr - a.mrr;
    }
    if (a.hitAt5 !== b.hitAt5) {
        return b.hitAt5 - a.hitAt5;
    }
    return b.hitAt3 - a.hitAt3;
}

function resolveCorpusView(metadata: readonly Metadata[]): CorpusView {
    const otMetadata = metadata.filter((item) => item.type === "OT");
    const kpMetadata = metadata.filter((item) => item.type === "KP");
    const otIndices: number[] = [];
    const kpIndices: number[] = [];

    metadata.forEach((item, index) => {
        if (item.type === "OT") {
            otIndices.push(index);
        }
        if (item.type === "KP") {
            kpIndices.push(index);
        }
    });

    return {
        otMetadata,
        kpMetadata,
        otBm25Stats: buildBM25Stats(otMetadata),
        kpBm25Stats: buildBM25Stats(kpMetadata),
        otIndices,
        kpIndices,
    };
}

async function loadEngine() {
    const engine = await loadFrontendEvalEngine();
    vocabMap = engine.vocabMap;
    metadataList = engine.metadataList;
    vectorMatrix = engine.vectorMatrix;
    dimensions = engine.dimensions;
    topicPartitionIndex = engine.topicPartitionIndex;
    extractor = engine.extractor;
    corpusView = resolveCorpusView(metadataList);
    allMetadataBm25Stats = engine.bm25Stats;
}

async function buildQueryCache(testCases: DatasetCase[]): Promise<QueryCacheItem[]> {
    if (!extractor) {
        throw new Error("Extractor not initialized.");
    }

    const queryVectors = await embedFrontendQueries(
        extractor,
        testCases.map((item) => item.query),
        dimensions,
    );

    return testCases.map((testCase, index) => {
        const queryIntent = parseQueryIntent(testCase.query);
        const queryWords = Array.from(
            new Set(fmmTokenize(testCase.query, vocabMap)),
        );
        const querySparse = getQuerySparse(queryWords, vocabMap);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        return {
            testCase,
            queryVector: queryVectors[index],
            queryWords,
            querySparse,
            queryIntent,
            queryYearWordIds,
        };
    });
}

function bm25Score(
    querySparse: Record<number, number>,
    meta: Metadata,
    stats: BM25Stats,
    metaIndex: number,
): number {
    if (!meta.sparse || meta.sparse.length === 0) {
        return 0;
    }

    let sparse = 0;
    const dl = stats.docLengths[metaIndex];
    const safeDl = Math.max(dl, stats.avgdl * 0.25);

    for (let entryIndex = 0; entryIndex < meta.sparse.length; entryIndex += 2) {
        const wordId = meta.sparse[entryIndex] as number;
        const tf = meta.sparse[entryIndex + 1] as number;
        const qWeight = querySparse[wordId];
        if (!qWeight) {
            continue;
        }

        const idf = stats.idfMap.get(wordId) || 0;
        const numerator = tf * (BM25_K1 + 1);
        const denominator =
            tf + BM25_K1 * (1 - BM25_B + BM25_B * (safeDl / stats.avgdl));
        sparse += qWeight * idf * (numerator / denominator);
    }

    return sparse;
}

function scoreDirectMetadata(
    query: QueryCacheItem,
    metadata: readonly Metadata[],
    stats: BM25Stats,
): {
    dense: ScoredMeta[];
    sparse: ScoredMeta[];
} {
    if (!vectorMatrix) {
        throw new Error("Vector matrix not initialized.");
    }

    const dense: ScoredMeta[] = [];
    const sparse: ScoredMeta[] = [];

    metadata.forEach((meta, index) => {
        let denseScore = dotProduct(
            query.queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null) {
            denseScore *= meta.scale;
        }
        dense.push({ meta, score: denseScore });
        sparse.push({
            meta,
            score: bm25Score(query.querySparse, meta, stats, index),
        });
    });

    dense.sort((a, b) => b.score - a.score);
    sparse.sort((a, b) => b.score - a.score);

    return { dense, sparse };
}

function buildDirectRanking(scored: readonly ScoredMeta[]): RankedDoc[] {
    return scored.map((item) => ({
        otid: item.meta.id,
        score: item.score,
    }));
}

function buildRrfRanking(
    dense: readonly ScoredMeta[],
    sparse: readonly ScoredMeta[],
): RankedDoc[] {
    const scores = new Map<string, number>();

    dense.forEach((item, index) => {
        scores.set(item.meta.id, (scores.get(item.meta.id) || 0) + 1 / (RRF_K + index + 1));
    });

    sparse.forEach((item, index) => {
        if (item.score <= 0) {
            return;
        }
        scores.set(item.meta.id, (scores.get(item.meta.id) || 0) + 1 / (RRF_K + index + 1));
    });

    return dense
        .map((item) => ({
            otid: item.meta.id,
            score: scores.get(item.meta.id) || 0,
        }))
        .sort((a, b) => b.score - a.score);
}

function filterOtMetadataByTopic(
    queryIntent: ReturnType<typeof parseQueryIntent>,
): Metadata[] {
    if (!topicPartitionIndex) {
        throw new Error("Topic partition index not initialized.");
    }

    const candidateIndices = getCandidateIndicesForQuery(
        queryIntent,
        topicPartitionIndex,
    );
    if (!candidateIndices || candidateIndices.length === 0) {
        return corpusView?.otMetadata || [];
    }

    const allowedSet = new Set(candidateIndices);
    const filtered = metadataList.filter(
        (item, index) => item.type === "OT" && allowedSet.has(index),
    );

    return filtered.length > 0 ? filtered : corpusView?.otMetadata || [];
}

function buildPassageHybridRanking(query: QueryCacheItem): RankedDoc[] {
    if (!corpusView) {
        throw new Error("Corpus view not initialized.");
    }

    const scored = scoreDirectMetadata(
        query,
        corpusView.kpMetadata,
        corpusView.kpBm25Stats,
    );
    const passageScores = new Map<string, { meta: Metadata; score: number }>();
    const docScores = new Map<string, RankedDoc>();

    scored.dense.forEach((item, index) => {
        passageScores.set(item.meta.id, {
            meta: item.meta,
            score: (passageScores.get(item.meta.id)?.score || 0) + 1 / (RRF_K + index + 1),
        });
    });

    scored.sparse.forEach((item, index) => {
        if (item.score <= 0) {
            return;
        }
        passageScores.set(item.meta.id, {
            meta: item.meta,
            score: (passageScores.get(item.meta.id)?.score || 0) + 1 / (RRF_K + index + 1),
        });
    });

    passageScores.forEach(({ meta, score }) => {
        const otid = meta.parent_otid;
        const current = docScores.get(otid);
        if (!current || score > current.score) {
            docScores.set(otid, {
                otid,
                score,
                best_kpid: meta.id,
            });
        }
    });

    return Array.from(docScores.values()).sort((a, b) => b.score - a.score);
}

function buildStructuredRanking(
    query: QueryCacheItem,
    weights: { Q: number; KP: number; OT: number },
): RankedDoc[] {
    if (!vectorMatrix || !corpusView || !allMetadataBm25Stats) {
        throw new Error("Structured ranking dependencies not initialized.");
    }

    const allowedTypes = (Object.entries(weights) as Array<
        [keyof typeof weights, number]
    >)
        .filter(([, weight]) => weight > 0)
        .map(([type]) => type);
    const filteredMetadata =
        allowedTypes.length === 3
            ? metadataList
            : metadataList.filter((item) =>
                  allowedTypes.includes(item.type),
              );
    const filteredBm25Stats =
        filteredMetadata === metadataList
            ? allMetadataBm25Stats
            : buildBM25Stats(filteredMetadata);

    const result = searchAndRank({
        queryVector: query.queryVector,
        querySparse: query.querySparse,
        queryYearWordIds: query.queryYearWordIds,
        queryIntent: query.queryIntent,
        queryScopeHint: query.testCase.query_scope,
        metadata: filteredMetadata,
        vectorMatrix,
        dimensions,
        currentTimestamp: CURRENT_TIMESTAMP,
        bm25Stats: filteredBm25Stats,
        weights,
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
    });

    return result.matches.map((item) => ({
        otid: item.otid,
        score: item.score,
        best_kpid: item.best_kpid,
    }));
}

function toPerCaseResult(
    query: QueryCacheItem,
    ranking: readonly RankedDoc[],
): PerCaseResult {
    const rankIndex = ranking.findIndex(
        (item) => item.otid === query.testCase.expected_otid,
    );
    const rank = rankIndex === -1 ? null : rankIndex + 1;
    return {
        id: query.testCase.id,
        query: query.testCase.query,
        expected_otid: query.testCase.expected_otid,
        rank,
        hitAt1: rank === 1,
        reciprocalRank: rank ? 1 / rank : 0,
        top1Otid: ranking[0]?.otid,
        topMatches: ranking.slice(0, SAFE_TOP_MATCH_COUNT).map((item, index) => ({
            rank: index + 1,
            otid: item.otid,
            score: item.score,
            best_kpid: item.best_kpid,
        })),
    };
}

function buildModelReport(
    label: string,
    family: ModelFamily,
    description: string,
    perCase: PerCaseResult[],
): ModelReport {
    const metrics = buildMetricSummary(perCase);
    const hitAt1Values = perCase.map((item) => (item.hitAt1 ? 1 : 0));
    const reciprocalRankValues = perCase.map((item) => item.reciprocalRank);

    return {
        label,
        family,
        description,
        metrics,
        hitAt1Interval95: bootstrapMetricInterval(
            hitAt1Values,
            SAFE_BOOTSTRAP_ITERATIONS,
            `${label}:hit1`,
            100,
        ),
        mrrInterval95: bootstrapMetricInterval(
            reciprocalRankValues,
            SAFE_BOOTSTRAP_ITERATIONS,
            `${label}:mrr`,
        ),
        perCase,
    };
}

function buildPairwiseComparisons(modelReports: readonly ModelReport[]): PairwiseComparison[] {
    const structuredModels = modelReports.filter(
        (item) => item.family === "structured",
    );
    const standardModels = modelReports.filter(
        (item) => item.family !== "structured",
    );
    const comparisons: PairwiseComparison[] = [];

    structuredModels.forEach((challenger) => {
        standardModels.forEach((baseline) => {
            const challengerHitAt1 = challenger.perCase.map((item) =>
                item.hitAt1 ? 1 : 0,
            );
            const baselineHitAt1 = baseline.perCase.map((item) =>
                item.hitAt1 ? 1 : 0,
            );
            const challengerMrr = challenger.perCase.map(
                (item) => item.reciprocalRank,
            );
            const baselineMrr = baseline.perCase.map(
                (item) => item.reciprocalRank,
            );

            comparisons.push({
                challenger: challenger.label,
                baseline: baseline.label,
                hitAt1: pairedBootstrapDelta(
                    challengerHitAt1,
                    baselineHitAt1,
                    SAFE_BOOTSTRAP_ITERATIONS,
                    `${challenger.label}:${baseline.label}:hit1`,
                    100,
                ),
                mrr: pairedBootstrapDelta(
                    challengerMrr,
                    baselineMrr,
                    SAFE_BOOTSTRAP_ITERATIONS,
                    `${challenger.label}:${baseline.label}:mrr`,
                ),
            });
        });
    });

    return comparisons;
}

function buildDatasetReport(
    target: DatasetTarget,
    queryCache: readonly QueryCacheItem[],
): DatasetReport {
    if (!corpusView) {
        throw new Error("Corpus view not initialized.");
    }

    const bm25OtPerCase = queryCache.map((query) => {
        const scored = scoreDirectMetadata(query, corpusView.otMetadata, corpusView.otBm25Stats);
        return toPerCaseResult(query, buildDirectRanking(scored.sparse));
    });

    const denseOtPerCase = queryCache.map((query) => {
        const scored = scoreDirectMetadata(query, corpusView.otMetadata, corpusView.otBm25Stats);
        return toPerCaseResult(query, buildDirectRanking(scored.dense));
    });

    const hybridOtPerCase = queryCache.map((query) => {
        const scored = scoreDirectMetadata(query, corpusView.otMetadata, corpusView.otBm25Stats);
        return toPerCaseResult(query, buildRrfRanking(scored.dense, scored.sparse));
    });

    const filteredHybridPerCase = queryCache.map((query) => {
        const filteredMetadata = filterOtMetadataByTopic(query.queryIntent);
        const filteredStats = buildBM25Stats(filteredMetadata);
        const scored = scoreDirectMetadata(query, filteredMetadata, filteredStats);
        return toPerCaseResult(query, buildRrfRanking(scored.dense, scored.sparse));
    });

    const passageHybridPerCase = queryCache.map((query) =>
        toPerCaseResult(query, buildPassageHybridRanking(query)),
    );

    const structuredKpOtPerCase = queryCache.map((query) =>
        toPerCaseResult(query, buildStructuredRanking(query, STRUCTURED_KP_OT_WEIGHTS)),
    );

    const structuredFullPerCase = queryCache.map((query) =>
        toPerCaseResult(
            query,
            buildStructuredRanking(query, STRUCTURED_Q_KP_OT_WEIGHTS),
        ),
    );

    const models: ModelReport[] = [
        buildModelReport(
            "BM25-OT",
            "sparse",
            "经典稀疏检索：仅在 OT 文档上执行 BM25。",
            bm25OtPerCase,
        ),
        buildModelReport(
            "Dense-OT",
            "dense",
            "语义检索：仅在 OT 文档上执行向量相似度排序。",
            denseOtPerCase,
        ),
        buildModelReport(
            "Hybrid-OT-RRF",
            "hybrid",
            "标准混合检索：OT 文档级 dense + BM25，经 RRF 融合。",
            hybridOtPerCase,
        ),
        buildModelReport(
            "Hybrid-OT-RRF+TopicFilter",
            "metadata_filter",
            "简单 metadata filter baseline：先按 topic 分区过滤，再执行 OT 文档级 RRF。",
            filteredHybridPerCase,
        ),
        buildModelReport(
            "Passage-KP-HybridMax",
            "passage",
            "passage-level baseline：在 KP 粒度上做 dense + BM25，再按 parent OT 聚合。",
            passageHybridPerCase,
        ),
        buildModelReport(
            "Structured-KP+OT",
            "structured",
            "论文冻结主组合：Q=0, KP=0.2857, OT=0.7143。",
            structuredKpOtPerCase,
        ),
        buildModelReport(
            "Structured-Q+KP+OT",
            "structured",
            "论文主集 top-line 组合：Q=0.3333, KP=0.1333, OT=0.5333。",
            structuredFullPerCase,
        ),
    ];

    const bestStandardBaseline = [...models]
        .filter((item) => item.family !== "structured")
        .sort((a, b) => compareMetrics(a.metrics, b.metrics))[0]?.label;

    if (!bestStandardBaseline) {
        throw new Error("Unable to resolve best standard baseline.");
    }

    return {
        datasetKey: target.key,
        datasetLabel: target.label,
        datasetFile: target.datasetFile,
        caseCount: queryCache.length,
        bestStandardBaseline,
        models,
        pairwiseComparisons: buildPairwiseComparisons(models),
    };
}

function printDatasetSummary(report: DatasetReport) {
    console.log(
        `\n[${report.datasetLabel}] best standard baseline: ${report.bestStandardBaseline}`,
    );
    report.models.forEach((model) => {
        console.log(
            `${model.label}: Hit@1=${model.metrics.hitAt1.toFixed(2)}% [${model.hitAt1Interval95.lower.toFixed(2)}, ${model.hitAt1Interval95.upper.toFixed(2)}] | MRR=${model.metrics.mrr.toFixed(4)} [${model.mrrInterval95.lower.toFixed(4)}, ${model.mrrInterval95.upper.toFixed(4)}]`,
        );
    });

    report.pairwiseComparisons
        .filter((item) => item.baseline === report.bestStandardBaseline)
        .forEach((item) => {
            console.log(
                `${item.challenger} vs ${item.baseline}: ΔHit@1=${item.hitAt1.observedDelta.toFixed(2)}pp [${item.hitAt1.interval95.lower.toFixed(2)}, ${item.hitAt1.interval95.upper.toFixed(2)}], p=${item.hitAt1.randomizationPValue.toFixed(4)} | ΔMRR=${item.mrr.observedDelta.toFixed(4)} [${item.mrr.interval95.lower.toFixed(4)}, ${item.mrr.interval95.upper.toFixed(4)}], p=${item.mrr.randomizationPValue.toFixed(4)}`,
            );
        });
}

async function main() {
    if (DATASET_TARGET_ORDER.length === 0) {
        throw new Error(
            "未解析到可用的 granularity 数据集目标，请先生成主集或 holdout 新文件。",
        );
    }

    console.log("Loading frontend eval engine...");
    await loadEngine();

    const datasetReports: DatasetReport[] = [];
    for (const target of DATASET_TARGET_ORDER) {
        console.log(`\nEvaluating ${target.label}...`);
        const datasetCases = loadDataset(target.datasetFile, {
            datasetLabel: target.key,
        }) as DatasetCase[];
        console.log(`Loaded ${datasetCases.length} cases from ${target.datasetFile}`);
        const queryCache = await buildQueryCache(datasetCases);
        const datasetReport = buildDatasetReport(target, queryCache);
        datasetReports.push(datasetReport);
        printDatasetSummary(datasetReport);
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        embeddingModel: FRONTEND_MODEL_NAME,
        bootstrapIterations: SAFE_BOOTSTRAP_ITERATIONS,
        randomizationIterations: SAFE_RANDOMIZATION_ITERATIONS,
        datasets: datasetReports,
    };

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(resultsDir, { recursive: true });
    const outputPath = path.resolve(
        resultsDir,
        `granularity_standard_baselines_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`\nReport saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

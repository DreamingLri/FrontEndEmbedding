import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';

import {
    buildBM25Stats,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type BM25Stats,
    type Metadata,
    type ParsedQueryIntent,
    type SearchRankOutput,
} from '../src/worker/vector_engine.ts';
import {
    buildTopicPartitionIndex,
    getCandidateIndicesForQuery,
    type TopicPartitionIndex,
} from '../src/worker/topic_partition.ts';
import { fmmTokenize } from '../src/worker/fmm_tokenize.ts';
import {
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    loadDataset,
    type EvalDatasetCase,
} from './eval_shared.ts';
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from './frontend_eval_engine.ts';

type DatasetCase = EvalDatasetCase;

type SearchCache = {
    testCase: DatasetCase;
    queryIntent: ParsedQueryIntent;
    partitionUsed: boolean;
    candidateCount: number;
    totalCount: number;
    scanRatio: number;
    partialRank: number;
    fullRank: number;
    partialTop1: number;
    partialTop2: number;
    partialMatchCount: number;
    partialRejectionReason?: string;
};

type ThresholdConfig = {
    scoreThreshold: number;
    gapThreshold: number;
    scanRatioThreshold: number;
};

type EvalMetrics = {
    label: string;
    hitAt1: number;
    hitAt5: number;
    mrr: number;
    avgItemsScanned: number;
    avgScanRatio: number;
    partitionUsedRate: number;
    fallbackRate: number;
    rescuedHitAt1: number;
    rescuedAnyRank: number;
    badFallbacks: number;
    totalCases: number;
};

type EvalMode = 'full_scan' | 'partition_only' | 'threshold';

const TUNE_DATASETS = [
    '../Backend/test/test_dataset_v2/test_dataset_standard.json',
    '../Backend/test/test_dataset_v2/test_dataset_short_keyword.json',
] as const;
const HOLDOUT_DATASETS = [
    '../Backend/test/test_dataset_v2/test_dataset_situational.json',
] as const;
const ENABLE_PARTITION_FALLBACK =
    process.env.SUASK_ENABLE_PARTITION_FALLBACK === '1';
const CURRENT_CONFIG: ThresholdConfig = {
    scoreThreshold: 1.05,
    gapThreshold: 0.03,
    scanRatioThreshold: 0.85,
};
const SCORE_THRESHOLDS = [1.05, 1.15, 1.25, 1.35, 1.45, 1.55];
const GAP_THRESHOLDS = [0.03, 0.05, 0.08, 0.11, 0.14];
const SCAN_RATIO_THRESHOLDS = [0.85, 0.92, 0.98];

let extractor: FeatureExtractionPipeline | null = null;
let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let globalBM25Stats: BM25Stats | null = null;
let dimensions = 768;
let topicPartitionIndex: TopicPartitionIndex = {
    topicCandidateIndex: new Map<string, number[]>(),
    unlabeledCandidateIndices: [],
    metadataCount: 0,
};

function getRank(result: SearchRankOutput, expectedOtid: string): number {
    const rankIndex = result.matches.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function shouldFallbackToFullScan(
    cache: SearchCache,
    config: ThresholdConfig,
): boolean {
    if (!ENABLE_PARTITION_FALLBACK) return false;
    if (!cache.partitionUsed) return false;
    if (cache.scanRatio >= config.scanRatioThreshold) return false;

    if (cache.queryIntent.confidence >= 1) {
        return false;
    }

    if (cache.partialMatchCount < 2) {
        return false;
    }

    const scoreGap = cache.partialTop1 - cache.partialTop2;
    return (
        cache.partialTop1 < config.scoreThreshold &&
        scoreGap < config.gapThreshold
    );
}

function evaluateConfig(
    cases: SearchCache[],
    config: ThresholdConfig | null,
    label: string,
    mode: EvalMode,
): EvalMetrics {
    let hitAt1 = 0;
    let hitAt5 = 0;
    let reciprocalRankSum = 0;
    let totalScanned = 0;
    let partitionUsedCount = 0;
    let fallbackCount = 0;
    let rescuedHitAt1 = 0;
    let rescuedAnyRank = 0;
    let badFallbacks = 0;

    for (const item of cases) {
        const fallback =
            mode === 'threshold' && config
                ? shouldFallbackToFullScan(item, config)
                : false;
        const finalRank =
            mode === 'full_scan'
                ? item.fullRank
                : fallback
                    ? item.fullRank
                    : item.partialRank;
        const scannedItems =
            mode === 'full_scan'
                ? item.totalCount
                : item.partitionUsed
                    ? fallback
                        ? item.candidateCount + item.totalCount
                        : item.candidateCount
                    : item.totalCount;

        if (item.partitionUsed) partitionUsedCount += 1;
        if (fallback) fallbackCount += 1;
        totalScanned += scannedItems;

        if (finalRank === 1) hitAt1 += 1;
        if (finalRank <= 5) hitAt5 += 1;
        if (Number.isFinite(finalRank)) reciprocalRankSum += 1 / finalRank;

        if (fallback) {
            if (item.partialRank !== 1 && item.fullRank === 1) {
                rescuedHitAt1 += 1;
            }
            if (item.fullRank < item.partialRank) {
                rescuedAnyRank += 1;
            } else {
                badFallbacks += 1;
            }
        }
    }

    const totalCases = cases.length || 1;
    return {
        label,
        hitAt1: (hitAt1 / totalCases) * 100,
        hitAt5: (hitAt5 / totalCases) * 100,
        mrr: reciprocalRankSum / totalCases,
        avgItemsScanned: totalScanned / totalCases,
        avgScanRatio: totalScanned / (totalCases * metadataList.length),
        partitionUsedRate: partitionUsedCount / totalCases,
        fallbackRate: partitionUsedCount === 0 ? 0 : fallbackCount / partitionUsedCount,
        rescuedHitAt1,
        rescuedAnyRank,
        badFallbacks,
        totalCases,
    };
}

function compareMetrics(a: EvalMetrics, b: EvalMetrics): number {
    if (a.hitAt1 !== b.hitAt1) return b.hitAt1 - a.hitAt1;
    if (a.mrr !== b.mrr) return b.mrr - a.mrr;
    if (a.hitAt5 !== b.hitAt5) return b.hitAt5 - a.hitAt5;
    if (a.avgScanRatio !== b.avgScanRatio) return a.avgScanRatio - b.avgScanRatio;
    return a.badFallbacks - b.badFallbacks;
}

function summarizeScoreDistribution(cases: SearchCache[]) {
    const values = cases
        .filter((item) => item.partitionUsed && item.partialMatchCount > 0 && item.queryIntent.confidence < 1)
        .map((item) => item.partialTop1)
        .sort((a, b) => a - b);

    if (values.length === 0) {
        return null;
    }

    const pick = (ratio: number) =>
        values[Math.min(values.length - 1, Math.floor(values.length * ratio))];

    return {
        count: values.length,
        p25: pick(0.25),
        p50: pick(0.5),
        p75: pick(0.75),
        p90: pick(0.9),
        max: values[values.length - 1],
    };
}

async function loadEngine() {
    console.log('Loading metadata and vectors...');
    const engine = await loadFrontendEvalEngine();
    extractor = engine.extractor;
    vocabMap = engine.vocabMap;
    metadataList = engine.metadataList;
    vectorMatrix = engine.vectorMatrix;
    globalBM25Stats = engine.bm25Stats;
    topicPartitionIndex = engine.topicPartitionIndex;
    dimensions = engine.dimensions;

    console.log(
        `Loaded ${metadataList.length} vectors, dimensions=${dimensions}, unlabeled=${topicPartitionIndex.unlabeledCandidateIndices.length}`,
    );
}

async function embedQueries(queries: string[]): Promise<Float32Array[]> {
    if (!extractor) throw new Error('Extractor not initialized');
    return embedFrontendQueries(extractor, queries, dimensions, {
        batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
        onProgress: (done, total) => {
            console.log(`Embedded ${done} / ${total} queries`);
        },
    });
}

async function buildSearchCache(testCases: DatasetCase[]): Promise<SearchCache[]> {
    if (!vectorMatrix || !globalBM25Stats) {
        throw new Error('Search engine not initialized');
    }

    console.log(`Precomputing ${testCases.length} queries...`);
    const queryVectors = await embedQueries(testCases.map((item) => item.query));

    const cache: SearchCache[] = [];
    for (let index = 0; index < testCases.length; index++) {
        const testCase = testCases[index];
        const queryIntent = parseQueryIntent(testCase.query);
        const candidateIndices = getCandidateIndicesForQuery(queryIntent, topicPartitionIndex);
        const queryWords = Array.from(
            new Set(fmmTokenize(testCase.query, vocabMap)),
        );
        const querySparse = getQuerySparse(queryWords, vocabMap);
        const queryYearWordIds: number[] = [];

        queryIntent.years.map(String).forEach((year) => {
            const wordId = vocabMap.get(year);
            if (wordId !== undefined) queryYearWordIds.push(wordId);
        });

        const commonParams = {
            queryVector: queryVectors[index],
            querySparse,
            queryYearWordIds,
            queryIntent,
            metadata: metadataList,
            vectorMatrix,
            dimensions,
            currentTimestamp: Date.now() / 1000,
            bm25Stats: globalBM25Stats,
        };

        const fullResult = searchAndRank(commonParams);
        const partialResult = candidateIndices
            ? searchAndRank({
                ...commonParams,
                candidateIndices,
            })
            : fullResult;

        const partialTop1 = partialResult.matches[0]?.score ?? Number.NEGATIVE_INFINITY;
        const partialTop2 = partialResult.matches[1]?.score ?? Number.NEGATIVE_INFINITY;

        cache.push({
            testCase,
            queryIntent,
            partitionUsed: Boolean(candidateIndices),
            candidateCount: candidateIndices?.length ?? metadataList.length,
            totalCount: metadataList.length,
            scanRatio: (candidateIndices?.length ?? metadataList.length) / metadataList.length,
            partialRank: getRank(partialResult, testCase.expected_otid),
            fullRank: getRank(fullResult, testCase.expected_otid),
            partialTop1,
            partialTop2,
            partialMatchCount: partialResult.matches.length,
            partialRejectionReason: partialResult.rejection?.reason,
        });

        const done = index + 1;
        if (done % 50 === 0 || done === testCases.length) {
            console.log(`Scored ${done} / ${testCases.length} queries`);
        }
    }

    return cache;
}

function formatMetrics(metrics: EvalMetrics): string {
    return [
        `${metrics.label}`,
        `  Hit@1=${metrics.hitAt1.toFixed(2)}%`,
        `Hit@5=${metrics.hitAt5.toFixed(2)}%`,
        `MRR=${metrics.mrr.toFixed(4)}`,
        `avgScan=${metrics.avgItemsScanned.toFixed(1)}`,
        `avgScanRatio=${(metrics.avgScanRatio * 100).toFixed(2)}%`,
        `partitionUsed=${(metrics.partitionUsedRate * 100).toFixed(1)}%`,
        `fallbackRate=${(metrics.fallbackRate * 100).toFixed(1)}%`,
        `rescuedHit1=${metrics.rescuedHitAt1}`,
        `rescuedAnyRank=${metrics.rescuedAnyRank}`,
        `badFallbacks=${metrics.badFallbacks}`,
    ].join(' | ');
}

async function main() {
    const startedAt = performance.now();
    await loadEngine();

    const tuneCases = TUNE_DATASETS.flatMap((dataset) => loadDataset(dataset));
    const holdoutCases = HOLDOUT_DATASETS.flatMap((dataset) => loadDataset(dataset));
    const allCases = [...tuneCases, ...holdoutCases];

    const cache = await buildSearchCache(allCases);
    const tuneCache = cache.slice(0, tuneCases.length);
    const holdoutCache = cache.slice(tuneCases.length);

    const scoreDistribution = summarizeScoreDistribution(tuneCache);
    if (scoreDistribution) {
        console.log('Low-confidence partition top1 score distribution:', scoreDistribution);
    }

    const currentMode: EvalMode = ENABLE_PARTITION_FALLBACK
        ? 'threshold'
        : 'partition_only';

    const baselines = [
        evaluateConfig(tuneCache, null, 'tune/full_scan', 'full_scan'),
        evaluateConfig(
            tuneCache,
            {
                scoreThreshold: Number.NEGATIVE_INFINITY,
                gapThreshold: Number.NEGATIVE_INFINITY,
                scanRatioThreshold: 0,
            },
            'tune/partition_only',
            'partition_only',
        ),
        evaluateConfig(tuneCache, CURRENT_CONFIG, 'tune/current_config', currentMode),
    ];

    console.log('\nTune baselines');
    baselines.forEach((item) => console.log(formatMetrics(item)));

    const candidates: Array<{
        config: ThresholdConfig;
        metrics: EvalMetrics;
    }> = [];

    for (const scoreThreshold of SCORE_THRESHOLDS) {
        for (const gapThreshold of GAP_THRESHOLDS) {
            for (const scanRatioThreshold of SCAN_RATIO_THRESHOLDS) {
                const config = {
                    scoreThreshold,
                    gapThreshold,
                    scanRatioThreshold,
                };
                candidates.push({
                    config,
                    metrics: evaluateConfig(
                        tuneCache,
                        config,
                        `score=${scoreThreshold},gap=${gapThreshold},scan=${scanRatioThreshold}`,
                        'threshold',
                    ),
                });
            }
        }
    }

    candidates.sort((a, b) => compareMetrics(a.metrics, b.metrics));
    const best = candidates[0];

    console.log('\nTop candidate configs on tune set');
    candidates.slice(0, 10).forEach((item, index) => {
        console.log(
            `#${index + 1} ${formatMetrics(item.metrics)}`
        );
    });

    const holdoutBaselines = [
        evaluateConfig(holdoutCache, null, 'holdout/full_scan', 'full_scan'),
        evaluateConfig(
            holdoutCache,
            {
                scoreThreshold: Number.NEGATIVE_INFINITY,
                gapThreshold: Number.NEGATIVE_INFINITY,
                scanRatioThreshold: 0,
            },
            'holdout/partition_only',
            'partition_only',
        ),
        evaluateConfig(holdoutCache, CURRENT_CONFIG, 'holdout/current_config', currentMode),
        evaluateConfig(holdoutCache, best.config, 'holdout/best_config', 'threshold'),
    ];

    console.log('\nHoldout comparison');
    holdoutBaselines.forEach((item) => console.log(formatMetrics(item)));

    const combinedCurrent = evaluateConfig(
        cache,
        CURRENT_CONFIG,
        'combined/current_config',
        currentMode,
    );
    const combinedBest = evaluateConfig(
        cache,
        best.config,
        'combined/best_config',
        'threshold',
    );

    console.log('\nCombined comparison');
    console.log(formatMetrics(combinedCurrent));
    console.log(formatMetrics(combinedBest));

    const report = {
        currentConfig: CURRENT_CONFIG,
        bestConfig: best.config,
        tuneBestMetrics: best.metrics,
        tuneBaselines: baselines,
        holdoutBaselines,
        combinedCurrent,
        combinedBest,
        scoreDistribution,
        elapsedMs: performance.now() - startedAt,
    };

    const resultsDir = path.resolve(process.cwd(), 'scripts/results');
    if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
    }

    const outputPath = path.join(
        resultsDir,
        `partition_calibration_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), 'utf-8');
    console.log(`\nSaved report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

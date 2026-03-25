import * as fs from 'fs';
import * as path from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';

import {
    buildBM25Stats as buildCurrentBM25Stats,
    getQuerySparse as getCurrentQuerySparse,
    parseQueryIntent,
    searchAndRank as searchCurrent,
    type BM25Stats as CurrentBM25Stats,
    type Metadata as CurrentMetadata,
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
import type {
    BM25Stats as LegacyBM25Stats,
    Metadata as LegacyMetadata,
    SearchResult as LegacySearchResult,
} from '../../Backend/test/vector_engine.ts';

type DatasetCase = EvalDatasetCase;

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryIntent: ParsedQueryIntent;
    fmmTokens: string[];
    jiebaTokens: string[];
    yearWordIds: number[];
};

type Metrics = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
    rejectionRate?: number;
    avgCandidateCount?: number;
};

type ModeResult = {
    label: string;
    metricsByDataset: Record<string, Metrics>;
    combined: Metrics;
};

type Report = {
    generatedAt: string;
    datasetSizes: Record<string, number>;
    queryEmbeddingBatchSize: number;
    modes: ModeResult[];
};

type ModeDefinition = {
    label: string;
    run: (item: QueryCacheItem) => {
        matches: { otid: string }[];
        rejected: boolean;
        candidateCount?: number;
    };
};

const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || 'v2';
const DATASET_DIR = `../Backend/test/test_dataset_${DATASET_VERSION}`;
const DATASETS = [
    `${DATASET_DIR}/test_dataset_standard.json`,
    `${DATASET_DIR}/test_dataset_short_keyword.json`,
    `${DATASET_DIR}/test_dataset_situational.json`,
] as const;
const CURRENT_TIMESTAMP = 0;

const require = createRequire(import.meta.url);
const nodejieba = require(path.resolve(process.cwd(), '../Backend/test/node_modules/nodejieba'));

let extractor: FeatureExtractionPipeline | null = null;
let vocabMap = new Map<string, number>();
let currentMetadataList: CurrentMetadata[] = [];
let legacyMetadataList: LegacyMetadata[] = [];
let vectorMatrix: Int8Array | null = null;
let dimensions = 768;
let currentBM25Stats: CurrentBM25Stats | null = null;
let legacyBM25Stats: LegacyBM25Stats | null = null;
let legacyEngine: typeof import('../../Backend/test/vector_engine.ts') | null = null;
let topicPartitionIndex: TopicPartitionIndex = {
    topicCandidateIndex: new Map<string, number[]>(),
    unlabeledCandidateIndices: [],
    metadataCount: 0,
};

function initializeJieba() {
    const dictPath = path.resolve(process.cwd(), '../Backend/data/campus_dict.txt');
    if (!fs.existsSync(dictPath)) return;

    const lines = fs.readFileSync(dictPath, 'utf-8').split(/\r?\n/);
    lines.forEach((line: string) => {
        const word = line.trim();
        if (word) {
            nodejieba.insertWord(word);
        }
    });
}

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
}

async function loadEngine() {
    legacyEngine = await import(
        pathToFileURL(path.resolve(process.cwd(), '../Backend/test/vector_engine.ts')).href
    );

    const engine = await loadFrontendEvalEngine();
    extractor = engine.extractor;
    vocabMap = engine.vocabMap;
    currentMetadataList = engine.metadataList;
    legacyMetadataList = engine.metadataList as LegacyMetadata[];
    vectorMatrix = engine.vectorMatrix;
    dimensions = engine.dimensions;
    currentBM25Stats = engine.bm25Stats;
    topicPartitionIndex = engine.topicPartitionIndex;
    legacyBM25Stats = legacyEngine.buildBM25Stats(legacyMetadataList);
    initializeJieba();
}

async function embedQueries(queries: string[]): Promise<Float32Array[]> {
    if (!extractor) throw new Error('Extractor not initialized');
    return embedFrontendQueries(extractor, queries, dimensions, {
        batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
    });
}

async function buildQueryCache(testCases: DatasetCase[]): Promise<QueryCacheItem[]> {
    const queryVectors = await embedQueries(testCases.map((item) => item.query));
    return testCases.map((testCase, index) => {
        const queryIntent = parseQueryIntent(testCase.query);
        const fmmTokens = fmmTokenize(testCase.query, vocabMap);
        const jiebaTokens = nodejieba.cut(testCase.query) as string[];
        const yearWordIds = queryIntent.years
            .map(String)
            .map((year) => vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        return {
            testCase,
            queryVector: queryVectors[index],
            queryIntent,
            fmmTokens,
            jiebaTokens,
            yearWordIds,
        };
    });
}

function rankOf(matches: readonly { otid: string }[], expectedOtid: string): number {
    const rankIndex = matches.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function searchLegacyMode(
    item: QueryCacheItem,
    words: string[],
): { matches: LegacySearchResult[]; rejected: boolean } {
    if (!vectorMatrix || !legacyBM25Stats || !legacyEngine) {
        throw new Error('Legacy engine not initialized');
    }

    const querySparse = legacyEngine.getQuerySparse(words, vocabMap);
    const matches = legacyEngine.searchAndRank({
        queryVector: item.queryVector,
        querySparse,
        queryYearWordIds: item.yearWordIds,
        metadata: legacyMetadataList,
        vectorMatrix,
        dimensions,
        currentTimestamp: CURRENT_TIMESTAMP,
        bm25Stats: legacyBM25Stats,
    });

    return { matches, rejected: false };
}

function searchCurrentMode(
    item: QueryCacheItem,
    words: string[],
    candidateIndices: number[] | undefined,
): {
    matches: SearchRankOutput['matches'];
    rejected: boolean;
    candidateCount?: number;
} {
    if (!vectorMatrix || !currentBM25Stats) {
        throw new Error('Current engine not initialized');
    }

    const querySparse = getCurrentQuerySparse(words, vocabMap);
    const result = searchCurrent({
        queryVector: item.queryVector,
        querySparse,
        queryYearWordIds: item.yearWordIds,
        queryIntent: item.queryIntent,
        metadata: currentMetadataList,
        vectorMatrix,
        dimensions,
        currentTimestamp: CURRENT_TIMESTAMP,
        bm25Stats: currentBM25Stats,
        candidateIndices,
    });

    return {
        matches: result.matches,
        rejected: Boolean(result.rejection),
        candidateCount: candidateIndices?.length ?? currentMetadataList.length,
    };
}

function buildMetrics(
    testCases: readonly DatasetCase[],
    mode: ModeDefinition,
    queryCache: readonly QueryCacheItem[],
): ModeResult {
    const metricsByDataset: Record<string, Metrics> = {};
    const metricsSeed: Record<
        string,
        {
            total: number;
            hitAt1: number;
            hitAt3: number;
            hitAt5: number;
            reciprocalRankSum: number;
            rejectedCount: number;
            candidateCountSum: number;
            candidateCountSeen: number;
        }
    > = {};

    const combinedSeed = {
        total: 0,
        hitAt1: 0,
        hitAt3: 0,
        hitAt5: 0,
        reciprocalRankSum: 0,
        rejectedCount: 0,
        candidateCountSum: 0,
        candidateCountSeen: 0,
    };

    queryCache.forEach((item) => {
        const result = mode.run(item);
        const rank = rankOf(result.matches, item.testCase.expected_otid);
        const bucket =
            metricsSeed[item.testCase.dataset] ||
            (metricsSeed[item.testCase.dataset] = {
                total: 0,
                hitAt1: 0,
                hitAt3: 0,
                hitAt5: 0,
                reciprocalRankSum: 0,
                rejectedCount: 0,
                candidateCountSum: 0,
                candidateCountSeen: 0,
            });

        const targets = [bucket, combinedSeed];
        targets.forEach((target) => {
            target.total += 1;
            if (rank === 1) target.hitAt1 += 1;
            if (rank <= 3) target.hitAt3 += 1;
            if (rank <= 5) target.hitAt5 += 1;
            if (Number.isFinite(rank)) {
                target.reciprocalRankSum += 1 / rank;
            }
            if (result.rejected) {
                target.rejectedCount += 1;
            }
            if (typeof result.candidateCount === 'number') {
                target.candidateCountSum += result.candidateCount;
                target.candidateCountSeen += 1;
            }
        });
    });

    Object.entries(metricsSeed).forEach(([dataset, seed]) => {
        metricsByDataset[dataset] = {
            total: seed.total,
            hitAt1: (seed.hitAt1 / seed.total) * 100,
            hitAt3: (seed.hitAt3 / seed.total) * 100,
            hitAt5: (seed.hitAt5 / seed.total) * 100,
            mrr: seed.reciprocalRankSum / seed.total,
            rejectionRate: seed.rejectedCount / seed.total,
            avgCandidateCount:
                seed.candidateCountSeen > 0
                    ? seed.candidateCountSum / seed.candidateCountSeen
                    : undefined,
        };
    });

    return {
        label: mode.label,
        metricsByDataset,
        combined: {
            total: combinedSeed.total,
            hitAt1: (combinedSeed.hitAt1 / combinedSeed.total) * 100,
            hitAt3: (combinedSeed.hitAt3 / combinedSeed.total) * 100,
            hitAt5: (combinedSeed.hitAt5 / combinedSeed.total) * 100,
            mrr: combinedSeed.reciprocalRankSum / combinedSeed.total,
            rejectionRate: combinedSeed.rejectedCount / combinedSeed.total,
            avgCandidateCount:
                combinedSeed.candidateCountSeen > 0
                    ? combinedSeed.candidateCountSum / combinedSeed.candidateCountSeen
                    : undefined,
        },
    };
}

function printModeSummary(result: ModeResult) {
    const combined = result.combined;
    const avgCandidateText =
        typeof combined.avgCandidateCount === 'number'
            ? ` | avgCandidates=${combined.avgCandidateCount.toFixed(1)}`
            : '';
    const rejectionText =
        typeof combined.rejectionRate === 'number'
            ? ` | rejectionRate=${(combined.rejectionRate * 100).toFixed(2)}%`
            : '';

    console.log(
        `${result.label}: Hit@1=${combined.hitAt1.toFixed(2)}% | Hit@3=${combined.hitAt3.toFixed(2)}% | Hit@5=${combined.hitAt5.toFixed(2)}% | MRR=${combined.mrr.toFixed(4)}${rejectionText}${avgCandidateText}`,
    );
}

async function main() {
    const testCases = DATASETS.flatMap(loadDataset);
    const datasetSizes = testCases.reduce<Record<string, number>>((acc, item) => {
        acc[item.dataset] = (acc[item.dataset] || 0) + 1;
        return acc;
    }, {});

    console.log('Loading engine and datasets...');
    await loadEngine();

    console.log(`Embedding ${testCases.length} queries for baseline comparison...`);
    const queryCache = await buildQueryCache(testCases);

    const modes: ModeDefinition[] = [
        {
            label: 'legacy_fullscan_fmm',
            run: (item) => searchLegacyMode(item, item.fmmTokens),
        },
        {
            label: 'legacy_fullscan_jieba',
            run: (item) => searchLegacyMode(item, item.jiebaTokens),
        },
        {
            label: 'current_fullscan_fmm',
            run: (item) => searchCurrentMode(item, item.fmmTokens, undefined),
        },
        {
            label: 'current_fullscan_actual',
            run: (item) =>
                searchCurrentMode(
                    item,
                    item.fmmTokens,
                    undefined,
                ),
        },
        {
            label: 'current_partition_actual',
            run: (item) =>
                searchCurrentMode(
                    item,
                    item.fmmTokens,
                    getCandidateIndicesForQuery(item.queryIntent, topicPartitionIndex),
                ),
        },
    ];

    const modeResults = modes.map((mode) => {
        const result = buildMetrics(testCases, mode, queryCache);
        printModeSummary(result);
        return result;
    });

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetSizes,
        queryEmbeddingBatchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
        modes: modeResults,
    };

    const resultsDir = path.resolve(process.cwd(), 'scripts/results');
    if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
    }

    const outputPath = path.resolve(
        resultsDir,
        `baseline_compare_${DATASET_VERSION}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), 'utf-8');
    console.log(`Report saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

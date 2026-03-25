import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';
import { env, pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

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
import {
    normalizeMinMax,
    normalizeSnippetScore,
    splitIntoSemanticChunks,
} from '../src/worker/rerank_helpers.ts';

type DatasetCase = {
    query: string;
    expected_otid: string;
    query_type?: string;
    dataset: string;
};

type RerankDocument = {
    otid: string;
    ot_text: string;
    coarseScore: number;
    best_kpid?: string;
};

type SearchCache = {
    testCase: DatasetCase;
    queryIntent: ParsedQueryIntent;
    queryVector: Float32Array;
    coarseRank: number;
    coarseTop15Rank: number;
    coarseMatches: RerankDocument[];
};

type ChunkDoc = {
    otid: string;
    chunkTexts: string[];
    chunkVectors: Float32Array[];
};

type ChunkScoreCache = {
    rawScore: number;
    normalizedScore: number;
};

type CaseRerankCache = {
    testCase: DatasetCase;
    coarseRank: number;
    coarseTop15Rank: number;
    coarseMatches: RerankDocument[];
    chunkScoresByLimit: Record<number, ChunkScoreCache[]>;
};

type RankingConfig = {
    label: string;
    rerankDocCount: number;
    maxChunksPerDoc: number;
    scoreMode: 'coarse_only' | 'current_ui' | 'snippet_only' | 'blend';
    blendAlpha?: number;
    adaptiveDocWindow?: {
        baseDocCount: number;
        expandedDocCount: number;
        trigger: 'top1_top2_gap' | 'top1_top5_gap';
        threshold: number;
    };
};

type ThresholdMetrics = {
    threshold: number;
    coverage: number;
    rejectRate: number;
    acceptedHitAt1: number;
    acceptedHitAt5: number;
    acceptedMRR: number;
    badRejects: number;
    goodRejects: number;
    totalCases: number;
};

type RankingMetrics = {
    label: string;
    hitAt1: number;
    hitAt5: number;
    mrr: number;
    rescueHitAt1: number;
    rescueAnyRank: number;
    regressionsFromCoarse: number;
    avgRerankedDocs: number;
    avgChunksScored: number;
    totalCases: number;
};

type ScoredRerankDoc = RerankDocument & {
    rawSnippetScore: number;
    normalizedSnippetScore: number;
    finalScore: number;
};

type ScoredCaseResult = {
    finalDocs: ScoredRerankDoc[];
    finalRank: number;
    topConfidence: number;
    rerankedDocCount: number;
    chunksScored: number;
};

type HardCaseDiagnostic = {
    query: string;
    expectedOtid: string;
    dataset: string;
    queryType?: string;
    coarseRank: number;
    coarseTop15Rank: number;
    currentUiRank: number;
    productionRank: number;
    offlineBestRank: number;
    dominantReason:
        | 'expected_missing_from_top15'
        | 'expected_outside_rerank_window'
        | 'rerank_demoted_correct_doc'
        | 'rerank_failed_to_promote'
        | 'coarse_already_wrong_after_top15';
    coarseTop3: Array<{ otid: string; coarseScore: number }>;
    productionTop3: Array<{
        otid: string;
        coarseScore: number;
        snippetScore: number;
        finalScore: number;
    }>;
    expectedDocSnapshot?: {
        coarseScore: number;
        snippetScore: number;
        finalScore: number;
        productionPosition: number;
    };
};

type HardCaseSummary = {
    productionLabel: string;
    offlineBestLabel: string;
    productionVsCoarse: {
        improvedHitAt1: number;
        regressedHitAt1: number;
        unchanged: number;
    };
    coarseRankBuckets: {
        top1: number;
        top2to5: number;
        top6to15: number;
        missTop15: number;
    };
    productionRankBuckets: {
        top1: number;
        top2to5: number;
        top6to15: number;
        missTop15: number;
    };
    rerankWindowBuckets: {
        expectedInsideTop5: number;
        expectedOutsideTop5ButInsideTop15: number;
        expectedMissingTop15: number;
    };
    dominantReasons: Record<HardCaseDiagnostic['dominantReason'], number>;
    topRegressions: HardCaseDiagnostic[];
    topMissedOpportunities: HardCaseDiagnostic[];
};

type SplitReport = {
    split: string;
    bestRanking: RankingMetrics;
    rankingResults: RankingMetrics[];
    thresholdResults: ThresholdMetrics[];
    selectedThreshold?: ThresholdMetrics;
    hardCaseSummary?: HardCaseSummary;
};

const MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';
const VECTOR_FILE = 'public/data/frontend_vectors_dmeta_small.bin';
const METADATA_FILE = 'public/data/frontend_metadata_dmeta_small.json';
const ARTICLE_SOURCE_FILE = '../Backend/data/embeddings_v2/flattened_json.json';
const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || 'v2';
const DATASET_DIR = `../Backend/test/test_dataset_${DATASET_VERSION}`;
const TUNE_DATASETS = [
    `${DATASET_DIR}/test_dataset_standard.json`,
    `${DATASET_DIR}/test_dataset_short_keyword.json`,
] as const;
const HOLDOUT_DATASETS = [
    `${DATASET_DIR}/test_dataset_situational.json`,
] as const;

const FETCH_DOC_LIMIT = 15;
const PARTITION_FALLBACK_ENABLED = false;
const QUERY_EMBED_BATCH_SIZE = 16;
const CHUNK_EMBED_BATCH_SIZE = 24;
const CHUNK_MAX_LEN = 150;
const CHUNK_LIMITS = [6, 10, 14] as const;
const THRESHOLD_CANDIDATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55] as const;
const PRODUCTION_RANKING_LABEL = 'adaptive015_top10_top5gap080_chunk14';
const OFFLINE_BEST_RANKING_LABEL = 'adaptive015_top10_top5gap080_chunk14';
const CURRENT_RANKING_CONFIG: RankingConfig = {
    label: 'current_ui',
    rerankDocCount: 3,
    maxChunksPerDoc: 10,
    scoreMode: 'current_ui',
};
const CANDIDATE_RANKING_CONFIGS: RankingConfig[] = [
    { label: 'coarse_only', rerankDocCount: 0, maxChunksPerDoc: 0, scoreMode: 'coarse_only' },
    CURRENT_RANKING_CONFIG,
    { label: 'snippet_top3_chunk6', rerankDocCount: 3, maxChunksPerDoc: 6, scoreMode: 'snippet_only' },
    { label: 'snippet_top3_chunk10', rerankDocCount: 3, maxChunksPerDoc: 10, scoreMode: 'snippet_only' },
    { label: 'snippet_top3_chunk14', rerankDocCount: 3, maxChunksPerDoc: 14, scoreMode: 'snippet_only' },
    { label: 'snippet_top5_chunk6', rerankDocCount: 5, maxChunksPerDoc: 6, scoreMode: 'snippet_only' },
    { label: 'snippet_top5_chunk10', rerankDocCount: 5, maxChunksPerDoc: 10, scoreMode: 'snippet_only' },
    { label: 'snippet_top5_chunk14', rerankDocCount: 5, maxChunksPerDoc: 14, scoreMode: 'snippet_only' },
    { label: 'snippet_top8_chunk10', rerankDocCount: 8, maxChunksPerDoc: 10, scoreMode: 'snippet_only' },
    { label: 'blend015_top3_chunk10', rerankDocCount: 3, maxChunksPerDoc: 10, scoreMode: 'blend', blendAlpha: 0.15 },
    { label: 'blend030_top3_chunk10', rerankDocCount: 3, maxChunksPerDoc: 10, scoreMode: 'blend', blendAlpha: 0.3 },
    { label: 'blend045_top3_chunk10', rerankDocCount: 3, maxChunksPerDoc: 10, scoreMode: 'blend', blendAlpha: 0.45 },
    { label: 'blend015_top5_chunk10', rerankDocCount: 5, maxChunksPerDoc: 10, scoreMode: 'blend', blendAlpha: 0.15 },
    { label: 'blend030_top5_chunk10', rerankDocCount: 5, maxChunksPerDoc: 10, scoreMode: 'blend', blendAlpha: 0.3 },
    { label: 'blend045_top5_chunk10', rerankDocCount: 5, maxChunksPerDoc: 10, scoreMode: 'blend', blendAlpha: 0.45 },
    { label: 'blend015_top5_chunk14', rerankDocCount: 5, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.15 },
    { label: 'blend030_top5_chunk14', rerankDocCount: 5, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.3 },
    { label: 'blend045_top5_chunk14', rerankDocCount: 5, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.45 },
    { label: 'blend010_top8_chunk14', rerankDocCount: 8, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.1 },
    { label: 'blend015_top8_chunk14', rerankDocCount: 8, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.15 },
    { label: 'blend030_top8_chunk14', rerankDocCount: 8, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.3 },
    { label: 'blend010_top10_chunk14', rerankDocCount: 10, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.1 },
    { label: 'blend015_top10_chunk14', rerankDocCount: 10, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.15 },
    { label: 'blend030_top10_chunk14', rerankDocCount: 10, maxChunksPerDoc: 14, scoreMode: 'blend', blendAlpha: 0.3 },
    {
        label: 'adaptive015_top8_top2gap012_chunk14',
        rerankDocCount: 8,
        maxChunksPerDoc: 14,
        scoreMode: 'blend',
        blendAlpha: 0.15,
        adaptiveDocWindow: {
            baseDocCount: 5,
            expandedDocCount: 8,
            trigger: 'top1_top2_gap',
            threshold: 0.12,
        },
    },
    {
        label: 'adaptive015_top8_top2gap020_chunk14',
        rerankDocCount: 8,
        maxChunksPerDoc: 14,
        scoreMode: 'blend',
        blendAlpha: 0.15,
        adaptiveDocWindow: {
            baseDocCount: 5,
            expandedDocCount: 8,
            trigger: 'top1_top2_gap',
            threshold: 0.2,
        },
    },
    {
        label: 'adaptive015_top10_top2gap012_chunk14',
        rerankDocCount: 10,
        maxChunksPerDoc: 14,
        scoreMode: 'blend',
        blendAlpha: 0.15,
        adaptiveDocWindow: {
            baseDocCount: 5,
            expandedDocCount: 10,
            trigger: 'top1_top2_gap',
            threshold: 0.12,
        },
    },
    {
        label: 'adaptive015_top10_top2gap020_chunk14',
        rerankDocCount: 10,
        maxChunksPerDoc: 14,
        scoreMode: 'blend',
        blendAlpha: 0.15,
        adaptiveDocWindow: {
            baseDocCount: 5,
            expandedDocCount: 10,
            trigger: 'top1_top2_gap',
            threshold: 0.2,
        },
    },
    {
        label: 'adaptive015_top10_top5gap080_chunk14',
        rerankDocCount: 10,
        maxChunksPerDoc: 14,
        scoreMode: 'blend',
        blendAlpha: 0.15,
        adaptiveDocWindow: {
            baseDocCount: 5,
            expandedDocCount: 10,
            trigger: 'top1_top5_gap',
            threshold: 0.8,
        },
    },
    {
        label: 'adaptive010_top10_top2gap012_chunk14',
        rerankDocCount: 10,
        maxChunksPerDoc: 14,
        scoreMode: 'blend',
        blendAlpha: 0.1,
        adaptiveDocWindow: {
            baseDocCount: 5,
            expandedDocCount: 10,
            trigger: 'top1_top2_gap',
            threshold: 0.12,
        },
    },
];

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = path.resolve(process.cwd(), '../Backend/models');

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
let articleMap = new Map<string, { otid: string; ot_text: string }>();

function loadDataset(datasetPath: string): DatasetCase[] {
    const absolutePath = path.resolve(process.cwd(), datasetPath);
    const raw = JSON.parse(fs.readFileSync(absolutePath, 'utf-8'));
    const datasetName = path.basename(datasetPath, '.json');
    return raw.map((item: Omit<DatasetCase, 'dataset'>) => ({
        ...item,
        dataset: datasetName,
    }));
}

function fmmTokenize(text: string): string[] {
    const tokens: string[] = [];
    let i = 0;
    while (i < text.length) {
        let matched = false;
        const maxLen = Math.min(10, text.length - i);
        for (let len = maxLen; len > 0; len--) {
            const word = text.substring(i, i + len);
            if (vocabMap.has(word)) {
                tokens.push(word);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) i++;
    }
    return tokens;
}

function getRank(result: SearchRankOutput, expectedOtid: string): number {
    const rankIndex = result.matches.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function getRankFromDocs(docs: readonly { otid: string }[], expectedOtid: string): number {
    const rankIndex = docs.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function compareRankingMetrics(a: RankingMetrics, b: RankingMetrics): number {
    if (a.mrr !== b.mrr) return b.mrr - a.mrr;
    if (a.hitAt1 !== b.hitAt1) return b.hitAt1 - a.hitAt1;
    if (a.hitAt5 !== b.hitAt5) return b.hitAt5 - a.hitAt5;
    return a.avgChunksScored - b.avgChunksScored;
}

function compareThresholdMetrics(a: ThresholdMetrics, b: ThresholdMetrics): number {
    if (a.badRejects !== b.badRejects) return a.badRejects - b.badRejects;
    if (a.goodRejects !== b.goodRejects) return b.goodRejects - a.goodRejects;
    if (a.coverage !== b.coverage) return b.coverage - a.coverage;
    return b.acceptedHitAt1 - a.acceptedHitAt1;
}

function dot(vecA: Float32Array, vecB: Float32Array): number {
    let sum = 0;
    const unrolledLimit = vecA.length - (vecA.length % 4);
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;

    for (let i = 0; i < unrolledLimit; i += 4) {
        s0 += vecA[i] * vecB[i];
        s1 += vecA[i + 1] * vecB[i + 1];
        s2 += vecA[i + 2] * vecB[i + 2];
        s3 += vecA[i + 3] * vecB[i + 3];
    }

    sum = s0 + s1 + s2 + s3;
    for (let i = unrolledLimit; i < vecA.length; i++) {
        sum += vecA[i] * vecB[i];
    }
    return sum;
}

async function loadEngine() {
    console.log('Loading metadata, vectors, and model...');
    const metadataPath = path.resolve(process.cwd(), METADATA_FILE);
    const vectorPath = path.resolve(process.cwd(), VECTOR_FILE);
    const metadataPayload = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));

    metadataList = Array.isArray(metadataPayload.data)
        ? metadataPayload.data
        : metadataPayload;
    const vocabList: string[] = metadataPayload.vocab || [];
    vocabMap.clear();
    vocabList.forEach((word, index) => vocabMap.set(word, index));

    const vectorBuffer = fs.readFileSync(vectorPath);
    vectorMatrix = new Int8Array(
        vectorBuffer.buffer,
        vectorBuffer.byteOffset,
        vectorBuffer.byteLength,
    );

    globalBM25Stats = buildBM25Stats(metadataList);
    topicPartitionIndex = buildTopicPartitionIndex(metadataList);

    if (metadataList.length > 0 && vectorMatrix.length > 0) {
        dimensions = Math.round(vectorMatrix.length / metadataList.length);
    }

    extractor = await pipeline('feature-extraction', MODEL_NAME, {
        dtype: 'q8',
        device: 'cpu',
    });

    console.log(
        `Loaded ${metadataList.length} vectors, dimensions=${dimensions}, unlabeled=${topicPartitionIndex.unlabeledCandidateIndices.length}`,
    );
}

async function embedQueries(queries: string[]): Promise<Float32Array[]> {
    if (!extractor) throw new Error('Extractor not initialized');

    const vectors: Float32Array[] = [];
    for (let start = 0; start < queries.length; start += QUERY_EMBED_BATCH_SIZE) {
        const batch = queries.slice(start, start + QUERY_EMBED_BATCH_SIZE);
        const output = await extractor(batch, {
            pooling: 'mean',
            normalize: true,
            truncation: true,
            max_length: 512,
        } as any);

        const data = output.data as Float32Array;
        for (let i = 0; i < batch.length; i++) {
            const begin = i * dimensions;
            const end = begin + dimensions;
            vectors.push(new Float32Array(data.slice(begin, end)));
        }

        const done = Math.min(start + batch.length, queries.length);
        console.log(`Embedded ${done} / ${queries.length} queries`);
    }

    return vectors;
}

function mergeCoarseMatchesIntoDocuments(
    coarseMatches: SearchRankOutput['matches'],
): RerankDocument[] {
    return coarseMatches
        .slice(0, FETCH_DOC_LIMIT)
        .map((match) => {
            const article = articleMap.get(match.otid);
            if (!article) return null;

            return {
                otid: match.otid,
                ot_text: article.ot_text,
                coarseScore: match.score,
                best_kpid: match.best_kpid,
            };
        })
        .filter((item): item is RerankDocument => Boolean(item));
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
        const candidateIndices = PARTITION_FALLBACK_ENABLED
            ? undefined
            : getCandidateIndicesForQuery(queryIntent, topicPartitionIndex);
        const queryWords = Array.from(
            new Set([...fmmTokenize(testCase.query), ...queryIntent.normalizedTerms]),
        );
        const querySparse = getQuerySparse(queryWords, vocabMap);
        const queryYearWordIds: number[] = [];

        queryIntent.years.map(String).forEach((year) => {
            const wordId = vocabMap.get(year);
            if (wordId !== undefined) queryYearWordIds.push(wordId);
        });

        const result = searchAndRank({
            queryVector: queryVectors[index],
            querySparse,
            queryYearWordIds,
            queryIntent,
            metadata: metadataList,
            vectorMatrix,
            dimensions,
            currentTimestamp: Date.now() / 1000,
            bm25Stats: globalBM25Stats,
            candidateIndices,
        });

        cache.push({
            testCase,
            queryIntent,
            queryVector: queryVectors[index],
            coarseRank: getRank(result, testCase.expected_otid),
            coarseTop15Rank: getRankFromDocs(
                result.matches.slice(0, FETCH_DOC_LIMIT),
                testCase.expected_otid,
            ),
            coarseMatches: result.matches.slice(0, FETCH_DOC_LIMIT).map((match) => ({
                otid: match.otid,
                ot_text: '',
                coarseScore: match.score,
                best_kpid: match.best_kpid,
            })),
        });

        if ((index + 1) % 50 === 0) {
            console.log(`Searched ${index + 1} / ${testCases.length} queries`);
        }
    }

    return cache;
}

function loadArticleMap(neededOtids: Set<string>) {
    console.log(`Loading article texts for ${neededOtids.size} OTIDs...`);
    const sourcePath = path.resolve(process.cwd(), ARTICLE_SOURCE_FILE);
    const raw = JSON.parse(fs.readFileSync(sourcePath, 'utf-8'));
    articleMap = new Map();

    raw.forEach((item: any) => {
        const otid = item.otid;
        if (!neededOtids.has(otid)) return;
        articleMap.set(otid, {
            otid,
            ot_text: item.ot_text || '',
        });
    });

    console.log(`Loaded ${articleMap.size} article texts`);
}

function attachArticles(searchCache: SearchCache[]): SearchCache[] {
    return searchCache.map((item) => ({
        ...item,
        coarseMatches: item.coarseMatches
            .map((match) => {
                const article = articleMap.get(match.otid);
                if (!article) return null;
                return {
                    ...match,
                    ot_text: article.ot_text,
                };
            })
            .filter((doc): doc is RerankDocument => Boolean(doc)),
    }));
}

async function buildChunkDocMap(searchCache: SearchCache[]): Promise<Map<string, ChunkDoc>> {
    if (!extractor) throw new Error('Extractor not initialized');

    const neededOtids = new Set<string>();
    searchCache.forEach((item) => {
        item.coarseMatches.forEach((doc) => neededOtids.add(doc.otid));
    });

    const docEntries: ChunkDoc[] = [];
    const batchTexts: string[] = [];
    const batchOffsets: Array<{ otid: string; text: string }> = [];
    const maxChunkLimit = Math.max(...CHUNK_LIMITS);

    neededOtids.forEach((otid) => {
        const article = articleMap.get(otid);
        if (!article) return;
        const chunkTexts = splitIntoSemanticChunks(article.ot_text || '', CHUNK_MAX_LEN)
            .map((chunk) => (chunk || '').trim())
            .filter(Boolean)
            .slice(0, maxChunkLimit);
        docEntries.push({
            otid,
            chunkTexts,
            chunkVectors: [],
        });
        chunkTexts.forEach((text) => {
            batchTexts.push(text);
            batchOffsets.push({ otid, text });
        });
    });

    const chunkDocMap = new Map<string, ChunkDoc>(
        docEntries.map((item) => [item.otid, item]),
    );

    console.log(
        `Embedding ${batchTexts.length} chunks across ${docEntries.length} documents...`,
    );

    let globalChunkIndex = 0;
    for (let start = 0; start < batchTexts.length; start += CHUNK_EMBED_BATCH_SIZE) {
        const batch = batchTexts.slice(start, start + CHUNK_EMBED_BATCH_SIZE);
        const output = await extractor(batch, {
            pooling: 'mean',
            normalize: true,
            truncation: true,
            max_length: 512,
        } as any);

        const data = output.data as Float32Array;
        for (let i = 0; i < batch.length; i++) {
            const begin = i * dimensions;
            const end = begin + dimensions;
            const vector = new Float32Array(data.slice(begin, end));
            const meta = batchOffsets[globalChunkIndex++];
            const doc = chunkDocMap.get(meta.otid);
            if (!doc) continue;
            doc.chunkVectors.push(vector);
        }

        const done = Math.min(start + batch.length, batchTexts.length);
        console.log(`Embedded ${done} / ${batchTexts.length} chunks`);
    }

    return chunkDocMap;
}

function buildCaseRerankCache(
    searchCache: SearchCache[],
    chunkDocMap: Map<string, ChunkDoc>,
): CaseRerankCache[] {
    return searchCache.map((item) => {
        const chunkScoresByLimit: Record<number, ChunkScoreCache[]> = {
            6: [],
            10: [],
            14: [],
        };

        item.coarseMatches.forEach((doc) => {
            const chunkDoc = chunkDocMap.get(doc.otid);
            const vectors = chunkDoc?.chunkVectors || [];

            for (const limit of CHUNK_LIMITS) {
                const usableVectors = vectors.slice(0, limit);
                let bestRawScore = Number.NEGATIVE_INFINITY;
                for (const chunkVector of usableVectors) {
                    const score = dot(item.queryVector, chunkVector);
                    if (score > bestRawScore) {
                        bestRawScore = score;
                    }
                }

                if (!Number.isFinite(bestRawScore)) {
                    bestRawScore = -1;
                }

                chunkScoresByLimit[limit].push({
                    rawScore: bestRawScore,
                    normalizedScore: normalizeSnippetScore(bestRawScore),
                });
            }
        });

        return {
            testCase: item.testCase,
            coarseRank: item.coarseRank,
            coarseTop15Rank: item.coarseTop15Rank,
            coarseMatches: item.coarseMatches,
            chunkScoresByLimit,
        };
    });
}

function scoreCaseWithConfig(
    item: CaseRerankCache,
    config: RankingConfig,
): ScoredCaseResult {
    if (config.scoreMode === 'coarse_only') {
        return {
            finalDocs: item.coarseMatches.map((doc) => ({
                ...doc,
                rawSnippetScore: -1,
                normalizedSnippetScore: 0,
                finalScore: doc.coarseScore,
            })),
            finalRank: item.coarseTop15Rank,
            topConfidence: item.coarseMatches[0]?.coarseScore ?? 0,
            rerankedDocCount: 0,
            chunksScored: 0,
        };
    }

    let targetDocCount = config.rerankDocCount;
    if (config.adaptiveDocWindow) {
        const top1 = item.coarseMatches[0]?.coarseScore ?? Number.POSITIVE_INFINITY;
        const top2 = item.coarseMatches[1]?.coarseScore ?? Number.NEGATIVE_INFINITY;
        const top5Index = Math.min(4, item.coarseMatches.length - 1);
        const top5 = top5Index >= 0
            ? item.coarseMatches[top5Index]?.coarseScore ?? Number.NEGATIVE_INFINITY
            : Number.NEGATIVE_INFINITY;

        const shouldExpand =
            config.adaptiveDocWindow.trigger === 'top1_top2_gap'
                ? top1 - top2 <= config.adaptiveDocWindow.threshold
                : top1 - top5 <= config.adaptiveDocWindow.threshold;

        targetDocCount = shouldExpand
            ? config.adaptiveDocWindow.expandedDocCount
            : config.adaptiveDocWindow.baseDocCount;
    }

    const rerankDocCount = Math.min(targetDocCount, item.coarseMatches.length);
    const rerankPool: ScoredRerankDoc[] = item.coarseMatches
        .slice(0, rerankDocCount)
        .map((doc, index) => ({
            ...doc,
            rawSnippetScore:
                item.chunkScoresByLimit[config.maxChunksPerDoc][index]?.rawScore ?? -1,
            normalizedSnippetScore:
                item.chunkScoresByLimit[config.maxChunksPerDoc][index]?.normalizedScore ?? 0,
            finalScore: doc.coarseScore,
        }));
    const coarseTail: ScoredRerankDoc[] = item.coarseMatches
        .slice(rerankDocCount)
        .map((doc) => ({
            ...doc,
            rawSnippetScore: -1,
            normalizedSnippetScore: 0,
            finalScore: doc.coarseScore,
        }));

    if (config.scoreMode === 'current_ui') {
        rerankPool.forEach((doc) => {
            doc.finalScore = Math.max(doc.coarseScore, doc.rawSnippetScore);
        });
    } else if (config.scoreMode === 'snippet_only') {
        rerankPool.forEach((doc) => {
            doc.finalScore = doc.normalizedSnippetScore;
        });
        rerankPool.sort((a, b) => b.finalScore - a.finalScore);
    } else if (config.scoreMode === 'blend') {
        const alpha = config.blendAlpha ?? 0.3;
        const coarseNorm = normalizeMinMax(rerankPool.map((doc) => doc.coarseScore));
        rerankPool.forEach((doc, index) => {
            doc.finalScore =
                alpha * coarseNorm[index] +
                (1 - alpha) * doc.normalizedSnippetScore;
        });
        rerankPool.sort((a, b) => b.finalScore - a.finalScore);
    }

    const finalDocs = rerankPool.concat(coarseTail);
    return {
        finalDocs,
        finalRank: getRankFromDocs(finalDocs, item.testCase.expected_otid),
        topConfidence: finalDocs[0]?.finalScore ?? 0,
        rerankedDocCount: rerankDocCount,
        chunksScored: rerankDocCount * config.maxChunksPerDoc,
    };
}

function evaluateRankingConfig(
    cases: CaseRerankCache[],
    config: RankingConfig,
): RankingMetrics {
    let hitAt1 = 0;
    let hitAt5 = 0;
    let reciprocalRankSum = 0;
    let rescueHitAt1 = 0;
    let rescueAnyRank = 0;
    let regressionsFromCoarse = 0;
    let totalRerankedDocs = 0;
    let totalChunksScored = 0;

    for (const item of cases) {
        const scored = scoreCaseWithConfig(item, config);
        const finalRank = scored.finalRank;
        totalRerankedDocs += scored.rerankedDocCount;
        totalChunksScored += scored.chunksScored;

        if (finalRank === 1) hitAt1 += 1;
        if (finalRank <= 5) hitAt5 += 1;
        if (Number.isFinite(finalRank)) reciprocalRankSum += 1 / finalRank;

        if (item.coarseTop15Rank !== 1 && finalRank === 1) rescueHitAt1 += 1;
        if (Number.isFinite(finalRank) && finalRank < item.coarseTop15Rank) rescueAnyRank += 1;
        if (finalRank > item.coarseTop15Rank) regressionsFromCoarse += 1;
    }

    const totalCases = cases.length || 1;
    return {
        label: config.label,
        hitAt1: (hitAt1 / totalCases) * 100,
        hitAt5: (hitAt5 / totalCases) * 100,
        mrr: reciprocalRankSum / totalCases,
        rescueHitAt1,
        rescueAnyRank,
        regressionsFromCoarse,
        avgRerankedDocs: totalRerankedDocs / totalCases,
        avgChunksScored: totalChunksScored / totalCases,
        totalCases,
    };
}

function evaluateThresholds(
    cases: CaseRerankCache[],
    config: RankingConfig,
): ThresholdMetrics[] {
    if (config.scoreMode === 'coarse_only') {
        return [];
    }

    const thresholds = [...THRESHOLD_CANDIDATES];
    const results: ThresholdMetrics[] = [];

    for (const threshold of thresholds) {
        let acceptedCount = 0;
        let acceptedHitAt1 = 0;
        let acceptedHitAt5 = 0;
        let reciprocalRankSum = 0;
        let badRejects = 0;
        let goodRejects = 0;

        for (const item of cases) {
            const scored = scoreCaseWithConfig(item, config);
            const finalRank = scored.finalRank;
            const topConfidence = scored.topConfidence;
            const rejected = topConfidence < threshold;

            if (rejected) {
                if (finalRank === 1) badRejects += 1;
                else goodRejects += 1;
                continue;
            }

            acceptedCount += 1;
            if (finalRank === 1) acceptedHitAt1 += 1;
            if (finalRank <= 5) acceptedHitAt5 += 1;
            if (Number.isFinite(finalRank)) reciprocalRankSum += 1 / finalRank;
        }

        const totalCases = cases.length || 1;
        results.push({
            threshold,
            coverage: acceptedCount / totalCases,
            rejectRate: 1 - acceptedCount / totalCases,
            acceptedHitAt1: acceptedCount === 0 ? 0 : acceptedHitAt1 / acceptedCount,
            acceptedHitAt5: acceptedCount === 0 ? 0 : acceptedHitAt5 / acceptedCount,
            acceptedMRR: acceptedCount === 0 ? 0 : reciprocalRankSum / acceptedCount,
            badRejects,
            goodRejects,
            totalCases,
        });
    }

    return results.sort(compareThresholdMetrics);
}

function buildHardCaseSummary(
    cases: CaseRerankCache[],
    productionConfig: RankingConfig,
    offlineBestConfig: RankingConfig,
): HardCaseSummary {
    const currentUiConfig = pickConfigByLabel(CURRENT_RANKING_CONFIG.label);
    const dominantReasons: HardCaseSummary['dominantReasons'] = {
        expected_missing_from_top15: 0,
        expected_outside_rerank_window: 0,
        rerank_demoted_correct_doc: 0,
        rerank_failed_to_promote: 0,
        coarse_already_wrong_after_top15: 0,
    };
    const topRegressions: HardCaseDiagnostic[] = [];
    const topMissedOpportunities: HardCaseDiagnostic[] = [];
    let improvedHitAt1 = 0;
    let regressedHitAt1 = 0;
    let unchanged = 0;
    const coarseRankBuckets = {
        top1: 0,
        top2to5: 0,
        top6to15: 0,
        missTop15: 0,
    };
    const productionRankBuckets = {
        top1: 0,
        top2to5: 0,
        top6to15: 0,
        missTop15: 0,
    };
    const rerankWindowBuckets = {
        expectedInsideTop5: 0,
        expectedOutsideTop5ButInsideTop15: 0,
        expectedMissingTop15: 0,
    };

    for (const item of cases) {
        const currentUi = scoreCaseWithConfig(item, currentUiConfig);
        const production = scoreCaseWithConfig(item, productionConfig);
        const offlineBest = scoreCaseWithConfig(item, offlineBestConfig);

        if (item.coarseTop15Rank === 1) coarseRankBuckets.top1 += 1;
        else if (item.coarseTop15Rank <= 5) coarseRankBuckets.top2to5 += 1;
        else if (item.coarseTop15Rank <= 15) coarseRankBuckets.top6to15 += 1;
        else coarseRankBuckets.missTop15 += 1;

        if (production.finalRank === 1) productionRankBuckets.top1 += 1;
        else if (production.finalRank <= 5) productionRankBuckets.top2to5 += 1;
        else if (production.finalRank <= 15) productionRankBuckets.top6to15 += 1;
        else productionRankBuckets.missTop15 += 1;

        if (!Number.isFinite(item.coarseTop15Rank)) {
            rerankWindowBuckets.expectedMissingTop15 += 1;
        } else if (item.coarseTop15Rank <= productionConfig.rerankDocCount) {
            rerankWindowBuckets.expectedInsideTop5 += 1;
        } else {
            rerankWindowBuckets.expectedOutsideTop5ButInsideTop15 += 1;
        }

        if (production.finalRank === 1 && item.coarseTop15Rank !== 1) {
            improvedHitAt1 += 1;
        } else if (production.finalRank !== 1 && item.coarseTop15Rank === 1) {
            regressedHitAt1 += 1;
        } else {
            unchanged += 1;
        }

        let dominantReason: HardCaseDiagnostic['dominantReason'];
        if (!Number.isFinite(item.coarseTop15Rank)) {
            dominantReason = 'expected_missing_from_top15';
        } else if (
            item.coarseTop15Rank > productionConfig.rerankDocCount &&
            production.finalRank === item.coarseTop15Rank
        ) {
            dominantReason = 'expected_outside_rerank_window';
        } else if (item.coarseTop15Rank === 1 && production.finalRank > 1) {
            dominantReason = 'rerank_demoted_correct_doc';
        } else if (production.finalRank === item.coarseTop15Rank) {
            dominantReason = 'rerank_failed_to_promote';
        } else {
            dominantReason = 'coarse_already_wrong_after_top15';
        }

        dominantReasons[dominantReason] += 1;

        const expectedDoc = production.finalDocs.find(
            (doc) => doc.otid === item.testCase.expected_otid,
        );
        const expectedPosition = production.finalDocs.findIndex(
            (doc) => doc.otid === item.testCase.expected_otid,
        );

        const diagnostic: HardCaseDiagnostic = {
            query: item.testCase.query,
            expectedOtid: item.testCase.expected_otid,
            dataset: item.testCase.dataset,
            queryType: item.testCase.query_type,
            coarseRank: item.coarseRank,
            coarseTop15Rank: item.coarseTop15Rank,
            currentUiRank: currentUi.finalRank,
            productionRank: production.finalRank,
            offlineBestRank: offlineBest.finalRank,
            dominantReason,
            coarseTop3: item.coarseMatches.slice(0, 3).map((doc) => ({
                otid: doc.otid,
                coarseScore: Number(doc.coarseScore.toFixed(4)),
            })),
            productionTop3: production.finalDocs.slice(0, 3).map((doc) => ({
                otid: doc.otid,
                coarseScore: Number(doc.coarseScore.toFixed(4)),
                snippetScore: Number(doc.normalizedSnippetScore.toFixed(4)),
                finalScore: Number(doc.finalScore.toFixed(4)),
            })),
            expectedDocSnapshot: expectedDoc
                ? {
                    coarseScore: Number(expectedDoc.coarseScore.toFixed(4)),
                    snippetScore: Number(expectedDoc.normalizedSnippetScore.toFixed(4)),
                    finalScore: Number(expectedDoc.finalScore.toFixed(4)),
                    productionPosition:
                        expectedPosition === -1 ? Number.POSITIVE_INFINITY : expectedPosition + 1,
                }
                : undefined,
        };

        if (item.coarseTop15Rank === 1 && production.finalRank > 1) {
            topRegressions.push(diagnostic);
        } else if (
            Number.isFinite(item.coarseTop15Rank) &&
            item.coarseTop15Rank > 1 &&
            production.finalRank >= item.coarseTop15Rank
        ) {
            topMissedOpportunities.push(diagnostic);
        }
    }

    topRegressions.sort((a, b) => b.productionRank - a.productionRank);
    topMissedOpportunities.sort((a, b) => {
        const gapA = a.productionRank - a.coarseTop15Rank;
        const gapB = b.productionRank - b.coarseTop15Rank;
        if (gapA !== gapB) return gapB - gapA;
        return b.coarseTop15Rank - a.coarseTop15Rank;
    });

    return {
        productionLabel: productionConfig.label,
        offlineBestLabel: offlineBestConfig.label,
        productionVsCoarse: {
            improvedHitAt1,
            regressedHitAt1,
            unchanged,
        },
        coarseRankBuckets,
        productionRankBuckets,
        rerankWindowBuckets,
        dominantReasons,
        topRegressions: topRegressions.slice(0, 20),
        topMissedOpportunities: topMissedOpportunities.slice(0, 20),
    };
}

function selectBestRanking(cases: CaseRerankCache[]): RankingMetrics[] {
    return CANDIDATE_RANKING_CONFIGS
        .map((config) => evaluateRankingConfig(cases, config))
        .sort(compareRankingMetrics);
}

function pickConfigByLabel(label: string): RankingConfig {
    const found = CANDIDATE_RANKING_CONFIGS.find((config) => config.label === label);
    if (!found) throw new Error(`Unknown config label: ${label}`);
    return found;
}

function buildSplitReport(split: string, cases: CaseRerankCache[]): SplitReport {
    const rankingResults = selectBestRanking(cases);
    const bestRanking = rankingResults[0];
    const bestConfig = pickConfigByLabel(bestRanking.label);
    const thresholdResults = evaluateThresholds(cases, bestConfig);
    const productionConfig = pickConfigByLabel(PRODUCTION_RANKING_LABEL);
    const offlineBestConfig = pickConfigByLabel(OFFLINE_BEST_RANKING_LABEL);

    return {
        split,
        bestRanking,
        rankingResults,
        thresholdResults,
        selectedThreshold: thresholdResults[0],
        hardCaseSummary:
            split === 'holdout'
                ? buildHardCaseSummary(cases, productionConfig, offlineBestConfig)
                : undefined,
    };
}

async function main() {
    const start = performance.now();
    await loadEngine();

    const tuneCases = TUNE_DATASETS.flatMap(loadDataset);
    const holdoutCases = HOLDOUT_DATASETS.flatMap(loadDataset);
    const allCases = [...tuneCases, ...holdoutCases];

    const searchCache = await buildSearchCache(allCases);
    const neededOtids = new Set<string>();
    searchCache.forEach((item) => {
        item.coarseMatches.forEach((match) => neededOtids.add(match.otid));
    });

    loadArticleMap(neededOtids);
    const attachedCache = attachArticles(searchCache);
    const chunkDocMap = await buildChunkDocMap(attachedCache);
    const rerankCache = buildCaseRerankCache(attachedCache, chunkDocMap);

    const tuneRerankCache = rerankCache.slice(0, tuneCases.length);
    const holdoutRerankCache = rerankCache.slice(tuneCases.length);

    const tuneReport = buildSplitReport('tune', tuneRerankCache);
    const holdoutReport = buildSplitReport('holdout', holdoutRerankCache);
    const combinedReport = buildSplitReport('combined', rerankCache);

    const report = {
        generatedAt: new Date().toISOString(),
        runtimeMs: performance.now() - start,
        datasets: {
            tune: tuneCases.map((item) => item.dataset),
            holdout: holdoutCases.map((item) => item.dataset),
        },
        metadataCount: metadataList.length,
        fetchedDocLimit: FETCH_DOC_LIMIT,
        rankingConfigs: CANDIDATE_RANKING_CONFIGS,
        tune: tuneReport,
        holdout: holdoutReport,
        combined: combinedReport,
    };

    const resultDir = path.resolve(process.cwd(), 'scripts/results');
    fs.mkdirSync(resultDir, { recursive: true });
    const outputPath = path.join(
        resultDir,
        `rerank_calibration_${DATASET_VERSION}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), 'utf-8');

    console.log('\n===== Rerank Calibration Summary =====');
    console.log(
        `Tune best: ${tuneReport.bestRanking.label} | Hit@1=${tuneReport.bestRanking.hitAt1.toFixed(2)}% | Hit@5=${tuneReport.bestRanking.hitAt5.toFixed(2)}% | MRR=${tuneReport.bestRanking.mrr.toFixed(4)}`,
    );
    console.log(
        `Holdout best: ${holdoutReport.bestRanking.label} | Hit@1=${holdoutReport.bestRanking.hitAt1.toFixed(2)}% | Hit@5=${holdoutReport.bestRanking.hitAt5.toFixed(2)}% | MRR=${holdoutReport.bestRanking.mrr.toFixed(4)}`,
    );
    console.log(
        `Combined best: ${combinedReport.bestRanking.label} | Hit@1=${combinedReport.bestRanking.hitAt1.toFixed(2)}% | Hit@5=${combinedReport.bestRanking.hitAt5.toFixed(2)}% | MRR=${combinedReport.bestRanking.mrr.toFixed(4)}`,
    );
    if (combinedReport.selectedThreshold) {
        console.log(
            `Suggested threshold: ${combinedReport.selectedThreshold.threshold.toFixed(2)} | coverage=${(combinedReport.selectedThreshold.coverage * 100).toFixed(2)}% | badRejects=${combinedReport.selectedThreshold.badRejects} | goodRejects=${combinedReport.selectedThreshold.goodRejects}`,
        );
    }
    console.log(`Report saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

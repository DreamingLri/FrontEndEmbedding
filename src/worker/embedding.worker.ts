import { pipeline, env } from '@huggingface/transformers';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';
import type {
    BM25Stats,
    Metadata,
    ParsedQueryIntent,
    SearchRankOutput
} from './vector_engine';
import {
    buildBM25Stats,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank
} from './vector_engine';
import {
    buildTopicPartitionIndex,
    getCandidateIndicesForQuery,
    type TopicPartitionIndex,
} from './topic_partition';
import {
    normalizeMinMax,
    normalizeSnippetScore,
    splitIntoSemanticChunks,
} from './rerank_helpers';

const MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';
const DEBUG_SEARCH = false;
const WORKER_BUILD_TAG = 'intent-debug-20260324-4';
const PARTITION_FALLBACK_ENABLED = false;
const PARTITION_FALLBACK_SCORE_THRESHOLD = 1.05;
const PARTITION_FALLBACK_GAP_THRESHOLD = 0.03;
const PARTITION_FALLBACK_SCAN_RATIO_THRESHOLD = 0.85;
const BASE_RERANK_DOC_COUNT = 5;
const EXPANDED_RERANK_DOC_COUNT = 10;
const ADAPTIVE_RERANK_TOP5_GAP_THRESHOLD = 0.8;
const RERANK_MAX_CHUNKS_PER_DOC = 14;
const RERANK_BLEND_ALPHA = 0.15;
const RERANK_BEST_SENTENCE_THRESHOLD = 0.4;
let DIMENSIONS = 768;

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL(import.meta.env.BASE_URL + 'models/', self.location.origin).href;

const isSecureContext = typeof self.caches !== 'undefined';
env.useBrowserCache = isSecureContext;

try {
    const onnx = (env as any).backends?.onnx;
    if (onnx) {
        onnx.wasm.wasmPaths = new URL(import.meta.env.BASE_URL + 'wasm/', self.location.origin).href;
        onnx.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 8);
        onnx.wasm.proxy = false;
    }
} catch (_) {
    // Ignore optional ONNX runtime configuration failures.
}

let extractor: FeatureExtractionPipeline | null = null;
let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let globalBM25Stats: BM25Stats | null = null;
let topicPartitionIndex: TopicPartitionIndex = {
    topicCandidateIndex: new Map<string, number[]>(),
    unlabeledCandidateIndices: [],
    metadataCount: 0,
};
let lastEmbeddedQuery = '';
let lastQueryVector: Float32Array | null = null;

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

function getAdaptiveRerankDocCount(
    results: Array<{ coarseScore?: number }>,
): number {
    if (results.length <= BASE_RERANK_DOC_COUNT) {
        return results.length;
    }

    const top1 = results[0]?.coarseScore ?? 0;
    const top5Index = Math.min(BASE_RERANK_DOC_COUNT - 1, results.length - 1);
    const top5 = results[top5Index]?.coarseScore ?? top1;
    const shouldExpand = top1 - top5 <= ADAPTIVE_RERANK_TOP5_GAP_THRESHOLD;

    return Math.min(
        shouldExpand ? EXPANDED_RERANK_DOC_COUNT : BASE_RERANK_DOC_COUNT,
        results.length
    );
}

async function embedQuery(query: string): Promise<Float32Array> {
    if (!extractor) throw new Error('\u67e5\u8be2\u6a21\u578b\u672a\u521d\u59cb\u5316');

    const output = await extractor(query, {
        pooling: 'mean', normalize: true, truncation: true, max_length: 512
    } as any);
    const queryVector = new Float32Array(output.data as Float32Array);
    lastEmbeddedQuery = query;
    lastQueryVector = queryVector;
    return queryVector;
}

function resetQueryCache() {
    lastEmbeddedQuery = '';
    lastQueryVector = null;
}

function getPartitionFallbackReason(
    queryIntent: ParsedQueryIntent,
    candidateIndices: readonly number[] | undefined,
    searchResult: SearchRankOutput
): string | undefined {
    if (!PARTITION_FALLBACK_ENABLED) {
        return undefined;
    }

    if (!candidateIndices || candidateIndices.length === 0 || metadataList.length === 0) {
        return undefined;
    }

    const scanRatio = candidateIndices.length / metadataList.length;
    if (scanRatio >= PARTITION_FALLBACK_SCAN_RATIO_THRESHOLD) {
        return undefined;
    }

    if (queryIntent.confidence >= 1) {
        return undefined;
    }

    if (searchResult.matches.length < 2) {
        return undefined;
    }

    const top1 = searchResult.matches[0]?.score ?? Number.NEGATIVE_INFINITY;
    const top2 = searchResult.matches[1]?.score ?? Number.NEGATIVE_INFINITY;
    const scoreGap = top1 - top2;

    if (
        top1 < PARTITION_FALLBACK_SCORE_THRESHOLD &&
        scoreGap < PARTITION_FALLBACK_GAP_THRESHOLD
    ) {
        return 'partition_ambiguous_top1';
    }

    return undefined;
}

self.onmessage = async (event: MessageEvent) => {
    const { type, payload, taskId } = event.data;

    try {
        switch (type) {
            case 'INIT':
                await handleInit(payload, taskId);
                break;
            case 'SEARCH':
                await handleSearch(payload, taskId);
                break;
            case 'RERANK':
                await handleRerank(payload, taskId);
                break;
        }
    } catch (e: any) {
        self.postMessage({ taskId, status: 'error', error: e.message });
    }
};

async function handleInit(payload: any, taskId?: string) {
    try {
        let rawMetadata: any;

        if (payload.metadata && payload.vectorMatrix) {
            self.postMessage({ taskId, status: 'loading', message: '解析已传入的元数据与向量矩阵...' });
            rawMetadata = payload.metadata;
            vectorMatrix = payload.vectorMatrix;
        } else if (payload.metadataUrl && payload.vectorsUrl) {
            self.postMessage({ taskId, status: 'loading', message: '加载元数据与向量矩阵...' });
            const [metaRes, vecRes] = await Promise.all([
                fetch(payload.metadataUrl),
                fetch(payload.vectorsUrl)
            ]);
            if (!metaRes.ok || !vecRes.ok) throw new Error('网络请求资源失败');
            rawMetadata = await metaRes.json();
            const vecBuffer = await vecRes.arrayBuffer();
            vectorMatrix = new Int8Array(vecBuffer);
        } else {
            throw new Error('INIT payload 格式错误');
        }

        metadataList = Array.isArray(rawMetadata) ? rawMetadata : (rawMetadata.data || []);
        const vocabList: string[] = rawMetadata.vocab || [];
        vocabMap.clear();
        vocabList.forEach((word, index) => vocabMap.set(word, index));
        topicPartitionIndex = buildTopicPartitionIndex(metadataList);
        resetQueryCache();

        self.postMessage({ taskId, status: 'loading', message: '构建 BM25 统计...' });
        globalBM25Stats = buildBM25Stats(metadataList);

        if (metadataList.length > 0 && vectorMatrix && vectorMatrix.length > 0) {
            DIMENSIONS = Math.round(vectorMatrix.length / metadataList.length);
        }

        self.postMessage({ taskId, status: 'loading', message: `探测推理后端 (${DIMENSIONS}维)...` });
        let device = 'wasm';
        try {
            if ((navigator as any).gpu) {
                const adapter = await (navigator as any).gpu.requestAdapter();
                if (adapter) device = 'webgpu';
            }
        } catch (_) {
            // Fall back to WASM.
        }

        const dtype = device === 'webgpu' ? 'fp16' : 'q8';
        self.postMessage({ taskId, status: 'loading', message: `加载 Embedding 模型 (${device}/${dtype})...` });

        extractor = await pipeline('feature-extraction', MODEL_NAME, {
            dtype: dtype as any,
            device: device as any
        });

        self.postMessage({
            taskId,
            status: 'ready',
            message: DEBUG_SEARCH
                ? `AI 引擎就绪 [${WORKER_BUILD_TAG}, ${device}/${dtype}, ${metadataList.length} 条, ${DIMENSIONS}维]`
                : `AI 引擎就绪 [${device}/${dtype}, ${metadataList.length} 条, ${DIMENSIONS}维]`
        });
    } catch (err: any) {
        self.postMessage({ taskId, status: 'error', error: `初始化失败: ${err.message}` });
    }
}

async function handleSearch(query: string, taskId?: string) {
    if (!vectorMatrix || !globalBM25Stats || !extractor) {
        throw new Error('引擎尚未初始化完成');
    }

    const t0 = performance.now();

    const queryVector = await embedQuery(query);
    const queryIntent = parseQueryIntent(query);
    const candidateIndices = getCandidateIndicesForQuery(queryIntent, topicPartitionIndex);
    const queryWords = Array.from(new Set([
        ...fmmTokenize(query),
        ...queryIntent.normalizedTerms
    ]));
    const querySparse = getQuerySparse(queryWords, vocabMap);

    const queryYearWordIds: number[] = [];
    queryIntent.years.map(String).forEach((year) => {
        const id = vocabMap.get(year);
        if (id !== undefined) queryYearWordIds.push(id);
    });

    const partialSearchResult: SearchRankOutput = searchAndRank({
        queryVector,
        querySparse,
        queryYearWordIds,
        queryIntent,
        metadata: metadataList,
        vectorMatrix,
        dimensions: DIMENSIONS,
        currentTimestamp: Date.now() / 1000,
        bm25Stats: globalBM25Stats,
        candidateIndices
    });

    const initialItemsScanned = candidateIndices?.length ?? metadataList.length;
    const fallbackReason = getPartitionFallbackReason(
        queryIntent,
        candidateIndices,
        partialSearchResult
    );
    const partitionFallbackTriggered = Boolean(fallbackReason);

    if (partitionFallbackTriggered) {
        self.postMessage({
            taskId,
            status: 'info',
            message: `涓婚鍒嗗尯鍙洖淇″績涓嶈冻锛屽洖閫€鍏ㄩ噺鎵弿 (${fallbackReason})`
        });
    }

    const searchResult: SearchRankOutput = partitionFallbackTriggered
        ? searchAndRank({
            queryVector,
            querySparse,
            queryYearWordIds,
            queryIntent,
            metadata: metadataList,
            vectorMatrix,
            dimensions: DIMENSIONS,
            currentTimestamp: Date.now() / 1000,
            bm25Stats: globalBM25Stats
        })
        : partialSearchResult;

    const finalItemsScanned = partitionFallbackTriggered
        ? metadataList.length
        : initialItemsScanned;
    const totalItemsScanned = partitionFallbackTriggered
        ? initialItemsScanned + finalItemsScanned
        : initialItemsScanned;

    self.postMessage({
        taskId,
        status: 'search_complete',
        result: searchResult,
        stats: {
            elapsedMs: (performance.now() - t0).toFixed(1),
            itemsScanned: totalItemsScanned,
            partitionUsed: Boolean(candidateIndices),
            partitionCandidateCount: candidateIndices?.length,
            partitionFallbackTriggered,
            partitionFallbackReason: fallbackReason,
            initialItemsScanned,
            finalItemsScanned,
            workerBuildTag: DEBUG_SEARCH ? WORKER_BUILD_TAG : undefined,
            queryIntent: DEBUG_SEARCH ? queryIntent : undefined,
            topMatches: DEBUG_SEARCH
                ? searchResult.matches.slice(0, 5).map((match) => ({
                    otid: match.otid,
                    score: Number(match.score.toFixed(4)),
                    best_kpid: match.best_kpid,
                }))
                : undefined,
        }
    });
}

async function handleRerank(payload: { query: string, documents: any[] }, taskId?: string) {
    if (!extractor) throw new Error('引擎尚未初始化');

    const { query, documents } = payload;
    const t0 = performance.now();

    const queryVector =
        lastEmbeddedQuery === query && lastQueryVector
            ? lastQueryVector
            : await embedQuery(query);

    self.postMessage({
        taskId,
        status: 'progress',
        message: '正在为高亮结果提取更相关的原文片段...'
    });

    const results = documents.map((doc: any) => {
        let defaultPoint = '暂无要点';
        if (doc.best_kpid && Array.isArray(doc.kps)) {
            const hitKp = doc.kps.find((kp: any) => kp.kpid === doc.best_kpid);
            if (hitKp?.kp_text) defaultPoint = hitKp.kp_text;
        }

        return {
            ...doc,
            coarseScore: doc.score,
            displayScore: doc.score,
            rerankScore: 0,
            snippetScore: 0,
            confidenceScore: 0,
            bestPoint: defaultPoint,
            bestSentence: ''
        };
    });

    const rerankDocCount = getAdaptiveRerankDocCount(results);
    const rerankDocs = results.slice(0, rerankDocCount);
    const batchChunks: { text: string, docIdx: number }[] = [];

    for (let j = 0; j < rerankDocs.length; j++) {
        const textChunks = splitIntoSemanticChunks(
            rerankDocs[j].ot_text || '',
            150
        ).slice(0, RERANK_MAX_CHUNKS_PER_DOC);
        textChunks.forEach((chunk) => {
            const normalizedChunk = (chunk || '').trim();
            if (normalizedChunk) {
                batchChunks.push({ text: normalizedChunk, docIdx: j });
            }
        });
    }

    if (batchChunks.length > 0) {
        const batchTexts = batchChunks.map((c) => c.text);
        const batchOutputs = await extractor(batchTexts, {
            pooling: 'mean', normalize: true, truncation: true, max_length: 512
        } as any);

        const pureData = (batchOutputs.data as Float32Array).subarray(0, batchChunks.length * DIMENSIONS);
        const rawDocumentScores = new Float32Array(rerankDocs.length).fill(-1);
        const documentBestSentence = new Array<string>(rerankDocs.length).fill('');

        for (let k = 0; k < batchChunks.length; k++) {
            const chunkVec = pureData.subarray(k * DIMENSIONS, (k + 1) * DIMENSIONS);
            let score = 0;
            for (let d = 0; d < DIMENSIONS; d++) {
                score += queryVector[d] * chunkVec[d];
            }

            const docIdx = batchChunks[k].docIdx;
            if (score > rawDocumentScores[docIdx]) {
                rawDocumentScores[docIdx] = score;
                documentBestSentence[docIdx] = batchChunks[k].text;
            }
        }

        const coarseNorm = normalizeMinMax(
            rerankDocs.map((doc) => doc.coarseScore ?? 0)
        );

        for (let j = 0; j < rerankDocs.length; j++) {
            const normalizedSnippetScore = normalizeSnippetScore(rawDocumentScores[j]);
            const blendedScore =
                RERANK_BLEND_ALPHA * coarseNorm[j] +
                (1 - RERANK_BLEND_ALPHA) * normalizedSnippetScore;

            rerankDocs[j].snippetScore = normalizedSnippetScore;
            rerankDocs[j].confidenceScore = blendedScore;
            rerankDocs[j].rerankScore = blendedScore;
            rerankDocs[j].displayScore = blendedScore;

            if (
                normalizedSnippetScore > RERANK_BEST_SENTENCE_THRESHOLD &&
                documentBestSentence[j]
            ) {
                rerankDocs[j].bestSentence = documentBestSentence[j];
            }
        }

        rerankDocs.sort((a, b) => {
            const scoreDiff = (b.rerankScore ?? 0) - (a.rerankScore ?? 0);
            if (Math.abs(scoreDiff) > 1e-9) return scoreDiff;
            return (b.coarseScore ?? 0) - (a.coarseScore ?? 0);
        });
    } else {
        rerankDocs.forEach((doc) => {
            doc.displayScore = 0;
            doc.rerankScore = 0;
            doc.snippetScore = 0;
            doc.confidenceScore = 0;
        });
    }

    const rerankedResults = rerankDocs.concat(results.slice(rerankDocCount));

    self.postMessage({
        taskId,
        status: 'rerank_complete',
        result: rerankedResults,
        stats: { elapsedMs: (performance.now() - t0).toFixed(1) }
    });
}

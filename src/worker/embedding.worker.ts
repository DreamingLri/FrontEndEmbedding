import { pipeline, env } from '@huggingface/transformers';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';
import type {
    BM25Stats,
    Metadata
} from './vector_engine.ts';
import {
    buildBM25Stats
} from './vector_engine.ts';
import {
    buildTopicPartitionIndex,
    type TopicPartitionIndex,
} from './topic_partition.ts';
import {
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    executeSearchPipeline,
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
} from './search_pipeline.ts';

const MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';

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
let scopeSpecificityWordIdToTerm = new Map<number, string>();
let topicPartitionIndex: TopicPartitionIndex = {
    topicCandidateIndex: new Map<string, number[]>(),
    unlabeledCandidateIndices: [],
    metadataCount: 0,
};

async function fetchDocumentsFromApi(query: string, otids: string[]) {
    const response = await fetch('/api/get_answers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, otids }),
    });

    if (!response.ok) {
        throw new Error(`后端响应异常: ${response.status}`);
    }

    const payload = await response.json();
    return Array.isArray(payload?.data) ? payload.data : [];
}

async function embedQuery(query: string): Promise<Float32Array> {
    if (!extractor) throw new Error('\u67e5\u8be2\u6a21\u578b\u672a\u521d\u59cb\u5316');

    const output = await extractor(query, {
        pooling: 'mean', normalize: true, truncation: true, max_length: 512
    } as any);
    return new Float32Array(output.data as Float32Array);
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
        }
    } catch (e: any) {
        self.postMessage({ taskId, status: 'error', error: e.message });
    }
};

async function handleInit(payload: any, taskId?: string) {
    try {
        let rawMetadata: any;

        if (!payload.metadata || !payload.vectorMatrix) {
            throw new Error('INIT payload 格式错误：缺少 metadata 或 vectorMatrix');
        }
        self.postMessage({ taskId, status: 'loading', message: '解析已传入的元数据与向量矩阵...' });
        rawMetadata = payload.metadata;
        vectorMatrix = payload.vectorMatrix;

        metadataList = Array.isArray(rawMetadata) ? rawMetadata : (rawMetadata.data || []);
        const vocabList: string[] = rawMetadata.vocab || [];
        vocabMap.clear();
        vocabList.forEach((word, index) => vocabMap.set(word, index));
        ({ scopeSpecificityWordIdToTerm } = buildPipelineTermMaps(vocabMap));
        topicPartitionIndex = buildTopicPartitionIndex(metadataList);

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
            message: `AI 引擎就绪 [${device}/${dtype}, ${metadataList.length} 条, ${DIMENSIONS}维]`
        });
    } catch (err: any) {
        self.postMessage({ taskId, status: 'error', error: `初始化失败: ${err.message}` });
    }
}

async function handleSearch(query: string, taskId?: string) {
    if (!vectorMatrix || !globalBM25Stats || !extractor) {
        throw new Error('引擎尚未初始化完成');
    }

    const queryVector = await embedQuery(query);
    const queryContext = buildSearchPipelineQueryContext(
        query,
        vocabMap,
        topicPartitionIndex,
    );
    const searchResult = await executeSearchPipeline({
        query,
        queryVector,
        queryContext,
        metadata: metadataList,
        vectorMatrix,
        dimensions: DIMENSIONS,
        currentTimestamp: Date.now() / 1000,
        bm25Stats: globalBM25Stats,
        extractor,
        documentLoader: ({ query: documentQuery, otids }) =>
            fetchDocumentsFromApi(documentQuery, otids),
        termMaps: {
            scopeSpecificityWordIdToTerm,
        },
        preset: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        onStatus: (message) => {
            self.postMessage({
                taskId,
                status: 'progress',
                message,
            });
        },
    });

    self.postMessage({
        taskId,
        status: 'search_complete',
        result: searchResult,
        stats: {
            elapsedMs: searchResult.trace.totalMs.toFixed(1),
            itemsScanned: searchResult.trace.candidateCount,
            partitionUsed: searchResult.trace.partitionUsed,
            partitionCandidateCount: searchResult.trace.partitionCandidateCount,
        }
    });
}

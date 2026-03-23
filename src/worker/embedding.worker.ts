import { pipeline, env } from '@huggingface/transformers';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';
import type { BM25Stats, Metadata } from './vector_engine';
import { buildBM25Stats, getQuerySparse, searchAndRank } from './vector_engine';

// ==========================================
// 0. 环境配置
// ==========================================
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
} catch (_) { /* ONNX 配置不可用时忽略 */ }

// ==========================================
// 1. 全局状态
// ==========================================
let extractor: FeatureExtractionPipeline | null = null;
let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let globalBM25Stats: BM25Stats | null = null;

// ==========================================
// 2. FMM 正向最大匹配分词器 (替代 nodejieba)
// ==========================================
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
        if (!matched) {
            i++; // 跳过未匹配的单字符
        }
    }
    return tokens;
}

// ==========================================
// 3. 语义切片器 (用于精排)
// ==========================================
function splitIntoSemanticChunks(text: string, maxLen = 150): string[] {
    const sentences = text.split(/([。！？\n]+)/g);
    const chunks: string[] = [];
    let currentChunk = '';

    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxLen && currentChunk.length > 0) {
            chunks.push(currentChunk);
            currentChunk = '';
        }
        currentChunk += sentence;
    }
    if (currentChunk) chunks.push(currentChunk);

    return chunks.slice(0, 4); // 每篇最多 4 个 chunk，防止算力爆炸
}

// ==========================================
// 4. 消息分发器
// ==========================================
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

// ==========================================
// 5. INIT — 加载数据 + 构建索引 + 加载模型
// ==========================================
// 支持两种初始化模式:
//   模式 A (SearchRAG.vue): payload = { metadata: JSON, vectorMatrix: Int8Array }
//   模式 B (RAGEvaluator.vue): payload = { metadataUrl: string, vectorsUrl: string }
async function handleInit(payload: any, taskId?: string) {
    try {
        let rawMetadata: any;

        if (payload.metadata && payload.vectorMatrix) {
            // 模式 A: 主线程已预加载数据，通过 Transferable 零拷贝传入
            self.postMessage({ taskId, status: 'loading', message: '解析已传递的词典与零拷贝矩阵内存...' });
            rawMetadata = payload.metadata;
            vectorMatrix = payload.vectorMatrix;
        } else if (payload.metadataUrl && payload.vectorsUrl) {
            // 模式 B: Worker 自行从网络拉取数据
            self.postMessage({ taskId, status: 'loading', message: '从网络加载核心数据...' });
            const [metaRes, vecRes] = await Promise.all([
                fetch(payload.metadataUrl),
                fetch(payload.vectorsUrl)
            ]);
            if (!metaRes.ok || !vecRes.ok) throw new Error('网络请求资源失败');
            rawMetadata = await metaRes.json();
            const vecBuffer = await vecRes.arrayBuffer();
            vectorMatrix = new Int8Array(vecBuffer);
        } else {
            throw new Error('INIT payload 格式错误：需要 {metadata, vectorMatrix} 或 {metadataUrl, vectorsUrl}');
        }

        // 解析元数据与词典
        metadataList = Array.isArray(rawMetadata) ? rawMetadata : (rawMetadata.data || []);
        const vocabList: string[] = rawMetadata.vocab || [];
        vocabMap.clear();
        vocabList.forEach((word, index) => vocabMap.set(word, index));

        // 构建 BM25 统计
        self.postMessage({ taskId, status: 'loading', message: '构建 BM25 统计索引...' });
        globalBM25Stats = buildBM25Stats(metadataList);

        // 动态推断向量维度
        if (metadataList.length > 0 && vectorMatrix && vectorMatrix.length > 0) {
            DIMENSIONS = Math.round(vectorMatrix.length / metadataList.length);
        }

        // 探测 WebGPU 支持
        self.postMessage({ taskId, status: 'loading', message: `探测计算后端 (${DIMENSIONS}维向量)...` });
        let device = 'wasm';
        try {
            if ((navigator as any).gpu) {
                const adapter = await (navigator as any).gpu.requestAdapter();
                if (adapter) device = 'webgpu';
            }
        } catch (_) { /* 降级到 WASM */ }

        const dtype = device === 'webgpu' ? 'fp16' : 'q8';
        self.postMessage({ taskId, status: 'loading', message: `加载 Embedding 模型 (${device}/${dtype})...` });

        extractor = await pipeline('feature-extraction', MODEL_NAME, {
            dtype: dtype as any,
            device: device as any
        });

        self.postMessage({
            taskId,
            status: 'ready',
            message: `AI 引擎就绪 [${device}/${dtype}, ${metadataList.length} 词条, ${DIMENSIONS}维]`
        });
    } catch (err: any) {
        self.postMessage({ taskId, status: 'error', error: `初始化失败: ${err.message}` });
    }
}

// ==========================================
// 6. SEARCH — FMM 分词 + 粗排
// ==========================================
async function handleSearch(query: string, taskId?: string) {
    if (!vectorMatrix || !globalBM25Stats || !extractor) {
        throw new Error('引擎未初始化完毕');
    }

    const t0 = performance.now();

    // 1. 生成 Query Embedding
    const output = await extractor(query, {
        pooling: 'mean', normalize: true, truncation: true, max_length: 512
    } as any);
    const queryVector = output.data as Float32Array;

    // 2. FMM 分词 → 构建稀疏向量
    const queryWords = fmmTokenize(query);
    const querySparse = getQuerySparse(queryWords, vocabMap);

    // 3. 提取年份 Word ID (用于年份精准匹配)
    const queryYears = query.match(/20\d{2}/g);
    const queryYearWordIds: number[] = [];
    if (queryYears) {
        queryYears.forEach(y => {
            const id = vocabMap.get(y);
            if (id !== undefined) queryYearWordIds.push(id);
        });
    }

    // 4. 执行粗排: Dense + BM25 RRF 融合
    const matches = searchAndRank({
        queryVector,
        querySparse,
        queryYearWordIds,
        metadata: metadataList,
        vectorMatrix,
        dimensions: DIMENSIONS,
        currentTimestamp: 0,
        bm25Stats: globalBM25Stats
    });

    self.postMessage({
        taskId,
        status: 'search_complete',
        result: matches,
        stats: {
            elapsedMs: (performance.now() - t0).toFixed(1),
            itemsScanned: metadataList.length
        }
    });
}

// ==========================================
// 7. RERANK — 语义切片 + Batch Embedding 精排
// ==========================================
async function handleRerank(payload: { query: string, documents: any[] }, taskId?: string) {
    if (!extractor) throw new Error('引擎未初始化');

    const { query, documents } = payload;
    const t0 = performance.now();

    // 1. 生成 Query Embedding
    const output = await extractor(query, {
        pooling: 'mean', normalize: true, truncation: true, max_length: 512
    } as any);
    const queryVector = output.data as Float32Array;

    // 2. 收集所有文档的语义切片
    const BATCH_SIZE = 8;
    const allChunks: { docIdx: number, text: string }[] = [];

    for (let i = 0; i < documents.length; i++) {
        const docChunks = splitIntoSemanticChunks(documents[i].ot_text || '', 150);
        for (const chunk of docChunks) {
            allChunks.push({ docIdx: i, text: chunk });
        }
    }

    const totalChunks = allChunks.length;
    const documentScores = new Float32Array(documents.length).fill(-Infinity);
    const documentBestSentence: string[] = new Array(documents.length).fill('');

    // 3. 按 Batch 并行计算每个 chunk 的 embedding 与相似度
    for (let i = 0; i < totalChunks; i += BATCH_SIZE) {
        const batchChunks = allChunks.slice(i, i + BATCH_SIZE);
        const batchTexts = batchChunks.map(c => (c.text || '').substring(0, 400));

        const batchOutputs = await extractor(batchTexts, {
            pooling: 'mean', normalize: true, truncation: true, max_length: 512
        } as any);

        const validElements = batchChunks.length * DIMENSIONS;
        const pureData = (batchOutputs.data as Float32Array).subarray(0, validElements);

        for (let k = 0; k < batchChunks.length; k++) {
            const chunkVec = pureData.subarray(k * DIMENSIONS, (k + 1) * DIMENSIONS);

            // 点积 (因 normalize=true, 等价于余弦相似度)
            let score = 0;
            for (let d = 0; d < DIMENSIONS; d++) {
                score += queryVector[d] * chunkVec[d];
            }

            const docIdx = batchChunks[k].docIdx;
            if (score > documentScores[docIdx]) {
                documentScores[docIdx] = score;
                documentBestSentence[docIdx] = batchChunks[k].text;
            }
        }

        self.postMessage({
            taskId,
            status: 'progress',
            message: `深度重排进行中：语义切片 ${Math.min(i + BATCH_SIZE, totalChunks)}/${totalChunks}`
        });
    }

    // 4. 组装最终结果
    const results = [];
    for (let j = 0; j < documents.length; j++) {
        results.push({
            ...documents[j],
            rerankScore: documentScores[j],
            bestSentence: documentBestSentence[j]
        });
    }

    results.sort((a, b) => b.rerankScore - a.rerankScore);

    self.postMessage({
        taskId,
        status: 'rerank_complete',
        result: results,
        stats: { elapsedMs: (performance.now() - t0).toFixed(1) }
    });
}

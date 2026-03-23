import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';
import { buildBM25Stats, searchAndRank, getQuerySparse, type BM25Stats } from '../utils/vector_engine.ts';

// --- 配置环境 ---
const MODEL_NAME = 'DMetaSoul/Dmeta-embedding-zh-small';
let DIMENSIONS = 768;

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL(import.meta.env.BASE_URL + 'models/', self.location.origin).href;

const isSecureContext = typeof self.caches !== 'undefined';
env.useBrowserCache = isSecureContext; 

const onnx = (env as any).backends?.onnx;
if (onnx) {
    onnx.wasm.wasmPaths = new URL(import.meta.env.BASE_URL + 'wasm/', self.location.origin).href;
    onnx.wasm.numThreads = 8; // PC 端多线程全量并发
    onnx.wasm.proxy = false;
}

// --- 状态与变量 ---
let extractor: FeatureExtractionPipeline | null = null;
let vocabMap = new Map<string, number>();
let metadataList: any[] = [];
let vectorMatrix: Int8Array | null = null;
let globalBM25Stats: BM25Stats | null = null;

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
            i++;
        }
    }
    return tokens;
}

self.onmessage = async (event: MessageEvent) => {
    const { type, payload, taskId } = event.data;

    try {
        switch (type) {
            case 'INIT':
                await handleInit({ ...payload, taskId });
                break;
            case 'SEARCH':
                await handleSearch({ query: payload, taskId });
                break;
            case 'RERANK':
                await handleRerank({ ...payload, taskId });
                break;
        }
    } catch (e: any) {
        self.postMessage({ taskId, status: 'error', error: e.message });
    }
};

async function handleInit({ metadata, vectorMatrix: vm, taskId }: { metadata: any, vectorMatrix: Int8Array, taskId?: string }) {
    try {
        self.postMessage({ taskId, status: 'loading', message: '解析已传递的词典与零拷贝矩阵内存...' });
        
        // 解析传递过来的元数据字典
        metadataList = Array.isArray(metadata) ? metadata : (metadata.data || []);
        const vocabList = (metadata.vocab || []); 
        vocabMap.clear();
        vocabList.forEach((word: string, index: number) => vocabMap.set(word, index));
        globalBM25Stats = buildBM25Stats(metadataList);

        // Zero-Copy 转移的矩阵直接可用！不再产生 Fetch 等多余内存分配
        vectorMatrix = vm;
        
        // 动态推断真实模型的向量维度，防止 offset is out of bounds！
        if (metadataList.length > 0 && vectorMatrix.length > 0) {
            DIMENSIONS = Math.round(vectorMatrix.length / metadataList.length);
        }

        self.postMessage({ taskId, status: 'loading', message: `唤醒 WebGPU (${DIMENSIONS}维)...` });
        let device = 'webgpu';
        try { if (!(navigator as any).gpu) device = 'wasm'; } catch(e) { device = 'wasm'; }
        
        const dtype = device === 'webgpu' ? 'fp16' : 'q8';
        extractor = await pipeline('feature-extraction', MODEL_NAME, {
            dtype: dtype as any,
            device: device as any
        });
        
        self.postMessage({ taskId, status: 'ready', message: 'AI 本地知识图谱加载完毕。' });
    } catch (err: any) {
        self.postMessage({ taskId, status: 'error', error: `初始化崩盘: ${err.message}` });
    }
}

async function handleSearch({ query, taskId }: { query: string, taskId?: string }) {
    if (!vectorMatrix || !globalBM25Stats || !extractor) throw new Error("引擎未初始化完毕");

    const t0 = performance.now();
    const output = await extractor(query, { pooling: 'mean', normalize: true, truncation: true, max_length: 512 } as any);
    const queryVector = output.data as Float32Array;
    const queryWords = fmmTokenize(query);
    const querySparse = getQuerySparse(queryWords, vocabMap);

    const queryYears = query.match(/20\d{2}/g);
    const queryYearWordIds: number[] = [];
    if (queryYears) {
        queryYears.forEach(y => {
            const id = vocabMap.get(y);
            if (id !== undefined) queryYearWordIds.push(id);
        });
    }

    const matches = searchAndRank({
        queryVector,
        querySparse,
        queryYearWordIds,
        metadata: metadataList,
        vectorMatrix: vectorMatrix,
        dimensions: DIMENSIONS,
        currentTimestamp: 0,
        bm25Stats: globalBM25Stats
    });

    self.postMessage({
        taskId,
        status: 'search_complete',
        result: matches,
        stats: { elapsedMs: (performance.now() - t0).toFixed(1), itemsScanned: metadataList.length }
    });
}

function splitIntoSemanticChunks(text: string, maxLen = 150): string[] {
    const sentences = text.split(/([。！？\n]+)/g); 
    const chunks: string[] = [];
    let currentChunk = "";

    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxLen && currentChunk.length > 0) {
            chunks.push(currentChunk);
            currentChunk = "";
        }
        currentChunk += sentence;
    }
    if (currentChunk) chunks.push(currentChunk);
    
    return chunks.slice(0, 4); // 依然保留防爆锁
}

async function handleRerank({ query, documents, taskId }: { query: string, documents: any[], taskId?: string }) {
    if (!extractor) throw new Error("引擎未初始化");

    const t0 = performance.now();
    const output = await extractor(query, { pooling: 'mean', normalize: true, truncation: true, max_length: 512 } as any);
    const queryVector = output.data as Float32Array;

    const BATCH_SIZE = 8; 
    const results = [];
    
    // 🌟 收集所有片段
    const allChunks: { docIdx: number, text: string }[] = [];
    for (let i = 0; i < documents.length; i++) {
        const docChunks = splitIntoSemanticChunks(documents[i].ot_text || "", 150);
        for (const chunk of docChunks) {
            allChunks.push({ docIdx: i, text: chunk });
        }
    }
    
    const totalChunks = allChunks.length;
    const documentScores = new Float32Array(documents.length).fill(-Infinity);
    const documentBestSentence: string[] = new Array(documents.length).fill("");

    for (let i = 0; i < totalChunks; i += BATCH_SIZE) {
        const batchChunks = allChunks.slice(i, i + BATCH_SIZE);
        const batchTexts = batchChunks.map(c => (c.text || "").substring(0, 400));
        const batchOutputs = await extractor(batchTexts, { pooling: 'mean', normalize: true, truncation: true, max_length: 512 } as any);
        
        const validElements = batchChunks.length * DIMENSIONS;
        const pureData = (batchOutputs.data as Float32Array).subarray(0, validElements);
        
        for (let k = 0; k < batchChunks.length; k++) {
            const chunkVec = pureData.subarray(k * DIMENSIONS, (k + 1) * DIMENSIONS);
            let score = 0;
            for (let d = 0; d < DIMENSIONS; d++) score += queryVector[d] * chunkVec[d];
            
            const docIdx = batchChunks[k].docIdx;
            if (score > documentScores[docIdx]) {
                documentScores[docIdx] = score;
                documentBestSentence[docIdx] = batchChunks[k].text;
            }
        }
        
        self.postMessage({ taskId, status: 'progress', message: `深度重排进行中：语义切片 ${Math.min(i + BATCH_SIZE, totalChunks)}/${totalChunks}` });
    }

    for (let j = 0; j < documents.length; j++) {
        results.push({ 
            ...documents[j], 
            rerankScore: documentScores[j],
            bestSentence: documentBestSentence[j]
        });
    }

    results.sort((a, b) => b.rerankScore - a.rerankScore);
    self.postMessage({ taskId, status: 'rerank_complete', result: results, stats: { elapsedMs: (performance.now() - t0).toFixed(1) } });
}

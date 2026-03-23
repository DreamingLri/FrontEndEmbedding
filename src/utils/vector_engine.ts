export interface Metadata {
    id: string;
    type: 'Q' | 'KP' | 'OT';
    parent_otid: string;
    timestamp?: number;
    vector_index: number;
    scale?: number;
    sparse?: number[];
}

export interface SearchResult {
    otid: string;
    score: number;
}

export interface BM25Stats {
    idfArray: Float32Array;      // 词表 -> IDF 分数 (极速内存寻址)
    docLengths: Int32Array;      // 每个文档的长度 (词总数)
    avgdl: number;               // 库内平均文档长度
}

export const DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33
};

export const DECAY_LAMBDA = 0.001;
export const SECONDS_IN_DAY = 86400;
export const RRF_K = 60;
const BM25_K1 = 1.2;
const BM25_B = 0.4;

export const CAMPUS_SYNONYMS: Record<string, string[]> = {
    "考研": ["研究生", "招生", "考试", "初试"],
    "保研": ["免试", "推免", "推荐"],
    "名额": ["计划", "人数"],
    "退课": ["退选"]
};

// 预热阶段调用：计算全局 BM25 极值
export function buildBM25Stats(metadata: Metadata[]): BM25Stats {
    const N = metadata.length;
    const dfMap = new Map<number, number>(); 
    const docLengths = new Int32Array(N);
    let totalLength = 0;

    for (let i = 0; i < N; i++) {
        const sparse = metadata[i].sparse;
        if (!sparse || sparse.length === 0) {
            docLengths[i] = 0;
            continue;
        }

        let dl = 0;
        for (let j = 0; j < sparse.length; j += 2) {
            const wordId = sparse[j];
            const tf = sparse[j + 1];
            dl += tf;
            dfMap.set(wordId, (dfMap.get(wordId) || 0) + 1);
        }
        docLengths[i] = dl;
        totalLength += dl;
    }

    const avgdl = totalLength / (N || 1);

    let maxWordId = 0;
    for (const wordId of dfMap.keys()) {
        if (wordId > maxWordId) maxWordId = wordId;
    }
    const idfArray = new Float32Array(maxWordId + 1);

    for (const [wordId, df] of dfMap.entries()) {
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        idfArray[wordId] = Math.max(idf, 0.01);
    }

    return { idfArray, docLengths, avgdl };
}

export function dotProduct(
    vecA: Float32Array,
    matrix: Int8Array, // 前端使用强类型的 8 位整型矩阵
    matrixIndex: number,
    dimensions: number
): number {
    let sum = 0;
    const offset = matrixIndex * dimensions;
    
    // CPU 多路循环展开 (Loop Unrolling, 4步长) 极限压榨标量指令极值
    let i = 0;
    for (; i <= dimensions - 4; i += 4) {
        sum += vecA[i] * matrix[offset + i]
             + vecA[i+1] * matrix[offset + i+1]
             + vecA[i+2] * matrix[offset + i+2]
             + vecA[i+3] * matrix[offset + i+3];
    }
    for (; i < dimensions; i++) {
        sum += vecA[i] * matrix[offset + i];
    }
    return sum;
}

export function getQuerySparse(
    words: string[], 
    vocabMap: Map<string, number>
): Record<number, number> {
    const sparse: Record<number, number> = {};

    words.forEach(word => {
        const index = vocabMap.get(word);
        if (index !== undefined) {
            sparse[index] = (sparse[index] || 0) + 1;
        }

        const synonyms = CAMPUS_SYNONYMS[word];
        if (synonyms) {
            synonyms.forEach(syn => {
                const sIndex = vocabMap.get(syn);
                if (sIndex !== undefined) {
                    sparse[sIndex] = (sparse[sIndex] || 0) + 1;
                }
            });
        }
    });

    return sparse;
}

export function searchAndRank(params: {
    queryVector: Float32Array,
    querySparse?: Record<number, number>,
    queryYearWordIds?: number[], 
    metadata: Metadata[],
    vectorMatrix: Int8Array,
    dimensions: number,
    currentTimestamp: number,
    bm25Stats: BM25Stats, 
    weights?: typeof DEFAULT_WEIGHTS
}): SearchResult[] {
    const {
        queryVector, querySparse, metadata, vectorMatrix, dimensions, currentTimestamp,
        bm25Stats, weights = DEFAULT_WEIGHTS, queryYearWordIds
    } = params;

    const n = metadata.length;

    const denseScores = new Float32Array(n);
    const sparseScores = new Float32Array(n);
    const denseIndices = new Int32Array(n);
    const sparseIndices = new Int32Array(n);
    const lexicalBonusMap = new Map<string, number>();
    const yearHitMap = new Map<string, boolean>();

    for (let i = 0; i < n; i++) {
        const meta = metadata[i];

        let dense = 0;
        if (meta.vector_index >= 0) {
            dense = dotProduct(queryVector, vectorMatrix, meta.vector_index, dimensions);
            if (meta.scale !== undefined && meta.scale !== null) dense *= meta.scale;
        }
        
        denseScores[i] = dense;
        denseIndices[i] = i;

        let sparse = 0;
        if (querySparse && meta.sparse && meta.sparse.length > 0) {
            const dl = bm25Stats.docLengths[i];
            const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25); 

            for (let j = 0; j < meta.sparse.length; j += 2) {
                const wordId = meta.sparse[j];
                const tf = meta.sparse[j + 1];

                if (queryYearWordIds && queryYearWordIds.includes(wordId)) {
                    const otid = meta.type === 'OT' ? meta.id : meta.parent_otid;
                    yearHitMap.set(otid, true);
                }

                if (querySparse[wordId]) {
                    const qWeight = querySparse[wordId] || 1;
                    const idf = bm25Stats.idfArray[wordId] || 0;
                    const numerator = tf * (BM25_K1 + 1);
                    const denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
                    sparse += qWeight * idf * (numerator / denominator);
                }
            }

            if (sparse > 0) {
                const otid = meta.type === 'OT' ? meta.id : meta.parent_otid;
                let currentBonus = lexicalBonusMap.get(otid) || 0;
                if (meta.type === 'Q') currentBonus += sparse * 1.5;
                else if (meta.type === 'KP') currentBonus += sparse * 1.2;
                else currentBonus += sparse;
                lexicalBonusMap.set(otid, currentBonus);
            }
        }
        sparseScores[i] = sparse;
        sparseIndices[i] = i;
    }

    // 就地排序 - 使用 TypedArray 的 sort 方法以获得极限重排速度
    denseIndices.sort((a, b) => denseScores[b] - denseScores[a]);
    
    // 消灭 Map<Metadata, number> 与其带来恐怖的 GC 开销，降维为 0 损耗 TypedArray
    const rrfScores = new Float32Array(n);

    for (let rank = 0; rank < Math.min(4000, n); rank++) {
        const originalIndex = denseIndices[rank];
        rrfScores[originalIndex] += (1 / (rank + RRF_K)) * 100;
    }

    if (querySparse) {
        const nonZeroSparseIndices: number[] = [];
        for (let i = 0; i < n; i++) {
            if (sparseScores[i] > 0) nonZeroSparseIndices.push(i);
        }
        
        // 仅对非零文档进行局部排序 (O(k log k) 代替 O(N log N))
        nonZeroSparseIndices.sort((a, b) => sparseScores[b] - sparseScores[a]);
        
        const limit = Math.min(4000, nonZeroSparseIndices.length);
        for (let rank = 0; rank < limit; rank++) {
            const originalIndex = nonZeroSparseIndices[rank];
            rrfScores[originalIndex] += (1.2 / (rank + RRF_K)) * 100; 
        }
    }

    const topHybridIndices: number[] = [];
    for (let i = 0; i < n; i++) {
        if (rrfScores[i] > 0) topHybridIndices.push(i);
    }
    topHybridIndices.sort((a, b) => rrfScores[b] - rrfScores[a]);

    const topHybrid = topHybridIndices.slice(0, 1000).map(i => [metadata[i], rrfScores[i]] as [Metadata, number]);

    const otidMap: Record<string, { max_q: number, max_kp: number, ot_score: number, timestamp?: number }> = {};

    for (const [meta, score] of topHybrid) {
        const otid = meta.type === 'OT' ? meta.id : meta.parent_otid;
        if (!otidMap[otid]) {
            otidMap[otid] = { max_q: 0, max_kp: 0, ot_score: 0, timestamp: meta.timestamp };
        }
        if (meta.type === 'Q') otidMap[otid].max_q = Math.max(otidMap[otid].max_q, score);
        else if (meta.type === 'KP') otidMap[otid].max_kp = Math.max(otidMap[otid].max_kp, score);
        else if (meta.type === 'OT') otidMap[otid].ot_score = Math.max(otidMap[otid].ot_score, score);
    }

    const finalRanking: SearchResult[] = [];
    for (const [otid, scores] of Object.entries(otidMap)) {
        
        const weightedQ = scores.max_q * weights.Q;
        const weightedKP = scores.max_kp * weights.KP;
        const weightedOT = scores.ot_score * weights.OT;
        
        const maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
        const unionBonus = (weightedQ * 0.1) + (weightedKP * 0.1) + (weightedOT * 0.1); 
        
        let finalScore = maxComponent + unionBonus;

        if (scores.timestamp) {
            const daysDiff = (currentTimestamp - scores.timestamp) / SECONDS_IN_DAY;
            if (daysDiff > 0) finalScore *= Math.exp(-DECAY_LAMBDA * daysDiff);
        }

        let boost = 1.0;
        
        if (queryYearWordIds && queryYearWordIds.length > 0) {
            if (!yearHitMap.get(otid)) {
                 boost = 0.01; 
            } else {
                 boost = 1.5; 
            }
        } else {
            const lexicalBonus = lexicalBonusMap.get(otid) || 0;
            if (lexicalBonus > 0) {
                boost = 1 + Math.log1p(lexicalBonus) / 4; 
            }
        }

        finalRanking.push({ otid, score: finalScore * boost });
    }

    return finalRanking.sort((a, b) => b.score - a.score).slice(0, 100);
}

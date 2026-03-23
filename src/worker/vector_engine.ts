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
    details?: {
        denseRRF: number;
        sparseRRF: number;
        lexicalBoost: number;
    }
}

export interface BM25Stats {
    idfMap: Map<number, number>; // 词表 -> IDF 分数
    docLengths: Int32Array;      // 每个文档的长度 (词总数)
    avgdl: number;               // 库内平均文档长度
}

// 核心融合权重配置 (语义分内部比例)
export const DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33
};

// 时间衰减相关常量
export const DECAY_LAMBDA = 0.001;
export const SECONDS_IN_DAY = 86400;

// RRF 融合常量
export const RRF_K = 60;

// BM25 经验常量
const BM25_K1 = 1.2; // 控制 TF 的饱和度
const BM25_B = 0.4;  // 控制文档长度的惩罚力度

// 校园黑话词典，用于词法扩展
export const CAMPUS_SYNONYMS: Record<string, string[]> = {
    "考研": ["研究生", "招生", "考试", "初试"],
    "保研": ["免试", "推免", "推荐"],
    "名额": ["计划", "人数"],
    "退课": ["退选"]
};

// 在数据加载时调用此函数，将结果缓存到内存中
export function buildBM25Stats(metadata: Metadata[]): BM25Stats {
    const N = metadata.length;
    const dfMap = new Map<number, number>(); // 记录包含某个词的文档数量 (Document Frequency)
    const docLengths = new Int32Array(N);
    let totalLength = 0;

    for (let i = 0; i < N; i++) {
        const sparse = metadata[i].sparse;
        if (!sparse || sparse.length === 0) {
            docLengths[i] = 0;
            continue;
        }

        let dl = 0;
        // sparse 数组结构是 [wordId1, tf1, wordId2, tf2, ...]
        for (let j = 0; j < sparse.length; j += 2) {
            const wordId = sparse[j];
            const tf = sparse[j + 1];
            dl += tf;
            // 统计 DF (每个词在这个文档出现过，文档频率+1)
            dfMap.set(wordId, (dfMap.get(wordId) || 0) + 1);
        }
        docLengths[i] = dl;
        totalLength += dl;
    }

    const avgdl = totalLength / (N || 1);
    const idfMap = new Map<number, number>();

    // 计算标准的 BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
    for (const [wordId, df] of dfMap.entries()) {
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        // 如果数据极度倾斜，IDF 可能为负，做一个底线保护
        idfMap.set(wordId, Math.max(idf, 0.01));
    }

    return { idfMap, docLengths, avgdl };
}

/**
 * 计算两个向量的点积
 */
export function dotProduct(
    vecA: Float32Array,
    matrix: Int8Array | Float32Array,
    matrixIndex: number,
    dimensions: number
): number {
    let sum = 0;
    const offset = matrixIndex * dimensions;
    for (let i = 0; i < dimensions; i++) {
        sum += vecA[i] * matrix[offset + i];
    }
    return sum;
}

/**
 * 将分词后的词列表转换为稀疏向量 (含校园同义词扩展)
 */
export function getQuerySparse(
    words: string[],
    vocabMap: Map<string, number> | Record<string, number>
): Record<number, number> {
    const sparse: Record<number, number> = {};
    const isMap = vocabMap instanceof Map;

    words.forEach(word => {
        // 1. 基础词 ID 匹配
        const index = isMap ? (vocabMap as Map<string, number>).get(word) : (vocabMap as Record<string, number>)[word];
        if (index !== undefined) {
            sparse[index] = (sparse[index] || 0) + 1;
        }

        // 2. 校园同义词扩展 (让"考研"也能匹配到"研究生招生")
        const synonyms = CAMPUS_SYNONYMS[word];
        if (synonyms) {
            synonyms.forEach(syn => {
                const sIndex = isMap ? (vocabMap as Map<string, number>).get(syn) : (vocabMap as Record<string, number>)[syn];
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
    queryYearWordIds?: number[], // 精确的年份词汇 ID 数组
    metadata: Metadata[],
    vectorMatrix: Int8Array | Float32Array,
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

    // 高性能 TypedArray 内存分配
    const denseScores = new Float32Array(n);
    const sparseScores = new Float32Array(n);
    const denseIndices = new Int32Array(n);
    const sparseIndices = new Int32Array(n);
    const lexicalBonusMap = new Map<string, number>();

    // 记录哪篇具体文章真正命中了特定的年份词
    const yearHitMap = new Map<string, boolean>();

    // 阶段 1：计算分数
    for (let i = 0; i < n; i++) {
        const meta = metadata[i];

        // 1. Dense (语义匹配)
        let dense = dotProduct(queryVector, vectorMatrix, meta.vector_index, dimensions);
        if (meta.scale !== undefined && meta.scale !== null) dense *= meta.scale;
        denseScores[i] = dense;
        denseIndices[i] = i;

        // 2. Sparse (BM25 词法匹配)
        let sparse = 0;
        if (querySparse && meta.sparse && meta.sparse.length > 0) {
            const dl = bm25Stats.docLengths[i];

            // 设定最小安全长度，防止短 Q 霸凌长 OT
            const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);

            for (let j = 0; j < meta.sparse.length; j += 2) {
                const wordId = meta.sparse[j];
                const tf = meta.sparse[j + 1];

                // 精确查杀：看这篇文章有没有命中特定的年份 ID
                if (queryYearWordIds && queryYearWordIds.includes(wordId)) {
                    const otid = meta.type === 'OT' ? meta.id : meta.parent_otid;
                    yearHitMap.set(otid, true);
                }

                if (querySparse[wordId]) {
                    const qWeight = querySparse[wordId] || 1;
                    const idf = bm25Stats.idfMap.get(wordId) || 0;

                    const numerator = tf * (BM25_K1 + 1);
                    // 使用 safeDl 替代 dl
                    const denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
                    sparse += qWeight * idf * (numerator / denominator);
                }
            }

            // 计算该文档对实体的词法增益 (用于阶段 5 重排)
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

    // 阶段 2：RRF 融合 (依旧保持高性能的就地排序)
    denseIndices.sort((a, b) => denseScores[b] - denseScores[a]);
    const rrfScores = new Map<Metadata, number>();

    // 扩大融合池到 4000，绝对不漏掉长尾文档
    for (let rank = 0; rank < Math.min(4000, n); rank++) {
        const meta = metadata[denseIndices[rank]];
        rrfScores.set(meta, (1 / (rank + RRF_K)) * 100);
    }

    if (querySparse) {
        sparseIndices.sort((a, b) => sparseScores[b] - sparseScores[a]);
        for (let rank = 0; rank < Math.min(4000, n); rank++) {
            const originalIndex = sparseIndices[rank];
            if (sparseScores[originalIndex] === 0) break;

            const meta = metadata[originalIndex];
            const current = rrfScores.get(meta) || 0;
            // Sparse 赋予 1.2 倍 RRF 权重，增强词法约束力
            rrfScores.set(meta, current + ((1.2 / (rank + RRF_K)) * 100));
        }
    }

    // 阶段 3：提取聚合
    const topHybrid = Array.from(rrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 1000);

    const otidMap: Record<string, {
        max_q: number, max_kp: number, ot_score: number,
        timestamp?: number
    }> = {};

    for (const [meta, score] of topHybrid) {
        const otid = meta.type === 'OT' ? meta.id : meta.parent_otid;
        if (!otidMap[otid]) {
            otidMap[otid] = { max_q: 0, max_kp: 0, ot_score: 0, timestamp: meta.timestamp };
        }

        if (meta.type === 'Q') otidMap[otid].max_q = Math.max(otidMap[otid].max_q, score);
        else if (meta.type === 'KP') otidMap[otid].max_kp = Math.max(otidMap[otid].max_kp, score);
        else if (meta.type === 'OT') otidMap[otid].ot_score = Math.max(otidMap[otid].ot_score, score);
    }

    // 阶段 4：分数合并逻辑
    const finalRanking: SearchResult[] = [];
    for (const [otid, scores] of Object.entries(otidMap)) {

        // 把 weights 真正用起来
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

        // 极其精确的年份硬过滤
        if (queryYearWordIds && queryYearWordIds.length > 0) {
            if (!yearHitMap.get(otid)) {
                // 绝杀降权：用户问了具体年份，但文章里没有该年份
                boost = 0.01;
            } else {
                // 提权：年份完美命中
                boost = 1.5;
            }
        } else {
            // 如果用户没有问特定年份，按常规词法奖励处理
            const lexicalBonus = lexicalBonusMap.get(otid) || 0;
            if (lexicalBonus > 0) {
                boost = 1 + Math.log1p(lexicalBonus) / 4;
            }
        }

        finalRanking.push({ otid, score: finalScore * boost });
    }

    // 阶段 5：对聚合合并后的真实文章 (OT) 进行排名，并切出 Top 100
    return finalRanking.sort((a, b) => b.score - a.score).slice(0, 100);
}

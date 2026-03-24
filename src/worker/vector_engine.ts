export interface Metadata {
    id: string;
    type: "Q" | "KP" | "OT";
    parent_otid: string;
    timestamp?: number;
    vector_index: number;
    scale?: number;
    sparse?: number[];
    target_year?: number;
    intent_ids?: string[];
}

export interface SearchResult {
    otid: string;
    best_kpid?: string;
    score: number;
    details?: {
        denseRRF: number;
        sparseRRF: number;
        lexicalBoost: number;
    };
}

export interface BM25Stats {
    idfMap: Map<number, number>;
    docLengths: Int32Array;
    avgdl: number;
}

export interface ParsedQueryIntent {
    rawQuery: string;
    years: number[];
    intentIds: string[];
    normalizedTerms: string[];
    confidence: number;
}

export interface IntentVectorItem {
    intent_id: string;
    intent_name: string;
    aliases: string[];
    negative_intents: string[];
    related_intents: string[];
    weight: number;
}

export const DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33,
};

export const DECAY_LAMBDA = 0.001;
export const SECONDS_IN_DAY = 86400;
export const RRF_K = 60;

const BM25_K1 = 1.2;
const BM25_B = 0.4;

export const CAMPUS_SYNONYMS: Record<string, string[]> = {
    考研: ["研究生", "招生", "考试", "初试"],
    保研: ["推免", "推荐免试", "推免生"],
    名额: ["计划", "人数"],
    退课: ["退选"],
};

export const INTENT_VECTOR_TABLE: IntentVectorItem[] = [
    {
        intent_id: "ug_recommend_admission",
        intent_name: "本科保送生",
        aliases: ["保送生", "外语类保送生", "竞赛保送"],
        negative_intents: [
            "master_recommend_exemption",
            "master_unified_exam",
        ],
        related_intents: [],
        weight: 1,
    },
    {
        intent_id: "master_recommend_exemption",
        intent_name: "推免",
        aliases: [
            "保研",
            "推免",
            "推荐免试",
            "推免生",
            "免试研究生",
            "预推免",
        ],
        negative_intents: ["ug_recommend_admission", "master_unified_exam"],
        related_intents: ["summer_camp", "pre_recommend"],
        weight: 1,
    },
    {
        intent_id: "master_unified_exam",
        intent_name: "统考硕士",
        aliases: [
            "考研",
            "硕士研究生招生考试",
            "全国硕士研究生招生考试",
            "硕士统考",
            "初试",
            "复试",
            "研招",
        ],
        negative_intents: [
            "ug_recommend_admission",
            "master_recommend_exemption",
        ],
        related_intents: ["master_adjustment"],
        weight: 1,
    },
    {
        intent_id: "master_adjustment",
        intent_name: "调剂",
        aliases: ["调剂", "硕士调剂", "接受调剂", "调剂复试"],
        negative_intents: [],
        related_intents: ["master_unified_exam"],
        weight: 1,
    },
    {
        intent_id: "phd_apply_assessment",
        intent_name: "博士申请考核",
        aliases: ["申请考核", "申请-考核", "博士申请考核"],
        negative_intents: ["phd_general_exam"],
        related_intents: [],
        weight: 1,
    },
    {
        intent_id: "phd_general_exam",
        intent_name: "博士普通招考",
        aliases: ["博士招考", "博士报名", "博士考试", "公开招考博士"],
        negative_intents: ["phd_apply_assessment"],
        related_intents: [],
        weight: 1,
    },
    {
        intent_id: "summer_camp",
        intent_name: "夏令营",
        aliases: ["夏令营", "优秀大学生夏令营"],
        negative_intents: [],
        related_intents: ["master_recommend_exemption"],
        weight: 1,
    },
    {
        intent_id: "pre_recommend",
        intent_name: "预推免",
        aliases: ["预推免", "预推免报名", "预推免考核"],
        negative_intents: ["master_unified_exam"],
        related_intents: ["master_recommend_exemption", "summer_camp"],
        weight: 1,
    },
];

const INTENT_CONFLICTS: Record<string, string[]> = Object.fromEntries(
    INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item.negative_intents]),
);

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
    const idfMap = new Map<number, number>();

    for (const [wordId, df] of dfMap.entries()) {
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        idfMap.set(wordId, Math.max(idf, 0.01));
    }

    return { idfMap, docLengths, avgdl };
}

export function dotProduct(
    vecA: Float32Array,
    matrix: Int8Array | Float32Array,
    matrixIndex: number,
    dimensions: number,
): number {
    let sum = 0;
    const offset = matrixIndex * dimensions;
    for (let i = 0; i < dimensions; i++) {
        sum += vecA[i] * matrix[offset + i];
    }
    return sum;
}

export function getQuerySparse(
    words: string[],
    vocabMap: Map<string, number> | Record<string, number>,
): Record<number, number> {
    const sparse: Record<number, number> = {};
    const isMap = vocabMap instanceof Map;

    words.forEach((word) => {
        const index = isMap
            ? (vocabMap as Map<string, number>).get(word)
            : (vocabMap as Record<string, number>)[word];
        if (index !== undefined) {
            sparse[index] = (sparse[index] || 0) + 1;
        }

        const synonyms = CAMPUS_SYNONYMS[word];
        if (synonyms) {
            synonyms.forEach((syn) => {
                const sIndex = isMap
                    ? (vocabMap as Map<string, number>).get(syn)
                    : (vocabMap as Record<string, number>)[syn];
                if (sIndex !== undefined) {
                    sparse[sIndex] = (sparse[sIndex] || 0) + 1;
                }
            });
        }
    });

    return sparse;
}

export function parseQueryIntent(query: string): ParsedQueryIntent {
    const lowered = query.toLowerCase();
    const years = Array.from(
        new Set((query.match(/20\d{2}/g) || []).map((year) => Number(year))),
    );
    const matchedRules = INTENT_VECTOR_TABLE.filter((rule) =>
        rule.aliases.some((alias) => lowered.includes(alias.toLowerCase())),
    );

    return {
        rawQuery: query,
        years,
        intentIds: matchedRules.map((rule) => rule.intent_id),
        normalizedTerms: Array.from(
            new Set(matchedRules.flatMap((rule) => rule.aliases)),
        ),
        confidence: matchedRules.length > 0 ? 1 : 0,
    };
}

function hasIntentConflict(
    queryIntentIds: string[],
    docIntentIds?: string[],
): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) =>
        (INTENT_CONFLICTS[queryIntentId] || []).some((conflictId) =>
            docIntentIds.includes(conflictId),
        ),
    );
}

function hasIntentMatch(queryIntentIds: string[], docIntentIds?: string[]): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) => docIntentIds.includes(queryIntentId));
}

export function searchAndRank(params: {
    queryVector: Float32Array;
    querySparse?: Record<number, number>;
    queryYearWordIds?: number[];
    queryIntent?: ParsedQueryIntent;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    weights?: typeof DEFAULT_WEIGHTS;
}): SearchResult[] {
    const {
        queryVector,
        querySparse,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        weights = DEFAULT_WEIGHTS,
        queryYearWordIds,
        queryIntent,
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

        let dense = dotProduct(
            queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null) dense *= meta.scale;
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
                    const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
                    yearHitMap.set(otid, true);
                }

                if (querySparse[wordId]) {
                    const qWeight = querySparse[wordId] || 1;
                    const idf = bm25Stats.idfMap.get(wordId) || 0;
                    const numerator = tf * (BM25_K1 + 1);
                    const denominator =
                        tf +
                        BM25_K1 *
                            (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
                    sparse += qWeight * idf * (numerator / denominator);
                }
            }

            if (sparse > 0) {
                const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
                let currentBonus = lexicalBonusMap.get(otid) || 0;
                if (meta.type === "Q") currentBonus += sparse * 1.5;
                else if (meta.type === "KP") currentBonus += sparse * 1.2;
                else currentBonus += sparse;
                lexicalBonusMap.set(otid, currentBonus);
            }
        }
        sparseScores[i] = sparse;
        sparseIndices[i] = i;
    }

    denseIndices.sort((a, b) => denseScores[b] - denseScores[a]);
    const rrfScores = new Map<Metadata, number>();

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
            rrfScores.set(meta, current + (1.2 / (rank + RRF_K)) * 100);
        }
    }

    const topHybrid = Array.from(rrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 1000);

    const otidMap: Record<
        string,
        {
            max_q: number;
            max_kp: number;
            ot_score: number;
            timestamp?: number;
            best_kpid?: string;
            target_year?: number;
            intent_ids?: string[];
        }
    > = {};

    for (const [meta, score] of topHybrid) {
        const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
        if (!otidMap[otid]) {
            otidMap[otid] = {
                max_q: 0,
                max_kp: 0,
                ot_score: 0,
                timestamp: meta.timestamp,
                target_year: meta.target_year,
                intent_ids: meta.intent_ids,
            };
        }

        if (
            otidMap[otid].target_year === undefined &&
            meta.target_year !== undefined
        ) {
            otidMap[otid].target_year = meta.target_year;
        }
        if (
            (!otidMap[otid].intent_ids || otidMap[otid].intent_ids!.length === 0) &&
            meta.intent_ids &&
            meta.intent_ids.length > 0
        ) {
            otidMap[otid].intent_ids = meta.intent_ids;
        }

        if (meta.type === "Q") {
            otidMap[otid].max_q = Math.max(otidMap[otid].max_q, score);
        } else if (meta.type === "KP") {
            if (score > otidMap[otid].max_kp) {
                otidMap[otid].max_kp = score;
                otidMap[otid].best_kpid = meta.id;
            }
        } else if (meta.type === "OT") {
            otidMap[otid].ot_score = Math.max(otidMap[otid].ot_score, score);
        }
    }

    const finalRanking: SearchResult[] = [];
    for (const [otid, scores] of Object.entries(otidMap)) {
        const weightedQ = scores.max_q * weights.Q;
        const weightedKP = scores.max_kp * weights.KP;
        const weightedOT = scores.ot_score * weights.OT;

        const maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
        const unionBonus =
            weightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;

        let finalScore = maxComponent + unionBonus;

        if (scores.timestamp) {
            const daysDiff =
                (currentTimestamp - scores.timestamp) / SECONDS_IN_DAY;
            if (daysDiff > 0) finalScore *= Math.exp(-DECAY_LAMBDA * daysDiff);
        }

        let boost = 1.0;
        const lexicalBonus = lexicalBonusMap.get(otid) || 0;
        if (lexicalBonus > 0) {
            boost *= 1 + Math.log1p(lexicalBonus) / 4;
        }

        if (queryYearWordIds && queryYearWordIds.length > 0) {
            const hasStructuredYearMatch =
                !!queryIntent?.years?.length &&
                scores.target_year !== undefined &&
                queryIntent.years.includes(scores.target_year);
            const hasLexicalYearMatch = yearHitMap.get(otid) === true;
            if (!hasStructuredYearMatch && !hasLexicalYearMatch) {
                boost *= 0.35;
            }
        }

        if (queryIntent && queryIntent.intentIds.length > 0) {
            if (hasIntentMatch(queryIntent.intentIds, scores.intent_ids)) {
                boost *= 1.2;
            } else if (hasIntentConflict(queryIntent.intentIds, scores.intent_ids)) {
                boost *= 0.3;
            }
        }

        finalRanking.push({
            otid,
            score: finalScore * boost,
            best_kpid: scores.best_kpid,
        });
    }

    return finalRanking.sort((a, b) => b.score - a.score).slice(0, 100);
}

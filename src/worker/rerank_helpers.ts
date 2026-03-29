export type AdaptiveRerankPlan = {
    rerankDocCount: number;
    reason: string;
    top1Top2Gap: number | null;
    clusterTopGap: number | null;
};

export type AdaptiveChunkPlan = {
    maxChunksPerDoc: number;
    reason: string;
    queryCharLength: number;
    termCount: number;
};

const MIN_RERANK_DOC_COUNT = 1;
const MAX_RERANK_DOC_COUNT = 6;
const ADAPTIVE_RERANK_TOP1_TOP2_GAP_THRESHOLD = 0.2;
const ADJACENT_CLUSTER_GAP_THRESHOLD = 0.12;
const CLUSTER_TOTAL_GAP_THRESHOLD = 0.35;

function countQueryTerms(query: string): number {
    return query
        .split(/[\s,，。；;、/]+/)
        .map((item) => item.trim())
        .filter(Boolean).length;
}

export function getAdaptiveRerankPlan(
    results: Array<{ coarseScore?: number }>,
): AdaptiveRerankPlan {
    if (results.length === 0) {
        return {
            rerankDocCount: 0,
            reason: "empty_results",
            top1Top2Gap: null,
            clusterTopGap: null,
        };
    }

    if (results.length === 1) {
        return {
            rerankDocCount: 1,
            reason: "single_candidate",
            top1Top2Gap: null,
            clusterTopGap: 0,
        };
    }

    const top1 = results[0]?.coarseScore ?? 0;
    const top2 = results[1]?.coarseScore ?? top1;
    const top1Top2Gap = top1 - top2;

    if (top1Top2Gap > ADAPTIVE_RERANK_TOP1_TOP2_GAP_THRESHOLD) {
        return {
            rerankDocCount: MIN_RERANK_DOC_COUNT,
            reason: "top1_clear_lead",
            top1Top2Gap,
            clusterTopGap: 0,
        };
    }

    let rerankDocCount = Math.min(2, results.length);
    let clusterTopGap = top1Top2Gap;

    for (
        let index = rerankDocCount;
        index < Math.min(MAX_RERANK_DOC_COUNT, results.length);
        index += 1
    ) {
        const previousScore = results[index - 1]?.coarseScore ?? top1;
        const currentScore = results[index]?.coarseScore ?? previousScore;
        const adjacentGap = previousScore - currentScore;
        const totalGap = top1 - currentScore;

        if (
            adjacentGap > ADJACENT_CLUSTER_GAP_THRESHOLD ||
            totalGap > CLUSTER_TOTAL_GAP_THRESHOLD
        ) {
            break;
        }

        rerankDocCount = index + 1;
        clusterTopGap = totalGap;
    }

    return {
        rerankDocCount,
        reason: rerankDocCount > 2 ? "low_gap_cluster" : "top2_near_tie",
        top1Top2Gap,
        clusterTopGap,
    };
}

export function getAdaptiveChunkPlan(
    query: string,
    rerankDocCount: number,
): AdaptiveChunkPlan {
    const normalizedQuery = query.trim();
    const queryCharLength = normalizedQuery.replace(/\s+/g, "").length;
    const termCount = countQueryTerms(normalizedQuery);
    const isShortQuery = queryCharLength <= 8;
    const isMediumQuery =
        queryCharLength <= 18 ||
        (queryCharLength <= 24 && termCount <= 4);

    if (rerankDocCount <= 1) {
        return {
            maxChunksPerDoc: isShortQuery ? 3 : 4,
            reason: isShortQuery ? "single_doc_short_query" : "single_doc",
            queryCharLength,
            termCount,
        };
    }

    if (rerankDocCount <= 2) {
        return {
            maxChunksPerDoc: isShortQuery ? 4 : 6,
            reason: isShortQuery ? "two_doc_short_query" : "two_doc",
            queryCharLength,
            termCount,
        };
    }

    if (rerankDocCount <= 4) {
        return {
            maxChunksPerDoc: isShortQuery ? 5 : isMediumQuery ? 6 : 7,
            reason: isShortQuery
                ? "small_cluster_short_query"
                : isMediumQuery
                  ? "small_cluster_medium_query"
                  : "small_cluster_long_query",
            queryCharLength,
            termCount,
        };
    }

    return {
        maxChunksPerDoc: isShortQuery ? 6 : isMediumQuery ? 8 : 10,
        reason: isShortQuery
            ? "wide_cluster_short_query"
            : isMediumQuery
              ? "wide_cluster_medium_query"
              : "wide_cluster_long_query",
        queryCharLength,
        termCount,
    };
}

export function splitIntoSemanticChunks(
    text: string,
    maxLen = 150,
    maxChunks?: number,
): string[] {
    const sentences =
        text.match(/[^\u3002\uff01\uff1f\n]+[\u3002\uff01\uff1f\n]*/g) || [text];
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

    return typeof maxChunks === "number" ? chunks.slice(0, maxChunks) : chunks;
}

export function normalizeSnippetScore(rawScore: number): number {
    const normalized = (rawScore + 1) / 2;
    return Math.min(1, Math.max(0, normalized));
}

export function normalizeMinMax(values: number[]): number[] {
    if (values.length === 0) return [];

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    if (Math.abs(maxValue - minValue) < 1e-9) {
        return values.map(() => 1);
    }

    return values.map((value) => (value - minValue) / (maxValue - minValue));
}

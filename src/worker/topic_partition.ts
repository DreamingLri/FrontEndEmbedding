import type { Metadata, ParsedQueryIntent } from "./vector_engine.ts";
import { resolveMetadataTopicIds } from "./vector_engine.ts";

export type TopicPartitionIndex = {
    topicCandidateIndex: Map<string, number[]>;
    unlabeledCandidateIndices: number[];
    metadataIntentIds: string[][];
    metadataCount: number;
    candidateIndicesCache: Map<string, number[] | null>;
    candidateVisitMarks: Uint32Array;
    candidateVisitGeneration: number;
};

export function createEmptyTopicPartitionIndex(): TopicPartitionIndex {
    return {
        topicCandidateIndex: new Map<string, number[]>(),
        unlabeledCandidateIndices: [],
        metadataIntentIds: [],
        metadataCount: 0,
        candidateIndicesCache: new Map<string, number[] | null>(),
        candidateVisitMarks: new Uint32Array(0),
        candidateVisitGeneration: 0,
    };
}

export function buildTopicPartitionIndex(
    metadata: readonly Metadata[],
): TopicPartitionIndex {
    const topicCandidateIndex = new Map<string, number[]>();
    const unlabeledCandidateIndices: number[] = [];
    const metadataIntentIds = metadata.map((meta) => meta.intent_ids || []);

    metadata.forEach((meta, index) => {
        const topicIds = resolveMetadataTopicIds(meta);
        if (topicIds.length === 0) {
            unlabeledCandidateIndices.push(index);
            return;
        }

        topicIds.forEach((topicId) => {
            const bucket = topicCandidateIndex.get(topicId);
            if (bucket) {
                bucket.push(index);
            } else {
                topicCandidateIndex.set(topicId, [index]);
            }
        });
    });

    return {
        topicCandidateIndex,
        unlabeledCandidateIndices,
        metadataIntentIds,
        metadataCount: metadata.length,
        candidateIndicesCache: new Map<string, number[] | null>(),
        candidateVisitMarks: new Uint32Array(metadata.length),
        candidateVisitGeneration: 0,
    };
}

function normalizePartitionKey(params: {
    topicIds: readonly string[];
    intentIds: readonly string[];
}): string {
    const normalizedTopics = Array.from(new Set(params.topicIds)).sort().join("|");
    const normalizedIntents = Array.from(new Set(params.intentIds))
        .sort()
        .join("|");
    return `${normalizedTopics}::${normalizedIntents}`;
}

function hasIntentOverlap(
    queryIntentIds: readonly string[],
    docIntentIds: readonly string[],
): boolean {
    return queryIntentIds.some((intentId) => docIntentIds.includes(intentId));
}

function nextCandidateVisitGeneration(
    partitionIndex: TopicPartitionIndex,
): number {
    const nextGeneration = partitionIndex.candidateVisitGeneration + 1;
    if (nextGeneration >= 0xffffffff) {
        partitionIndex.candidateVisitMarks.fill(0);
        partitionIndex.candidateVisitGeneration = 1;
        return 1;
    }

    partitionIndex.candidateVisitGeneration = nextGeneration;
    return nextGeneration;
}

export function getCandidateIndicesForQuery(
    queryIntent: Pick<ParsedQueryIntent, "topicIds" | "intentIds">,
    partitionIndex: TopicPartitionIndex,
): number[] | undefined {
    if (queryIntent.topicIds.length === 0) return undefined;

    const partitionKey = normalizePartitionKey({
        topicIds: queryIntent.topicIds,
        intentIds: queryIntent.intentIds,
    });
    if (!partitionKey) {
        return undefined;
    }

    const cachedCandidates = partitionIndex.candidateIndicesCache.get(partitionKey);
    if (cachedCandidates !== undefined) {
        return cachedCandidates || undefined;
    }

    const visitGeneration = nextCandidateVisitGeneration(partitionIndex);
    const candidateIndices: number[] = [];
    const visitMarks = partitionIndex.candidateVisitMarks;

    partitionIndex.unlabeledCandidateIndices.forEach((index) => {
        visitMarks[index] = visitGeneration;
        candidateIndices.push(index);
    });

    queryIntent.topicIds.forEach((topicId) => {
        const bucket = partitionIndex.topicCandidateIndex.get(topicId);
        if (!bucket) return;
        bucket.forEach((index) => {
            const docIntentIds = partitionIndex.metadataIntentIds[index] || [];
            if (
                queryIntent.intentIds.length > 0 &&
                docIntentIds.length > 0 &&
                !hasIntentOverlap(queryIntent.intentIds, docIntentIds)
            ) {
                return;
            }
            if (visitMarks[index] === visitGeneration) {
                return;
            }
            visitMarks[index] = visitGeneration;
            candidateIndices.push(index);
        });
    });

    if (
        candidateIndices.length === 0 ||
        candidateIndices.length >= partitionIndex.metadataCount
    ) {
        partitionIndex.candidateIndicesCache.set(partitionKey, null);
        return undefined;
    }

    partitionIndex.candidateIndicesCache.set(partitionKey, candidateIndices);
    return candidateIndices;
}

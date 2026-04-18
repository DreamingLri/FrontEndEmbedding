import type { Metadata, ParsedQueryIntent } from "./vector_engine.ts";
import { resolveMetadataTopicIds } from "./vector_engine.ts";

export type TopicPartitionIndex = {
    topicCandidateIndex: Map<string, number[]>;
    unlabeledCandidateIndices: number[];
    metadataCount: number;
    candidateIndicesCache: Map<string, number[] | null>;
    candidateVisitMarks: Uint32Array;
    candidateVisitGeneration: number;
};

export function createEmptyTopicPartitionIndex(): TopicPartitionIndex {
    return {
        topicCandidateIndex: new Map<string, number[]>(),
        unlabeledCandidateIndices: [],
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
        metadataCount: metadata.length,
        candidateIndicesCache: new Map<string, number[] | null>(),
        candidateVisitMarks: new Uint32Array(metadata.length),
        candidateVisitGeneration: 0,
    };
}

function normalizeTopicKey(topicIds: readonly string[]): string {
    return Array.from(new Set(topicIds)).sort().join("|");
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
    queryIntent: Pick<ParsedQueryIntent, "topicIds">,
    partitionIndex: TopicPartitionIndex,
): number[] | undefined {
    if (queryIntent.topicIds.length === 0) return undefined;

    const topicKey = normalizeTopicKey(queryIntent.topicIds);
    if (!topicKey) {
        return undefined;
    }

    const cachedCandidates = partitionIndex.candidateIndicesCache.get(topicKey);
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

    topicKey.split("|").forEach((topicId) => {
        const bucket = partitionIndex.topicCandidateIndex.get(topicId);
        if (!bucket) return;
        bucket.forEach((index) => {
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
        partitionIndex.candidateIndicesCache.set(topicKey, null);
        return undefined;
    }

    partitionIndex.candidateIndicesCache.set(topicKey, candidateIndices);
    return candidateIndices;
}

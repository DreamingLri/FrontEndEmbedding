import type { Metadata, ParsedQueryIntent } from "./vector_engine";
import { resolveMetadataTopicIds } from "./vector_engine";

export type TopicPartitionIndex = {
    topicCandidateIndex: Map<string, number[]>;
    unlabeledCandidateIndices: number[];
    metadataCount: number;
};

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
    };
}

export function getCandidateIndicesForQuery(
    queryIntent: Pick<ParsedQueryIntent, "topicIds">,
    partitionIndex: TopicPartitionIndex,
): number[] | undefined {
    if (queryIntent.topicIds.length === 0) return undefined;

    const candidateSet = new Set<number>(
        partitionIndex.unlabeledCandidateIndices,
    );
    queryIntent.topicIds.forEach((topicId) => {
        const bucket = partitionIndex.topicCandidateIndex.get(topicId);
        if (!bucket) return;
        bucket.forEach((index) => candidateSet.add(index));
    });

    if (
        candidateSet.size === 0 ||
        candidateSet.size >= partitionIndex.metadataCount
    ) {
        return undefined;
    }

    return Array.from(candidateSet);
}

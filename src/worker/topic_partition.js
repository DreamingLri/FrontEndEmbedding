"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.buildTopicPartitionIndex = buildTopicPartitionIndex;
exports.getCandidateIndicesForQuery = getCandidateIndicesForQuery;
var vector_engine_1 = require("./vector_engine");
function buildTopicPartitionIndex(metadata) {
    var topicCandidateIndex = new Map();
    var unlabeledCandidateIndices = [];
    metadata.forEach(function (meta, index) {
        var topicIds = (0, vector_engine_1.resolveMetadataTopicIds)(meta);
        if (topicIds.length === 0) {
            unlabeledCandidateIndices.push(index);
            return;
        }
        topicIds.forEach(function (topicId) {
            var bucket = topicCandidateIndex.get(topicId);
            if (bucket) {
                bucket.push(index);
            }
            else {
                topicCandidateIndex.set(topicId, [index]);
            }
        });
    });
    return {
        topicCandidateIndex: topicCandidateIndex,
        unlabeledCandidateIndices: unlabeledCandidateIndices,
        metadataCount: metadata.length,
    };
}
function getCandidateIndicesForQuery(queryIntent, partitionIndex) {
    if (queryIntent.topicIds.length === 0)
        return undefined;
    var candidateSet = new Set(partitionIndex.unlabeledCandidateIndices);
    queryIntent.topicIds.forEach(function (topicId) {
        var bucket = partitionIndex.topicCandidateIndex.get(topicId);
        if (!bucket)
            return;
        bucket.forEach(function (index) { return candidateSet.add(index); });
    });
    if (candidateSet.size === 0 ||
        candidateSet.size >= partitionIndex.metadataCount) {
        return undefined;
    }
    return Array.from(candidateSet);
}

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAdaptiveRerankPlan = getAdaptiveRerankPlan;
exports.getAdaptiveChunkPlan = getAdaptiveChunkPlan;
exports.splitIntoSemanticChunks = splitIntoSemanticChunks;
exports.normalizeSnippetScore = normalizeSnippetScore;
exports.normalizeMinMax = normalizeMinMax;
var MIN_RERANK_DOC_COUNT = 1;
var MAX_RERANK_DOC_COUNT = 6;
var ADAPTIVE_RERANK_TOP1_TOP2_GAP_THRESHOLD = 0.2;
var ADJACENT_CLUSTER_GAP_THRESHOLD = 0.12;
var CLUSTER_TOTAL_GAP_THRESHOLD = 0.35;
function countQueryTerms(query) {
    return query
        .split(/[\s,，。；;、/]+/)
        .map(function (item) { return item.trim(); })
        .filter(Boolean).length;
}
function getAdaptiveRerankPlan(results) {
    var _a, _b, _c, _d, _e, _f, _g, _h;
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
    var top1 = (_b = (_a = results[0]) === null || _a === void 0 ? void 0 : _a.coarseScore) !== null && _b !== void 0 ? _b : 0;
    var top2 = (_d = (_c = results[1]) === null || _c === void 0 ? void 0 : _c.coarseScore) !== null && _d !== void 0 ? _d : top1;
    var top1Top2Gap = top1 - top2;
    if (top1Top2Gap > ADAPTIVE_RERANK_TOP1_TOP2_GAP_THRESHOLD) {
        return {
            rerankDocCount: MIN_RERANK_DOC_COUNT,
            reason: "top1_clear_lead",
            top1Top2Gap: top1Top2Gap,
            clusterTopGap: 0,
        };
    }
    var rerankDocCount = Math.min(2, results.length);
    var clusterTopGap = top1Top2Gap;
    for (var index = rerankDocCount; index < Math.min(MAX_RERANK_DOC_COUNT, results.length); index += 1) {
        var previousScore = (_f = (_e = results[index - 1]) === null || _e === void 0 ? void 0 : _e.coarseScore) !== null && _f !== void 0 ? _f : top1;
        var currentScore = (_h = (_g = results[index]) === null || _g === void 0 ? void 0 : _g.coarseScore) !== null && _h !== void 0 ? _h : previousScore;
        var adjacentGap = previousScore - currentScore;
        var totalGap = top1 - currentScore;
        if (adjacentGap > ADJACENT_CLUSTER_GAP_THRESHOLD ||
            totalGap > CLUSTER_TOTAL_GAP_THRESHOLD) {
            break;
        }
        rerankDocCount = index + 1;
        clusterTopGap = totalGap;
    }
    return {
        rerankDocCount: rerankDocCount,
        reason: rerankDocCount > 2 ? "low_gap_cluster" : "top2_near_tie",
        top1Top2Gap: top1Top2Gap,
        clusterTopGap: clusterTopGap,
    };
}
function getAdaptiveChunkPlan(query, rerankDocCount) {
    var normalizedQuery = query.trim();
    var queryCharLength = normalizedQuery.replace(/\s+/g, "").length;
    var termCount = countQueryTerms(normalizedQuery);
    var isShortQuery = queryCharLength <= 8;
    var isMediumQuery = queryCharLength <= 18 ||
        (queryCharLength <= 24 && termCount <= 4);
    if (rerankDocCount <= 1) {
        return {
            maxChunksPerDoc: isShortQuery ? 3 : 4,
            reason: isShortQuery ? "single_doc_short_query" : "single_doc",
            queryCharLength: queryCharLength,
            termCount: termCount,
        };
    }
    if (rerankDocCount <= 2) {
        return {
            maxChunksPerDoc: isShortQuery ? 4 : 6,
            reason: isShortQuery ? "two_doc_short_query" : "two_doc",
            queryCharLength: queryCharLength,
            termCount: termCount,
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
            queryCharLength: queryCharLength,
            termCount: termCount,
        };
    }
    return {
        maxChunksPerDoc: isShortQuery ? 6 : isMediumQuery ? 8 : 10,
        reason: isShortQuery
            ? "wide_cluster_short_query"
            : isMediumQuery
                ? "wide_cluster_medium_query"
                : "wide_cluster_long_query",
        queryCharLength: queryCharLength,
        termCount: termCount,
    };
}
function splitIntoSemanticChunks(text, maxLen, maxChunks) {
    if (maxLen === void 0) { maxLen = 150; }
    var sentences = text.match(/[^\u3002\uff01\uff1f\n]+[\u3002\uff01\uff1f\n]*/g) || [text];
    var chunks = [];
    var currentChunk = "";
    for (var _i = 0, sentences_1 = sentences; _i < sentences_1.length; _i++) {
        var sentence = sentences_1[_i];
        if ((currentChunk + sentence).length > maxLen && currentChunk.length > 0) {
            chunks.push(currentChunk);
            currentChunk = "";
        }
        currentChunk += sentence;
    }
    if (currentChunk)
        chunks.push(currentChunk);
    return typeof maxChunks === "number" ? chunks.slice(0, maxChunks) : chunks;
}
function normalizeSnippetScore(rawScore) {
    var normalized = (rawScore + 1) / 2;
    return Math.min(1, Math.max(0, normalized));
}
function normalizeMinMax(values) {
    if (values.length === 0)
        return [];
    var minValue = Math.min.apply(Math, values);
    var maxValue = Math.max.apply(Math, values);
    if (Math.abs(maxValue - minValue) < 1e-9) {
        return values.map(function () { return 1; });
    }
    return values.map(function (value) { return (value - minValue) / (maxValue - minValue); });
}

"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CANONICAL_PIPELINE_PRESET = void 0;
exports.buildPipelineTermMaps = buildPipelineTermMaps;
exports.buildSearchPipelineQueryContext = buildSearchPipelineQueryContext;
exports.mergeCoarseMatchesIntoDocuments = mergeCoarseMatchesIntoDocuments;
exports.executeRetrievalStage = executeRetrievalStage;
exports.executeSearchPipeline = executeSearchPipeline;
var fmm_tokenize_1 = require("./fmm_tokenize");
var direct_answer_display_1 = require("./direct_answer_display");
var vector_engine_1 = require("./vector_engine");
var topic_partition_1 = require("./topic_partition");
exports.CANONICAL_PIPELINE_PRESET = {
    name: "canonical_full_v1",
    retrieval: {
        weights: {
            Q: 0.3333333333333333,
            KP: 0.13333333333333333,
            OT: 0.5333333333333333,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
    },
    display: {
        rejectThreshold: 0.4,
        rerankBlendAlpha: 0.15,
        bestSentenceThreshold: 0.4,
        fetchMatchLimit: 15,
        fetchWeakMatchLimit: 10,
    },
};
var ROUTE_ENTRY_TOPIC_BY_PATTERN = [
    {
        pattern: /录取|拟录取|考上|录取结果|拿到.*录取/,
        entryTopic: "新生录取后手续总入口",
    },
    {
        pattern: /新生|入学前|正式入学|入学以后|报到前/,
        entryTopic: "新生入学总入口",
    },
];
function nowMs() {
    if (typeof performance !== "undefined" && performance.now) {
        return performance.now();
    }
    return Date.now();
}
function dedupe(items) {
    return Array.from(new Set(items));
}
var QUERY_EXPANSION_RULES = [
    {
        pattern: /现场确认/,
        terms: ["网上确认"],
        intentTerms: ["网上确认"],
    },
];
function buildExpandedQueryWords(query, vocabMap) {
    var baseWords = (0, fmm_tokenize_1.fmmTokenize)(query, vocabMap);
    var expandedWords = QUERY_EXPANSION_RULES.flatMap(function (rule) {
        if (!rule.pattern.test(query)) {
            return [];
        }
        return rule.terms.flatMap(function (term) { return (0, fmm_tokenize_1.fmmTokenize)(term, vocabMap); });
    });
    return dedupe(__spreadArray(__spreadArray([], baseWords, true), expandedWords, true));
}
function buildExpandedIntentQuery(query) {
    var expandedTerms = dedupe(QUERY_EXPANSION_RULES.flatMap(function (rule) {
        return rule.pattern.test(query) ? rule.intentTerms || [] : [];
    }));
    if (expandedTerms.length === 0) {
        return query;
    }
    return "".concat(query, " ").concat(expandedTerms.join(" "));
}
function buildPipelineTermMaps(vocabMap) {
    var scopeSpecificityWordIdToTerm = new Map();
    vector_engine_1.QUERY_SCOPE_SPECIFICITY_TERMS.forEach(function (term) {
        var wordId = vocabMap.get(term);
        if (wordId !== undefined) {
            scopeSpecificityWordIdToTerm.set(wordId, term);
        }
    });
    var directAnswerEvidenceWordIdToTerm = new Map();
    vector_engine_1.DIRECT_ANSWER_EVIDENCE_TERMS.forEach(function (term) {
        var wordId = vocabMap.get(term);
        if (wordId !== undefined) {
            directAnswerEvidenceWordIdToTerm.set(wordId, term);
        }
    });
    return {
        scopeSpecificityWordIdToTerm: scopeSpecificityWordIdToTerm,
        directAnswerEvidenceWordIdToTerm: directAnswerEvidenceWordIdToTerm,
    };
}
function buildSearchPipelineQueryContext(query, vocabMap, topicPartitionIndex) {
    var expandedIntentQuery = buildExpandedIntentQuery(query);
    var parsedQueryIntent = (0, vector_engine_1.parseQueryIntent)(expandedIntentQuery);
    var queryIntent = expandedIntentQuery === query
        ? parsedQueryIntent
        : __assign(__assign({}, parsedQueryIntent), { rawQuery: query });
    var candidateIndices = (0, topic_partition_1.getCandidateIndicesForQuery)(queryIntent, topicPartitionIndex);
    var queryWords = buildExpandedQueryWords(query, vocabMap);
    var querySparse = (0, vector_engine_1.getQuerySparse)(queryWords, vocabMap);
    var queryYearWordIds = queryIntent.years
        .map(String)
        .map(function (year) { return vocabMap.get(year); })
        .filter(function (item) { return item !== undefined; });
    return {
        query: query,
        queryIntent: queryIntent,
        queryWords: queryWords,
        querySparse: querySparse,
        queryYearWordIds: queryYearWordIds,
        candidateIndices: candidateIndices,
    };
}
function inferEntryTopic(query) {
    for (var _i = 0, ROUTE_ENTRY_TOPIC_BY_PATTERN_1 = ROUTE_ENTRY_TOPIC_BY_PATTERN; _i < ROUTE_ENTRY_TOPIC_BY_PATTERN_1.length; _i++) {
        var item = ROUTE_ENTRY_TOPIC_BY_PATTERN_1[_i];
        if (item.pattern.test(query)) {
            return item.entryTopic;
        }
    }
    return undefined;
}
function inferClarifyOrRouteBehavior(query, queryIntent) {
    var normalizedQuery = query.replace(/\s+/g, "");
    var hasPendingOfferCue = /拟录取/.test(normalizedQuery);
    var hasOnboardingCue = /新生|入学|录取|考上|录取结果/.test(normalizedQuery) ||
        /拿到.*录取/.test(normalizedQuery);
    var hasClarifyCue = /审核|初审|资格审核|材料|补交|申请|获批|过审|通过了|通知我通过|学校通知我通过/.test(normalizedQuery);
    if (hasPendingOfferCue) {
        return "clarify";
    }
    if (/新生|入学前|正式入学/.test(normalizedQuery) &&
        queryIntent.signals.hasGenericNextStep) {
        return "route_to_entry";
    }
    if (hasOnboardingCue &&
        queryIntent.signals.hasGenericNextStep &&
        !hasClarifyCue &&
        !queryIntent.signals.hasStrongDetailAnchor) {
        return "route_to_entry";
    }
    return "clarify";
}
function buildPipelineDecision(params) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
    var query = params.query, queryIntent = params.queryIntent, searchOutput = params.searchOutput;
    var rawMode = ((_a = searchOutput.responseDecision) === null || _a === void 0 ? void 0 : _a.mode) ||
        (searchOutput.rejection ? "reject" : "direct_answer");
    var behavior = rawMode === "clarify_or_route"
        ? inferClarifyOrRouteBehavior(query, queryIntent)
        : rawMode === "reject"
            ? "reject"
            : "direct_answer";
    return {
        behavior: behavior,
        rawMode: rawMode,
        confidence: (_c = (_b = searchOutput.responseDecision) === null || _b === void 0 ? void 0 : _b.confidence) !== null && _c !== void 0 ? _c : 0.62,
        reason: ((_d = searchOutput.responseDecision) === null || _d === void 0 ? void 0 : _d.reason) ||
            ((_e = searchOutput.rejection) === null || _e === void 0 ? void 0 : _e.reason) ||
            "scored_pipeline_behavior",
        entryTopic: behavior === "route_to_entry" ? inferEntryTopic(query) : undefined,
        preferLatestWithinTopic: (_g = (_f = searchOutput.responseDecision) === null || _f === void 0 ? void 0 : _f.preferLatestWithinTopic) !== null && _g !== void 0 ? _g : false,
        useWeakMatches: behavior !== "direct_answer" &&
            ((_j = (_h = searchOutput.responseDecision) === null || _h === void 0 ? void 0 : _h.useWeakMatches) !== null && _j !== void 0 ? _j : searchOutput.weakMatches.length > 0),
        rejectionReason: ((_k = searchOutput.rejection) === null || _k === void 0 ? void 0 : _k.reason) || null,
        displayRejected: false,
    };
}
function mergeCoarseMatchesIntoDocuments(documents, coarseMatches) {
    var documentMap = new Map(documents.map(function (doc) { return [doc.otid || doc.id || "", doc]; }));
    return coarseMatches
        .map(function (match) {
        var _a, _b, _c, _d, _e, _f;
        var doc = documentMap.get(match.otid);
        if (!doc) {
            return null;
        }
        return __assign(__assign({}, doc), { score: (_a = match.score) !== null && _a !== void 0 ? _a : doc.score, coarseScore: (_c = (_b = match.score) !== null && _b !== void 0 ? _b : doc.coarseScore) !== null && _c !== void 0 ? _c : doc.score, displayScore: (_e = (_d = match.score) !== null && _d !== void 0 ? _d : doc.displayScore) !== null && _e !== void 0 ? _e : doc.score, best_kpid: (_f = match.best_kpid) !== null && _f !== void 0 ? _f : doc.best_kpid });
    })
        .filter(Boolean);
}
function executeRetrievalStage(params) {
    var _a, _b;
    var query = params.query, queryVector = params.queryVector, queryContext = params.queryContext, metadata = params.metadata, vectorMatrix = params.vectorMatrix, dimensions = params.dimensions, currentTimestamp = params.currentTimestamp, bm25Stats = params.bm25Stats, termMaps = params.termMaps, _c = params.preset, preset = _c === void 0 ? exports.CANONICAL_PIPELINE_PRESET : _c;
    var startedAt = nowMs();
    var searchOutput = (0, vector_engine_1.searchAndRank)({
        queryVector: queryVector,
        querySparse: queryContext.querySparse,
        queryWords: queryContext.queryWords,
        queryYearWordIds: queryContext.queryYearWordIds,
        queryIntent: queryContext.queryIntent,
        metadata: metadata,
        vectorMatrix: vectorMatrix,
        dimensions: dimensions,
        currentTimestamp: currentTimestamp,
        bm25Stats: bm25Stats,
        candidateIndices: queryContext.candidateIndices,
        scopeSpecificityWordIdToTerm: termMaps === null || termMaps === void 0 ? void 0 : termMaps.scopeSpecificityWordIdToTerm,
        directAnswerEvidenceWordIdToTerm: termMaps === null || termMaps === void 0 ? void 0 : termMaps.directAnswerEvidenceWordIdToTerm,
        weights: preset.retrieval.weights,
        topHybridLimit: preset.retrieval.topHybridLimit,
        kpAggregationMode: preset.retrieval.kpAggregationMode,
        kpTopN: preset.retrieval.kpTopN,
        kpTailWeight: preset.retrieval.kpTailWeight,
        lexicalBonusMode: preset.retrieval.lexicalBonusMode,
        kpRoleRerankMode: preset.retrieval.kpRoleRerankMode,
        kpRoleDocWeight: preset.retrieval.kpRoleDocWeight,
    });
    return {
        queryContext: queryContext,
        searchOutput: searchOutput,
        retrievalDecision: buildPipelineDecision({
            query: query,
            queryIntent: queryContext.queryIntent,
            searchOutput: searchOutput,
        }),
        candidateCount: (_b = (_a = queryContext.candidateIndices) === null || _a === void 0 ? void 0 : _a.length) !== null && _b !== void 0 ? _b : metadata.length,
        searchMs: nowMs() - startedAt,
    };
}
function executeSearchPipeline(params) {
    return __awaiter(this, void 0, void 0, function () {
        var query, queryVector, queryContext, metadata, vectorMatrix, dimensions, currentTimestamp, bm25Stats, extractor, documentLoader, termMaps, _a, preset, onStatus, pipelineStartedAt, retrievalStage, searchOutput, retrievalDecision, shouldFetchWeakResults, matchIds, weakMatchIds, fetchIds, fetchMs, rerankMs, fetchedDocumentCount, results, weakResults, finalDecision, rerankStats, initialTopConfidence, directAnswerRescue, fetchStartedAt, documents, directDocuments, rerankStartedAt, displayStage, displayRejected;
        var _b, _c, _d, _e, _f, _g;
        return __generator(this, function (_h) {
            switch (_h.label) {
                case 0:
                    query = params.query, queryVector = params.queryVector, queryContext = params.queryContext, metadata = params.metadata, vectorMatrix = params.vectorMatrix, dimensions = params.dimensions, currentTimestamp = params.currentTimestamp, bm25Stats = params.bm25Stats, extractor = params.extractor, documentLoader = params.documentLoader, termMaps = params.termMaps, _a = params.preset, preset = _a === void 0 ? exports.CANONICAL_PIPELINE_PRESET : _a, onStatus = params.onStatus;
                    pipelineStartedAt = nowMs();
                    retrievalStage = executeRetrievalStage({
                        query: query,
                        queryVector: queryVector,
                        queryContext: queryContext,
                        metadata: metadata,
                        vectorMatrix: vectorMatrix,
                        dimensions: dimensions,
                        currentTimestamp: currentTimestamp,
                        bm25Stats: bm25Stats,
                        termMaps: termMaps,
                        preset: preset,
                    });
                    searchOutput = retrievalStage.searchOutput, retrievalDecision = retrievalStage.retrievalDecision;
                    shouldFetchWeakResults = retrievalDecision.behavior === "clarify" ||
                        retrievalDecision.behavior === "route_to_entry" ||
                        ((_b = searchOutput.rejection) === null || _b === void 0 ? void 0 : _b.reason) === "low_topic_coverage";
                    matchIds = retrievalDecision.behavior === "direct_answer"
                        ? searchOutput.matches
                            .slice(0, preset.display.fetchMatchLimit)
                            .map(function (item) { return item.otid; })
                        : [];
                    weakMatchIds = shouldFetchWeakResults
                        ? searchOutput.weakMatches
                            .slice(0, preset.display.fetchWeakMatchLimit)
                            .map(function (item) { return item.otid; })
                        : [];
                    fetchIds = dedupe(__spreadArray(__spreadArray([], matchIds, true), weakMatchIds, true));
                    fetchMs = 0;
                    rerankMs = 0;
                    fetchedDocumentCount = 0;
                    results = [];
                    weakResults = [];
                    finalDecision = retrievalDecision;
                    rerankStats = {
                        rerankedDocCount: 0,
                        chunksScored: 0,
                        topConfidence: null,
                    };
                    initialTopConfidence = null;
                    if (!(fetchIds.length > 0)) return [3 /*break*/, 4];
                    onStatus === null || onStatus === void 0 ? void 0 : onStatus("正在请求原文数据...");
                    fetchStartedAt = nowMs();
                    return [4 /*yield*/, documentLoader({
                            query: query,
                            otids: fetchIds,
                        })];
                case 1:
                    documents = _h.sent();
                    fetchMs = nowMs() - fetchStartedAt;
                    fetchedDocumentCount = documents.length;
                    if (!(retrievalDecision.behavior === "direct_answer")) return [3 /*break*/, 3];
                    directDocuments = mergeCoarseMatchesIntoDocuments(documents, searchOutput.matches
                        .slice(0, preset.display.fetchMatchLimit)
                        .map(function (item) { return ({
                        otid: item.otid,
                        score: item.score,
                        best_kpid: item.best_kpid,
                    }); }));
                    onStatus === null || onStatus === void 0 ? void 0 : onStatus("正在重排并提炼可信原话...");
                    rerankStartedAt = nowMs();
                    return [4 /*yield*/, (0, direct_answer_display_1.runDirectAnswerDisplayStage)({
                            query: query,
                            queryVector: queryVector,
                            documents: directDocuments,
                            extractor: extractor,
                            dimensions: dimensions,
                            preset: preset,
                            querySignals: (_c = searchOutput.diagnostics) === null || _c === void 0 ? void 0 : _c.querySignals,
                            retrievalSignals: (_d = searchOutput.diagnostics) === null || _d === void 0 ? void 0 : _d.retrievalSignals,
                        })];
                case 2:
                    displayStage = _h.sent();
                    rerankMs = nowMs() - rerankStartedAt;
                    rerankStats = displayStage.stats;
                    initialTopConfidence = displayStage.initialTopConfidence;
                    directAnswerRescue = displayStage.directAnswerRescue;
                    displayRejected = displayStage.displayRejected;
                    finalDecision = displayRejected
                        ? __assign(__assign({}, retrievalDecision), { behavior: "reject", rejectionReason: "display_threshold", reason: "display_threshold_reject", displayRejected: true, useWeakMatches: false }) : retrievalDecision;
                    results = displayRejected ? [] : displayStage.documents;
                    _h.label = 3;
                case 3:
                    if (shouldFetchWeakResults) {
                        weakResults = mergeCoarseMatchesIntoDocuments(documents, searchOutput.weakMatches
                            .slice(0, preset.display.fetchWeakMatchLimit)
                            .map(function (item) { return ({
                            otid: item.otid,
                            score: item.score,
                            best_kpid: item.best_kpid,
                        }); }));
                    }
                    _h.label = 4;
                case 4: return [2 /*return*/, {
                        query: query,
                        presetName: preset.name,
                        queryContext: queryContext,
                        searchOutput: searchOutput,
                        responseDecision: searchOutput.responseDecision,
                        retrievalDecision: retrievalDecision,
                        finalDecision: finalDecision,
                        rejection: searchOutput.rejection,
                        results: results,
                        weakResults: weakResults,
                        trace: {
                            totalMs: nowMs() - pipelineStartedAt,
                            searchMs: retrievalStage.searchMs,
                            fetchMs: fetchMs,
                            rerankMs: rerankMs,
                            candidateCount: retrievalStage.candidateCount,
                            partitionUsed: Boolean(queryContext.candidateIndices),
                            partitionCandidateCount: (_e = queryContext.candidateIndices) === null || _e === void 0 ? void 0 : _e.length,
                            matchCount: searchOutput.matches.length,
                            weakMatchCount: searchOutput.weakMatches.length,
                            fetchedDocumentCount: fetchedDocumentCount,
                            rerankedDocCount: rerankStats.rerankedDocCount,
                            chunksScored: rerankStats.chunksScored,
                            rerankWindowReason: rerankStats.windowReason,
                            maxChunksPerDoc: rerankStats.maxChunksPerDoc,
                            chunkPlanReason: rerankStats.chunkPlanReason,
                            initialTopConfidence: initialTopConfidence,
                            topConfidence: rerankStats.topConfidence,
                            rejectionThreshold: preset.display.rejectThreshold,
                            querySignals: (_f = searchOutput.diagnostics) === null || _f === void 0 ? void 0 : _f.querySignals,
                            retrievalSignals: (_g = searchOutput.diagnostics) === null || _g === void 0 ? void 0 : _g.retrievalSignals,
                            directAnswerRescue: directAnswerRescue,
                        },
                    }];
            }
        });
    });
}

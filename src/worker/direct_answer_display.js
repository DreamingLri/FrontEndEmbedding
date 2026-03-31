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
Object.defineProperty(exports, "__esModule", { value: true });
exports.runDirectAnswerDisplayStage = runDirectAnswerDisplayStage;
var rerank_helpers_1 = require("./rerank_helpers");
var NEAR_TIE_COARSE_WINDOW = 0.025;
var DIRECT_ANSWER_RESCUE_MARGIN = 0.12;
var MIN_RESCUE_CHUNK_LIMIT = 6;
var MAX_RESCUE_CHUNK_LIMIT = 14;
var MAX_RESCUE_RERANK_DOC_COUNT = 6;
var RESCUE_ACCEPT_DELTA = 0.005;
function clamp01(value) {
    return Math.min(1, Math.max(0, value));
}
function inferQueryDisambiguationFloor(query) {
    var normalizedQuery = query.replace(/\s+/g, "");
    if (/考什么|考哪些|考试内容|考试科目|科目|题型/.test(normalizedQuery)) {
        return {
            rerankDocCount: 3,
            reason: "exam_content_disambiguation",
        };
    }
    if (/(广州国家实验室|广州实验室|鹏城国家实验室|鹏城)/.test(normalizedQuery) &&
        /联合培养|博士/.test(normalizedQuery)) {
        return {
            rerankDocCount: 3,
            reason: "lab_entity_disambiguation",
        };
    }
    if (/临床医学博士/.test(normalizedQuery) &&
        /同等学力|答辩/.test(normalizedQuery)) {
        return {
            rerankDocCount: 3,
            reason: "clinical_doctoral_defense_disambiguation",
        };
    }
    return {
        rerankDocCount: 0,
    };
}
function computeTitleAdjustment(query, docTitle) {
    var normalizedQuery = query.replace(/\s+/g, "");
    var normalizedTitle = (docTitle || "").replace(/\s+/g, "");
    if (!normalizedTitle) {
        return 0;
    }
    var adjustment = 0;
    if (/考什么|考哪些|考试内容|考试科目|科目|题型/.test(normalizedQuery)) {
        if (/章程|专业目录/.test(normalizedTitle)) {
            adjustment += 0.14;
        }
        if (/成绩查询|合格成绩要求|复试/.test(normalizedTitle)) {
            adjustment -= 0.2;
        }
    }
    if (/(广州国家实验室|广州实验室)/.test(normalizedQuery)) {
        if (/(广州国家实验室|广州实验室)/.test(normalizedTitle)) {
            adjustment += 0.16;
        }
        if (/鹏城/.test(normalizedTitle)) {
            adjustment -= 0.22;
        }
    }
    if (/(鹏城国家实验室|鹏城)/.test(normalizedQuery)) {
        if (/鹏城/.test(normalizedTitle)) {
            adjustment += 0.16;
        }
        if (/(广州国家实验室|广州实验室)/.test(normalizedTitle)) {
            adjustment -= 0.18;
        }
    }
    if (/临床医学博士/.test(normalizedQuery)) {
        if (/临床医学博士/.test(normalizedTitle)) {
            adjustment += 0.16;
        }
        if (/硕士学位/.test(normalizedTitle)) {
            adjustment -= 0.24;
        }
        else if (/博士/.test(normalizedTitle)) {
            adjustment += 0.08;
        }
    }
    if (/同等学力/.test(normalizedQuery) && /同等学力/.test(normalizedTitle)) {
        adjustment += 0.05;
    }
    return Math.max(-0.28, Math.min(0.22, adjustment));
}
function rerankDocuments(params) {
    var _this = this;
    return (function () { return __awaiter(_this, void 0, void 0, function () {
        var query, queryVector, documents, extractor, dimensions, preset, overrides, results, adaptiveRerankPlan, disambiguationFloor, rerankDocCount, adaptiveChunkPlan, maxChunksPerDoc, blendAlpha, rerankDocs, batchChunks, _loop_1, index, batchTexts, batchOutputs, pureData, rawDocumentScores, documentBestSentence, chunkIndex, chunkVec, score, dimensionIndex, docIdx, coarseNorm, index, normalizedSnippetScore, blendedScore, titleAdjustment, finalScore, documentsWithRerank, topConfidence;
        var _a, _b, _c, _d, _e, _f, _g;
        return __generator(this, function (_h) {
            switch (_h.label) {
                case 0:
                    query = params.query, queryVector = params.queryVector, documents = params.documents, extractor = params.extractor, dimensions = params.dimensions, preset = params.preset, overrides = params.overrides;
                    results = documents.map(function (doc) {
                        var _a, _b, _c, _d;
                        var defaultPoint = "暂无要点";
                        if (doc.best_kpid && Array.isArray(doc.kps)) {
                            var hitKp = doc.kps.find(function (kp) { return kp.kpid === doc.best_kpid; });
                            if (hitKp === null || hitKp === void 0 ? void 0 : hitKp.kp_text) {
                                defaultPoint = hitKp.kp_text;
                            }
                        }
                        return __assign(__assign({}, doc), { coarseScore: (_b = (_a = doc.coarseScore) !== null && _a !== void 0 ? _a : doc.score) !== null && _b !== void 0 ? _b : 0, displayScore: (_d = (_c = doc.displayScore) !== null && _c !== void 0 ? _c : doc.score) !== null && _d !== void 0 ? _d : 0, rerankScore: 0, snippetScore: 0, confidenceScore: 0, bestPoint: defaultPoint, bestSentence: "" });
                    });
                    adaptiveRerankPlan = (0, rerank_helpers_1.getAdaptiveRerankPlan)(results);
                    disambiguationFloor = inferQueryDisambiguationFloor(query);
                    rerankDocCount = Math.min(results.length, Math.max(0, disambiguationFloor.rerankDocCount, (_a = overrides === null || overrides === void 0 ? void 0 : overrides.rerankDocCount) !== null && _a !== void 0 ? _a : adaptiveRerankPlan.rerankDocCount));
                    adaptiveChunkPlan = (0, rerank_helpers_1.getAdaptiveChunkPlan)(query, rerankDocCount);
                    maxChunksPerDoc = (_b = overrides === null || overrides === void 0 ? void 0 : overrides.maxChunksPerDoc) !== null && _b !== void 0 ? _b : adaptiveChunkPlan.maxChunksPerDoc;
                    blendAlpha = (_c = overrides === null || overrides === void 0 ? void 0 : overrides.blendAlpha) !== null && _c !== void 0 ? _c : preset.display.rerankBlendAlpha;
                    rerankDocs = results.slice(0, rerankDocCount);
                    batchChunks = [];
                    _loop_1 = function (index) {
                        var textChunks = (0, rerank_helpers_1.splitIntoSemanticChunks)(rerankDocs[index].ot_text || "", 150, maxChunksPerDoc);
                        textChunks.forEach(function (chunk) {
                            var normalizedChunk = (chunk || "").trim();
                            if (normalizedChunk) {
                                batchChunks.push({
                                    text: normalizedChunk,
                                    docIdx: index,
                                });
                            }
                        });
                    };
                    for (index = 0; index < rerankDocs.length; index += 1) {
                        _loop_1(index);
                    }
                    if (!(batchChunks.length > 0)) return [3 /*break*/, 2];
                    batchTexts = batchChunks.map(function (item) { return item.text; });
                    return [4 /*yield*/, extractor(batchTexts, {
                            pooling: "mean",
                            normalize: true,
                            truncation: true,
                            max_length: 512,
                        })];
                case 1:
                    batchOutputs = _h.sent();
                    pureData = batchOutputs.data.subarray(0, batchChunks.length * dimensions);
                    rawDocumentScores = new Float32Array(rerankDocs.length).fill(-1);
                    documentBestSentence = new Array(rerankDocs.length).fill("");
                    for (chunkIndex = 0; chunkIndex < batchChunks.length; chunkIndex += 1) {
                        chunkVec = pureData.subarray(chunkIndex * dimensions, (chunkIndex + 1) * dimensions);
                        score = 0;
                        for (dimensionIndex = 0; dimensionIndex < dimensions; dimensionIndex += 1) {
                            score += queryVector[dimensionIndex] * chunkVec[dimensionIndex];
                        }
                        docIdx = batchChunks[chunkIndex].docIdx;
                        if (score > rawDocumentScores[docIdx]) {
                            rawDocumentScores[docIdx] = score;
                            documentBestSentence[docIdx] = batchChunks[chunkIndex].text;
                        }
                    }
                    coarseNorm = (0, rerank_helpers_1.normalizeMinMax)(rerankDocs.map(function (doc) { var _a; return (_a = doc.coarseScore) !== null && _a !== void 0 ? _a : 0; }));
                    for (index = 0; index < rerankDocs.length; index += 1) {
                        normalizedSnippetScore = (0, rerank_helpers_1.normalizeSnippetScore)(rawDocumentScores[index]);
                        blendedScore = blendAlpha * coarseNorm[index] +
                            (1 - blendAlpha) * normalizedSnippetScore;
                        titleAdjustment = computeTitleAdjustment(query, rerankDocs[index].ot_title);
                        finalScore = clamp01(blendedScore + titleAdjustment);
                        rerankDocs[index].snippetScore = normalizedSnippetScore;
                        rerankDocs[index].confidenceScore = finalScore;
                        rerankDocs[index].rerankScore = finalScore;
                        rerankDocs[index].displayScore = finalScore;
                        if (normalizedSnippetScore > preset.display.bestSentenceThreshold &&
                            documentBestSentence[index]) {
                            rerankDocs[index].bestSentence = documentBestSentence[index];
                        }
                    }
                    rerankDocs.sort(function (a, b) {
                        var _a, _b, _c, _d;
                        var scoreDiff = ((_a = b.rerankScore) !== null && _a !== void 0 ? _a : 0) - ((_b = a.rerankScore) !== null && _b !== void 0 ? _b : 0);
                        if (Math.abs(scoreDiff) <= NEAR_TIE_COARSE_WINDOW) {
                            return ((_c = b.coarseScore) !== null && _c !== void 0 ? _c : 0) - ((_d = a.coarseScore) !== null && _d !== void 0 ? _d : 0);
                        }
                        return scoreDiff;
                    });
                    return [3 /*break*/, 3];
                case 2:
                    rerankDocs.forEach(function (doc) {
                        doc.displayScore = 0;
                        doc.rerankScore = 0;
                        doc.snippetScore = 0;
                        doc.confidenceScore = 0;
                    });
                    _h.label = 3;
                case 3:
                    documentsWithRerank = rerankDocs.concat(results.slice(rerankDocCount));
                    topConfidence = (_g = (_e = (_d = documentsWithRerank[0]) === null || _d === void 0 ? void 0 : _d.confidenceScore) !== null && _e !== void 0 ? _e : (_f = documentsWithRerank[0]) === null || _f === void 0 ? void 0 : _f.rerankScore) !== null && _g !== void 0 ? _g : null;
                    return [2 /*return*/, {
                            documents: documentsWithRerank,
                            stats: {
                                rerankedDocCount: rerankDocCount,
                                chunksScored: batchChunks.length,
                                windowReason: (overrides === null || overrides === void 0 ? void 0 : overrides.overrideReason) ||
                                    [adaptiveRerankPlan.reason, disambiguationFloor.reason]
                                        .filter(Boolean)
                                        .join("+"),
                                maxChunksPerDoc: maxChunksPerDoc,
                                chunkPlanReason: adaptiveChunkPlan.reason,
                                topConfidence: topConfidence,
                                blendAlpha: blendAlpha,
                            },
                        }];
            }
        });
    }); })();
}
function buildDirectAnswerRescuePlan(params) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m;
    var documentCount = params.documentCount, initialTopConfidence = params.initialTopConfidence, rejectThreshold = params.rejectThreshold, initialStats = params.initialStats, querySignals = params.querySignals, retrievalSignals = params.retrievalSignals;
    if (initialTopConfidence >= rejectThreshold) {
        return null;
    }
    var thresholdGap = rejectThreshold - initialTopConfidence;
    var strongAnchors = (querySignals === null || querySignals === void 0 ? void 0 : querySignals.hasExplicitTopicOrIntent) ||
        (querySignals === null || querySignals === void 0 ? void 0 : querySignals.hasStrongDetailAnchor) ||
        (querySignals === null || querySignals === void 0 ? void 0 : querySignals.hasExplicitYear);
    var stableRetrieval = ((_a = retrievalSignals === null || retrievalSignals === void 0 ? void 0 : retrievalSignals.top1Top2Gap) !== null && _a !== void 0 ? _a : 0) >= 0.04 ||
        (((_b = retrievalSignals === null || retrievalSignals === void 0 ? void 0 : retrievalSignals.distinctTopicCount) !== null && _b !== void 0 ? _b : 99) <= 2 &&
            ((_c = retrievalSignals === null || retrievalSignals === void 0 ? void 0 : retrievalSignals.dominantTopicRatio) !== null && _c !== void 0 ? _c : 0) >= 0.45);
    if (!strongAnchors && !stableRetrieval) {
        return null;
    }
    if (thresholdGap > DIRECT_ANSWER_RESCUE_MARGIN) {
        return null;
    }
    var queryLength = (_d = querySignals === null || querySignals === void 0 ? void 0 : querySignals.queryLength) !== null && _d !== void 0 ? _d : 999;
    var isShortQuery = queryLength <= 14;
    var isMediumQuery = queryLength <= 28;
    var veryStableRetrieval = ((_e = retrievalSignals === null || retrievalSignals === void 0 ? void 0 : retrievalSignals.top1Top2Gap) !== null && _e !== void 0 ? _e : 0) >= 1.0 &&
        ((_f = retrievalSignals === null || retrievalSignals === void 0 ? void 0 : retrievalSignals.dominantTopicRatio) !== null && _f !== void 0 ? _f : 0) >= 0.75;
    var docGrowth = (querySignals === null || querySignals === void 0 ? void 0 : querySignals.hasStrongDetailAnchor)
        ? 6
        : veryStableRetrieval
            ? 4
            : 3;
    var nextRerankDocCount = documentCount > ((_g = initialStats.rerankedDocCount) !== null && _g !== void 0 ? _g : 0)
        ? Math.min(documentCount, Math.max(Math.min(MAX_RESCUE_RERANK_DOC_COUNT, ((_h = initialStats.rerankedDocCount) !== null && _h !== void 0 ? _h : 0) +
            docGrowth), 3))
        : undefined;
    var chunkGrowth = (querySignals === null || querySignals === void 0 ? void 0 : querySignals.hasStrongDetailAnchor) || queryLength > 24
        ? 5
        : veryStableRetrieval
            ? 6
            : 4;
    var nextMaxChunksPerDoc = Math.min(MAX_RESCUE_CHUNK_LIMIT, Math.max(((_j = initialStats.maxChunksPerDoc) !== null && _j !== void 0 ? _j : 0) + chunkGrowth, MIN_RESCUE_CHUNK_LIMIT));
    var rescueBlendAlpha = (_k = initialStats.blendAlpha) !== null && _k !== void 0 ? _k : 0.15;
    if (thresholdGap <= 0.03 && stableRetrieval) {
        rescueBlendAlpha = Math.max(rescueBlendAlpha, 0.24);
    }
    if (isMediumQuery && strongAnchors && stableRetrieval) {
        rescueBlendAlpha = Math.max(rescueBlendAlpha, 0.28);
    }
    if (isShortQuery && strongAnchors && veryStableRetrieval) {
        rescueBlendAlpha = Math.max(rescueBlendAlpha, 0.36);
    }
    if (nextRerankDocCount === undefined &&
        nextMaxChunksPerDoc <= ((_l = initialStats.maxChunksPerDoc) !== null && _l !== void 0 ? _l : 0) &&
        rescueBlendAlpha <= ((_m = initialStats.blendAlpha) !== null && _m !== void 0 ? _m : 0)) {
        return null;
    }
    var reasonParts = [];
    if (thresholdGap <= 0.06)
        reasonParts.push("near_threshold");
    if (strongAnchors)
        reasonParts.push("strong_anchor");
    if (stableRetrieval)
        reasonParts.push("stable_retrieval");
    return {
        rerankDocCount: nextRerankDocCount,
        maxChunksPerDoc: nextMaxChunksPerDoc,
        blendAlpha: rescueBlendAlpha,
        reason: reasonParts.join("+") || "direct_answer_rescue",
    };
}
function runDirectAnswerDisplayStage(params) {
    return __awaiter(this, void 0, void 0, function () {
        var query, queryVector, documents, extractor, dimensions, preset, querySignals, retrievalSignals, initialResult, initialTopConfidence, selectedResult, rescueTrace, rescuePlan, rescueResult, finalTopConfidence, displayRejected;
        var _a, _b, _c, _d, _e;
        return __generator(this, function (_f) {
            switch (_f.label) {
                case 0:
                    query = params.query, queryVector = params.queryVector, documents = params.documents, extractor = params.extractor, dimensions = params.dimensions, preset = params.preset, querySignals = params.querySignals, retrievalSignals = params.retrievalSignals;
                    return [4 /*yield*/, rerankDocuments({
                            query: query,
                            queryVector: queryVector,
                            documents: documents,
                            extractor: extractor,
                            dimensions: dimensions,
                            preset: preset,
                        })];
                case 1:
                    initialResult = _f.sent();
                    initialTopConfidence = (_a = initialResult.stats.topConfidence) !== null && _a !== void 0 ? _a : null;
                    selectedResult = initialResult;
                    rescueTrace = {
                        attempted: false,
                        accepted: false,
                        succeeded: false,
                        initialTopConfidence: initialTopConfidence,
                        initialRerankDocCount: initialResult.stats.rerankedDocCount,
                        initialMaxChunksPerDoc: initialResult.stats.maxChunksPerDoc,
                        initialBlendAlpha: initialResult.stats.blendAlpha,
                    };
                    rescuePlan = initialTopConfidence !== null
                        ? buildDirectAnswerRescuePlan({
                            documentCount: documents.length,
                            initialTopConfidence: initialTopConfidence,
                            rejectThreshold: preset.display.rejectThreshold,
                            initialStats: initialResult.stats,
                            querySignals: querySignals,
                            retrievalSignals: retrievalSignals,
                        })
                        : null;
                    if (!rescuePlan) return [3 /*break*/, 3];
                    rescueTrace.attempted = true;
                    rescueTrace.reason = rescuePlan.reason;
                    rescueTrace.rescueRerankDocCount = rescuePlan.rerankDocCount;
                    rescueTrace.rescueMaxChunksPerDoc = rescuePlan.maxChunksPerDoc;
                    rescueTrace.rescueBlendAlpha = rescuePlan.blendAlpha;
                    return [4 /*yield*/, rerankDocuments({
                            query: query,
                            queryVector: queryVector,
                            documents: documents,
                            extractor: extractor,
                            dimensions: dimensions,
                            preset: preset,
                            overrides: {
                                rerankDocCount: rescuePlan.rerankDocCount,
                                maxChunksPerDoc: rescuePlan.maxChunksPerDoc,
                                blendAlpha: rescuePlan.blendAlpha,
                                overrideReason: "rescue:".concat(rescuePlan.reason),
                            },
                        })];
                case 2:
                    rescueResult = _f.sent();
                    rescueTrace.rescueTopConfidence = (_b = rescueResult.stats.topConfidence) !== null && _b !== void 0 ? _b : null;
                    if (((_c = rescueResult.stats.topConfidence) !== null && _c !== void 0 ? _c : -999) >=
                        preset.display.rejectThreshold ||
                        ((_d = rescueResult.stats.topConfidence) !== null && _d !== void 0 ? _d : -999) >=
                            (initialTopConfidence !== null && initialTopConfidence !== void 0 ? initialTopConfidence : -999) + RESCUE_ACCEPT_DELTA) {
                        selectedResult = rescueResult;
                        rescueTrace.accepted = true;
                    }
                    _f.label = 3;
                case 3:
                    finalTopConfidence = (_e = selectedResult.stats.topConfidence) !== null && _e !== void 0 ? _e : null;
                    displayRejected = selectedResult.documents.length > 0 &&
                        (finalTopConfidence !== null && finalTopConfidence !== void 0 ? finalTopConfidence : -999) < preset.display.rejectThreshold;
                    rescueTrace.succeeded =
                        rescueTrace.attempted &&
                            !displayRejected &&
                            (initialTopConfidence !== null && initialTopConfidence !== void 0 ? initialTopConfidence : -999) < preset.display.rejectThreshold;
                    return [2 /*return*/, {
                            documents: selectedResult.documents,
                            stats: selectedResult.stats,
                            initialTopConfidence: initialTopConfidence,
                            finalTopConfidence: finalTopConfidence,
                            displayRejected: displayRejected,
                            directAnswerRescue: rescueTrace,
                        }];
            }
        });
    });
}

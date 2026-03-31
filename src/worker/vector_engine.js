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
exports.DIRECT_ANSWER_EVIDENCE_TERMS = exports.QUERY_SCOPE_SPECIFICITY_TERMS = exports.RRF_K = exports.DEFAULT_WEIGHTS = void 0;
exports.resolveMetadataTopicIds = resolveMetadataTopicIds;
exports.buildBM25Stats = buildBM25Stats;
exports.dotProduct = dotProduct;
exports.getQuerySparse = getQuerySparse;
exports.parseQueryIntent = parseQueryIntent;
exports.extractRetrievalSignals = extractRetrievalSignals;
exports.classifyResponseMode = classifyResponseMode;
exports.searchAndRank = searchAndRank;
var search_topic_config_1 = require("./search_topic_config");
var aggregated_doc_scores_1 = require("./aggregated_doc_scores");
exports.DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33,
};
exports.RRF_K = 60;
var BM25_K1 = 1.2;
var BM25_B = 0.4;
var EVENT_TYPE_MISMATCH_PENALTY = 0.95;
var LATEST_YEAR_BOOST_BASE = 0.82;
var LATEST_POLICY_TIMESTAMP_BOOST_BASE = 0.97;
var CURRENT_PROCESS_TIMESTAMP_BOOST_BASE = 0.984;
var DEFAULT_KP_ROLE_DOC_WEIGHT = 0.35;
var DEFAULT_KP_ROLE_CANDIDATE_LIMIT = 5;
var CURRENT_PROCESS_EVENT_TYPES = [
    "招生章程",
    "报名通知",
    "考试安排",
    "材料提交",
    "资格要求",
];
exports.QUERY_SCOPE_SPECIFICITY_TERMS = [
    "港澳台",
    "海外",
    "直博",
    "单独",
    "士兵",
    "改报",
    "报考点",
    "联合培养",
    "专项",
    "调剂",
    "夏令营",
    "推免",
    "保研",
];
var QUERY_SCOPE_SPECIFICITY_TERM_SET = new Set(exports.QUERY_SCOPE_SPECIFICITY_TERMS);
exports.DIRECT_ANSWER_EVIDENCE_TERMS = [
    "夏令营",
    "调剂",
    "港澳台",
    "报名",
    "申请",
    "招生简章",
    "简章",
    "招生章程",
    "章程",
    "外语类",
    "保送生",
    "综合评价",
    "免修",
    "选课",
    "补退选",
    "退选",
    "转专业",
    "优惠",
    "优秀营员",
    "缺额",
    "缺额专业",
];
var BROAD_LATEST_SCOPE_CUE_PATTERN = /完整流程|完整|通用|一般|总流程|怎么报名|如何报名|条件.*报名|条件.*流程|条件.*操作/;
var INTENT_CONFLICTS = Object.fromEntries(search_topic_config_1.INTENT_VECTOR_TABLE.map(function (item) { return [item.intent_id, item.negative_intents]; }));
var INTENT_RULE_MAP = new Map(search_topic_config_1.INTENT_VECTOR_TABLE.map(function (item) { return [item.intent_id, item]; }));
var TOPIC_RULE_MAP = new Map(search_topic_config_1.TOPIC_CONFIGS.map(function (item) { return [item.topic_id, item]; }));
function isOutOfScopeTopic(topicId) {
    var _a;
    return ((_a = TOPIC_RULE_MAP.get(topicId)) === null || _a === void 0 ? void 0 : _a.scope) === "out_of_scope";
}
function hasOnlyOutOfScopeTopics(topicIds) {
    return (topicIds.length > 0 &&
        topicIds.every(function (topicId) { return isOutOfScopeTopic(topicId); }));
}
function deriveTopicIdsFromIntents(intentIds) {
    return dedupe(intentIds
        .map(function (intentId) { var _a; return (_a = INTENT_RULE_MAP.get(intentId)) === null || _a === void 0 ? void 0 : _a.topic_id; })
        .filter(function (topicId) { return Boolean(topicId); }));
}
function resolveMetadataTopicIds(meta) {
    if (meta.topic_ids && meta.topic_ids.length > 0) {
        return meta.topic_ids;
    }
    return deriveTopicIdsFromIntents(meta.subtopic_ids || meta.intent_ids || []);
}
function matchTopicIds(query) {
    return dedupe(search_topic_config_1.TOPIC_CONFIGS.filter(function (topic) {
        return topic.aliases.some(function (alias) { return query.includes(alias); });
    }).map(function (topic) { return topic.topic_id; }));
}
function dedupe(items) {
    return Array.from(new Set(items));
}
function hasGenericNextStepCue(query) {
    return /怎么办|怎么做|怎么处理|怎么操作|如何办理|如何操作|接下来|下一步|要做什么|需要做什么|还要做什么|还需要做什么|还需要再操作什么|后面该怎么处理|后面怎么办|后续怎么办|应该怎么办|应该怎么|要办哪些事|下一步是什么|怎么弄|怎么搞|怎么整|处理什么|该做什么|准备什么|干什么/.test(query);
}
function hasClarificationStateCue(query) {
    return /考上|录取|录取结果|拟录取|收到通知书|拿到通知书|审核通过|审核没通过|审核未通过|审核不通过|通过初审|初审通过|学校通知我通过|通知我通过|通过了|过审|没过审|未过审|获批|评上|提交完材料|提交完申请|提交材料后|补交完材料|补交完|成了新生|已经是新生|新生以后|录取后|拟录取后|考上后|收到通知书后|拿到通知书后|审核通过后|通过初审后|获批后/.test(query);
}
function hasLatestPolicyStateCue(query) {
    return /考上|录取|录取结果|拟录取|收到通知书|拿到通知书|成了新生|已经是新生|新生以后|录取后|拟录取后|考上后|收到通知书后|拿到通知书后/.test(query);
}
function hasPostOutcomeConditionCue(query) {
    return /最终有效|最终有效性|最终录取|录取.*有效|拟录取.*有效|审核.*为准|审批.*为准/.test(query);
}
function hasStrongDetailAnchorCue(query) {
    return /录取通知书|通知书|报到|宿舍|党团关系|奖助金|档案|调档|政审|网上确认|现场确认|答辩|报名|考试|考试内容|考试科目|科目|题型|缴费|申请书|复试|面试|邮寄|地址|银行卡|学费/.test(query);
}
function hasEntryLikeAnchorCue(query) {
    return /新生|入学|录取|拟录取|审核|初审|资格审核|申请|材料|网上确认|现场确认/.test(query);
}
function hasLatestPolicyFallbackCue(querySignals) {
    return querySignals.hasGenericNextStep && querySignals.hasLatestPolicyState;
}
function buildQuerySignals(params) {
    var query = params.query, years = params.years, topicIds = params.topicIds, intentIds = params.intentIds, hasHistoricalHint = params.hasHistoricalHint;
    return {
        hasExplicitTopicOrIntent: topicIds.length > 0 || intentIds.length > 0,
        hasExplicitYear: years.length > 0,
        hasHistoricalHint: hasHistoricalHint,
        hasStrongDetailAnchor: hasStrongDetailAnchorCue(query),
        hasEntryLikeAnchor: hasEntryLikeAnchorCue(query),
        hasResultState: hasClarificationStateCue(query),
        hasLatestPolicyState: hasLatestPolicyStateCue(query),
        hasGenericNextStep: hasGenericNextStepCue(query),
        queryLength: query.length,
    };
}
function withQueryTokenCount(signals, querySparse) {
    return __assign(__assign({}, signals), { tokenCount: querySparse ? Object.keys(querySparse).length : 0 });
}
function extractQueryMonths(query) {
    var months = new Set();
    var matches = query.matchAll(/(?:^|[^\d])(1[0-2]|0?[1-9])月(?:份)?/g);
    for (var _i = 0, matches_1 = matches; _i < matches_1.length; _i++) {
        var match = matches_1[_i];
        var value = Number.parseInt(match[1] || "", 10);
        if (Number.isFinite(value) && value >= 1 && value <= 12) {
            months.add(value);
        }
    }
    return Array.from(months);
}
function buildBM25Stats(metadata) {
    var N = metadata.length;
    var dfMap = new Map();
    var docLengths = new Int32Array(N);
    var totalLength = 0;
    for (var i = 0; i < N; i++) {
        var sparse = metadata[i].sparse;
        if (!sparse || sparse.length === 0) {
            docLengths[i] = 0;
            continue;
        }
        var dl = 0;
        for (var j = 0; j < sparse.length; j += 2) {
            var wordId = sparse[j];
            var tf = sparse[j + 1];
            dl += tf;
            dfMap.set(wordId, (dfMap.get(wordId) || 0) + 1);
        }
        docLengths[i] = dl;
        totalLength += dl;
    }
    var avgdl = totalLength / (N || 1);
    var idfMap = new Map();
    for (var _i = 0, _a = dfMap.entries(); _i < _a.length; _i++) {
        var _b = _a[_i], wordId = _b[0], df = _b[1];
        var idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        idfMap.set(wordId, Math.max(idf, 0.01));
    }
    return { idfMap: idfMap, docLengths: docLengths, avgdl: avgdl };
}
function dotProduct(vecA, matrix, matrixIndex, dimensions) {
    var offset = matrixIndex * dimensions;
    var unrolledLimit = dimensions - (dimensions % 4);
    var s0 = 0;
    var s1 = 0;
    var s2 = 0;
    var s3 = 0;
    for (var i = 0; i < unrolledLimit; i += 4) {
        s0 += vecA[i] * matrix[offset + i];
        s1 += vecA[i + 1] * matrix[offset + i + 1];
        s2 += vecA[i + 2] * matrix[offset + i + 2];
        s3 += vecA[i + 3] * matrix[offset + i + 3];
    }
    var sum = s0 + s1 + s2 + s3;
    for (var i = unrolledLimit; i < dimensions; i++) {
        sum += vecA[i] * matrix[offset + i];
    }
    return sum;
}
function getQuerySparse(words, vocabMap) {
    var sparse = {};
    var isMap = vocabMap instanceof Map;
    words.forEach(function (word) {
        var index = isMap
            ? vocabMap.get(word)
            : vocabMap[word];
        if (index !== undefined) {
            sparse[index] = (sparse[index] || 0) + 1;
        }
    });
    return sparse;
}
function parseQueryIntent(query) {
    var years = dedupe((query.match(/20\d{2}/g) || []).map(function (year) { return Number(year); }));
    var months = extractQueryMonths(query);
    var matchedRules = search_topic_config_1.INTENT_VECTOR_TABLE.filter(function (rule) {
        return rule.aliases.some(function (alias) { return query.includes(alias); });
    });
    var intentIds = dedupe(matchedRules.map(function (rule) { return rule.intent_id; }));
    var matchedTopicIds = matchTopicIds(query);
    var topicIds = dedupe(__spreadArray(__spreadArray([], deriveTopicIdsFromIntents(intentIds), true), matchedTopicIds, true));
    var degreeLevels = dedupe(__spreadArray(__spreadArray([], search_topic_config_1.DEGREE_LEVEL_TABLE.filter(function (level) { return query.includes(level); }), true), (query.includes("直博") ? ["博士"] : []), true));
    var eventTypes = dedupe(search_topic_config_1.EVENT_TYPE_TABLE.filter(function (eventType) { return query.includes(eventType); }));
    var normalizedTerms = dedupe(matchedRules.map(function (rule) { return rule.intent_name; }));
    var hasHistoricalHint = search_topic_config_1.HISTORICAL_QUERY_HINTS.some(function (hint) {
        return query.includes(hint);
    });
    var hasLatestHint = search_topic_config_1.LATEST_QUERY_HINTS.some(function (hint) {
        return query.includes(hint);
    });
    var signals = buildQuerySignals({
        query: query,
        years: years,
        topicIds: topicIds,
        intentIds: intentIds,
        hasHistoricalHint: hasHistoricalHint,
    });
    var preferLatestStrong = years.length === 0 &&
        !hasHistoricalHint &&
        (hasLatestHint || hasLatestPolicyFallbackCue(signals));
    var preferLatest = years.length === 0 &&
        !hasHistoricalHint &&
        (topicIds.some(function (topicId) { var _a; return (_a = TOPIC_RULE_MAP.get(topicId)) === null || _a === void 0 ? void 0 : _a.prefer_latest; }) ||
            hasLatestHint ||
            preferLatestStrong);
    return {
        rawQuery: query,
        years: years,
        months: months,
        topicIds: topicIds,
        subtopicIds: intentIds,
        intentIds: intentIds,
        degreeLevels: degreeLevels,
        eventTypes: eventTypes,
        normalizedTerms: normalizedTerms,
        confidence: intentIds.length > 0 ? 1 : 0,
        preferLatest: preferLatest,
        preferLatestStrong: preferLatestStrong,
        signals: signals,
    };
}
function hasIntentConflict(queryIntentIds, docIntentIds) {
    if (!docIntentIds || docIntentIds.length === 0)
        return false;
    return queryIntentIds.some(function (queryIntentId) {
        return (INTENT_CONFLICTS[queryIntentId] || []).some(function (conflictId) {
            return docIntentIds.includes(conflictId);
        });
    });
}
function hasIntentMatch(queryIntentIds, docIntentIds) {
    if (!docIntentIds || docIntentIds.length === 0)
        return false;
    return queryIntentIds.some(function (queryIntentId) {
        return docIntentIds.includes(queryIntentId);
    });
}
function hasAnyOverlap(a, b) {
    if (!b || b.length === 0)
        return false;
    return a.some(function (item) { return b.includes(item); });
}
function getCoverageComparableTopicIds(doc) {
    if (doc.primary_topic_ids && doc.primary_topic_ids.length > 0) {
        return doc.primary_topic_ids;
    }
    if (doc.secondary_topic_ids && doc.secondary_topic_ids.length > 0) {
        return doc.secondary_topic_ids;
    }
    return doc.topic_ids || [];
}
function getRelatedIntentTypes(intentIds) {
    return dedupe(intentIds.flatMap(function (intentId) { var _a; return ((_a = INTENT_RULE_MAP.get(intentId)) === null || _a === void 0 ? void 0 : _a.related_intents) || []; }));
}
function getMatchedSpecificityTf(querySpecificityTerms, scopeSpecificityStats) {
    if (!scopeSpecificityStats || querySpecificityTerms.length === 0) {
        return 0;
    }
    return querySpecificityTerms.reduce(function (sum, term) { return sum + (scopeSpecificityStats.termTf[term] || 0); }, 0);
}
function extractQuerySpecificityTerms(queryWords) {
    return dedupe(queryWords.filter(function (word) { return QUERY_SCOPE_SPECIFICITY_TERM_SET.has(word); }));
}
function buildEvidenceCoverageRequirement(rawQuery) {
    if (/夏令营/.test(rawQuery) &&
        /(录取优惠|优惠|优秀营员)/.test(rawQuery)) {
        return {
            label: "summer_camp_benefit",
            requiredGroups: [["夏令营"], ["优惠", "优秀营员"]],
            requireIntentAlignment: true,
        };
    }
    if (/夏令营/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)) {
        return {
            label: "summer_camp_apply",
            requiredGroups: [["夏令营"], ["报名", "申请"]],
            requireIntentAlignment: true,
        };
    }
    if (/港澳台/.test(rawQuery) && /调剂/.test(rawQuery)) {
        return {
            label: "hongkong_macau_taiwan_adjustment",
            requiredGroups: [["港澳台"], ["调剂"]],
            requireIntentAlignment: true,
        };
    }
    if (/调剂/.test(rawQuery) && /缺额专业/.test(rawQuery)) {
        return {
            label: "adjustment_vacancy",
            requiredGroups: [["调剂"], ["缺额", "缺额专业"]],
            requireIntentAlignment: true,
        };
    }
    if (/调剂/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)) {
        return {
            label: "adjustment_apply",
            requiredGroups: [["调剂"], ["报名", "申请"]],
            requireIntentAlignment: true,
        };
    }
    if (/外语类保送生/.test(rawQuery) &&
        /(招生简章|简章|招生章程|章程)/.test(rawQuery)) {
        return {
            label: "foreign_language_recommend_brochure",
            requiredGroups: [
                ["外语类", "保送生"],
                ["招生简章", "简章", "招生章程", "章程"],
            ],
        };
    }
    if (/综合评价/.test(rawQuery) &&
        /(招生简章|简章|招生章程|章程)/.test(rawQuery)) {
        return {
            label: "comprehensive_evaluation_brochure",
            requiredGroups: [["综合评价"], ["招生简章", "简章", "招生章程", "章程"]],
        };
    }
    if (/选课/.test(rawQuery) &&
        /补退选/.test(rawQuery) &&
        /(时间|什么时候|何时|截止|流程|步骤)/.test(rawQuery)) {
        return {
            label: "course_add_drop",
            requiredGroups: [["选课"], ["补退选", "退选"]],
        };
    }
    if (/转专业/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)) {
        return {
            label: "major_transfer",
            requiredGroups: [["转专业"], ["报名", "申请"]],
        };
    }
    if (/免修/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)) {
        return {
            label: "course_exemption",
            requiredGroups: [["免修"], ["报名", "申请"]],
        };
    }
    return undefined;
}
function topDocumentSatisfiesEvidenceRequirement(sortedRanking, docEvidenceStatsMap, requirement) {
    var topDoc = sortedRanking[0];
    if (!topDoc) {
        return false;
    }
    var stats = docEvidenceStatsMap.get(topDoc.otid);
    if (!stats) {
        return false;
    }
    return requirement.requiredGroups.every(function (group) {
        return group.some(function (term) { return (stats.termTf[term] || 0) > 0; });
    });
}
function shouldRejectForMissingInDomainEvidence(params) {
    var _a, _b, _c, _d;
    var requirement = buildEvidenceCoverageRequirement(params.rawQuery);
    if (!requirement) {
        return { shouldReject: false };
    }
    var topDoc = params.sortedRanking[0];
    if (requirement.requireIntentAlignment &&
        topDoc &&
        (((_a = params.queryIntent) === null || _a === void 0 ? void 0 : _a.intentIds.length) || 0) > 0) {
        var topDocIntentIds = ((_b = params.otidMap[topDoc.otid]) === null || _b === void 0 ? void 0 : _b.intent_ids) ||
            ((_c = params.otidMap[topDoc.otid]) === null || _c === void 0 ? void 0 : _c.subtopic_ids) ||
            [];
        if (!hasAnyOverlap(((_d = params.queryIntent) === null || _d === void 0 ? void 0 : _d.intentIds) || [], topDocIntentIds)) {
            return {
                shouldReject: true,
                label: "".concat(requirement.label, "_intent_mismatch"),
            };
        }
    }
    if (topDocumentSatisfiesEvidenceRequirement(params.sortedRanking, params.docEvidenceStatsMap, requirement)) {
        return { shouldReject: false };
    }
    return {
        shouldReject: true,
        label: requirement.label,
    };
}
function createQueryIntentContext(queryIntent, queryWords) {
    if (queryWords === void 0) { queryWords = []; }
    var years = (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.years) || [];
    var intentIds = (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.intentIds) || [];
    var rawQuery = (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.rawQuery) || "";
    var querySpecificityTerms = extractQuerySpecificityTerms(queryWords);
    var discourageUnexpectedSpecificity = querySpecificityTerms.length === 0 &&
        Boolean(queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.preferLatestStrong) &&
        BROAD_LATEST_SCOPE_CUE_PATTERN.test(rawQuery);
    return {
        rawQuery: rawQuery,
        years: years,
        months: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.months) || [],
        hasExplicitYear: years.length > 0,
        hasExplicitMonth: ((queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.months) || []).length > 0,
        hasHistoricalHint: Boolean(queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.signals.hasHistoricalHint),
        hasStrongDetailAnchor: Boolean(queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.signals.hasStrongDetailAnchor),
        topicIds: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.topicIds) || [],
        intentIds: intentIds,
        relatedIntentIds: getRelatedIntentTypes(intentIds),
        degreeLevels: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.degreeLevels) || [],
        eventTypes: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.eventTypes) || [],
        hasPostOutcomeCondition: hasPostOutcomeConditionCue(rawQuery),
        preferLatest: Boolean(queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.preferLatest),
        preferLatestStrong: Boolean(queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.preferLatestStrong),
        querySpecificityTerms: querySpecificityTerms,
        discourageUnexpectedSpecificity: discourageUnexpectedSpecificity,
    };
}
function getTimestampMonth(timestamp) {
    if (typeof timestamp !== "number" || !Number.isFinite(timestamp)) {
        return undefined;
    }
    var date = new Date(timestamp * 1000);
    if (Number.isNaN(date.getTime())) {
        return undefined;
    }
    return date.getUTCMonth() + 1;
}
function getTimestampYear(timestamp) {
    if (typeof timestamp !== "number" || !Number.isFinite(timestamp)) {
        return undefined;
    }
    var date = new Date(timestamp * 1000);
    if (Number.isNaN(date.getTime())) {
        return undefined;
    }
    return date.getUTCFullYear();
}
function getDocQuerySignals(otid, scores, intentContext, yearHitMap) {
    var docMonth = getTimestampMonth(scores.timestamp);
    var docPublishYear = getTimestampYear(scores.timestamp);
    var hasSuspiciousStructuredYear = scores.target_year !== undefined &&
        docPublishYear !== undefined &&
        Math.abs(scores.target_year - docPublishYear) >= 2;
    return {
        hasStructuredYearMatch: intentContext.hasExplicitYear &&
            scores.target_year !== undefined &&
            intentContext.years.includes(scores.target_year),
        hasLexicalYearMatch: yearHitMap.get(otid) === true,
        hasPublishYearMatch: intentContext.hasExplicitYear &&
            docPublishYear !== undefined &&
            intentContext.years.includes(docPublishYear),
        hasSuspiciousStructuredYear: hasSuspiciousStructuredYear,
        docPublishYear: docPublishYear,
        hasStructuredMonthMatch: intentContext.hasExplicitYear &&
            intentContext.hasExplicitMonth &&
            docMonth !== undefined &&
            intentContext.months.includes(docMonth),
        docMonth: docMonth,
    };
}
function shouldSkipForExplicitYear(scores, intentContext, signals) {
    if (!intentContext.hasExplicitYear) {
        return false;
    }
    if (scores.target_year !== undefined &&
        !signals.hasStructuredYearMatch &&
        !(signals.hasLexicalYearMatch && signals.hasSuspiciousStructuredYear)) {
        return true;
    }
    return (scores.target_year === undefined &&
        !signals.hasLexicalYearMatch &&
        !signals.hasPublishYearMatch);
}
function computeBaseScore(scores, weights, options) {
    var _a;
    var kpAggregationMode = (options === null || options === void 0 ? void 0 : options.kpAggregationMode) || "max";
    var kpTopN = Math.max(1, (options === null || options === void 0 ? void 0 : options.kpTopN) || 3);
    var kpTailWeight = (_a = options === null || options === void 0 ? void 0 : options.kpTailWeight) !== null && _a !== void 0 ? _a : 0.35;
    var topKpScores = scores.kp_scores && scores.kp_scores.length > 0
        ? scores.kp_scores.slice(0, kpTopN)
        : scores.max_kp > 0
            ? [scores.max_kp]
            : [];
    var aggregatedKpScore = kpAggregationMode === "max_plus_topn" && topKpScores.length > 1
        ? topKpScores[0] +
            topKpScores.slice(1).reduce(function (sum, item) { return sum + item; }, 0) *
                kpTailWeight
        : topKpScores[0] || 0;
    var weightedQ = scores.max_q * weights.Q;
    var weightedKP = aggregatedKpScore * weights.KP;
    var weightedOT = scores.ot_score * weights.OT;
    var maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
    var unionBonus = weightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;
    return maxComponent + unionBonus;
}
function hasKpRoleTag(candidate, tag) {
    var _a;
    return ((_a = candidate === null || candidate === void 0 ? void 0 : candidate.kp_role_tags) === null || _a === void 0 ? void 0 : _a.includes(tag)) === true;
}
function deriveQueryRoleSignals(rawQuery, queryScopeHint) {
    return {
        asksTime: /什么时候|何时|哪几天|几号|截止|到账|时间|公示期/.test(rawQuery) || queryScopeHint === "time_location",
        asksCondition: /条件|满足|资格/.test(rawQuery) ||
            queryScopeHint === "eligibility_condition",
        asksPostOutcomeCondition: hasPostOutcomeConditionCue(rawQuery),
        asksMaterials: /材料|扫描件|电子版|邮箱|mail/i.test(rawQuery),
        asksProcedure: /怎么办|怎么处理|不通过|补交|补充|流程|步骤/.test(rawQuery),
        asksExamContent: /考什么|考哪些|考试内容|考试科目|科目|题型|综合能力/.test(rawQuery),
        asksAnnouncementPeriod: /公示期|哪几天/.test(rawQuery),
        asksApplicationStage: /申请|报名|确认|提交/.test(rawQuery) &&
            !/通过后|答辩通过|审批后|获得学位/.test(rawQuery),
        mentionsThesis: /论文/.test(rawQuery),
        mentionsPrintedDocument: /准考证|打印|纸质/.test(rawQuery),
        mentionsCollectionOrArchive: /领取|证书|档案/.test(rawQuery),
        mentionsReviewOrReissue: /资格|评审|补发/.test(rawQuery),
    };
}
function computeKpRoleBonus(candidate, signals, rawQuery) {
    var bonus = 0;
    if (signals.asksTime) {
        if (hasKpRoleTag(candidate, "arrival")
            || hasKpRoleTag(candidate, "deadline")
            || hasKpRoleTag(candidate, "announcement_period")
            || hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.9;
        }
        if (hasKpRoleTag(candidate, "time_expression")) {
            bonus += 0.45;
        }
    }
    if (signals.asksCondition) {
        if (hasKpRoleTag(candidate, "condition")) {
            bonus += 1.1;
        }
        // Some eligibility constraints are encoded as cutoff deadlines.
        if (hasKpRoleTag(candidate, "deadline")) {
            bonus += 0.55;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus += signals.asksPostOutcomeCondition ? 1.0 : -0.7;
        }
    }
    if (signals.asksMaterials) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 0.8;
        }
        if (hasKpRoleTag(candidate, "materials")
            && hasKpRoleTag(candidate, "email")) {
            bonus += 0.9;
        }
        if (/申请|答辩/.test(rawQuery) && hasKpRoleTag(candidate, "application_stage")) {
            bonus += 0.9;
        }
        if (!signals.mentionsThesis
            && (hasKpRoleTag(candidate, "post_outcome")
                || hasKpRoleTag(candidate, "thesis"))) {
            bonus -= 1.2;
        }
    }
    if (signals.mentionsPrintedDocument) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 0.55;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.35;
        }
    }
    if (signals.asksApplicationStage) {
        if (hasKpRoleTag(candidate, "application_stage")) {
            bonus += 1.1;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 1.1;
        }
    }
    if (signals.mentionsThesis &&
        signals.asksApplicationStage &&
        !signals.asksCondition) {
        if (hasKpRoleTag(candidate, "condition")) {
            bonus -= 0.55;
        }
        if (hasKpRoleTag(candidate, "application_stage")) {
            bonus += 0.3;
        }
    }
    if (signals.asksProcedure) {
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 1.25;
        }
        if (hasKpRoleTag(candidate, "reminder")
            || hasKpRoleTag(candidate, "background")) {
            bonus -= 0.8;
        }
    }
    if (signals.asksExamContent) {
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 1.1;
        }
        if (hasKpRoleTag(candidate, "time_expression")) {
            bonus += 0.25;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 1.35;
        }
        if (hasKpRoleTag(candidate, "announcement_period")) {
            bonus -= 0.9;
        }
        if (hasKpRoleTag(candidate, "deadline")) {
            bonus -= 0.55;
        }
        if (hasKpRoleTag(candidate, "publish")) {
            bonus -= 0.45;
        }
    }
    if (signals.asksAnnouncementPeriod) {
        if (hasKpRoleTag(candidate, "announcement_period")) {
            bonus += 1.2;
        }
        if (hasKpRoleTag(candidate, "publish")
            && !hasKpRoleTag(candidate, "announcement_period")) {
            bonus -= 0.6;
        }
    }
    if (/到账/.test(rawQuery)) {
        if (hasKpRoleTag(candidate, "arrival")) {
            bonus += 1.0;
        }
        if (hasKpRoleTag(candidate, "distribution")) {
            bonus -= 0.5;
        }
    }
    if (signals.mentionsCollectionOrArchive) {
        if (hasKpRoleTag(candidate, "reminder")) {
            bonus += 0.7;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus += 0.35;
        }
        if (hasKpRoleTag(candidate, "materials")) {
            bonus -= 0.35;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.35;
        }
    }
    if (signals.mentionsReviewOrReissue) {
        if (hasKpRoleTag(candidate, "deadline")) {
            bonus += 0.7;
        }
        if (hasKpRoleTag(candidate, "distribution")) {
            bonus -= 0.45;
        }
        if (hasKpRoleTag(candidate, "publish")) {
            bonus -= 0.25;
        }
    }
    if (!signals.asksCondition && hasKpRoleTag(candidate, "condition")) {
        bonus -= 0.25;
    }
    if (!signals.asksMaterials && hasKpRoleTag(candidate, "materials")) {
        bonus -= 0.35;
    }
    if (!signals.asksAnnouncementPeriod &&
        !signals.asksPostOutcomeCondition &&
        hasKpRoleTag(candidate, "publish")) {
        bonus -= 0.2;
    }
    if (!/到账|发放|补发/.test(rawQuery) &&
        hasKpRoleTag(candidate, "distribution")) {
        bonus -= 0.35;
    }
    if (hasKpRoleTag(candidate, "background")) {
        bonus -= 0.15;
    }
    return bonus;
}
function rerankKpCandidatesByRole(params) {
    var _a, _b;
    var kpCandidates = params.kpCandidates, bestKpid = params.bestKpid, rawQuery = params.rawQuery, queryScopeHint = params.queryScopeHint, _c = params.mode, mode = _c === void 0 ? "off" : _c;
    var orderedCandidates = __spreadArray([], kpCandidates, true);
    if (mode !== "feature" || orderedCandidates.length === 0) {
        return {
            bestKpid: bestKpid,
            orderedCandidates: orderedCandidates,
            docScoreDelta: 0,
        };
    }
    var rerankWindow = orderedCandidates.slice(0, DEFAULT_KP_ROLE_CANDIDATE_LIMIT);
    var rawTopScore = (_b = (_a = rerankWindow[0]) === null || _a === void 0 ? void 0 : _a.score) !== null && _b !== void 0 ? _b : Number.NEGATIVE_INFINITY;
    var signals = deriveQueryRoleSignals(rawQuery, queryScopeHint);
    var reranked = rerankWindow
        .map(function (candidate) { return ({
        candidate: candidate,
        rerankedScore: candidate.score + computeKpRoleBonus(candidate, signals, rawQuery),
    }); })
        .sort(function (a, b) { return b.rerankedScore - a.rerankedScore; });
    var topCandidate = reranked[0];
    return {
        bestKpid: (topCandidate === null || topCandidate === void 0 ? void 0 : topCandidate.candidate.kpid) || bestKpid,
        orderedCandidates: __spreadArray(__spreadArray([], reranked.map(function (item) { return item.candidate; }), true), orderedCandidates.slice(rerankWindow.length), true),
        docScoreDelta: Number.isFinite(rawTopScore) && Number.isFinite(topCandidate === null || topCandidate === void 0 ? void 0 : topCandidate.rerankedScore)
            ? Math.max(0, topCandidate.rerankedScore - rawTopScore)
            : 0,
    };
}
function applyLexicalBonusBoost(boost, lexicalBonus) {
    if (lexicalBonus <= 0) {
        return boost;
    }
    return boost * (1 + Math.log1p(lexicalBonus) / 4);
}
function applyYearConstraintBoost(boost, queryYearWordIds, intentContext, scores, signals) {
    if (!queryYearWordIds || queryYearWordIds.length === 0) {
        return boost;
    }
    if (intentContext.years.length > 0) {
        if (signals.hasStructuredYearMatch) {
            if (!signals.hasSuspiciousStructuredYear) {
                return boost * 1.04;
            }
            if (signals.hasPublishYearMatch) {
                return boost * 0.96;
            }
            return boost * 0.78;
        }
        if (signals.hasLexicalYearMatch && signals.hasSuspiciousStructuredYear) {
            if (signals.hasPublishYearMatch) {
                return boost * 1.03;
            }
            return boost * 0.9;
        }
        if (scores.target_year !== undefined) {
            return boost * 0.01;
        }
    }
    if (!signals.hasStructuredYearMatch &&
        !signals.hasLexicalYearMatch &&
        !signals.hasPublishYearMatch) {
        return boost * 0.12;
    }
    return boost;
}
function applyMonthConstraintBoost(boost, intentContext, signals) {
    if (!intentContext.hasExplicitYear || !intentContext.hasExplicitMonth) {
        return boost;
    }
    if (signals.docMonth === undefined) {
        return boost * 0.94;
    }
    if (signals.hasStructuredMonthMatch) {
        return boost * 1.12;
    }
    return boost * 0.82;
}
function applyIntentBoost(boost, intentContext, scores) {
    if (intentContext.intentIds.length === 0) {
        return boost;
    }
    var nextBoost = boost;
    if (hasIntentMatch(intentContext.intentIds, scores.intent_ids)) {
        nextBoost *= 1.12;
    }
    else if (hasIntentConflict(intentContext.intentIds, scores.intent_ids)) {
        nextBoost *= 0.85;
    }
    else if (intentContext.relatedIntentIds.length > 0 &&
        hasAnyOverlap(intentContext.relatedIntentIds, scores.intent_ids)) {
        nextBoost *= 1.04;
    }
    return nextBoost;
}
function applyTopicCoverageBoost(boost, intentContext, scores, signals) {
    var _a;
    if (intentContext.topicIds.length === 0) {
        return boost;
    }
    var docTopicIds = dedupe(getCoverageComparableTopicIds(scores));
    if (docTopicIds.length > 0) {
        if (hasAnyOverlap(intentContext.topicIds, docTopicIds)) {
            return boost * 1.08;
        }
        return boost * 0.9;
    }
    if (scores.weak_topic_ids &&
        hasAnyOverlap(intentContext.topicIds, scores.weak_topic_ids)) {
        return boost * 1.02;
    }
    var hasStructuredFallbackEvidence = intentContext.querySpecificityTerms.length > 0 &&
        (signals.hasStructuredYearMatch ||
            signals.hasLexicalYearMatch ||
            signals.hasPublishYearMatch) &&
        (hasAnyOverlap(intentContext.degreeLevels, scores.degree_levels) ||
            (((_a = scores.event_types) === null || _a === void 0 ? void 0 : _a.length) || 0) > 0);
    if (hasStructuredFallbackEvidence) {
        return boost * 0.98;
    }
    return boost * 0.84;
}
function applyDegreeBoost(boost, intentContext, scores) {
    var _a;
    if (intentContext.degreeLevels.length === 0) {
        return boost;
    }
    if (hasAnyOverlap(intentContext.degreeLevels, scores.degree_levels)) {
        return boost * 1.05;
    }
    if ((((_a = scores.degree_levels) === null || _a === void 0 ? void 0 : _a.length) || 0) > 0) {
        return boost * 0.93;
    }
    return boost;
}
function applyEventBoost(boost, intentContext, scores) {
    var _a, _b;
    if (hasAnyOverlap(intentContext.eventTypes, scores.event_types)) {
        boost *= 1.05;
    }
    else if (intentContext.eventTypes.length > 0 &&
        (((_a = scores.event_types) === null || _a === void 0 ? void 0 : _a.length) || 0) > 0) {
        boost *= EVENT_TYPE_MISMATCH_PENALTY;
    }
    if (intentContext.hasPostOutcomeCondition &&
        (((_b = scores.event_types) === null || _b === void 0 ? void 0 : _b.length) || 0) > 0) {
        if (hasAnyOverlap(["录取公示"], scores.event_types)) {
            boost *= 1.1;
        }
        if (hasAnyOverlap(["复试通知"], scores.event_types)) {
            boost *= 0.78;
        }
        if (hasAnyOverlap(["招生章程", "报名通知"], scores.event_types)) {
            boost *= 0.82;
        }
    }
    var asksConditionOnly = /条件|满足|资格/.test(intentContext.rawQuery) &&
        !/怎么|流程|报名|操作|步骤/.test(intentContext.rawQuery) &&
        !/初试|复试|成绩|分数/.test(intentContext.rawQuery);
    if (asksConditionOnly &&
        hasAnyOverlap(["复试通知"], scores.event_types)) {
        boost *= 0.35;
    }
    return boost;
}
function applyLatestYearBoost(boost, intentContext, scores, latestTargetYear) {
    if (!intentContext.preferLatest ||
        latestTargetYear === undefined ||
        scores.target_year === undefined) {
        return boost;
    }
    var yearGap = Math.max(0, latestTargetYear - scores.target_year);
    return boost * Math.pow(LATEST_YEAR_BOOST_BASE, yearGap);
}
function applyLatestTimestampBoost(boost, intentContext, scores, latestTimestamp) {
    if (!intentContext.preferLatestStrong || latestTimestamp === undefined) {
        return boost;
    }
    if (scores.timestamp === undefined) {
        return boost * 0.82;
    }
    var gapSeconds = Math.max(0, latestTimestamp - scores.timestamp);
    if (gapSeconds <= 0) {
        return boost;
    }
    var gapMonths = gapSeconds / (60 * 60 * 24 * 30);
    return boost * Math.pow(LATEST_POLICY_TIMESTAMP_BOOST_BASE, gapMonths);
}
function shouldPreferCurrentProcessVersion(intentContext) {
    if (intentContext.hasExplicitYear || intentContext.hasHistoricalHint) {
        return false;
    }
    if (intentContext.hasPostOutcomeCondition) {
        return false;
    }
    var hasProcessCue = /报名|确认|提交|材料|申请|答辩|流程|步骤|怎么办|如何|接下来/.test(intentContext.rawQuery);
    if (!hasProcessCue) {
        return false;
    }
    return (intentContext.hasStrongDetailAnchor ||
        /流程|步骤|怎么办|如何|接下来/.test(intentContext.rawQuery));
}
function applyCurrentProcessTimestampBoost(boost, intentContext, scores, latestTimestamp) {
    if (!shouldPreferCurrentProcessVersion(intentContext) ||
        latestTimestamp === undefined) {
        return boost;
    }
    if (scores.timestamp === undefined) {
        return boost * 0.9;
    }
    var gapSeconds = Math.max(0, latestTimestamp - scores.timestamp);
    if (gapSeconds <= 0) {
        return boost;
    }
    var gapMonths = gapSeconds / (60 * 60 * 24 * 30);
    var hasProcessLikeEvent = hasAnyOverlap(__spreadArray([], CURRENT_PROCESS_EVENT_TYPES, true), scores.event_types);
    var decayBase = hasProcessLikeEvent
        ? CURRENT_PROCESS_TIMESTAMP_BOOST_BASE
        : 0.992;
    return boost * Math.pow(decayBase, gapMonths);
}
function applyScopeSpecificityBoost(boost, intentContext, scopeSpecificityStats) {
    if (!scopeSpecificityStats) {
        return boost;
    }
    var querySpecificityTerms = intentContext.querySpecificityTerms;
    if (querySpecificityTerms.length > 0) {
        var matchedTf = getMatchedSpecificityTf(querySpecificityTerms, scopeSpecificityStats);
        var matchedTerms = querySpecificityTerms.filter(function (term) { return (scopeSpecificityStats.termTf[term] || 0) > 0; }).length;
        if (matchedTerms === 0 || matchedTf === 0) {
            return boost * 0.45;
        }
        var coverageRatio = matchedTerms / querySpecificityTerms.length;
        var focusRatio = matchedTf / Math.max(matchedTf, scopeSpecificityStats.totalTf, 1);
        var nextBoost = boost;
        nextBoost *= 0.88 + coverageRatio * 0.24;
        nextBoost *= 0.55 + focusRatio * 0.95;
        return nextBoost;
    }
    if (!intentContext.discourageUnexpectedSpecificity) {
        return boost;
    }
    var unexpectedTf = Object.entries(scopeSpecificityStats.termTf).reduce(function (sum, _a) {
        var term = _a[0], tf = _a[1];
        return intentContext.querySpecificityTerms.includes(term) ? sum : sum + tf;
    }, 0);
    if (unexpectedTf <= 0) {
        return boost;
    }
    return boost * Math.max(0.72, 1 - Math.log1p(unexpectedTf) / 10);
}
function applySpecificityLocalFreshnessBoost(boost, intentContext, scores, scopeSpecificityStats, latestFocusedSpecificityTimestamp) {
    if (!intentContext.preferLatestStrong ||
        intentContext.querySpecificityTerms.length === 0 ||
        latestFocusedSpecificityTimestamp === undefined ||
        scores.timestamp === undefined) {
        return boost;
    }
    if (!/怎么|流程|报名|操作|步骤/.test(intentContext.rawQuery)) {
        return boost;
    }
    var matchedTf = getMatchedSpecificityTf(intentContext.querySpecificityTerms, scopeSpecificityStats);
    if (matchedTf < 10) {
        return boost;
    }
    var gapSeconds = Math.max(0, latestFocusedSpecificityTimestamp - scores.timestamp);
    if (gapSeconds <= 0) {
        return boost;
    }
    var gapMonths = gapSeconds / (60 * 60 * 24 * 30);
    return boost * Math.pow(0.75, gapMonths);
}
function computeBoostMultiplier(params) {
    var otid = params.otid, scores = params.scores, lexicalBonusMap = params.lexicalBonusMap, yearHitMap = params.yearHitMap, queryYearWordIds = params.queryYearWordIds, intentContext = params.intentContext, latestTargetYear = params.latestTargetYear, latestTimestamp = params.latestTimestamp, scopeSpecificityStats = params.scopeSpecificityStats, latestFocusedSpecificityTimestamp = params.latestFocusedSpecificityTimestamp;
    var signals = getDocQuerySignals(otid, scores, intentContext, yearHitMap);
    var lexicalBonus = lexicalBonusMap.get(otid) || 0;
    var boost = 1.0;
    boost = applyLexicalBonusBoost(boost, lexicalBonus);
    boost = applyYearConstraintBoost(boost, queryYearWordIds, intentContext, scores, signals);
    boost = applyMonthConstraintBoost(boost, intentContext, signals);
    boost = applyIntentBoost(boost, intentContext, scores);
    boost = applyTopicCoverageBoost(boost, intentContext, scores, signals);
    boost = applyDegreeBoost(boost, intentContext, scores);
    boost = applyEventBoost(boost, intentContext, scores);
    boost = applyLatestYearBoost(boost, intentContext, scores, latestTargetYear);
    boost = applyLatestTimestampBoost(boost, intentContext, scores, latestTimestamp);
    boost = applyCurrentProcessTimestampBoost(boost, intentContext, scores, latestTimestamp);
    boost = applyScopeSpecificityBoost(boost, intentContext, scopeSpecificityStats);
    boost = applySpecificityLocalFreshnessBoost(boost, intentContext, scores, scopeSpecificityStats, latestFocusedSpecificityTimestamp);
    return boost;
}
function extractRetrievalSignals(sortedRanking, otidMap) {
    var _a, _b, _c, _d, _e, _f, _g;
    var consistencyWindow = sortedRanking.slice(0, 10);
    var topicHistogram = new Map();
    var labeledCount = 0;
    for (var _i = 0, consistencyWindow_1 = consistencyWindow; _i < consistencyWindow_1.length; _i++) {
        var item = consistencyWindow_1[_i];
        var scores = otidMap[item.otid];
        var topicIds = dedupe(getCoverageComparableTopicIds(scores));
        if (topicIds.length === 0)
            continue;
        labeledCount += 1;
        topicIds.forEach(function (topicId) {
            topicHistogram.set(topicId, (topicHistogram.get(topicId) || 0) + 1);
        });
    }
    var top1Score = ((_a = sortedRanking[0]) === null || _a === void 0 ? void 0 : _a.score) || 0;
    var top2Score = (_c = (_b = sortedRanking[1]) === null || _b === void 0 ? void 0 : _b.score) !== null && _c !== void 0 ? _c : top1Score;
    var top5Score = (_g = (_e = (_d = sortedRanking[4]) === null || _d === void 0 ? void 0 : _d.score) !== null && _e !== void 0 ? _e : (_f = sortedRanking.at(-1)) === null || _f === void 0 ? void 0 : _f.score) !== null && _g !== void 0 ? _g : top1Score;
    var dominantCount = topicHistogram.size > 0 ? Math.max.apply(Math, topicHistogram.values()) : 0;
    var dominantRatio = consistencyWindow.length > 0 ? dominantCount / consistencyWindow.length : 0;
    return {
        candidateCount: sortedRanking.length,
        top1Score: top1Score,
        top1Top2Gap: top1Score - top2Score,
        top1Top5Gap: top1Score - top5Score,
        distinctTopicCount: topicHistogram.size,
        dominantTopicCount: dominantCount,
        dominantTopicRatio: dominantRatio,
        labeledTopicCount: labeledCount,
    };
}
function classifyResponseMode(querySignals, retrievalSignals) {
    var _a, _b;
    var scores = {
        direct_answer: 0.5,
        clarify_or_route: 0,
        reject: 0,
    };
    var reasons = new Set();
    if (querySignals.hasExplicitTopicOrIntent) {
        scores.direct_answer += 2.6;
        scores.clarify_or_route -= 1.1;
        scores.reject -= 1.3;
        reasons.add("explicit_topic_or_intent");
    }
    if (querySignals.hasStrongDetailAnchor) {
        scores.direct_answer += 3.0;
        scores.clarify_or_route -= 2.0;
        scores.reject -= 1.1;
        reasons.add("strong_detail_anchor");
    }
    if (querySignals.hasExplicitYear) {
        scores.direct_answer += 0.8;
        reasons.add("explicit_year");
    }
    if (querySignals.hasResultState) {
        scores.clarify_or_route += 0.55;
        scores.reject -= 0.9;
        reasons.add("result_state");
    }
    if (querySignals.hasGenericNextStep) {
        scores.clarify_or_route += 1.35;
        scores.reject += 1.1;
        scores.direct_answer -= 0.75;
        reasons.add("generic_next_step");
    }
    if (querySignals.hasEntryLikeAnchor) {
        scores.clarify_or_route += 0.5;
        scores.reject -= 0.25;
        reasons.add("entry_like_anchor");
    }
    if (querySignals.hasGenericNextStep &&
        !querySignals.hasResultState &&
        !querySignals.hasEntryLikeAnchor &&
        !querySignals.hasStrongDetailAnchor) {
        scores.reject += 1.6;
        scores.clarify_or_route -= 0.45;
        reasons.add("empty_next_step_without_state");
    }
    if (querySignals.hasResultState &&
        querySignals.hasGenericNextStep &&
        !querySignals.hasStrongDetailAnchor) {
        scores.clarify_or_route += 1.35;
        scores.direct_answer -= 0.3;
        reasons.add("result_state_needs_clarification");
    }
    if (querySignals.hasResultState &&
        querySignals.hasGenericNextStep &&
        querySignals.hasStrongDetailAnchor) {
        scores.direct_answer += 1.0;
        reasons.add("detail_anchor_overrides_generic_state");
    }
    if (querySignals.hasExplicitYear &&
        !querySignals.hasGenericNextStep &&
        (querySignals.hasEntryLikeAnchor || querySignals.hasResultState)) {
        scores.direct_answer += 0.8;
        scores.clarify_or_route -= 0.25;
        reasons.add("time_anchor_supports_direct_answer");
    }
    if (!querySignals.hasExplicitTopicOrIntent &&
        !querySignals.hasStrongDetailAnchor &&
        !querySignals.hasEntryLikeAnchor) {
        scores.reject += 0.45;
        reasons.add("anchorless_query");
    }
    if ((querySignals.tokenCount || 0) === 0) {
        scores.reject += 0.45;
        scores.direct_answer -= 0.15;
        reasons.add("zero_sparse_token");
    }
    if (retrievalSignals.candidateCount === 0) {
        scores.reject += 1.4;
        scores.direct_answer -= 0.7;
        reasons.add("no_candidates");
    }
    if (retrievalSignals.labeledTopicCount === 0) {
        scores.reject += 1.1;
        scores.direct_answer -= 0.4;
        reasons.add("no_labeled_topics");
    }
    if (retrievalSignals.distinctTopicCount >= 3 &&
        retrievalSignals.dominantTopicRatio < 0.45) {
        scores.reject += 1.15;
        scores.direct_answer -= 0.35;
        reasons.add("low_topic_consistency");
    }
    if (retrievalSignals.top1Top2Gap >= 0.12) {
        scores.direct_answer += 0.35;
        reasons.add("stable_top1_gap");
    }
    if (retrievalSignals.labeledTopicCount >= 3 &&
        retrievalSignals.distinctTopicCount <= 2 &&
        retrievalSignals.dominantTopicRatio >= 0.5) {
        scores.direct_answer += 0.35;
        reasons.add("stable_topic_cluster");
    }
    var rankedModes = Object.entries(scores)
        .map(function (_a) {
        var mode = _a[0], score = _a[1];
        return [mode, score];
    })
        .sort(function (a, b) { return b[1] - a[1]; });
    var _c = rankedModes[0], mode = _c[0], topScore = _c[1];
    var secondScore = (_b = (_a = rankedModes[1]) === null || _a === void 0 ? void 0 : _a[1]) !== null && _b !== void 0 ? _b : topScore;
    var confidence = Math.max(0.55, Math.min(0.98, 0.62 + (topScore - secondScore) * 0.14));
    return {
        mode: mode,
        confidence: confidence,
        reason: Array.from(reasons).slice(0, 3).join("+") ||
            "scored_response_mode",
        preferLatestWithinTopic: querySignals.hasLatestPolicyState && !querySignals.hasExplicitYear,
        useWeakMatches: mode === "clarify_or_route",
    };
}
function searchAndRank(params) {
    var queryVector = params.queryVector, querySparse = params.querySparse, _a = params.queryWords, queryWords = _a === void 0 ? [] : _a, metadata = params.metadata, vectorMatrix = params.vectorMatrix, dimensions = params.dimensions, _currentTimestamp = params.currentTimestamp, bm25Stats = params.bm25Stats, _b = params.weights, weights = _b === void 0 ? exports.DEFAULT_WEIGHTS : _b, queryYearWordIds = params.queryYearWordIds, queryIntent = params.queryIntent, queryScopeHint = params.queryScopeHint, candidateIndices = params.candidateIndices, scopeSpecificityWordIdToTerm = params.scopeSpecificityWordIdToTerm, directAnswerEvidenceWordIdToTerm = params.directAnswerEvidenceWordIdToTerm, _c = params.topHybridLimit, topHybridLimit = _c === void 0 ? 1000 : _c, _d = params.kpAggregationMode, kpAggregationMode = _d === void 0 ? "max" : _d, _e = params.kpTopN, kpTopN = _e === void 0 ? 3 : _e, _f = params.kpTailWeight, kpTailWeight = _f === void 0 ? 0.35 : _f, _g = params.lexicalBonusMode, lexicalBonusMode = _g === void 0 ? "sum" : _g, _h = params.kpRoleRerankMode, kpRoleRerankMode = _h === void 0 ? "off" : _h, _j = params.kpRoleDocWeight, kpRoleDocWeight = _j === void 0 ? DEFAULT_KP_ROLE_DOC_WEIGHT : _j;
    var n = metadata.length;
    var activeCandidateIndices = candidateIndices && candidateIndices.length > 0 ? candidateIndices : undefined;
    var candidateCount = activeCandidateIndices
        ? activeCandidateIndices.length
        : n;
    var denseScores = new Float32Array(candidateCount);
    var sparseScores = new Float32Array(candidateCount);
    var denseOrder = new Int32Array(candidateCount);
    var sparseOrder = new Int32Array(candidateCount);
    var lexicalBonusMap = new Map();
    var yearHitMap = new Map();
    var docScopeSpecificityStatsMap = new Map();
    var docDirectAnswerEvidenceStatsMap = new Map();
    for (var localIndex = 0; localIndex < candidateCount; localIndex++) {
        var metaIndex = activeCandidateIndices
            ? activeCandidateIndices[localIndex]
            : localIndex;
        var meta = metadata[metaIndex];
        var dense = dotProduct(queryVector, vectorMatrix, meta.vector_index, dimensions);
        if (meta.scale !== undefined && meta.scale !== null)
            dense *= meta.scale;
        denseScores[localIndex] = dense;
        denseOrder[localIndex] = localIndex;
        var sparse = 0;
        if (querySparse && meta.sparse && meta.sparse.length > 0) {
            var dl = bm25Stats.docLengths[metaIndex];
            var safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);
            var otid = meta.type === "OT" ? meta.id : meta.parent_otid;
            for (var j = 0; j < meta.sparse.length; j += 2) {
                var wordId = meta.sparse[j];
                var tf = meta.sparse[j + 1];
                var specificityTerm = scopeSpecificityWordIdToTerm === null || scopeSpecificityWordIdToTerm === void 0 ? void 0 : scopeSpecificityWordIdToTerm.get(wordId);
                if (specificityTerm) {
                    var existing = docScopeSpecificityStatsMap.get(otid) || {
                        termTf: {},
                        totalTf: 0,
                    };
                    existing.termTf[specificityTerm] =
                        (existing.termTf[specificityTerm] || 0) + tf;
                    existing.totalTf += tf;
                    docScopeSpecificityStatsMap.set(otid, existing);
                }
                var directAnswerEvidenceTerm = directAnswerEvidenceWordIdToTerm === null || directAnswerEvidenceWordIdToTerm === void 0 ? void 0 : directAnswerEvidenceWordIdToTerm.get(wordId);
                if (directAnswerEvidenceTerm) {
                    var existing = docDirectAnswerEvidenceStatsMap.get(otid) || {
                        termTf: {},
                        totalTf: 0,
                    };
                    existing.termTf[directAnswerEvidenceTerm] =
                        (existing.termTf[directAnswerEvidenceTerm] || 0) + tf;
                    existing.totalTf += tf;
                    docDirectAnswerEvidenceStatsMap.set(otid, existing);
                }
                if (queryYearWordIds && queryYearWordIds.includes(wordId)) {
                    yearHitMap.set(otid, true);
                }
                if (querySparse[wordId]) {
                    var qWeight = querySparse[wordId] || 1;
                    var idf = bm25Stats.idfMap.get(wordId) || 0;
                    var numerator = tf * (BM25_K1 + 1);
                    var denominator = tf +
                        BM25_K1 *
                            (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
                    sparse += qWeight * idf * (numerator / denominator);
                }
            }
            if (sparse > 0) {
                var otid_1 = meta.type === "OT" ? meta.id : meta.parent_otid;
                var weightedBonus = meta.type === "Q"
                    ? sparse * 1.5
                    : meta.type === "KP"
                        ? sparse * 1.2
                        : sparse;
                var currentBonus = lexicalBonusMap.get(otid_1) || 0;
                var nextBonus = lexicalBonusMode === "max"
                    ? Math.max(currentBonus, weightedBonus)
                    : currentBonus + weightedBonus;
                lexicalBonusMap.set(otid_1, nextBonus);
            }
        }
        sparseScores[localIndex] = sparse;
        sparseOrder[localIndex] = localIndex;
    }
    denseOrder.sort(function (a, b) { return denseScores[b] - denseScores[a]; });
    var rrfScores = new Map();
    for (var rank = 0; rank < Math.min(4000, candidateCount); rank++) {
        var metaIndex = activeCandidateIndices
            ? activeCandidateIndices[denseOrder[rank]]
            : denseOrder[rank];
        var meta = metadata[metaIndex];
        rrfScores.set(meta, (1 / (rank + exports.RRF_K)) * 100);
    }
    if (querySparse) {
        sparseOrder.sort(function (a, b) { return sparseScores[b] - sparseScores[a]; });
        for (var rank = 0; rank < Math.min(4000, candidateCount); rank++) {
            var localIndex = sparseOrder[rank];
            if (sparseScores[localIndex] === 0)
                break;
            var metaIndex = activeCandidateIndices
                ? activeCandidateIndices[localIndex]
                : localIndex;
            var meta = metadata[metaIndex];
            var current = rrfScores.get(meta) || 0;
            rrfScores.set(meta, current + (1.2 / (rank + exports.RRF_K)) * 100);
        }
    }
    var topHybrid = Array.from(rrfScores.entries())
        .sort(function (a, b) { return b[1] - a[1]; })
        .slice(0, Math.max(1, topHybridLimit));
    var otidMap = {};
    for (var _i = 0, topHybrid_1 = topHybrid; _i < topHybrid_1.length; _i++) {
        var _k = topHybrid_1[_i], meta = _k[0], score = _k[1];
        var otid = meta.type === "OT" ? meta.id : meta.parent_otid;
        var topicIds = resolveMetadataTopicIds(meta);
        if (!otidMap[otid]) {
            otidMap[otid] = (0, aggregated_doc_scores_1.createAggregatedDocScores)(meta, topicIds);
        }
        (0, aggregated_doc_scores_1.mergeAggregatedDocMetadata)(otidMap[otid], meta, topicIds);
        (0, aggregated_doc_scores_1.applyScoreToAggregatedDocScores)(otidMap[otid], meta, score);
    }
    var finalRanking = [];
    var candidateTargetYears = Object.values(otidMap)
        .map(function (scores) { return scores.target_year; })
        .filter(function (year) { return typeof year === "number"; });
    var candidateTimestamps = Object.values(otidMap)
        .map(function (scores) { return scores.timestamp; })
        .filter(function (timestamp) { return typeof timestamp === "number"; });
    var latestTargetYear = candidateTargetYears.length > 0
        ? Math.max.apply(Math, candidateTargetYears) : undefined;
    var latestTimestamp = candidateTimestamps.length > 0
        ? Math.max.apply(Math, candidateTimestamps) : undefined;
    var intentContext = createQueryIntentContext(queryIntent, queryWords);
    var latestFocusedSpecificityTimestamp = intentContext.querySpecificityTerms.length > 0
        ? Object.entries(otidMap)
            .map(function (_a) {
            var otid = _a[0], scores = _a[1];
            var matchedTf = getMatchedSpecificityTf(intentContext.querySpecificityTerms, docScopeSpecificityStatsMap.get(otid));
            return matchedTf >= 10 ? scores.timestamp : undefined;
        })
            .filter(function (timestamp) {
            return typeof timestamp === "number";
        })
            .reduce(function (latest, timestamp) {
            return latest === undefined || timestamp > latest
                ? timestamp
                : latest;
        }, undefined)
        : undefined;
    for (var _l = 0, _m = Object.entries(otidMap); _l < _m.length; _l++) {
        var _o = _m[_l], otid = _o[0], scores = _o[1];
        var signals = getDocQuerySignals(otid, scores, intentContext, yearHitMap);
        if (shouldSkipForExplicitYear(scores, intentContext, signals)) {
            continue;
        }
        var finalScore = computeBaseScore(scores, weights, {
            kpAggregationMode: kpAggregationMode,
            kpTopN: kpTopN,
            kpTailWeight: kpTailWeight,
        });
        var kpRoleSelection = rerankKpCandidatesByRole({
            kpCandidates: scores.kp_candidates,
            bestKpid: scores.best_kpid,
            rawQuery: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.rawQuery) || "",
            queryScopeHint: queryScopeHint,
            mode: kpRoleRerankMode,
        });
        var boost = computeBoostMultiplier({
            otid: otid,
            scores: scores,
            lexicalBonusMap: lexicalBonusMap,
            yearHitMap: yearHitMap,
            queryYearWordIds: queryYearWordIds,
            intentContext: intentContext,
            latestTargetYear: latestTargetYear,
            latestTimestamp: latestTimestamp,
            scopeSpecificityStats: docScopeSpecificityStatsMap.get(otid),
            latestFocusedSpecificityTimestamp: latestFocusedSpecificityTimestamp,
        });
        finalRanking.push({
            otid: otid,
            score: finalScore * boost + kpRoleSelection.docScoreDelta * kpRoleDocWeight,
            best_kpid: kpRoleSelection.bestKpid,
            kp_candidates: kpRoleSelection.orderedCandidates.slice(0, 5),
        });
    }
    var sortedRanking = finalRanking.sort(function (a, b) { return b.score - a.score; });
    var defaultQuerySignals = {
        hasExplicitTopicOrIntent: false,
        hasExplicitYear: false,
        hasHistoricalHint: false,
        hasStrongDetailAnchor: false,
        hasEntryLikeAnchor: false,
        hasResultState: false,
        hasLatestPolicyState: false,
        hasGenericNextStep: false,
        queryLength: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.rawQuery.length) || 0,
        tokenCount: 0,
    };
    var querySignals = withQueryTokenCount((queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.signals) || defaultQuerySignals, querySparse);
    var retrievalSignals = extractRetrievalSignals(sortedRanking, otidMap);
    var responseDecision = classifyResponseMode(querySignals, retrievalSignals);
    var explicitOutOfScopeOnly = ((queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.intentIds.length) || 0) === 0 &&
        hasOnlyOutOfScopeTopics((queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.topicIds) || []);
    var inDomainEvidenceReject = shouldRejectForMissingInDomainEvidence({
        rawQuery: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.rawQuery) || "",
        queryIntent: queryIntent,
        sortedRanking: sortedRanking,
        docEvidenceStatsMap: docDirectAnswerEvidenceStatsMap,
        otidMap: otidMap,
    });
    var diagnostics = {
        querySignals: querySignals,
        retrievalSignals: retrievalSignals,
        explicitOutOfScopeOnly: explicitOutOfScopeOnly,
        inDomainEvidenceRejectLabel: inDomainEvidenceReject.label || null,
    };
    if (explicitOutOfScopeOnly) {
        return {
            matches: [],
            weakMatches: sortedRanking.slice(0, 5),
            rejection: {
                reason: "low_topic_coverage",
                topicIds: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.topicIds) || [],
            },
            responseDecision: __assign(__assign({}, responseDecision), { mode: "reject", confidence: Math.max(responseDecision.confidence, 0.92), reason: "explicit_out_of_scope_topic", preferLatestWithinTopic: false, useWeakMatches: true }),
            diagnostics: diagnostics,
        };
    }
    if (responseDecision.mode === "direct_answer" &&
        inDomainEvidenceReject.shouldReject) {
        return {
            matches: [],
            weakMatches: sortedRanking.slice(0, 5),
            rejection: {
                reason: "low_consistency",
                topicIds: (queryIntent === null || queryIntent === void 0 ? void 0 : queryIntent.topicIds) || [],
            },
            responseDecision: __assign(__assign({}, responseDecision), { mode: "reject", confidence: Math.max(responseDecision.confidence, 0.9), reason: "missing_in_domain_evidence:".concat(inDomainEvidenceReject.label || "unknown"), preferLatestWithinTopic: false, useWeakMatches: true }),
            diagnostics: diagnostics,
        };
    }
    if (responseDecision.mode === "reject") {
        return {
            matches: [],
            weakMatches: [],
            rejection: {
                reason: "low_consistency",
                topicIds: [],
            },
            responseDecision: responseDecision,
            diagnostics: diagnostics,
        };
    }
    if (responseDecision.mode === "clarify_or_route") {
        return {
            matches: [],
            weakMatches: responseDecision.useWeakMatches
                ? sortedRanking.slice(0, 5)
                : [],
            rejection: {
                reason: "weak_anchor_needs_clarification",
                topicIds: [],
            },
            responseDecision: responseDecision,
            diagnostics: diagnostics,
        };
    }
    return {
        matches: sortedRanking.slice(0, 100),
        weakMatches: [],
        responseDecision: responseDecision,
        diagnostics: diagnostics,
    };
}

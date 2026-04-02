import {
    DEGREE_LEVEL_TABLE as CONFIG_DEGREE_LEVEL_TABLE,
    EVENT_TYPE_TABLE as CONFIG_EVENT_TYPE_TABLE,
    HISTORICAL_QUERY_HINTS as CONFIG_HISTORICAL_QUERY_HINTS,
    INTENT_VECTOR_TABLE as CONFIG_INTENT_VECTOR_TABLE,
    LATEST_QUERY_HINTS as CONFIG_LATEST_QUERY_HINTS,
    TOPIC_CONFIGS as CONFIG_TOPIC_CONFIGS,
} from "./search_topic_config";
import {
    applyScoreToAggregatedDocScores,
    createAggregatedDocScores,
    mergeAggregatedDocMetadata,
    type AggregatedDocScores,
    type KPCandidate,
} from "./aggregated_doc_scores";

export interface Metadata {
    id: string;
    type: "Q" | "KP" | "OT";
    parent_pkid?: string;
    parent_otid: string;
    timestamp?: number;
    vector_index: number;
    scale?: number;
    sparse?: number[];
    target_year?: number;
    topic_ids?: string[];
    subtopic_ids?: string[];
    primary_topic_ids?: string[];
    secondary_topic_ids?: string[];
    weak_topic_ids?: string[];
    primary_subtopic_ids?: string[];
    secondary_subtopic_ids?: string[];
    weak_subtopic_ids?: string[];
    intent_ids?: string[];
    degree_levels?: string[];
    event_types?: string[];
    kp_role_tags?: string[];
}

export interface SearchResult {
    otid: string;
    best_kpid?: string;
    kp_candidates?: KPCandidate[];
    score: number;
    details?: {
        denseRRF: number;
        sparseRRF: number;
        lexicalBoost: number;
    };
}

export interface SearchRejection {
    reason:
        | "low_topic_coverage"
        | "low_consistency"
        | "weak_anchor_needs_clarification";
    topicIds: string[];
}

export type ResponseMode =
    | "direct_answer"
    | "clarify_or_route"
    | "reject";

export interface QuerySignals {
    hasExplicitTopicOrIntent: boolean;
    hasExplicitYear: boolean;
    hasHistoricalHint: boolean;
    hasStrongDetailAnchor: boolean;
    hasEntryLikeAnchor: boolean;
    hasResultState: boolean;
    hasLatestPolicyState: boolean;
    hasGenericNextStep: boolean;
    queryLength: number;
    tokenCount?: number;
}

export interface RetrievalSignals {
    candidateCount: number;
    top1Score: number;
    top1Top2Gap: number;
    top1Top5Gap: number;
    distinctTopicCount: number;
    dominantTopicCount: number;
    dominantTopicRatio: number;
    labeledTopicCount: number;
}

export interface ResponseDecision {
    mode: ResponseMode;
    confidence: number;
    reason: string;
    preferLatestWithinTopic: boolean;
    useWeakMatches: boolean;
}

type ResponseModeScores = Record<ResponseMode, number>;

export interface SearchRankDiagnostics {
    querySignals: QuerySignals;
    retrievalSignals: RetrievalSignals;
    explicitOutOfScopeOnly: boolean;
    inDomainEvidenceRejectLabel: string | null;
}

export interface SearchRankOutput {
    matches: SearchResult[];
    weakMatches: SearchResult[];
    rejection?: SearchRejection;
    responseDecision?: ResponseDecision;
    diagnostics?: SearchRankDiagnostics;
}

export interface BM25Stats {
    idfMap: Map<number, number>;
    docLengths: Int32Array;
    avgdl: number;
}

export interface ParsedQueryIntent {
    rawQuery: string;
    years: number[];
    months: number[];
    topicIds: string[];
    subtopicIds: string[];
    intentIds: string[];
    degreeLevels: string[];
    eventTypes: string[];
    normalizedTerms: string[];
    confidence: number;
    preferLatest: boolean;
    preferLatestStrong: boolean;
    signals: QuerySignals;
}

export type KPAggregationMode = "max" | "max_plus_topn";
export type LexicalBonusMode = "sum" | "max";
export type KPRoleRerankMode = "off" | "feature";

export interface IntentVectorItem {
    intent_id: string;
    topic_id: string;
    intent_name: string;
    aliases: readonly string[];
    negative_intents: readonly string[];
    related_intents: readonly string[];
}

export const DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33,
};

export const RRF_K = 60;

const BM25_K1 = 1.2;
const BM25_B = 0.4;
const EVENT_TYPE_MISMATCH_PENALTY = 0.95;
const LATEST_YEAR_BOOST_BASE = 0.82;
const LATEST_POLICY_TIMESTAMP_BOOST_BASE = 0.97;
const CURRENT_PROCESS_TIMESTAMP_BOOST_BASE = 0.984;
const DEFAULT_KP_ROLE_DOC_WEIGHT = 0.35;
const DEFAULT_KP_ROLE_CANDIDATE_LIMIT = 5;
const CURRENT_PROCESS_EVENT_TYPES = [
    "招生章程",
    "报名通知",
    "考试安排",
    "材料提交",
    "资格要求",
] as const;
export const QUERY_SCOPE_SPECIFICITY_TERMS = [
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
] as const;
const QUERY_SCOPE_SPECIFICITY_TERM_SET: ReadonlySet<string> = new Set(
    QUERY_SCOPE_SPECIFICITY_TERMS,
);
export const DIRECT_ANSWER_EVIDENCE_TERMS = [
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
] as const;
const BROAD_LATEST_SCOPE_CUE_PATTERN =
    /完整流程|完整|通用|一般|总流程|怎么报名|如何报名|条件.*报名|条件.*流程|条件.*操作/;



const INTENT_CONFLICTS: Record<string, readonly string[]> = Object.fromEntries(
    CONFIG_INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item.negative_intents]),
);

const INTENT_RULE_MAP: Map<string, IntentVectorItem> = new Map(
    CONFIG_INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item]),
);

const TOPIC_RULE_MAP: Map<string, (typeof CONFIG_TOPIC_CONFIGS)[number]> =
    new Map(CONFIG_TOPIC_CONFIGS.map((item) => [item.topic_id, item] as const));

function isOutOfScopeTopic(topicId: string): boolean {
    return TOPIC_RULE_MAP.get(topicId)?.scope === "out_of_scope";
}

function hasOnlyOutOfScopeTopics(topicIds: string[]): boolean {
    return (
        topicIds.length > 0 &&
        topicIds.every((topicId) => isOutOfScopeTopic(topicId))
    );
}

function deriveTopicIdsFromIntents(intentIds: string[]): string[] {
    return dedupe(
        intentIds
            .map((intentId) => INTENT_RULE_MAP.get(intentId)?.topic_id)
            .filter((topicId): topicId is string => Boolean(topicId)),
    );
}

export function resolveMetadataTopicIds(
    meta: Pick<Metadata, "topic_ids" | "subtopic_ids" | "intent_ids">,
): string[] {
    if (meta.topic_ids && meta.topic_ids.length > 0) {
        return meta.topic_ids;
    }
    return deriveTopicIdsFromIntents(meta.subtopic_ids || meta.intent_ids || []);
}

function resolveDocOtid(meta: Pick<Metadata, "id" | "parent_otid">): string {
    return meta.parent_otid || meta.id;
}

function matchTopicIds(query: string): string[] {
    return dedupe(
        CONFIG_TOPIC_CONFIGS.filter((topic) =>
            topic.aliases.some((alias) => query.includes(alias)),
        ).map((topic) => topic.topic_id),
    );
}

function dedupe<T>(items: T[]): T[] {
    return Array.from(new Set(items));
}

function hasGenericNextStepCue(query: string): boolean {
    return /怎么办|怎么做|怎么处理|怎么操作|如何办理|如何操作|接下来|下一步|要做什么|需要做什么|还要做什么|还需要做什么|还需要再操作什么|后面该怎么处理|后面怎么办|后续怎么办|应该怎么办|应该怎么|要办哪些事|下一步是什么|怎么弄|怎么搞|怎么整|处理什么|该做什么|准备什么|干什么/.test(
        query,
    );
}

function hasClarificationStateCue(query: string): boolean {
    return /考上|录取|录取结果|拟录取|收到通知书|拿到通知书|审核通过|审核没通过|审核未通过|审核不通过|通过初审|初审通过|学校通知我通过|通知我通过|通过了|过审|没过审|未过审|获批|评上|提交完材料|提交完申请|提交材料后|补交完材料|补交完|成了新生|已经是新生|新生以后|录取后|拟录取后|考上后|收到通知书后|拿到通知书后|审核通过后|通过初审后|获批后/.test(
        query,
    );
}

function hasLatestPolicyStateCue(query: string): boolean {
    return /考上|录取|录取结果|拟录取|收到通知书|拿到通知书|成了新生|已经是新生|新生以后|录取后|拟录取后|考上后|收到通知书后|拿到通知书后/.test(
        query,
    );
}

function hasPostOutcomeConditionCue(query: string): boolean {
    return /最终有效|最终有效性|最终录取|录取.*有效|拟录取.*有效|审核.*为准|审批.*为准/.test(
        query,
    );
}

function hasStrongDetailAnchorCue(query: string): boolean {
    return /录取通知书|通知书|报到|宿舍|党团关系|奖助金|档案|调档|政审|网上确认|现场确认|答辩|报名|考试|考试内容|考试科目|科目|题型|缴费|申请书|复试|面试|邮寄|地址|银行卡|学费/.test(
        query,
    );
}

function hasEntryLikeAnchorCue(query: string): boolean {
    return /新生|入学|录取|拟录取|审核|初审|资格审核|申请|材料|网上确认|现场确认/.test(
        query,
    );
}

function hasLatestPolicyFallbackCue(querySignals: QuerySignals): boolean {
    return querySignals.hasGenericNextStep && querySignals.hasLatestPolicyState;
}

function buildQuerySignals(params: {
    query: string;
    years: number[];
    topicIds: string[];
    intentIds: string[];
    hasHistoricalHint: boolean;
}): QuerySignals {
    const { query, years, topicIds, intentIds, hasHistoricalHint } = params;
    return {
        hasExplicitTopicOrIntent: topicIds.length > 0 || intentIds.length > 0,
        hasExplicitYear: years.length > 0,
        hasHistoricalHint,
        hasStrongDetailAnchor: hasStrongDetailAnchorCue(query),
        hasEntryLikeAnchor: hasEntryLikeAnchorCue(query),
        hasResultState: hasClarificationStateCue(query),
        hasLatestPolicyState: hasLatestPolicyStateCue(query),
        hasGenericNextStep: hasGenericNextStepCue(query),
        queryLength: query.length,
    };
}

function withQueryTokenCount(
    signals: QuerySignals,
    querySparse?: Record<number, number>,
): QuerySignals {
    return {
        ...signals,
        tokenCount: querySparse ? Object.keys(querySparse).length : 0,
    };
}

function extractQueryMonths(query: string): number[] {
    const months = new Set<number>();
    const matches = query.matchAll(/(?:^|[^\d])(1[0-2]|0?[1-9])月(?:份)?/g);
    for (const match of matches) {
        const value = Number.parseInt(match[1] || "", 10);
        if (Number.isFinite(value) && value >= 1 && value <= 12) {
            months.add(value);
        }
    }
    return Array.from(months);
}

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
    const offset = matrixIndex * dimensions;
    const unrolledLimit = dimensions - (dimensions % 4);
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;

    for (let i = 0; i < unrolledLimit; i += 4) {
        s0 += vecA[i] * matrix[offset + i];
        s1 += vecA[i + 1] * matrix[offset + i + 1];
        s2 += vecA[i + 2] * matrix[offset + i + 2];
        s3 += vecA[i + 3] * matrix[offset + i + 3];
    }

    let sum = s0 + s1 + s2 + s3;
    for (let i = unrolledLimit; i < dimensions; i++) {
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
    });

    return sparse;
}

export function parseQueryIntent(query: string): ParsedQueryIntent {
    const years = dedupe(
        (query.match(/20\d{2}/g) || []).map((year) => Number(year)),
    );
    const months = extractQueryMonths(query);
    const matchedRules = CONFIG_INTENT_VECTOR_TABLE.filter((rule) =>
        rule.aliases.some((alias) => query.includes(alias)),
    );
    const intentIds = dedupe(matchedRules.map((rule) => rule.intent_id));
    const matchedTopicIds = matchTopicIds(query);
    const topicIds = dedupe([
        ...deriveTopicIdsFromIntents(intentIds),
        ...matchedTopicIds,
    ]);
    const degreeLevels = dedupe([
        ...CONFIG_DEGREE_LEVEL_TABLE.filter((level) => query.includes(level)),
        ...(query.includes("直博") ? ["博士"] : []),
    ]);
    const eventTypes = dedupe(
        CONFIG_EVENT_TYPE_TABLE.filter((eventType) => query.includes(eventType)),
    );
    const normalizedTerms = dedupe(
        matchedRules.map((rule) => rule.intent_name),
    );
    const hasHistoricalHint = CONFIG_HISTORICAL_QUERY_HINTS.some((hint) =>
        query.includes(hint),
    );
    const hasLatestHint = CONFIG_LATEST_QUERY_HINTS.some((hint) =>
        query.includes(hint),
    );
    const signals = buildQuerySignals({
        query,
        years,
        topicIds,
        intentIds,
        hasHistoricalHint,
    });
    const preferLatestStrong =
        years.length === 0 &&
        !hasHistoricalHint &&
        (hasLatestHint || hasLatestPolicyFallbackCue(signals));
    const preferLatest =
        years.length === 0 &&
        !hasHistoricalHint &&
        (topicIds.some(
            (topicId) => TOPIC_RULE_MAP.get(topicId)?.prefer_latest,
        ) ||
            hasLatestHint ||
            preferLatestStrong);

    return {
        rawQuery: query,
        years,
        months,
        topicIds,
        subtopicIds: intentIds,
        intentIds,
        degreeLevels,
        eventTypes,
        normalizedTerms,
        confidence: intentIds.length > 0 ? 1 : 0,
        preferLatest,
        preferLatestStrong,
        signals,
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

function hasIntentMatch(
    queryIntentIds: string[],
    docIntentIds?: string[],
): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) =>
        docIntentIds.includes(queryIntentId),
    );
}

function hasAnyOverlap(a: string[], b?: string[]): boolean {
    if (!b || b.length === 0) return false;
    return a.some((item) => b.includes(item));
}

function getCoverageComparableTopicIds(doc: {
    primary_topic_ids?: string[];
    secondary_topic_ids?: string[];
    topic_ids?: string[];
}): string[] {
    if (doc.primary_topic_ids && doc.primary_topic_ids.length > 0) {
        return doc.primary_topic_ids;
    }
    if (doc.secondary_topic_ids && doc.secondary_topic_ids.length > 0) {
        return doc.secondary_topic_ids;
    }
    return doc.topic_ids || [];
}

function getRelatedIntentTypes(intentIds: string[]): string[] {
    return dedupe(
        intentIds.flatMap(
            (intentId) => INTENT_RULE_MAP.get(intentId)?.related_intents || [],
        ),
    );
}

type QueryIntentContext = {
    rawQuery: string;
    years: number[];
    months: number[];
    hasExplicitYear: boolean;
    hasExplicitMonth: boolean;
    hasHistoricalHint: boolean;
    hasStrongDetailAnchor: boolean;
    topicIds: string[];
    intentIds: string[];
    relatedIntentIds: string[];
    degreeLevels: string[];
    eventTypes: string[];
    hasPostOutcomeCondition: boolean;
    asksRuleDocument: boolean;
    asksOutcomeDocument: boolean;
    preferLatest: boolean;
    preferLatestStrong: boolean;
    querySpecificityTerms: string[];
    discourageUnexpectedSpecificity: boolean;
};

type DocQuerySignals = {
    hasStructuredYearMatch: boolean;
    hasLexicalYearMatch: boolean;
    hasPublishYearMatch: boolean;
    hasSuspiciousStructuredYear: boolean;
    docPublishYear?: number;
    hasStructuredMonthMatch: boolean;
    docMonth?: number;
};

type ScopeSpecificityStats = {
    termTf: Record<string, number>;
    totalTf: number;
};

type EvidenceCoverageRequirement = {
    label: string;
    requiredGroups: readonly (readonly string[])[];
    requireIntentAlignment?: boolean;
};

function getMatchedSpecificityTf(
    querySpecificityTerms: string[],
    scopeSpecificityStats?: ScopeSpecificityStats,
): number {
    if (!scopeSpecificityStats || querySpecificityTerms.length === 0) {
        return 0;
    }
    return querySpecificityTerms.reduce(
        (sum, term) => sum + (scopeSpecificityStats.termTf[term] || 0),
        0,
    );
}

function extractQuerySpecificityTerms(queryWords: string[]): string[] {
    return dedupe(
        queryWords.filter((word) => QUERY_SCOPE_SPECIFICITY_TERM_SET.has(word)),
    );
}

function queryAsksRuleDocument(rawQuery: string): boolean {
    return (
        /(招生简章|简章|招生章程|章程|实施细则|细则|实施办法|办法|接收办法|录取方案|方案)/.test(
            rawQuery,
        ) &&
        !/(结果|公示|名单|递补|增补|拟录取|录取结果)/.test(rawQuery)
    );
}

function queryAsksOutcomeDocument(rawQuery: string): boolean {
    return /(结果|公示|名单|递补|增补|拟录取|录取结果)/.test(rawQuery);
}

function buildEvidenceCoverageRequirement(
    rawQuery: string,
): EvidenceCoverageRequirement | undefined {
    if (
        /夏令营/.test(rawQuery) &&
        /(录取优惠|优惠|优秀营员)/.test(rawQuery)
    ) {
        return {
            label: "summer_camp_benefit",
            requiredGroups: [["夏令营"], ["优惠", "优秀营员"]],
            requireIntentAlignment: true,
        };
    }

    if (
        /夏令营/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)
    ) {
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

    if (
        /调剂/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)
    ) {
        return {
            label: "adjustment_apply",
            requiredGroups: [["调剂"], ["报名", "申请"]],
            requireIntentAlignment: true,
        };
    }

    if (
        /外语类保送生/.test(rawQuery) &&
        /(招生简章|简章|招生章程|章程)/.test(rawQuery)
    ) {
        return {
            label: "foreign_language_recommend_brochure",
            requiredGroups: [
                ["外语类", "保送生"],
                ["招生简章", "简章", "招生章程", "章程"],
            ],
        };
    }

    if (
        /综合评价/.test(rawQuery) &&
        /(招生简章|简章|招生章程|章程)/.test(rawQuery)
    ) {
        return {
            label: "comprehensive_evaluation_brochure",
            requiredGroups: [["综合评价"], ["招生简章", "简章", "招生章程", "章程"]],
        };
    }

    if (
        /选课/.test(rawQuery) &&
        /补退选/.test(rawQuery) &&
        /(时间|什么时候|何时|截止|流程|步骤)/.test(rawQuery)
    ) {
        return {
            label: "course_add_drop",
            requiredGroups: [["选课"], ["补退选", "退选"]],
        };
    }

    if (
        /转专业/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)
    ) {
        return {
            label: "major_transfer",
            requiredGroups: [["转专业"], ["报名", "申请"]],
        };
    }

    if (
        /免修/.test(rawQuery) &&
        /(报名|申请|流程|步骤|怎么办|如何)/.test(rawQuery)
    ) {
        return {
            label: "course_exemption",
            requiredGroups: [["免修"], ["报名", "申请"]],
        };
    }

    return undefined;
}

function topDocumentSatisfiesEvidenceRequirement(
    sortedRanking: SearchResult[],
    docEvidenceStatsMap: Map<string, ScopeSpecificityStats>,
    requirement: EvidenceCoverageRequirement,
): boolean {
    const topDoc = sortedRanking[0];
    if (!topDoc) {
        return false;
    }

    const stats = docEvidenceStatsMap.get(topDoc.otid);
    if (!stats) {
        return false;
    }

    return requirement.requiredGroups.every((group) =>
        group.some((term) => (stats.termTf[term] || 0) > 0),
    );
}

function shouldRejectForMissingInDomainEvidence(params: {
    rawQuery: string;
    queryIntent?: ParsedQueryIntent;
    sortedRanking: SearchResult[];
    docEvidenceStatsMap: Map<string, ScopeSpecificityStats>;
    otidMap: Record<string, AggregatedDocScores>;
}): { shouldReject: boolean; label?: string } {
    const requirement = buildEvidenceCoverageRequirement(params.rawQuery);
    if (!requirement) {
        return { shouldReject: false };
    }

    const topDoc = params.sortedRanking[0];
    if (
        requirement.requireIntentAlignment &&
        topDoc &&
        (params.queryIntent?.intentIds.length || 0) > 0
    ) {
        const topDocIntentIds =
            params.otidMap[topDoc.otid]?.intent_ids ||
            params.otidMap[topDoc.otid]?.subtopic_ids ||
            [];
        if (!hasAnyOverlap(params.queryIntent?.intentIds || [], topDocIntentIds)) {
            return {
                shouldReject: true,
                label: `${requirement.label}_intent_mismatch`,
            };
        }
    }

    if (
        topDocumentSatisfiesEvidenceRequirement(
            params.sortedRanking,
            params.docEvidenceStatsMap,
            requirement,
        )
    ) {
        return { shouldReject: false };
    }

    return {
        shouldReject: true,
        label: requirement.label,
    };
}

function createQueryIntentContext(
    queryIntent?: ParsedQueryIntent,
    queryWords: string[] = [],
): QueryIntentContext {
    const years = queryIntent?.years || [];
    const intentIds = queryIntent?.intentIds || [];
    const rawQuery = queryIntent?.rawQuery || "";
    const querySpecificityTerms = extractQuerySpecificityTerms(queryWords);
    const discourageUnexpectedSpecificity =
        querySpecificityTerms.length === 0 &&
        Boolean(queryIntent?.preferLatestStrong) &&
        BROAD_LATEST_SCOPE_CUE_PATTERN.test(rawQuery);

    return {
        rawQuery,
        years,
        months: queryIntent?.months || [],
        hasExplicitYear: years.length > 0,
        hasExplicitMonth: (queryIntent?.months || []).length > 0,
        hasHistoricalHint: Boolean(queryIntent?.signals.hasHistoricalHint),
        hasStrongDetailAnchor: Boolean(queryIntent?.signals.hasStrongDetailAnchor),
        topicIds: queryIntent?.topicIds || [],
        intentIds,
        relatedIntentIds: getRelatedIntentTypes(intentIds),
        degreeLevels: queryIntent?.degreeLevels || [],
        eventTypes: queryIntent?.eventTypes || [],
        hasPostOutcomeCondition: hasPostOutcomeConditionCue(
            rawQuery,
        ),
        asksRuleDocument: queryAsksRuleDocument(rawQuery),
        asksOutcomeDocument: queryAsksOutcomeDocument(rawQuery),
        preferLatest: Boolean(queryIntent?.preferLatest),
        preferLatestStrong: Boolean(queryIntent?.preferLatestStrong),
        querySpecificityTerms,
        discourageUnexpectedSpecificity,
    };
}

function getTimestampMonth(timestamp?: number): number | undefined {
    if (typeof timestamp !== "number" || !Number.isFinite(timestamp)) {
        return undefined;
    }
    const date = new Date(timestamp * 1000);
    if (Number.isNaN(date.getTime())) {
        return undefined;
    }
    return date.getUTCMonth() + 1;
}

function getTimestampYear(timestamp?: number): number | undefined {
    if (typeof timestamp !== "number" || !Number.isFinite(timestamp)) {
        return undefined;
    }
    const date = new Date(timestamp * 1000);
    if (Number.isNaN(date.getTime())) {
        return undefined;
    }
    return date.getUTCFullYear();
}

function getDocQuerySignals(
    otid: string,
    scores: AggregatedDocScores,
    intentContext: QueryIntentContext,
    yearHitMap: Map<string, boolean>,
): DocQuerySignals {
    const docMonth = getTimestampMonth(scores.timestamp);
    const docPublishYear = getTimestampYear(scores.timestamp);
    const hasSuspiciousStructuredYear =
        scores.target_year !== undefined &&
        docPublishYear !== undefined &&
        Math.abs(scores.target_year - docPublishYear) >= 2;
    return {
        hasStructuredYearMatch:
            intentContext.hasExplicitYear &&
            scores.target_year !== undefined &&
            intentContext.years.includes(scores.target_year),
        hasLexicalYearMatch: yearHitMap.get(otid) === true,
        hasPublishYearMatch:
            intentContext.hasExplicitYear &&
            docPublishYear !== undefined &&
            intentContext.years.includes(docPublishYear),
        hasSuspiciousStructuredYear,
        docPublishYear,
        hasStructuredMonthMatch:
            intentContext.hasExplicitYear &&
            intentContext.hasExplicitMonth &&
            docMonth !== undefined &&
            intentContext.months.includes(docMonth),
        docMonth,
    };
}

function shouldSkipForExplicitYear(
    scores: AggregatedDocScores,
    intentContext: QueryIntentContext,
    signals: DocQuerySignals,
): boolean {
    if (!intentContext.hasExplicitYear) {
        return false;
    }

    if (
        scores.target_year !== undefined &&
        !signals.hasStructuredYearMatch &&
        !(signals.hasLexicalYearMatch && signals.hasSuspiciousStructuredYear)
    ) {
        return true;
    }

    return (
        scores.target_year === undefined &&
        !signals.hasLexicalYearMatch &&
        !signals.hasPublishYearMatch
    );
}

function computeBaseScore(
    scores: AggregatedDocScores,
    weights: typeof DEFAULT_WEIGHTS,
    options?: {
        kpAggregationMode?: KPAggregationMode;
        kpTopN?: number;
        kpTailWeight?: number;
    },
): number {
    const kpAggregationMode = options?.kpAggregationMode || "max";
    const kpTopN = Math.max(1, options?.kpTopN || 3);
    const kpTailWeight = options?.kpTailWeight ?? 0.35;
    const topKpScores =
        scores.kp_scores && scores.kp_scores.length > 0
            ? scores.kp_scores.slice(0, kpTopN)
            : scores.max_kp > 0
              ? [scores.max_kp]
              : [];
    const aggregatedKpScore =
        kpAggregationMode === "max_plus_topn" && topKpScores.length > 1
            ? topKpScores[0] +
              topKpScores.slice(1).reduce((sum, item) => sum + item, 0) *
                  kpTailWeight
            : topKpScores[0] || 0;

    const weightedQ = scores.max_q * weights.Q;
    const weightedKP = aggregatedKpScore * weights.KP;
    const weightedOT = scores.ot_score * weights.OT;

    const maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
    const unionBonus =
        weightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;

    return maxComponent + unionBonus;
}

type QueryRoleSignals = {
    asksTime: boolean;
    asksCondition: boolean;
    asksPostOutcomeCondition: boolean;
    asksMaterials: boolean;
    asksProcedure: boolean;
    asksExamContent: boolean;
    asksAnnouncementPeriod: boolean;
    asksApplicationStage: boolean;
    mentionsThesis: boolean;
    mentionsPrintedDocument: boolean;
    mentionsCollectionOrArchive: boolean;
    mentionsReviewOrReissue: boolean;
};

function hasKpRoleTag(
    candidate: Pick<KPCandidate, "kp_role_tags"> | undefined,
    tag: string,
): boolean {
    return candidate?.kp_role_tags?.includes(tag) === true;
}

function deriveQueryRoleSignals(
    rawQuery: string,
    queryScopeHint?: string,
): QueryRoleSignals {
    return {
        asksTime:
            /什么时候|何时|哪几天|几号|截止|到账|时间|公示期/.test(
                rawQuery,
            ) || queryScopeHint === "time_location",
        asksCondition:
            /条件|满足|资格/.test(rawQuery) ||
            queryScopeHint === "eligibility_condition",
        asksPostOutcomeCondition: hasPostOutcomeConditionCue(rawQuery),
        asksMaterials: /材料|扫描件|电子版|邮箱|mail|提交|携带/i.test(rawQuery),
        asksProcedure: /怎么办|怎么处理|不通过|补交|补充|流程|步骤|报到|候考|地点|现场|资格审查/.test(
            rawQuery,
        ),
        asksExamContent:
            /考什么|考哪些|考试内容|考试科目|科目|题型|综合能力|分值|权重|占比|比例/.test(
                rawQuery,
            ),
        asksAnnouncementPeriod: /公示期|哪几天/.test(rawQuery),
        asksApplicationStage:
            /申请|报名|确认|提交/.test(rawQuery) &&
            !/通过后|答辩通过|审批后|获得学位/.test(rawQuery),
        mentionsThesis: /论文/.test(rawQuery),
        mentionsPrintedDocument: /准考证|打印|纸质/.test(rawQuery),
        mentionsCollectionOrArchive: /领取|证书|档案/.test(rawQuery),
        mentionsReviewOrReissue: /资格|评审|补发/.test(rawQuery),
    };
}

function computeKpRoleBonus(
    candidate: KPCandidate,
    signals: QueryRoleSignals,
    rawQuery: string,
): number {
    let bonus = 0;
    const asksOperationalEvidence =
        signals.asksCondition ||
        signals.asksMaterials ||
        signals.asksProcedure ||
        signals.asksTime ||
        /资格审查|报到|候考|地点|现场|安排/.test(rawQuery);
    const asksRuleDocument =
        /(招生简章|简章|招生章程|章程|实施细则|细则|实施办法|办法|接收办法|录取方案|方案)/.test(
            rawQuery,
        ) && !/(结果|公示|名单|递补|增补|拟录取|录取结果)/.test(rawQuery);

    if (signals.asksTime) {
        if (
            hasKpRoleTag(candidate, "arrival")
            || hasKpRoleTag(candidate, "deadline")
            || hasKpRoleTag(candidate, "announcement_period")
            || hasKpRoleTag(candidate, "schedule")
        ) {
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
            bonus += 1.0;
        }
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 0.35;
        }
        if (
            hasKpRoleTag(candidate, "materials")
            && hasKpRoleTag(candidate, "email")
        ) {
            bonus += 0.9;
        }
        if (/申请|答辩/.test(rawQuery) && hasKpRoleTag(candidate, "application_stage")) {
            bonus += 0.9;
        }
        if (
            !signals.mentionsThesis
            && (hasKpRoleTag(candidate, "post_outcome")
                || hasKpRoleTag(candidate, "thesis"))
        ) {
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

    if (
        signals.mentionsThesis &&
        signals.asksApplicationStage &&
        !signals.asksCondition
    ) {
        if (hasKpRoleTag(candidate, "condition")) {
            bonus -= 0.55;
        }
        if (hasKpRoleTag(candidate, "application_stage")) {
            bonus += 0.3;
        }
    }

    if (signals.asksProcedure) {
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 1.35;
        }
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.6;
        }
        if (
            hasKpRoleTag(candidate, "reminder")
            || hasKpRoleTag(candidate, "background")
        ) {
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

    if (/(权重|占比|比例|分值)/.test(rawQuery)) {
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.45;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.4;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.6;
        }
    }

    if (/资格审查|报到|候考|地点|现场|安排/.test(rawQuery)) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 0.55;
        }
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 0.85;
        }
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.7;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.7;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.6;
        }
    }

    if (asksRuleDocument) {
        if (
            hasKpRoleTag(candidate, "condition") ||
            hasKpRoleTag(candidate, "materials") ||
            hasKpRoleTag(candidate, "procedure") ||
            hasKpRoleTag(candidate, "application_stage")
        ) {
            bonus += 0.25;
        }
        if (
            hasKpRoleTag(candidate, "background") ||
            hasKpRoleTag(candidate, "publish") ||
            hasKpRoleTag(candidate, "post_outcome")
        ) {
            bonus -= 0.55;
        }
    }

    if (signals.asksAnnouncementPeriod) {
        if (hasKpRoleTag(candidate, "announcement_period")) {
            bonus += 1.2;
        }
        if (
            hasKpRoleTag(candidate, "publish")
            && !hasKpRoleTag(candidate, "announcement_period")
        ) {
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

    if (
        !signals.asksAnnouncementPeriod &&
        !signals.asksPostOutcomeCondition &&
        hasKpRoleTag(candidate, "publish")
    ) {
        bonus -= 0.2;
    }

    if (
        !/到账|发放|补发/.test(rawQuery) &&
        hasKpRoleTag(candidate, "distribution")
    ) {
        bonus -= 0.35;
    }

    if (hasKpRoleTag(candidate, "background")) {
        bonus -= 0.15;
    }

    if (asksOperationalEvidence) {
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.45;
        }
        if (hasKpRoleTag(candidate, "publish")) {
            bonus -= 0.35;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.6;
        }
    }

    return bonus;
}

function rerankKpCandidatesByRole(params: {
    kpCandidates: readonly KPCandidate[];
    bestKpid?: string;
    rawQuery: string;
    queryScopeHint?: string;
    mode?: KPRoleRerankMode;
}): {
    bestKpid?: string;
    orderedCandidates: KPCandidate[];
    docScoreDelta: number;
} {
    const {
        kpCandidates,
        bestKpid,
        rawQuery,
        queryScopeHint,
        mode = "off",
    } = params;

    const orderedCandidates = [...kpCandidates];
    if (mode !== "feature" || orderedCandidates.length === 0) {
        return {
            bestKpid,
            orderedCandidates,
            docScoreDelta: 0,
        };
    }

    const rerankWindow = orderedCandidates.slice(
        0,
        DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    );
    const rawTopScore = rerankWindow[0]?.score ?? Number.NEGATIVE_INFINITY;
    const signals = deriveQueryRoleSignals(rawQuery, queryScopeHint);
    const reranked = rerankWindow
        .map((candidate) => ({
            candidate,
            rerankedScore:
                candidate.score + computeKpRoleBonus(candidate, signals, rawQuery),
        }))
        .sort((a, b) => b.rerankedScore - a.rerankedScore);

    const topCandidate = reranked[0];
    return {
        bestKpid: topCandidate?.candidate.kpid || bestKpid,
        orderedCandidates: [
            ...reranked.map((item) => item.candidate),
            ...orderedCandidates.slice(rerankWindow.length),
        ],
        docScoreDelta:
            Number.isFinite(rawTopScore) && Number.isFinite(topCandidate?.rerankedScore)
                ? Math.max(0, topCandidate.rerankedScore - rawTopScore)
                : 0,
    };
}

function applyLexicalBonusBoost(boost: number, lexicalBonus: number): number {
    if (lexicalBonus <= 0) {
        return boost;
    }

    return boost * (1 + Math.log1p(lexicalBonus) / 4);
}

function applyYearConstraintBoost(
    boost: number,
    queryYearWordIds: number[] | undefined,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    signals: DocQuerySignals,
): number {
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

    if (
        !signals.hasStructuredYearMatch &&
        !signals.hasLexicalYearMatch &&
        !signals.hasPublishYearMatch
    ) {
        return boost * 0.12;
    }

    return boost;
}

function applyMonthConstraintBoost(
    boost: number,
    intentContext: QueryIntentContext,
    signals: DocQuerySignals,
): number {
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

function applyIntentBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (intentContext.intentIds.length === 0) {
        return boost;
    }

    let nextBoost = boost;

    if (hasIntentMatch(intentContext.intentIds, scores.intent_ids)) {
        nextBoost *= 1.12;
    } else if (hasIntentConflict(intentContext.intentIds, scores.intent_ids)) {
        nextBoost *= 0.85;
    } else if (
        intentContext.relatedIntentIds.length > 0 &&
        hasAnyOverlap(intentContext.relatedIntentIds, scores.intent_ids)
    ) {
        nextBoost *= 1.04;
    }

    return nextBoost;
}

function applyTopicCoverageBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    signals: DocQuerySignals,
): number {
    if (intentContext.topicIds.length === 0) {
        return boost;
    }

    const docTopicIds = dedupe(getCoverageComparableTopicIds(scores));
    if (docTopicIds.length > 0) {
        if (hasAnyOverlap(intentContext.topicIds, docTopicIds)) {
            return boost * 1.08;
        }
        return boost * 0.9;
    }

    if (
        scores.weak_topic_ids &&
        hasAnyOverlap(intentContext.topicIds, scores.weak_topic_ids)
    ) {
        return boost * 1.02;
    }

    const hasStructuredFallbackEvidence =
        intentContext.querySpecificityTerms.length > 0 &&
        (signals.hasStructuredYearMatch ||
            signals.hasLexicalYearMatch ||
            signals.hasPublishYearMatch) &&
        (hasAnyOverlap(intentContext.degreeLevels, scores.degree_levels) ||
            (scores.event_types?.length || 0) > 0);
    if (hasStructuredFallbackEvidence) {
        return boost * 0.98;
    }

    return boost * 0.84;
}

function applyDegreeBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (intentContext.degreeLevels.length === 0) {
        return boost;
    }

    if (hasAnyOverlap(intentContext.degreeLevels, scores.degree_levels)) {
        return boost * 1.05;
    }

    if ((scores.degree_levels?.length || 0) > 0) {
        return boost * 0.93;
    }

    return boost;
}

function applyEventBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (hasAnyOverlap(intentContext.eventTypes, scores.event_types)) {
        boost *= 1.05;
    } else if (
        intentContext.eventTypes.length > 0 &&
        (scores.event_types?.length || 0) > 0
    ) {
        boost *= EVENT_TYPE_MISMATCH_PENALTY;
    }

    if (
        intentContext.hasPostOutcomeCondition &&
        (scores.event_types?.length || 0) > 0
    ) {
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

    const asksConditionOnly =
        /条件|满足|资格/.test(intentContext.rawQuery) &&
        !/怎么|流程|报名|操作|步骤/.test(intentContext.rawQuery) &&
        !/初试|复试|成绩|分数/.test(intentContext.rawQuery);
    if (
        asksConditionOnly &&
        hasAnyOverlap(["复试通知"], scores.event_types)
    ) {
        boost *= 0.35;
    }

    if (intentContext.asksRuleDocument) {
        const asksBrochure =
            /(招生简章|简章|招生章程|章程)/.test(intentContext.rawQuery);
        const asksImplementationRule =
            /(实施细则|细则|实施办法|办法|接收办法|录取方案|方案)/.test(
                intentContext.rawQuery,
            );

        if (asksBrochure && hasAnyOverlap(["招生章程"], scores.event_types)) {
            boost *= 1.22;
        }

        if (
            asksImplementationRule &&
            hasAnyOverlap(
                ["推免实施办法", "资格要求", "材料提交", "考试安排", "复试通知"],
                scores.event_types,
            )
        ) {
            boost *= 1.14;
        }

        if (hasAnyOverlap(["录取公示", "推免资格公示"], scores.event_types)) {
            boost *= 0.62;
        }

        if (asksBrochure && hasAnyOverlap(["复试通知"], scores.event_types)) {
            boost *= 0.78;
        }
    }

    if (intentContext.asksOutcomeDocument) {
        if (hasAnyOverlap(["录取公示", "推免资格公示"], scores.event_types)) {
            boost *= 1.14;
        }
        if (
            hasAnyOverlap(
                ["招生章程", "推免实施办法", "资格要求", "材料提交"],
                scores.event_types,
            )
        ) {
            boost *= 0.86;
        }
    }

    if (
        /调剂/.test(intentContext.rawQuery) &&
        !intentContext.asksOutcomeDocument &&
        hasAnyOverlap(["录取公示"], scores.event_types)
    ) {
        boost *= 0.82;
    }

    return boost;
}

function applyLatestYearBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    latestTargetYear?: number,
): number {
    if (
        !intentContext.preferLatest ||
        latestTargetYear === undefined ||
        scores.target_year === undefined
    ) {
        return boost;
    }

    const yearGap = Math.max(0, latestTargetYear - scores.target_year);
    return boost * Math.pow(LATEST_YEAR_BOOST_BASE, yearGap);
}

function applyExplicitYearAlignmentBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    signals: DocQuerySignals,
): number {
    if (!intentContext.hasExplicitYear) {
        return boost;
    }

    if (signals.hasStructuredYearMatch && signals.hasPublishYearMatch) {
        return boost * 1.1;
    }

    if (signals.hasStructuredYearMatch && !signals.hasSuspiciousStructuredYear) {
        return boost * 1.06;
    }

    if (signals.hasPublishYearMatch) {
        return boost * 1.04;
    }

    if (
        signals.hasLexicalYearMatch &&
        !signals.hasStructuredYearMatch &&
        !signals.hasPublishYearMatch
    ) {
        return boost * 0.9;
    }

    if (
        scores.target_year !== undefined ||
        signals.docPublishYear !== undefined
    ) {
        return boost * 0.78;
    }

    return boost;
}

function applyLatestTimestampBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    latestTimestamp?: number,
): number {
    if (!intentContext.preferLatestStrong || latestTimestamp === undefined) {
        return boost;
    }

    if (scores.timestamp === undefined) {
        return boost * 0.82;
    }

    const gapSeconds = Math.max(0, latestTimestamp - scores.timestamp);
    if (gapSeconds <= 0) {
        return boost;
    }

    const gapMonths = gapSeconds / (60 * 60 * 24 * 30);
    return boost * Math.pow(LATEST_POLICY_TIMESTAMP_BOOST_BASE, gapMonths);
}

function shouldPreferCurrentProcessVersion(
    intentContext: QueryIntentContext,
): boolean {
    if (intentContext.hasExplicitYear || intentContext.hasHistoricalHint) {
        return false;
    }

    if (intentContext.hasPostOutcomeCondition) {
        return false;
    }

    const hasProcessCue =
        /报名|确认|提交|材料|申请|答辩|流程|步骤|怎么办|如何|接下来/.test(
            intentContext.rawQuery,
        );
    if (!hasProcessCue) {
        return false;
    }

    return (
        intentContext.hasStrongDetailAnchor ||
        /流程|步骤|怎么办|如何|接下来/.test(intentContext.rawQuery)
    );
}

function applyCurrentProcessTimestampBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    latestTimestamp?: number,
): number {
    if (
        !shouldPreferCurrentProcessVersion(intentContext) ||
        latestTimestamp === undefined
    ) {
        return boost;
    }

    if (scores.timestamp === undefined) {
        return boost * 0.9;
    }

    const gapSeconds = Math.max(0, latestTimestamp - scores.timestamp);
    if (gapSeconds <= 0) {
        return boost;
    }

    const gapMonths = gapSeconds / (60 * 60 * 24 * 30);
    const hasProcessLikeEvent = hasAnyOverlap(
        [...CURRENT_PROCESS_EVENT_TYPES],
        scores.event_types,
    );
    const decayBase = hasProcessLikeEvent
        ? CURRENT_PROCESS_TIMESTAMP_BOOST_BASE
        : 0.992;

    return boost * Math.pow(decayBase, gapMonths);
}

function applyScopeSpecificityBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scopeSpecificityStats?: ScopeSpecificityStats,
): number {
    if (!scopeSpecificityStats) {
        return boost;
    }

    const querySpecificityTerms = intentContext.querySpecificityTerms;
    if (querySpecificityTerms.length > 0) {
        const matchedTf = getMatchedSpecificityTf(
            querySpecificityTerms,
            scopeSpecificityStats,
        );
        const matchedTerms = querySpecificityTerms.filter(
            (term) => (scopeSpecificityStats.termTf[term] || 0) > 0,
        ).length;

        if (matchedTerms === 0 || matchedTf === 0) {
            return boost * 0.45;
        }

        const coverageRatio = matchedTerms / querySpecificityTerms.length;
        const focusRatio =
            matchedTf / Math.max(matchedTf, scopeSpecificityStats.totalTf, 1);

        let nextBoost = boost;
        nextBoost *= 0.88 + coverageRatio * 0.24;
        nextBoost *= 0.55 + focusRatio * 0.95;
        return nextBoost;
    }

    if (!intentContext.discourageUnexpectedSpecificity) {
        return boost;
    }

    const unexpectedTf = Object.entries(scopeSpecificityStats.termTf).reduce(
        (sum, [term, tf]) =>
            intentContext.querySpecificityTerms.includes(term) ? sum : sum + tf,
        0,
    );
    if (unexpectedTf <= 0) {
        return boost;
    }

    return boost * Math.max(0.72, 1 - Math.log1p(unexpectedTf) / 10);
}

function applySpecificityLocalFreshnessBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    scopeSpecificityStats?: ScopeSpecificityStats,
    latestFocusedSpecificityTimestamp?: number,
): number {
    if (
        !intentContext.preferLatestStrong ||
        intentContext.querySpecificityTerms.length === 0 ||
        latestFocusedSpecificityTimestamp === undefined ||
        scores.timestamp === undefined
    ) {
        return boost;
    }

    if (!/怎么|流程|报名|操作|步骤/.test(intentContext.rawQuery)) {
        return boost;
    }

    const matchedTf = getMatchedSpecificityTf(
        intentContext.querySpecificityTerms,
        scopeSpecificityStats,
    );
    if (matchedTf < 10) {
        return boost;
    }

    const gapSeconds = Math.max(
        0,
        latestFocusedSpecificityTimestamp - scores.timestamp,
    );
    if (gapSeconds <= 0) {
        return boost;
    }

    const gapMonths = gapSeconds / (60 * 60 * 24 * 30);
    return boost * Math.pow(0.75, gapMonths);
}

function hasAnyRoleEvidence(
    candidates: readonly KPCandidate[],
    tags: readonly string[],
): boolean {
    return candidates.some((candidate) =>
        tags.some((tag) => hasKpRoleTag(candidate, tag)),
    );
}

function applyMultiEvidenceKpBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (!scores.kp_candidates || scores.kp_candidates.length === 0) {
        return boost;
    }

    const signals = deriveQueryRoleSignals(intentContext.rawQuery);
    const window = scores.kp_candidates.slice(0, DEFAULT_KP_ROLE_CANDIDATE_LIMIT);
    const matchedCondition =
        signals.asksCondition &&
        hasAnyRoleEvidence(window, ["condition", "deadline"]);
    const matchedMaterials =
        signals.asksMaterials &&
        hasAnyRoleEvidence(window, ["materials", "email"]);
    const matchedProcedure =
        signals.asksProcedure &&
        hasAnyRoleEvidence(window, ["procedure", "application_stage", "schedule"]);
    const matchedTime =
        signals.asksTime &&
        hasAnyRoleEvidence(window, [
            "schedule",
            "arrival",
            "deadline",
            "announcement_period",
            "time_expression",
        ]);
    const matchedExam =
        signals.asksExamContent && hasAnyRoleEvidence(window, ["schedule"]);

    const matchedGroupCount = [
        matchedCondition,
        matchedMaterials,
        matchedProcedure,
        matchedTime,
        matchedExam,
    ].filter(Boolean).length;

    let nextBoost = boost;
    if (matchedGroupCount >= 2) {
        nextBoost *= 1 + Math.min(0.12, matchedGroupCount * 0.03);
    }

    if (matchedCondition && matchedMaterials) {
        nextBoost *= 1.05;
    }

    if (matchedMaterials && matchedProcedure) {
        nextBoost *= 1.06;
    }

    if (matchedTime && matchedProcedure) {
        nextBoost *= 1.04;
    }

    const topCandidate = window[0];
    const asksOperationalEvidence =
        signals.asksCondition ||
        signals.asksMaterials ||
        signals.asksProcedure ||
        signals.asksTime;
    const topCandidateIsBackgroundLike =
        hasKpRoleTag(topCandidate, "background") ||
        hasKpRoleTag(topCandidate, "publish") ||
        hasKpRoleTag(topCandidate, "post_outcome");
    const hasOperationalAlternative = hasAnyRoleEvidence(
        window.slice(1),
        [
            "condition",
            "materials",
            "email",
            "procedure",
            "application_stage",
            "schedule",
            "arrival",
            "deadline",
            "announcement_period",
        ],
    );

    if (
        asksOperationalEvidence &&
        topCandidateIsBackgroundLike &&
        hasOperationalAlternative
    ) {
        nextBoost *= 0.96;
    }

    return nextBoost;
}

function computeBoostMultiplier(params: {
    otid: string;
    scores: AggregatedDocScores;
    lexicalBonusMap: Map<string, number>;
    yearHitMap: Map<string, boolean>;
    queryYearWordIds?: number[];
    intentContext: QueryIntentContext;
    latestTargetYear?: number;
    latestTimestamp?: number;
    scopeSpecificityStats?: ScopeSpecificityStats;
    latestFocusedSpecificityTimestamp?: number;
}): number {
    const {
        otid,
        scores,
        lexicalBonusMap,
        yearHitMap,
        queryYearWordIds,
        intentContext,
        latestTargetYear,
        latestTimestamp,
        scopeSpecificityStats,
        latestFocusedSpecificityTimestamp,
    } = params;
    const signals = getDocQuerySignals(otid, scores, intentContext, yearHitMap);
    const lexicalBonus = lexicalBonusMap.get(otid) || 0;

    let boost = 1.0;
    boost = applyLexicalBonusBoost(boost, lexicalBonus);
    boost = applyYearConstraintBoost(
        boost,
        queryYearWordIds,
        intentContext,
        scores,
        signals,
    );
    boost = applyMonthConstraintBoost(
        boost,
        intentContext,
        signals,
    );
    boost = applyIntentBoost(boost, intentContext, scores);
    boost = applyTopicCoverageBoost(boost, intentContext, scores, signals);
    boost = applyDegreeBoost(boost, intentContext, scores);
    boost = applyEventBoost(boost, intentContext, scores);
    boost = applyMultiEvidenceKpBoost(boost, intentContext, scores);
    boost = applyLatestYearBoost(
        boost,
        intentContext,
        scores,
        latestTargetYear,
    );
    boost = applyExplicitYearAlignmentBoost(
        boost,
        intentContext,
        scores,
        signals,
    );
    boost = applyLatestTimestampBoost(
        boost,
        intentContext,
        scores,
        latestTimestamp,
    );
    boost = applyCurrentProcessTimestampBoost(
        boost,
        intentContext,
        scores,
        latestTimestamp,
    );
    boost = applyScopeSpecificityBoost(
        boost,
        intentContext,
        scopeSpecificityStats,
    );
    boost = applySpecificityLocalFreshnessBoost(
        boost,
        intentContext,
        scores,
        scopeSpecificityStats,
        latestFocusedSpecificityTimestamp,
    );

    return boost;
}

export function extractRetrievalSignals(
    sortedRanking: SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
): RetrievalSignals {
    const consistencyWindow = sortedRanking.slice(0, 10);
    const topicHistogram = new Map<string, number>();
    let labeledCount = 0;

    for (const item of consistencyWindow) {
        const scores = otidMap[item.otid];
        const topicIds = dedupe(getCoverageComparableTopicIds(scores));
        if (topicIds.length === 0) continue;

        labeledCount += 1;
        topicIds.forEach((topicId) => {
            topicHistogram.set(topicId, (topicHistogram.get(topicId) || 0) + 1);
        });
    }

    const top1Score = sortedRanking[0]?.score || 0;
    const top2Score = sortedRanking[1]?.score ?? top1Score;
    const top5Score = sortedRanking[4]?.score ?? sortedRanking.at(-1)?.score ?? top1Score;
    const dominantCount =
        topicHistogram.size > 0 ? Math.max(...topicHistogram.values()) : 0;
    const dominantRatio =
        consistencyWindow.length > 0 ? dominantCount / consistencyWindow.length : 0;

    return {
        candidateCount: sortedRanking.length,
        top1Score,
        top1Top2Gap: top1Score - top2Score,
        top1Top5Gap: top1Score - top5Score,
        distinctTopicCount: topicHistogram.size,
        dominantTopicCount: dominantCount,
        dominantTopicRatio: dominantRatio,
        labeledTopicCount: labeledCount,
    };
}

export function classifyResponseMode(
    querySignals: QuerySignals,
    retrievalSignals: RetrievalSignals,
): ResponseDecision {
    const scores: ResponseModeScores = {
        direct_answer: 0.5,
        clarify_or_route: 0,
        reject: 0,
    };
    const reasons = new Set<string>();

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

    if (
        querySignals.hasGenericNextStep &&
        !querySignals.hasResultState &&
        !querySignals.hasEntryLikeAnchor &&
        !querySignals.hasStrongDetailAnchor
    ) {
        scores.reject += 1.6;
        scores.clarify_or_route -= 0.45;
        reasons.add("empty_next_step_without_state");
    }

    if (
        querySignals.hasResultState &&
        querySignals.hasGenericNextStep &&
        !querySignals.hasStrongDetailAnchor
    ) {
        scores.clarify_or_route += 1.35;
        scores.direct_answer -= 0.3;
        reasons.add("result_state_needs_clarification");
    }

    if (
        querySignals.hasResultState &&
        querySignals.hasGenericNextStep &&
        querySignals.hasStrongDetailAnchor
    ) {
        scores.direct_answer += 1.0;
        reasons.add("detail_anchor_overrides_generic_state");
    }

    if (
        querySignals.hasExplicitYear &&
        !querySignals.hasGenericNextStep &&
        (querySignals.hasEntryLikeAnchor || querySignals.hasResultState)
    ) {
        scores.direct_answer += 0.8;
        scores.clarify_or_route -= 0.25;
        reasons.add("time_anchor_supports_direct_answer");
    }

    if (
        !querySignals.hasExplicitTopicOrIntent &&
        !querySignals.hasStrongDetailAnchor &&
        !querySignals.hasEntryLikeAnchor
    ) {
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

    if (
        retrievalSignals.distinctTopicCount >= 3 &&
        retrievalSignals.dominantTopicRatio < 0.45
    ) {
        scores.reject += 1.15;
        scores.direct_answer -= 0.35;
        reasons.add("low_topic_consistency");
    }

    if (retrievalSignals.top1Top2Gap >= 0.12) {
        scores.direct_answer += 0.35;
        reasons.add("stable_top1_gap");
    }

    if (
        retrievalSignals.labeledTopicCount >= 3 &&
        retrievalSignals.distinctTopicCount <= 2 &&
        retrievalSignals.dominantTopicRatio >= 0.5
    ) {
        scores.direct_answer += 0.35;
        reasons.add("stable_topic_cluster");
    }

    const rankedModes = Object.entries(scores)
        .map(([mode, score]) => [mode as ResponseMode, score] as const)
        .sort((a, b) => b[1] - a[1]);
    const [mode, topScore] = rankedModes[0];
    const secondScore = rankedModes[1]?.[1] ?? topScore;
    const confidence = Math.max(
        0.55,
        Math.min(0.98, 0.62 + (topScore - secondScore) * 0.14),
    );

    return {
        mode,
        confidence,
        reason:
            Array.from(reasons).slice(0, 3).join("+") ||
            "scored_response_mode",
        preferLatestWithinTopic:
            querySignals.hasLatestPolicyState && !querySignals.hasExplicitYear,
        useWeakMatches: mode === "clarify_or_route",
    };
}

export function searchAndRank(params: {
    queryVector: Float32Array;
    querySparse?: Record<number, number>;
    queryWords?: string[];
    queryYearWordIds?: number[];
    queryIntent?: ParsedQueryIntent;
    queryScopeHint?: string;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    weights?: typeof DEFAULT_WEIGHTS;
    candidateIndices?: readonly number[];
    scopeSpecificityWordIdToTerm?: Map<number, string>;
    directAnswerEvidenceWordIdToTerm?: Map<number, string>;
    topHybridLimit?: number;
    kpAggregationMode?: KPAggregationMode;
    kpTopN?: number;
    kpTailWeight?: number;
    lexicalBonusMode?: LexicalBonusMode;
    qLexicalMultiplier?: number;
    kpLexicalMultiplier?: number;
    otLexicalMultiplier?: number;
    denseScoreOverrides?: ReadonlyMap<string, number>;
    kpRoleRerankMode?: KPRoleRerankMode;
    kpRoleDocWeight?: number;
    otDenseScoreOverrides?: ReadonlyMap<string, number>;
}): SearchRankOutput {
    const {
        queryVector,
        querySparse,
        queryWords = [],
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp: _currentTimestamp,
        bm25Stats,
        weights = DEFAULT_WEIGHTS,
        queryYearWordIds,
        queryIntent,
        queryScopeHint,
        candidateIndices,
        scopeSpecificityWordIdToTerm,
        directAnswerEvidenceWordIdToTerm,
        topHybridLimit = 1000,
        kpAggregationMode = "max",
        kpTopN = 3,
        kpTailWeight = 0.35,
        lexicalBonusMode = "sum",
        qLexicalMultiplier = 1.5,
        kpLexicalMultiplier = 1.2,
        otLexicalMultiplier = 1.0,
        denseScoreOverrides,
        kpRoleRerankMode = "off",
        kpRoleDocWeight = DEFAULT_KP_ROLE_DOC_WEIGHT,
        otDenseScoreOverrides,
    } = params;
    const safeQLexicalMultiplier = Number.isFinite(qLexicalMultiplier)
        ? qLexicalMultiplier
        : 1.5;
    const safeKpLexicalMultiplier = Number.isFinite(kpLexicalMultiplier)
        ? kpLexicalMultiplier
        : 1.2;
    const safeOtLexicalMultiplier = Number.isFinite(otLexicalMultiplier)
        ? otLexicalMultiplier
        : 1.0;

    const n = metadata.length;
    const activeCandidateIndices =
        candidateIndices && candidateIndices.length > 0 ? candidateIndices : undefined;
    const candidateCount = activeCandidateIndices
        ? activeCandidateIndices.length
        : n;
    const denseScores = new Float32Array(candidateCount);
    const sparseScores = new Float32Array(candidateCount);
    const denseOrder = new Int32Array(candidateCount);
    const sparseOrder = new Int32Array(candidateCount);
    const lexicalBonusMap = new Map<string, number>();
    const yearHitMap = new Map<string, boolean>();
    const docScopeSpecificityStatsMap = new Map<string, ScopeSpecificityStats>();
    const docDirectAnswerEvidenceStatsMap = new Map<string, ScopeSpecificityStats>();

    for (let localIndex = 0; localIndex < candidateCount; localIndex++) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[localIndex]
            : localIndex;
        const meta = metadata[metaIndex];

        let dense = dotProduct(
            queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null)
            dense *= meta.scale;
        const overriddenDense =
            denseScoreOverrides?.get(meta.id) ??
            (meta.type === "OT" ? otDenseScoreOverrides?.get(meta.id) : undefined);
        if (overriddenDense !== undefined) {
            dense = overriddenDense;
        }
        denseScores[localIndex] = dense;
        denseOrder[localIndex] = localIndex;

        let sparse = 0;
        if (querySparse && meta.sparse && meta.sparse.length > 0) {
            const dl = bm25Stats.docLengths[metaIndex];
            const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);
            const otid = resolveDocOtid(meta);

            for (let j = 0; j < meta.sparse.length; j += 2) {
                const wordId: number = meta.sparse[j] as number;
                const tf: number = meta.sparse[j + 1] as number;
                const specificityTerm = scopeSpecificityWordIdToTerm?.get(wordId);

                if (specificityTerm) {
                    const existing =
                        docScopeSpecificityStatsMap.get(otid) || {
                            termTf: {},
                            totalTf: 0,
                        };
                    existing.termTf[specificityTerm] =
                        (existing.termTf[specificityTerm] || 0) + tf;
                    existing.totalTf += tf;
                    docScopeSpecificityStatsMap.set(otid, existing);
                }

                const directAnswerEvidenceTerm =
                    directAnswerEvidenceWordIdToTerm?.get(wordId);
                if (directAnswerEvidenceTerm) {
                    const existing =
                        docDirectAnswerEvidenceStatsMap.get(otid) || {
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
                const otid = resolveDocOtid(meta);
                const weightedBonus =
                    meta.type === "Q"
                        ? sparse * safeQLexicalMultiplier
                        : meta.type === "KP"
                          ? sparse * safeKpLexicalMultiplier
                          : sparse * safeOtLexicalMultiplier;
                const currentBonus = lexicalBonusMap.get(otid) || 0;
                const nextBonus =
                    lexicalBonusMode === "max"
                        ? Math.max(currentBonus, weightedBonus)
                        : currentBonus + weightedBonus;
                lexicalBonusMap.set(otid, nextBonus);
            }
        }
        sparseScores[localIndex] = sparse;
        sparseOrder[localIndex] = localIndex;
    }

    denseOrder.sort((a, b) => denseScores[b] - denseScores[a]);
    const rrfScores = new Map<Metadata, number>();

    for (let rank = 0; rank < Math.min(4000, candidateCount); rank++) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[denseOrder[rank]]
            : denseOrder[rank];
        const meta = metadata[metaIndex];
        rrfScores.set(meta, (1 / (rank + RRF_K)) * 100);
    }

    if (querySparse) {
        sparseOrder.sort((a, b) => sparseScores[b] - sparseScores[a]);
        for (let rank = 0; rank < Math.min(4000, candidateCount); rank++) {
            const localIndex = sparseOrder[rank];
            if (sparseScores[localIndex] === 0) break;

            const metaIndex = activeCandidateIndices
                ? activeCandidateIndices[localIndex]
                : localIndex;
            const meta = metadata[metaIndex];
            const current = rrfScores.get(meta) || 0;
            rrfScores.set(meta, current + (1.2 / (rank + RRF_K)) * 100);
        }
    }

    const topHybrid = Array.from(rrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(1, topHybridLimit));

    const otidMap: Record<string, AggregatedDocScores> = {};

    for (const [meta, score] of topHybrid) {
        const otid = resolveDocOtid(meta);
        const topicIds = resolveMetadataTopicIds(meta);
        if (!otidMap[otid]) {
            otidMap[otid] = createAggregatedDocScores(meta, topicIds);
        }

        mergeAggregatedDocMetadata(otidMap[otid], meta, topicIds);
        applyScoreToAggregatedDocScores(otidMap[otid], meta, score);
    }

    const finalRanking: SearchResult[] = [];
    const candidateTargetYears = Object.values(otidMap)
        .map((scores) => scores.target_year)
        .filter((year): year is number => typeof year === "number");
    const candidateTimestamps = Object.values(otidMap)
        .map((scores) => scores.timestamp)
        .filter(
            (timestamp): timestamp is number => typeof timestamp === "number",
        );
    const latestTargetYear =
        candidateTargetYears.length > 0
            ? Math.max(...candidateTargetYears)
            : undefined;
    const latestTimestamp =
        candidateTimestamps.length > 0
            ? Math.max(...candidateTimestamps)
            : undefined;
    const intentContext = createQueryIntentContext(queryIntent, queryWords);
    const latestFocusedSpecificityTimestamp =
        intentContext.querySpecificityTerms.length > 0
            ? Object.entries(otidMap)
                  .map(([otid, scores]) => {
                      const matchedTf = getMatchedSpecificityTf(
                          intentContext.querySpecificityTerms,
                          docScopeSpecificityStatsMap.get(otid),
                      );
                      return matchedTf >= 10 ? scores.timestamp : undefined;
                  })
                  .filter(
                      (timestamp): timestamp is number =>
                          typeof timestamp === "number",
                  )
                  .reduce<number | undefined>(
                      (latest, timestamp) =>
                          latest === undefined || timestamp > latest
                              ? timestamp
                              : latest,
                      undefined,
                  )
            : undefined;

    for (const [otid, scores] of Object.entries(otidMap)) {
        const signals = getDocQuerySignals(
            otid,
            scores,
            intentContext,
            yearHitMap,
        );

        if (shouldSkipForExplicitYear(scores, intentContext, signals)) {
            continue;
        }

        const finalScore = computeBaseScore(scores, weights, {
            kpAggregationMode,
            kpTopN,
            kpTailWeight,
        });
        const kpRoleSelection = rerankKpCandidatesByRole({
            kpCandidates: scores.kp_candidates,
            bestKpid: scores.best_kpid,
            rawQuery: queryIntent?.rawQuery || "",
            queryScopeHint,
            mode: kpRoleRerankMode,
        });
        const boost = computeBoostMultiplier({
            otid,
            scores,
            lexicalBonusMap,
            yearHitMap,
            queryYearWordIds,
            intentContext,
            latestTargetYear,
            latestTimestamp,
            scopeSpecificityStats: docScopeSpecificityStatsMap.get(otid),
            latestFocusedSpecificityTimestamp,
        });

        finalRanking.push({
            otid,
            score: finalScore * boost + kpRoleSelection.docScoreDelta * kpRoleDocWeight,
            best_kpid: kpRoleSelection.bestKpid,
            kp_candidates: kpRoleSelection.orderedCandidates.slice(0, 5),
        });
    }

    const sortedRanking = finalRanking.sort((a, b) => b.score - a.score);
    const defaultQuerySignals: QuerySignals = {
        hasExplicitTopicOrIntent: false,
        hasExplicitYear: false,
        hasHistoricalHint: false,
        hasStrongDetailAnchor: false,
        hasEntryLikeAnchor: false,
        hasResultState: false,
        hasLatestPolicyState: false,
        hasGenericNextStep: false,
        queryLength: queryIntent?.rawQuery.length || 0,
        tokenCount: 0,
    };
    const querySignals = withQueryTokenCount(
        queryIntent?.signals || defaultQuerySignals,
        querySparse,
    );
    const retrievalSignals = extractRetrievalSignals(sortedRanking, otidMap);
    const responseDecision = classifyResponseMode(
        querySignals,
        retrievalSignals,
    );

    const explicitOutOfScopeOnly =
        (queryIntent?.intentIds.length || 0) === 0 &&
        hasOnlyOutOfScopeTopics(queryIntent?.topicIds || []);

    const inDomainEvidenceReject = shouldRejectForMissingInDomainEvidence({
        rawQuery: queryIntent?.rawQuery || "",
        queryIntent,
        sortedRanking,
        docEvidenceStatsMap: docDirectAnswerEvidenceStatsMap,
        otidMap,
    });
    const diagnostics: SearchRankDiagnostics = {
        querySignals,
        retrievalSignals,
        explicitOutOfScopeOnly,
        inDomainEvidenceRejectLabel: inDomainEvidenceReject.label || null,
    };

    if (explicitOutOfScopeOnly) {
        return {
            matches: [],
            weakMatches: sortedRanking.slice(0, 5),
            rejection: {
                reason: "low_topic_coverage",
                topicIds: queryIntent?.topicIds || [],
            },
            responseDecision: {
                ...responseDecision,
                mode: "reject",
                confidence: Math.max(responseDecision.confidence, 0.92),
                reason: "explicit_out_of_scope_topic",
                preferLatestWithinTopic: false,
                useWeakMatches: true,
            },
            diagnostics,
        };
    }

    if (
        responseDecision.mode === "direct_answer" &&
        inDomainEvidenceReject.shouldReject
    ) {
        return {
            matches: [],
            weakMatches: sortedRanking.slice(0, 5),
            rejection: {
                reason: "low_consistency",
                topicIds: queryIntent?.topicIds || [],
            },
            responseDecision: {
                ...responseDecision,
                mode: "reject",
                confidence: Math.max(responseDecision.confidence, 0.9),
                reason: `missing_in_domain_evidence:${inDomainEvidenceReject.label || "unknown"}`,
                preferLatestWithinTopic: false,
                useWeakMatches: true,
            },
            diagnostics,
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
            responseDecision,
            diagnostics,
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
            responseDecision,
            diagnostics,
        };
    }

    return {
        matches: sortedRanking.slice(0, 100),
        weakMatches: [],
        responseDecision,
        diagnostics,
    };
}

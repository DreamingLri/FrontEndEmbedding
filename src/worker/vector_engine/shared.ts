import {
    DEGREE_LEVEL_TABLE as CONFIG_DEGREE_LEVEL_TABLE,
    EVENT_TYPE_TABLE as CONFIG_EVENT_TYPE_TABLE,
    HISTORICAL_QUERY_HINTS as CONFIG_HISTORICAL_QUERY_HINTS,
    INTENT_VECTOR_TABLE as CONFIG_INTENT_VECTOR_TABLE,
    LATEST_QUERY_HINTS as CONFIG_LATEST_QUERY_HINTS,
    TOPIC_CONFIGS as CONFIG_TOPIC_CONFIGS,
} from "../search_topic_config.ts";
import type { KPCandidate } from "../aggregated_doc_scores.ts";
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
        | "invalid_input";
    topicIds: string[];
}

export type ResponseMode = "answer" | "reject";
export type RejectTier =
    | "invalid_input"
    | "hard_reject"
    | "boundary_uncertain";

export interface QuerySignals {
    hasExplicitTopicOrIntent: boolean;
    hasExplicitYear: boolean;
    hasHistoricalHint: boolean;
    hasStrongDetailAnchor: boolean;
    hasEntryLikeAnchor: boolean;
    hasResultState: boolean;
    hasLatestPolicyState: boolean;
    hasGenericNextStep: boolean;
    hasPostOutcomeOperationalCue: boolean;
    hasMultiSlotConstraintCue: boolean;
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

export interface EvidenceSignals {
    topRoleTags: string[];
    topCandidateCount: number;
    strongRoleCount: number;
    weakRoleCount: number;
    strongRoleRatio: number;
    weakOnly: boolean;
    hasOperationalRoleEvidence: boolean;
}

export interface ResponseDecision {
    mode: ResponseMode;
    confidence: number;
    reason: string;
    preferLatestWithinTopic: boolean;
    useWeakMatches: boolean;
    rejectScore?: number;
    rejectTier?: RejectTier | null;
}

export interface SearchRankDiagnostics {
    querySignals: QuerySignals;
    retrievalSignals: RetrievalSignals;
    evidenceSignals: EvidenceSignals;
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

export type KPAggregationMode = "max" | "max_plus_topn" | "mean" | "sum";
export type LexicalBonusMode = "sum" | "max";
export type KPRoleRerankMode = "off" | "feature";
export type FusionMode = "default" | "max_q_vs_kpot";
export type QConfusionMode =
    | "off"
    | "consensus"
    | "consensus_no_year"
    | "competition"
    | "combined";

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
export const RRF_RANK_LIMIT = 4000;

export const BM25_K1 = 1.2;
export const BM25_B = 0.4;
export const EVENT_TYPE_MISMATCH_PENALTY = 0.95;
export const LATEST_YEAR_BOOST_BASE = 0.82;
export const LATEST_POLICY_TIMESTAMP_BOOST_BASE = 0.97;
export const CURRENT_PROCESS_TIMESTAMP_BOOST_BASE = 0.984;
export const DEFAULT_KP_ROLE_DOC_WEIGHT = 0.35;
export const DEFAULT_Q_CONFUSION_WEIGHT = 0.2;
export const DEFAULT_KP_ROLE_CANDIDATE_LIMIT = 5;
export const CURRENT_PROCESS_EVENT_TYPES = [
    "招生章程",
    "报名通知",
    "考试安排",
    "材料提交",
    "资格要求",
] as const;
export const STRONG_EVIDENCE_ROLE_TAGS: ReadonlySet<string> = new Set([
    "condition",
    "materials",
    "email",
    "procedure",
    "application_stage",
    "contact",
    "location",
]);
export const WEAK_EVIDENCE_ROLE_TAGS: ReadonlySet<string> = new Set([
    "background",
    "publish",
    "post_outcome",
    "thesis",
    "reminder",
]);
export const HARD_REJECT_SCORE_THRESHOLD = 0.68;
export const BOUNDARY_REJECT_SCORE_THRESHOLD = 0.5;
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
export const QUERY_SCOPE_SPECIFICITY_TERM_SET: ReadonlySet<string> = new Set(
    QUERY_SCOPE_SPECIFICITY_TERMS,
);
export const BROAD_LATEST_SCOPE_CUE_PATTERN =
    /完整流程|完整|通用|一般|总流程|怎么报名|如何报名|条件.*报名|条件.*流程|条件.*操作/;



export const INTENT_CONFLICTS: Record<string, readonly string[]> = Object.fromEntries(
    CONFIG_INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item.negative_intents]),
);

export const INTENT_RULE_MAP: Map<string, IntentVectorItem> = new Map(
    CONFIG_INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item]),
);

const TOPIC_RULE_MAP: Map<string, (typeof CONFIG_TOPIC_CONFIGS)[number]> =
    new Map(CONFIG_TOPIC_CONFIGS.map((item) => [item.topic_id, item] as const));

export function clamp01(value: number): number {
    return Math.min(1, Math.max(0, value));
}

function isOutOfScopeTopic(topicId: string): boolean {
    return TOPIC_RULE_MAP.get(topicId)?.scope === "out_of_scope";
}

export function hasOnlyOutOfScopeTopics(topicIds: string[]): boolean {
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

export function resolveDocOtid(meta: Pick<Metadata, "id" | "parent_otid">): string {
    return meta.parent_otid || meta.id;
}

function matchTopicIds(query: string): string[] {
    return dedupe(
        CONFIG_TOPIC_CONFIGS.filter((topic) =>
            topic.aliases.some((alias) => query.includes(alias)),
        ).map((topic) => topic.topic_id),
    );
}

export function dedupe<T>(items: T[]): T[] {
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

export function hasPostOutcomeConditionCue(query: string): boolean {
    return /最终有效|最终有效性|最终录取|录取.*有效|拟录取.*有效|审核.*为准|审批.*为准/.test(
        query,
    );
}

function hasContextualContactDetailCue(query: string): boolean {
    return (
        /联系方式|联系电话|邮箱|邮件/.test(query) &&
        /学院|系|老师|办公室|研究生招生|研究生院|复试|录取|拟录取|公示|结果|通知|强基计划|夏令营|调剂/.test(
            query,
        )
    );
}

function hasStrongDetailAnchorCue(query: string): boolean {
    return (
        /录取通知书|通知书|报到|宿舍|党团关系|奖助金|档案|调档|政审|网上确认|现场确认|答辩|报名|考试|考试内容|考试科目|科目|题型|缴费|申请书|复试|面试|邮寄|地址|银行卡|学费|体检表|复审表|递补|书面说明|签字/.test(
            query,
        ) || hasContextualContactDetailCue(query)
    );
}

function hasPostOutcomeActionCue(query: string): boolean {
    return /书面说明|签字|递补|增补|体检表|复审表|总成绩|排序|放弃录取/.test(
        query,
    );
}

function hasPostOutcomeContactCue(query: string): boolean {
    return /联系方式|联系电话|邮箱|邮件|联系学院|联系老师/.test(query);
}

function hasPostOutcomeCommunicationContextCue(query: string): boolean {
    return /拟录取|录取|复试|调剂|结果|公示|名单|监督|申诉|沟通/.test(query);
}

function hasPostOutcomeOperationalCue(query: string): boolean {
    return (
        hasPostOutcomeActionCue(query) ||
        (hasPostOutcomeContactCue(query) &&
            hasPostOutcomeCommunicationContextCue(query))
    );
}

function hasMultiSlotConstraintCue(query: string): boolean {
    return /分别|另外|以及|同时|并说明|并描述|另行/.test(query);
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
        hasPostOutcomeOperationalCue: hasPostOutcomeOperationalCue(query),
        hasMultiSlotConstraintCue: hasMultiSlotConstraintCue(query),
        queryLength: query.length,
    };
}

export function withQueryTokenCount(
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

function compareMinHeapScoreIndex(
    left: number,
    right: number,
    scores: Float32Array,
): number {
    const scoreDiff = scores[left] - scores[right];
    if (scoreDiff !== 0) {
        return scoreDiff;
    }
    return right - left;
}

function siftUpMinHeap(
    heap: number[],
    scores: Float32Array,
    startIndex: number,
) {
    let index = startIndex;
    while (index > 0) {
        const parentIndex = Math.floor((index - 1) / 2);
        if (
            compareMinHeapScoreIndex(
                heap[index] as number,
                heap[parentIndex] as number,
                scores,
            ) >= 0
        ) {
            break;
        }
        [heap[index], heap[parentIndex]] = [heap[parentIndex], heap[index]];
        index = parentIndex;
    }
}

function siftDownMinHeap(
    heap: number[],
    scores: Float32Array,
    startIndex: number,
) {
    let index = startIndex;
    while (true) {
        const leftChildIndex = index * 2 + 1;
        const rightChildIndex = leftChildIndex + 1;
        let smallestIndex = index;

        if (
            leftChildIndex < heap.length &&
            compareMinHeapScoreIndex(
                heap[leftChildIndex] as number,
                heap[smallestIndex] as number,
                scores,
            ) < 0
        ) {
            smallestIndex = leftChildIndex;
        }

        if (
            rightChildIndex < heap.length &&
            compareMinHeapScoreIndex(
                heap[rightChildIndex] as number,
                heap[smallestIndex] as number,
                scores,
            ) < 0
        ) {
            smallestIndex = rightChildIndex;
        }

        if (smallestIndex === index) {
            return;
        }

        [heap[index], heap[smallestIndex]] = [
            heap[smallestIndex],
            heap[index],
        ];
        index = smallestIndex;
    }
}

export function selectTopLocalIndices(
    scores: Float32Array,
    limit: number,
    options?: {
        minimumScoreExclusive?: number;
    },
): number[] {
    if (limit <= 0 || scores.length === 0) {
        return [];
    }

    const effectiveLimit = Math.min(limit, scores.length);
    const minimumScoreExclusive =
        options?.minimumScoreExclusive ?? Number.NEGATIVE_INFINITY;
    const heap: number[] = [];

    for (let localIndex = 0; localIndex < scores.length; localIndex += 1) {
        const score = scores[localIndex] as number;
        if (
            !Number.isFinite(score) ||
            score <= minimumScoreExclusive
        ) {
            continue;
        }

        if (heap.length < effectiveLimit) {
            heap.push(localIndex);
            siftUpMinHeap(heap, scores, heap.length - 1);
            continue;
        }

        const worstIndex = heap[0] as number;
        if (compareMinHeapScoreIndex(localIndex, worstIndex, scores) <= 0) {
            continue;
        }

        heap[0] = localIndex;
        siftDownMinHeap(heap, scores, 0);
    }

    return heap.sort((left, right) => {
        const scoreDiff = scores[right] - scores[left];
        if (scoreDiff !== 0) {
            return scoreDiff;
        }
        return left - right;
    });
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



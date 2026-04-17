import type { FeatureExtractionPipeline } from "@huggingface/transformers";

import { fmmTokenize } from "./fmm_tokenize.ts";
import {
    buildQueryPlan,
    inferDocumentRolesFromTitle,
    type QueryPlan,
} from "./query_planner.ts";
import {
    getQuerySparse,
    parseQueryIntent,
    QUERY_SCOPE_SPECIFICITY_TERMS,
    searchAndRank,
    type BM25Stats,
    type KPAggregationMode,
    type KPRoleRerankMode,
    type LexicalBonusMode,
    type Metadata,
    type ParsedQueryIntent,
    type QConfusionMode,
    type QuerySignals,
    type RejectTier,
    type RetrievalSignals,
    type ResponseDecision,
    type ResponseMode,
    type SearchRankOutput,
    type SearchRejection,
} from "./vector_engine.ts";
import {
    getCandidateIndicesForQuery,
    type TopicPartitionIndex,
} from "./topic_partition.ts";

export type PipelineBehavior = "answer" | "reject";

export type PipelinePreset = {
    name: string;
    retrieval: {
        weights: {
            Q: number;
            KP: number;
            OT: number;
        };
        topHybridLimit: number;
        kpAggregationMode: KPAggregationMode;
        kpTopN: number;
        kpTailWeight: number;
        lexicalBonusMode: LexicalBonusMode;
        kpRoleRerankMode: KPRoleRerankMode;
        kpRoleDocWeight: number;
        qConfusionMode: QConfusionMode;
        qConfusionWeight: number;
        useQueryExpansion: boolean;
        useTopicPartition: boolean;
        enableExplicitYearFilter: boolean;
        enablePhaseAnchorBoost: boolean;
        minimalMode: boolean;
    };
    display: {
        rejectThreshold: number;
        rerankBlendAlpha: number;
        bestSentenceThreshold: number;
        fetchMatchLimit: number;
        fetchWeakMatchLimit: number;
        useYearPhaseTitleAdjustment: boolean;
        enableQueryPlanner: boolean;
    };
};

const DEFAULT_DISPLAY_CONFIG: PipelinePreset["display"] = {
    rejectThreshold: 0.4,
    rerankBlendAlpha: 0.15,
    bestSentenceThreshold: 0.4,
    fetchMatchLimit: 15,
    fetchWeakMatchLimit: 10,
    useYearPhaseTitleAdjustment: false,
    enableQueryPlanner: false,
};

export const PAPER_FROZEN_MAIN_PIPELINE_PRESET: PipelinePreset = {
    name: "paper_frozen_main_v1",
    retrieval: {
        weights: {
            Q: 0,
            KP: 0.28571428571428575,
            OT: 0.7142857142857143,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
        qConfusionMode: "off",
        qConfusionWeight: 0.2,
        useQueryExpansion: true,
        useTopicPartition: true,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: false,
        minimalMode: false,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

// Historical preset retained only for explicit compatibility / replay.
// Do not use as the default runtime preset.
export const PRODUCT_CANONICAL_FULL_PIPELINE_PRESET: PipelinePreset = {
    name: "product_canonical_full_v1",
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
        qConfusionMode: "off",
        qConfusionWeight: 0.2,
        useQueryExpansion: true,
        useTopicPartition: true,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: false,
        minimalMode: false,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PAPER_TAIL_TOP3_W020_PIPELINE_PRESET: PipelinePreset = {
    name: "paper_tail_top3_w020_v1",
    retrieval: {
        ...PAPER_FROZEN_MAIN_PIPELINE_PRESET.retrieval,
        kpAggregationMode: "max_plus_topn",
        kpTopN: 3,
        kpTailWeight: 0.2,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET: PipelinePreset = {
    name: "product_tail_top3_w020_v1",
    retrieval: {
        ...PRODUCT_CANONICAL_FULL_PIPELINE_PRESET.retrieval,
        kpAggregationMode: "max_plus_topn",
        kpTopN: 3,
        kpTailWeight: 0.2,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const MINIMAL_BASELINE_PIPELINE_PRESET: PipelinePreset = {
    name: "minimal_q_kp_ot_v1",
    retrieval: {
        weights: {
            Q: 0.3333333333333333,
            KP: 0.3333333333333333,
            OT: 0.3333333333333333,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "off",
        kpRoleDocWeight: 0,
        qConfusionMode: "off",
        qConfusionWeight: 0.2,
        useQueryExpansion: false,
        useTopicPartition: false,
        enableExplicitYearFilter: false,
        enablePhaseAnchorBoost: false,
        minimalMode: true,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET: PipelinePreset = {
    name: "frontend_research_sync_v1",
    retrieval: {
        ...MINIMAL_BASELINE_PIPELINE_PRESET.retrieval,
        qConfusionMode: "combined",
        qConfusionWeight: 0.2,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: true,
    },
    display: {
        ...DEFAULT_DISPLAY_CONFIG,
        fetchMatchLimit: 20,
        fetchWeakMatchLimit: 12,
        useYearPhaseTitleAdjustment: true,
    },
};

export const FRONTEND_RESEARCH_SYNC_QUERY_PLANNER_PIPELINE_PRESET: PipelinePreset =
    {
        name: "frontend_research_sync_query_planner_v1",
        retrieval: { ...FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval },
        display: {
            ...FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display,
            enableQueryPlanner: true,
        },
    };

export const PIPELINE_PRESET_REGISTRY = {
    paper_frozen_main_v1: PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    product_canonical_full_v1: PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    paper_tail_top3_w020_v1: PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    product_tail_top3_w020_v1: PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    minimal_q_kp_ot_v1: MINIMAL_BASELINE_PIPELINE_PRESET,
    frontend_research_sync_v1: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    frontend_research_sync_query_planner_v1:
        FRONTEND_RESEARCH_SYNC_QUERY_PLANNER_PIPELINE_PRESET,
} as const;

export type PipelinePresetName = keyof typeof PIPELINE_PRESET_REGISTRY;

export const CANONICAL_PIPELINE_PRESET =
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET;

export function clonePipelinePreset(preset: PipelinePreset): PipelinePreset {
    return {
        ...preset,
        retrieval: {
            ...preset.retrieval,
            weights: { ...preset.retrieval.weights },
        },
        display: { ...preset.display },
    };
}

export function resolvePipelinePresetByName(
    presetName?: string,
): PipelinePreset {
    if (!presetName) {
        return clonePipelinePreset(CANONICAL_PIPELINE_PRESET);
    }

    const preset = PIPELINE_PRESET_REGISTRY[
        presetName as PipelinePresetName
    ];
    return clonePipelinePreset(preset || CANONICAL_PIPELINE_PRESET);
}

export type PipelineTermMaps = {
    scopeSpecificityWordIdToTerm: Map<number, string>;
};

export type SearchPipelineQueryContext = {
    query: string;
    queryIntent: ParsedQueryIntent;
    queryPlan: QueryPlan;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryYearWordIds: number[];
    candidateIndices?: number[];
};

export type PipelineDocumentRecord = {
    id?: string;
    otid?: string;
    ot_title?: string;
    ot_text?: string;
    link?: string;
    publish_time?: string;
    bestSentence?: string;
    bestPoint?: string;
    best_kpid?: string;
    kps?: Array<{ kpid?: string; kp_text?: string }>;
    score?: number;
    coarseScore?: number;
    displayScore?: number;
    rerankScore?: number;
    confidenceScore?: number;
    snippetScore?: number;
};

export type PipelineDecision = {
    behavior: PipelineBehavior;
    rawMode: ResponseMode;
    confidence: number;
    reason: string;
    preferLatestWithinTopic: boolean;
    useWeakMatches: boolean;
    rejectionReason: SearchRejection["reason"] | null;
    rejectScore?: number;
    rejectTier?: RejectTier | null;
};

export type PipelineTrace = {
    totalMs: number;
    searchMs: number;
    fetchMs: number;
    candidateCount: number;
    partitionUsed: boolean;
    partitionCandidateCount?: number;
    matchCount: number;
    weakMatchCount: number;
    fetchedDocumentCount: number;
    querySignals?: QuerySignals;
    retrievalSignals?: RetrievalSignals;
    queryPlan?: QueryPlan;
};

export type RetrievalStageResult = {
    queryContext: SearchPipelineQueryContext;
    searchOutput: SearchRankOutput;
    retrievalDecision: PipelineDecision;
    candidateCount: number;
    searchMs: number;
};

export type SearchPipelineResult = {
    query: string;
    presetName: string;
    queryContext: SearchPipelineQueryContext;
    searchOutput: SearchRankOutput;
    responseDecision?: ResponseDecision;
    retrievalDecision: PipelineDecision;
    finalDecision: PipelineDecision;
    rejection?: SearchRejection;
    results: PipelineDocumentRecord[];
    weakResults: PipelineDocumentRecord[];
    trace: PipelineTrace;
};

export type PipelineDocumentLoader = (params: {
    query: string;
    otids: string[];
}) => Promise<PipelineDocumentRecord[]>;

function nowMs(): number {
    if (typeof performance !== "undefined" && performance.now) {
        return performance.now();
    }
    return Date.now();
}

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
}

const QUERY_EXPANSION_RULES: Array<{
    pattern: RegExp;
    terms: string[];
    intentTerms?: string[];
}> = [
    {
        pattern: /现场确认/,
        terms: ["网上确认"],
        intentTerms: ["网上确认"],
    },
];

function buildExpandedQueryWords(
    query: string,
    vocabMap: Map<string, number>,
): string[] {
    const baseWords = fmmTokenize(query, vocabMap);
    const expandedWords = QUERY_EXPANSION_RULES.flatMap((rule) => {
        if (!rule.pattern.test(query)) {
            return [];
        }
        return rule.terms.flatMap((term) => fmmTokenize(term, vocabMap));
    });

    return dedupe([...baseWords, ...expandedWords]);
}

function buildExpandedIntentQuery(query: string): string {
    const expandedTerms = dedupe(
        QUERY_EXPANSION_RULES.flatMap((rule) =>
            rule.pattern.test(query) ? rule.intentTerms || [] : [],
        ),
    );
    if (expandedTerms.length === 0) {
        return query;
    }
    return `${query} ${expandedTerms.join(" ")}`;
}

export function buildPipelineTermMaps(
    vocabMap: Map<string, number>,
): PipelineTermMaps {
    const scopeSpecificityWordIdToTerm = new Map<number, string>();
    QUERY_SCOPE_SPECIFICITY_TERMS.forEach((term) => {
        const wordId = vocabMap.get(term);
        if (wordId !== undefined) {
            scopeSpecificityWordIdToTerm.set(wordId, term);
        }
    });

    return {
        scopeSpecificityWordIdToTerm,
    };
}

export function buildSearchPipelineQueryContext(
    query: string,
    vocabMap: Map<string, number>,
    topicPartitionIndex: TopicPartitionIndex,
    preset: PipelinePreset = CANONICAL_PIPELINE_PRESET,
): SearchPipelineQueryContext {
    const expandedIntentQuery = preset.retrieval.useQueryExpansion
        ? buildExpandedIntentQuery(query)
        : query;
    const parsedQueryIntent = parseQueryIntent(expandedIntentQuery);
    const queryIntent =
        expandedIntentQuery === query
            ? parsedQueryIntent
            : {
                  ...parsedQueryIntent,
                  rawQuery: query,
              };
    const candidateIndices = preset.retrieval.useTopicPartition
        ? getCandidateIndicesForQuery(queryIntent, topicPartitionIndex)
        : undefined;
    const queryWords = preset.retrieval.useQueryExpansion
        ? buildExpandedQueryWords(query, vocabMap)
        : Array.from(new Set(fmmTokenize(query, vocabMap)));
    const querySparse = getQuerySparse(queryWords, vocabMap);
    const queryYearWordIds = queryIntent.years
        .map(String)
        .map((year) => vocabMap.get(year))
        .filter((item): item is number => item !== undefined);
    const queryPlan = buildQueryPlan(query, queryIntent);

    return {
        query,
        queryIntent,
        queryPlan,
        queryWords,
        querySparse,
        queryYearWordIds,
        candidateIndices,
    };
}

function buildPipelineDecision(params: {
    query: string;
    queryIntent: ParsedQueryIntent;
    searchOutput: SearchRankOutput;
}): PipelineDecision {
    const { searchOutput } = params;
    const rawMode =
        searchOutput.responseDecision?.mode ||
        (searchOutput.rejection ? "reject" : "answer");
    const behavior = rawMode === "reject" ? "reject" : "answer";

    return {
        behavior,
        rawMode,
        confidence: searchOutput.responseDecision?.confidence ?? 0.62,
        reason:
            searchOutput.responseDecision?.reason ||
            searchOutput.rejection?.reason ||
            "scored_pipeline_behavior",
        preferLatestWithinTopic:
            searchOutput.responseDecision?.preferLatestWithinTopic ?? false,
        useWeakMatches:
            behavior === "reject" &&
            (searchOutput.responseDecision?.useWeakMatches ??
                searchOutput.weakMatches.length > 0),
        rejectionReason:
            behavior === "reject" ? searchOutput.rejection?.reason || null : null,
        rejectScore: searchOutput.responseDecision?.rejectScore,
        rejectTier: searchOutput.responseDecision?.rejectTier ?? null,
    };
}

type PhaseAnchor = {
    half?: "上半年" | "下半年";
    batch?: string;
    stages: string[];
};

const PHASE_ANCHOR_DOC_WEIGHT = 0.35;
const TITLE_INTENT_DOC_WEIGHT = 0.28;
const TITLE_COVERAGE_DOC_WEIGHT = 0.18;
const TITLE_DIVERSITY_DUPLICATE_PENALTY = 0.34;
const PHASE_STAGE_RULES: Array<{ stage: string; pattern: RegExp }> = [
    { stage: "预报名", pattern: /预报名/ },
    { stage: "报名通知", pattern: /报名通知|网上报名/ },
    { stage: "工作方案", pattern: /工作方案/ },
    { stage: "接收办法", pattern: /接收办法/ },
    { stage: "实施办法", pattern: /实施办法/ },
    { stage: "录取方案", pattern: /录取方案/ },
    { stage: "招生简章", pattern: /招生简章/ },
    { stage: "招生章程", pattern: /招生章程/ },
    { stage: "综合考核", pattern: /综合考核/ },
    { stage: "复试", pattern: /复试/ },
    { stage: "调剂", pattern: /调剂/ },
] as const;
const TITLE_RULE_DOC_PATTERN =
    /(招生简章|招生章程|实施细则|实施办法|接收办法|工作方案|录取方案|章程|简章)/;
const TITLE_PROCESS_NOTICE_PATTERN =
    /(活动报名通知|报名通知|预报名|综合考核通知|复试通知|考核通知|考核安排|申请通知|报名安排)/;
const TITLE_OUTCOME_PATTERN =
    /(结果公示|录取结果|拟录取|公示|名单|递补|增补|入营通知)/;
const TITLE_AI_SCHOOL_PATTERN = /人工智能学院/;
const TITLE_OTHER_PROGRAM_PATTERN =
    /(软件工程学院|海洋工程与技术学院|附属医院|广州实验室|鹏城实验室|大湾区大学|鹏城国家实验室)/;
const TITLE_PREAPPLY_PATTERN = /预报名/;
const TITLE_TRANSFER_PATTERN = /调剂/;
const TITLE_CANDIDATE_LIST_PATTERN =
    /进入综合考核考生名单|综合考核考生名单|进入综合考核名单/;
const TITLE_REVIEW_RESULT_PATTERN =
    /(复试结果|综合考核结果|结果公示|拟录取|录取结果|公示|名单)/;
const TITLE_SUMMER_CAMP_PATTERN = /夏令营/;
const TITLE_TUIMIAN_PATTERN = /推免|推荐免试/;
const TITLE_DOCTORAL_PATTERN = /博士/;
const TITLE_MASTER_PATTERN = /硕士/;
const TITLE_SYSTEM_NOTICE_PATTERN = /录取通知书|邮寄地址校对/;
const QUERY_ASPECT_RULES = [
    {
        query: /条件|资格|要求/,
        doc: /条件|资格|要求|申请人基本条件|报考条件|身体健康/,
    },
    {
        query: /材料|提交|准备什么材料/,
        doc: /材料|提交|成绩单|证明|推荐信|纸质版/,
    },
    {
        query: /时间|日期|什么时候|何时|截止/,
        doc: /时间|日期|截止|月|日|24:00|开通|关闭/,
    },
    {
        query: /系统|报名功能|录取功能|操作/,
        doc: /系统|报名功能|录取功能|确认|注册|填报/,
    },
    {
        query: /流程|步骤|经过哪些|过程|程序|考核/,
        doc: /流程|步骤|考核|审核|报名|录取|复试/,
    },
    {
        query: /评分|打分|成绩|单科/,
        doc: /评分|成绩|总成绩|单科|综合考核成绩/,
    },
] as const;

function normalizePatternText(text: string): string {
    return text.replace(/\s+/g, "");
}

function queryAsksOutcomeLikeTitle(query: string): boolean {
    return /结果|公示|名单|拟录取|递补|增补|录取结果|入营/.test(query);
}

function queryAsksProcedureLikeTitle(query: string): boolean {
    return /流程|步骤|环节|程序|过程|考核步骤|需要经过|怎么申请|如何申请|怎么报名|如何报名|关键时间节点|时间节点|时间安排|什么时候|何时|截止日期|截止时间|系统操作|报到/.test(
        query,
    );
}

function queryAsksRequirementLikeTitle(query: string): boolean {
    return /条件|要求|资格|材料|评分|怎么评分|如何评分|关键要求|整体政策|政策|规则/.test(
        query,
    );
}

function queryAsksEventDateLikeTitle(query: string): boolean {
    return /举办日期|举办时间|举行时间|活动时间|什么时候举办|何时举办|哪天举办/.test(
        query,
    );
}

function queryAsksPolicyOverviewLikeTitle(query: string): boolean {
    return /整体政策|主要要求|关键要求|政策|总体要求/.test(query);
}

function queryAsksSystemTimelineLikeTitle(query: string): boolean {
    return /关键时间节点|时间节点|截止时间|截止日期|系统操作|报名功能|录取功能|服务系统/.test(
        query,
    );
}

function queryNeedsCoverageLikeTitle(query: string): boolean {
    return /分别|以及|并描述|整个流程|申请和录取过程|从预报名到录取|从准备材料到完成面试|条件.*评分|条件.*材料|材料.*时间|申请.*录取过程/.test(
        query,
    );
}

function getDocumentEvidenceText(document: PipelineDocumentRecord): string {
    const parts: string[] = [];
    if (document.ot_title) {
        parts.push(document.ot_title);
    }
    if (Array.isArray(document.kps)) {
        const bestKp = document.kps.find((item) => item.kpid === document.best_kpid);
        if (bestKp?.kp_text) {
            parts.push(bestKp.kp_text);
        }
        document.kps
            .filter((item) => item.kpid !== document.best_kpid && item.kp_text)
            .slice(0, 4)
            .forEach((item) => {
                if (item.kp_text) {
                    parts.push(item.kp_text);
                }
            });
    }
    return normalizePatternText(parts.join(" "));
}

function normalizeTitleDedupKey(title: string): string {
    return normalizePatternText(title).replace(/20\d{2}年/g, "");
}

function computeQueryPlanDocRoleDelta(
    document: PipelineDocumentRecord,
    queryPlan: QueryPlan,
): number {
    if (
        queryPlan.difficultyTier !== "high" &&
        !queryPlan.asksOutcomeLike &&
        !queryPlan.asksCoverageLike &&
        !queryPlan.asksSystemTimelineLike
    ) {
        return 0;
    }

    const roles = inferDocumentRolesFromTitle(document.ot_title || "");
    let delta = 0;

    if (roles.some((role) => queryPlan.preferredDocRoles.includes(role))) {
        delta += 0.22;
    }
    if (
        (queryPlan.asksOutcomeLike || queryPlan.difficultyTier === "high") &&
        roles.some((role) => queryPlan.avoidedDocRoles.includes(role))
    ) {
        delta -= 0.18;
    }

    if (queryPlan.asksCoverageLike) {
        if (roles.includes("stage_list") || roles.includes("rule_doc")) {
            delta += 0.1;
        }
        if (
            !queryPlan.asksOutcomeLike &&
            queryPlan.difficultyTier === "high" &&
            roles.includes("result_notice")
        ) {
            delta -= 0.08;
        }
    }

    if (queryPlan.intentType === "outcome" && roles.includes("result_notice")) {
        delta += 0.12;
    }
    if (
        queryPlan.intentType === "time_location" &&
        roles.includes("stage_list")
    ) {
        delta += 0.08;
    }
    if (
        queryPlan.intentType === "policy_overview" &&
        roles.includes("rule_doc")
    ) {
        delta += 0.08;
    }

    return delta;
}

function computeTitleIntentDocDelta(
    query: string,
    document: PipelineDocumentRecord,
): number {
    const normalizedQuery = normalizePatternText(query);
    const normalizedTitle = normalizePatternText(document.ot_title || "");
    if (!normalizedTitle) {
        return 0;
    }

    const asksOutcomeLikeTitle = queryAsksOutcomeLikeTitle(normalizedQuery);
    const asksProcedureLikeTitle = queryAsksProcedureLikeTitle(normalizedQuery);
    const asksRequirementLikeTitle =
        queryAsksRequirementLikeTitle(normalizedQuery);
    const asksEventDateLikeTitle =
        queryAsksEventDateLikeTitle(normalizedQuery);
    const asksPolicyOverviewLikeTitle =
        queryAsksPolicyOverviewLikeTitle(normalizedQuery);
    const asksSystemTimelineLikeTitle =
        queryAsksSystemTimelineLikeTitle(normalizedQuery);
    const isRuleDocTitle = TITLE_RULE_DOC_PATTERN.test(normalizedTitle);
    const isProcessNoticeTitle =
        TITLE_PROCESS_NOTICE_PATTERN.test(normalizedTitle);
    const isOutcomeTitle = TITLE_OUTCOME_PATTERN.test(normalizedTitle);
    const isAiSchoolTitle = TITLE_AI_SCHOOL_PATTERN.test(normalizedTitle);
    const isOtherProgramTitle =
        TITLE_OTHER_PROGRAM_PATTERN.test(normalizedTitle);
    const isPreapplyTitle = TITLE_PREAPPLY_PATTERN.test(normalizedTitle);
    const isTransferTitle = TITLE_TRANSFER_PATTERN.test(normalizedTitle);
    const isCandidateListTitle =
        TITLE_CANDIDATE_LIST_PATTERN.test(normalizedTitle);
    const isReviewResultTitle =
        TITLE_REVIEW_RESULT_PATTERN.test(normalizedTitle);
    const isSummerCampTitle = TITLE_SUMMER_CAMP_PATTERN.test(normalizedTitle);
    const isTuimianTitle = TITLE_TUIMIAN_PATTERN.test(normalizedTitle);
    const isDoctoralTitle = TITLE_DOCTORAL_PATTERN.test(normalizedTitle);
    const isMasterOnlyTitle =
        TITLE_MASTER_PATTERN.test(normalizedTitle) &&
        !isDoctoralTitle &&
        !isTuimianTitle;
    const isSystemNoticeTitle =
        TITLE_SYSTEM_NOTICE_PATTERN.test(normalizedTitle);
    const mentionsAiSchool = /人工智能学院|AI学院/.test(normalizedQuery);
    const mentionsDoctoral = /博士/.test(normalizedQuery);
    const mentionsTuimian = /推免|推荐免试/.test(normalizedQuery);
    const mentionsSummerCamp = /夏令营/.test(normalizedQuery);
    const mentionsTransfer = /调剂/.test(normalizedQuery);
    const asksPostOutcomeAdmission =
        /通过考核后|还会被录取吗|确保.*录取|被.*录取/.test(normalizedQuery);
    const asksMaterialReviewTiming =
        /材料审核.*公示|通过材料审核/.test(normalizedQuery);

    let delta = 0;

    if (!asksOutcomeLikeTitle && isOutcomeTitle) {
        delta -= 0.95;
    }
    if ((asksProcedureLikeTitle || asksRequirementLikeTitle) && isRuleDocTitle) {
        delta += asksProcedureLikeTitle && asksRequirementLikeTitle ? 0.95 : 0.75;
    }
    if (asksProcedureLikeTitle && isProcessNoticeTitle) {
        delta += 0.55;
    }

    if (mentionsAiSchool) {
        if (isAiSchoolTitle) {
            delta += 0.45;
        } else if (isOtherProgramTitle) {
            delta -= 0.9;
        } else if (/学院/.test(normalizedTitle)) {
            delta -= 0.55;
        } else {
            delta -= 0.2;
        }
    }

    if (mentionsDoctoral) {
        if (isMasterOnlyTitle) {
            delta -= 0.95;
        } else if (isDoctoralTitle) {
            delta += 0.2;
        }
    }

    if (mentionsTuimian) {
        if (isTuimianTitle) {
            delta += 0.45;
        }
        if (isDoctoralTitle && !isTuimianTitle) {
            delta -= 0.6;
        }
    }

    if (mentionsSummerCamp) {
        if (isSummerCampTitle) {
            delta += 0.35;
        }
        if (!asksEventDateLikeTitle && /活动报名通知|报名通知/.test(normalizedTitle)) {
            delta += 0.45;
        }
        if (
            !asksOutcomeLikeTitle &&
            !asksEventDateLikeTitle &&
            /入营通知/.test(normalizedTitle)
        ) {
            delta -= 0.45;
        }
        if (asksEventDateLikeTitle && /入营通知|活动通知/.test(normalizedTitle)) {
            delta += 0.95;
        }
        if (
            asksEventDateLikeTitle &&
            !/报名/.test(normalizedQuery) &&
            /活动报名通知|报名通知/.test(normalizedTitle)
        ) {
            delta -= 0.45;
        }
        if (
            !isSummerCampTitle &&
            (isDoctoralTitle || isTuimianTitle || isMasterOnlyTitle)
        ) {
            delta -= 0.65;
        }
    }

    if (mentionsTuimian && /接收办法|工作方案/.test(normalizedTitle)) {
        delta += 0.45;
    }

    if (mentionsDoctoral && /实施办法|招生简章|综合考核通知/.test(normalizedTitle)) {
        delta += 0.25;
    }

    if (asksPolicyOverviewLikeTitle) {
        if (/招生简章|接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.55;
        }
        if (isPreapplyTitle) {
            delta -= 0.45;
        }
        if (isTransferTitle && !mentionsTransfer) {
            delta -= 0.65;
        }
        if (isReviewResultTitle && !asksOutcomeLikeTitle) {
            delta -= 0.55;
        }
    }

    if (asksSystemTimelineLikeTitle) {
        if (/接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.45;
        }
        if (isPreapplyTitle && /录取过程|系统操作/.test(normalizedQuery)) {
            delta -= 0.2;
        }
        if (isSystemNoticeTitle) {
            delta -= 0.55;
        }
    } else if (
        /关键时间节点|时间节点|截止日期|截止时间/.test(normalizedQuery) &&
        /报名通知|综合考核通知|复试通知/.test(normalizedTitle)
    ) {
        delta += 0.4;
    }

    if (asksPostOutcomeAdmission) {
        if (/招生简章|实施办法/.test(normalizedTitle)) {
            delta += 0.4;
        }
        if (isReviewResultTitle) {
            delta -= 0.55;
        }
    }

    if (asksMaterialReviewTiming) {
        if (isCandidateListTitle) {
            delta += 0.8;
        }
        if (/综合考核结果/.test(normalizedTitle)) {
            delta -= 0.35;
        }
    }

    return delta;
}

function computeCoverageDocDelta(
    query: string,
    document: PipelineDocumentRecord,
): number {
    const normalizedQuery = normalizePatternText(query);
    const requestedAspects = QUERY_ASPECT_RULES.filter((rule) =>
        rule.query.test(normalizedQuery),
    );
    if (requestedAspects.length < 2) {
        return 0;
    }

    const evidenceText = getDocumentEvidenceText(document);
    if (!evidenceText) {
        return -0.35;
    }

    const coveredCount = requestedAspects.filter((rule) =>
        rule.doc.test(evidenceText),
    ).length;

    if (coveredCount === 0) {
        return -0.4;
    }

    let delta = Math.min(coveredCount, 3) * 0.2;
    if (coveredCount === requestedAspects.length) {
        delta += 0.2;
    }
    return delta;
}

function normalizeBatchToken(token: string): string | undefined {
    switch (token) {
        case "1":
        case "一":
            return "1";
        case "2":
        case "二":
            return "2";
        case "3":
        case "三":
            return "3";
        case "4":
        case "四":
            return "4";
        default:
            return undefined;
    }
}

function extractPhaseAnchor(text: string): PhaseAnchor {
    const normalized = text.replace(/\s+/g, "");
    const halfMatch = normalized.match(/上半年|下半年/);
    const batchMatch = normalized.match(/第?\s*([一二三四1234])\s*批/);

    return {
        half:
            halfMatch?.[0] === "上半年" || halfMatch?.[0] === "下半年"
                ? halfMatch[0]
                : undefined,
        batch: batchMatch?.[1]
            ? normalizeBatchToken(batchMatch[1])
            : undefined,
        stages: PHASE_STAGE_RULES.filter((rule) => rule.pattern.test(normalized)).map(
            (rule) => rule.stage,
        ),
    };
}

function hasExplicitPhaseAnchor(anchor: PhaseAnchor): boolean {
    return Boolean(anchor.half || anchor.batch || anchor.stages.length > 0);
}

function computePhaseAnchorDocDelta(
    query: string,
    document: PipelineDocumentRecord,
): number {
    const queryPhase = extractPhaseAnchor(query);
    if (!hasExplicitPhaseAnchor(queryPhase)) {
        return 0;
    }

    const title = (document.ot_title || "").replace(/\s+/g, "");
    if (!title) {
        return -0.15;
    }

    const articlePhase = extractPhaseAnchor(title);
    let delta = 0;

    if (queryPhase.half) {
        if (articlePhase.half === queryPhase.half) {
            delta += 0.9;
        } else if (articlePhase.half) {
            delta -= 0.9;
        } else {
            delta -= 0.2;
        }
    }

    if (queryPhase.batch) {
        if (articlePhase.batch === queryPhase.batch) {
            delta += 1.0;
        } else if (articlePhase.batch) {
            delta -= 1.0;
        } else {
            delta -= 0.2;
        }
    }

    if (queryPhase.stages.length > 0) {
        const hasExactStage = queryPhase.stages.some((stage) =>
            articlePhase.stages.includes(stage),
        );
        if (hasExactStage) {
            delta += 0.8;
        } else if (articlePhase.stages.length > 0) {
            delta -= 0.8;
        } else {
            delta -= 0.15;
        }
    }

    return delta;
}

function applyPhaseAnchorBoostToDocuments(
    query: string,
    documents: PipelineDocumentRecord[],
    queryPlan?: QueryPlan,
): PipelineDocumentRecord[] {
    const weight = PHASE_ANCHOR_DOC_WEIGHT * (queryPlan?.phaseAnchorWeightScale ?? 1);
    return [...documents]
        .map((document) => {
            const baseScore =
                document.displayScore ?? document.coarseScore ?? document.score ?? 0;
            const delta = computePhaseAnchorDocDelta(query, document);
            const nextScore = baseScore + delta * weight;

            return {
                ...document,
                score: nextScore,
                coarseScore:
                    (document.coarseScore ?? document.score ?? baseScore) +
                    delta * weight,
                displayScore: nextScore,
            };
        })
        .sort(
            (left, right) =>
                (right.displayScore ?? right.score ?? 0) -
                (left.displayScore ?? left.score ?? 0),
        );
}

function applyTitleIntentBoostToDocuments(
    query: string,
    documents: PipelineDocumentRecord[],
    queryPlan?: QueryPlan,
): PipelineDocumentRecord[] {
    const weight = TITLE_INTENT_DOC_WEIGHT * (queryPlan?.titleIntentWeightScale ?? 1);
    return [...documents]
        .map((document) => {
            const baseScore =
                document.displayScore ?? document.coarseScore ?? document.score ?? 0;
            const delta =
                computeTitleIntentDocDelta(query, document) +
                (queryPlan
                    ? computeQueryPlanDocRoleDelta(document, queryPlan)
                    : 0);
            const nextScore = baseScore + delta * weight;

            return {
                ...document,
                score: nextScore,
                coarseScore:
                    (document.coarseScore ?? document.score ?? baseScore) +
                    delta * weight,
                displayScore: nextScore,
            };
        })
        .sort(
            (left, right) =>
                (right.displayScore ?? right.score ?? 0) -
                (left.displayScore ?? left.score ?? 0),
        );
}

function applyCoverageBoostToDocuments(
    query: string,
    documents: PipelineDocumentRecord[],
    queryPlan?: QueryPlan,
): PipelineDocumentRecord[] {
    const weight = TITLE_COVERAGE_DOC_WEIGHT * (queryPlan?.coverageWeightScale ?? 1);
    const scoredDocuments = [...documents]
        .map((document) => {
            const baseScore =
                document.displayScore ?? document.coarseScore ?? document.score ?? 0;
            const delta = computeCoverageDocDelta(query, document);
            const nextScore = baseScore + delta * weight;

            return {
                ...document,
                score: nextScore,
                displayScore: nextScore,
            };
        })
        .sort(
            (left, right) =>
                (right.displayScore ?? right.score ?? 0) -
                (left.displayScore ?? left.score ?? 0),
        );

    if (
        !(queryPlan?.asksCoverageLike ?? false) &&
        !queryNeedsCoverageLikeTitle(normalizePatternText(query))
    ) {
        return scoredDocuments;
    }

    const remaining = [...scoredDocuments];
    const selected: PipelineDocumentRecord[] = [];
    const seenTitleKeys = new Map<string, number>();

    while (remaining.length > 0) {
        let bestIndex = 0;
        let bestAdjustedScore = Number.NEGATIVE_INFINITY;

        remaining.forEach((candidate, index) => {
            const baseScore =
                candidate.displayScore ?? candidate.coarseScore ?? candidate.score ?? 0;
            const titleKey = normalizeTitleDedupKey(candidate.ot_title || "");
            const seenCount = titleKey ? (seenTitleKeys.get(titleKey) ?? 0) : 0;
            const adjustedScore =
                baseScore - seenCount * TITLE_DIVERSITY_DUPLICATE_PENALTY;

            if (adjustedScore > bestAdjustedScore) {
                bestAdjustedScore = adjustedScore;
                bestIndex = index;
            }
        });

        const chosen = remaining.splice(bestIndex, 1)[0];
        const titleKey = normalizeTitleDedupKey(chosen.ot_title || "");
        if (titleKey) {
            seenTitleKeys.set(titleKey, (seenTitleKeys.get(titleKey) ?? 0) + 1);
        }
        selected.push({
            ...chosen,
            score: bestAdjustedScore,
            displayScore: bestAdjustedScore,
        });
    }

    return selected;
}

function resolveDynamicFetchLimit(
    baseLimit: number,
    delta: number,
    maxLimit: number,
): number {
    return Math.max(baseLimit, Math.min(baseLimit + delta, maxLimit));
}

export function mergeCoarseMatchesIntoDocuments(
    documents: PipelineDocumentRecord[],
    coarseMatches: Array<{ otid: string; score: number; best_kpid?: string }>,
): PipelineDocumentRecord[] {
    const documentMap = new Map(
        documents.map((doc) => [doc.otid || doc.id || "", doc]),
    );

    return coarseMatches
        .map((match) => {
            const doc = documentMap.get(match.otid);
            if (!doc) {
                return null;
            }

            return {
                ...doc,
                score: match.score ?? doc.score,
                coarseScore: match.score ?? doc.coarseScore ?? doc.score,
                displayScore: match.score ?? doc.displayScore ?? doc.score,
                best_kpid: match.best_kpid ?? doc.best_kpid,
            };
        })
        .filter(Boolean) as PipelineDocumentRecord[];
}

export function executeRetrievalStage(params: {
    query: string;
    queryVector: Float32Array;
    queryContext: SearchPipelineQueryContext;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    termMaps?: PipelineTermMaps;
    preset?: PipelinePreset;
}): RetrievalStageResult {
    const {
        query,
        queryVector,
        queryContext,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        termMaps,
        preset = CANONICAL_PIPELINE_PRESET,
    } = params;

    const startedAt = nowMs();
    const searchOutput = searchAndRank({
        queryVector,
        querySparse: queryContext.querySparse,
        queryWords: queryContext.queryWords,
        queryYearWordIds: queryContext.queryYearWordIds,
        queryIntent: queryContext.queryIntent,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        candidateIndices: queryContext.candidateIndices,
        scopeSpecificityWordIdToTerm:
            termMaps?.scopeSpecificityWordIdToTerm,
        weights: preset.retrieval.weights,
        topHybridLimit: preset.retrieval.topHybridLimit,
        kpAggregationMode: preset.retrieval.kpAggregationMode,
        kpTopN: preset.retrieval.kpTopN,
        kpTailWeight: preset.retrieval.kpTailWeight,
        lexicalBonusMode: preset.retrieval.lexicalBonusMode,
        kpRoleRerankMode: preset.retrieval.kpRoleRerankMode,
        kpRoleDocWeight: preset.retrieval.kpRoleDocWeight,
        qConfusionMode: preset.retrieval.qConfusionMode,
        qConfusionWeight: preset.retrieval.qConfusionWeight,
        enableExplicitYearFilter: preset.retrieval.enableExplicitYearFilter,
        queryPlan: queryContext.queryPlan,
        enableQueryPlanner: preset.display.enableQueryPlanner,
        minimalMode: preset.retrieval.minimalMode,
    });

    return {
        queryContext,
        searchOutput,
        retrievalDecision: buildPipelineDecision({
            query,
            queryIntent: queryContext.queryIntent,
            searchOutput,
        }),
        candidateCount: queryContext.candidateIndices?.length ?? metadata.length,
        searchMs: nowMs() - startedAt,
    };
}

export async function executeSearchPipeline(params: {
    query: string;
    queryVector: Float32Array;
    queryContext: SearchPipelineQueryContext;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    extractor: FeatureExtractionPipeline;
    documentLoader: PipelineDocumentLoader;
    termMaps?: PipelineTermMaps;
    preset?: PipelinePreset;
    onStatus?: (message: string) => void;
}): Promise<SearchPipelineResult> {
    const {
        query,
        queryVector,
        queryContext,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        extractor: _extractor,
        documentLoader,
        termMaps,
        preset = CANONICAL_PIPELINE_PRESET,
        onStatus,
    } = params;

    const pipelineStartedAt = nowMs();
    const retrievalStage = executeRetrievalStage({
        query,
        queryVector,
        queryContext,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        termMaps,
        preset,
    });

    const { searchOutput, retrievalDecision } = retrievalStage;
    const plannerEnabled = preset.display.enableQueryPlanner;
    const fetchMatchLimit = resolveDynamicFetchLimit(
        preset.display.fetchMatchLimit,
        plannerEnabled ? queryContext.queryPlan.fetchMatchLimitDelta : 0,
        28,
    );
    const fetchWeakMatchLimit = resolveDynamicFetchLimit(
        preset.display.fetchWeakMatchLimit,
        plannerEnabled ? queryContext.queryPlan.fetchWeakMatchLimitDelta : 0,
        18,
    );
    const shouldFetchWeakResults =
        retrievalDecision.behavior === "reject" &&
        searchOutput.rejection?.reason === "low_topic_coverage";
    const matchIds =
        retrievalDecision.behavior === "answer"
            ? searchOutput.matches
                  .slice(0, fetchMatchLimit)
                  .map((item) => item.otid)
            : [];
    const weakMatchIds = shouldFetchWeakResults
        ? searchOutput.weakMatches
              .slice(0, fetchWeakMatchLimit)
              .map((item) => item.otid)
        : [];
    const fetchIds = dedupe([...matchIds, ...weakMatchIds]);

    let fetchMs = 0;
    let fetchedDocumentCount = 0;
    let results: PipelineDocumentRecord[] = [];
    let weakResults: PipelineDocumentRecord[] = [];
    let finalDecision: PipelineDecision = retrievalDecision;

    if (fetchIds.length > 0) {
        onStatus?.("正在请求原文数据...");
        const fetchStartedAt = nowMs();
        const documents = await documentLoader({
            query,
            otids: fetchIds,
        });
        fetchMs = nowMs() - fetchStartedAt;
        fetchedDocumentCount = documents.length;

        if (retrievalDecision.behavior === "answer") {
            const directDocuments = mergeCoarseMatchesIntoDocuments(
                documents,
                searchOutput.matches
                    .slice(0, fetchMatchLimit)
                    .map((item) => ({
                        otid: item.otid,
                        score: item.score,
                        best_kpid: item.best_kpid,
                    })),
            );
            const phaseAdjustedDocuments =
                preset.retrieval.enablePhaseAnchorBoost
                    ? applyPhaseAnchorBoostToDocuments(
                          query,
                          directDocuments,
                          plannerEnabled ? queryContext.queryPlan : undefined,
                      )
                    : directDocuments;
            const titleAdjustedDocuments =
                preset.display.useYearPhaseTitleAdjustment
                    ? applyTitleIntentBoostToDocuments(
                          query,
                          phaseAdjustedDocuments,
                          plannerEnabled ? queryContext.queryPlan : undefined,
                      )
                    : phaseAdjustedDocuments;
            const coverageAdjustedDocuments =
                preset.display.useYearPhaseTitleAdjustment
                    ? applyCoverageBoostToDocuments(
                          query,
                          titleAdjustedDocuments,
                          plannerEnabled ? queryContext.queryPlan : undefined,
                      )
                    : titleAdjustedDocuments;
            results = coverageAdjustedDocuments;
        }

        if (shouldFetchWeakResults) {
            weakResults = mergeCoarseMatchesIntoDocuments(
                documents,
                searchOutput.weakMatches
                    .slice(0, fetchWeakMatchLimit)
                    .map((item) => ({
                        otid: item.otid,
                        score: item.score,
                        best_kpid: item.best_kpid,
                    })),
            );
        }
    }

    return {
        query,
        presetName: preset.name,
        queryContext,
        searchOutput,
        responseDecision: searchOutput.responseDecision,
        retrievalDecision,
        finalDecision,
        rejection: searchOutput.rejection,
        results,
        weakResults,
        trace: {
            totalMs: nowMs() - pipelineStartedAt,
            searchMs: retrievalStage.searchMs,
            fetchMs,
            candidateCount: retrievalStage.candidateCount,
            partitionUsed: Boolean(queryContext.candidateIndices),
            partitionCandidateCount: queryContext.candidateIndices?.length,
            matchCount: searchOutput.matches.length,
            weakMatchCount: searchOutput.weakMatches.length,
            fetchedDocumentCount,
            querySignals: searchOutput.diagnostics?.querySignals,
            retrievalSignals: searchOutput.diagnostics?.retrievalSignals,
            queryPlan: plannerEnabled ? queryContext.queryPlan : undefined,
        },
    };
}

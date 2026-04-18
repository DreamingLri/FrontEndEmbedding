import { fmmTokenize } from "./fmm_tokenize.ts";
import {
    buildQueryPlan,
    inferDocumentRolesFromTitle,
    type QueryPlan,
    type QueryPlanDocRole,
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
const LATEST_VERSION_DOC_WEIGHT = 0.38;
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

type QueryAspectRule = (typeof QUERY_ASPECT_RULES)[number];

type DocumentRerankQuerySignals = {
    normalizedQuery: string;
    asksOutcomeLikeTitle: boolean;
    asksProcedureLikeTitle: boolean;
    asksRequirementLikeTitle: boolean;
    asksEventDateLikeTitle: boolean;
    asksPolicyOverviewLikeTitle: boolean;
    asksSystemTimelineLikeTitle: boolean;
    asksBroadRuleDocLikeTitle: boolean;
    isCompressedKeywordQuery: boolean;
    mentionsAiSchool: boolean;
    mentionsDoctoral: boolean;
    mentionsTuimian: boolean;
    mentionsSummerCamp: boolean;
    mentionsTransfer: boolean;
    asksPostOutcomeAdmission: boolean;
    asksMaterialReviewTiming: boolean;
    asksPostOutcomeOperationalDetail: boolean;
    asksCampExecutionDetail: boolean;
    asksCompressedNoticeLike: boolean;
    asksCompressedOutcomeLike: boolean;
    asksCompressedConstraintLike: boolean;
    hasCompressedThemeCue: boolean;
    hasCompressedIntentCue: boolean;
    asksTimelineNodeLike: boolean;
    asksSystemOperationLike: boolean;
    mentionsRegistration: boolean;
    requestedAspects: QueryAspectRule[];
    queryPhase: PhaseAnchor;
    hasExplicitPhaseAnchor: boolean;
    wantsLatestVersion: boolean;
    roleSensitiveLatestVersion: boolean;
    wantsCoverageDiversity: boolean;
    phaseAnchorWeight: number;
    titleIntentWeight: number;
    coverageWeight: number;
    latestVersionWeight: number;
};

type DocumentRerankMetadata = {
    normalizedTitle: string;
    titleDedupKey: string;
    latestVersionFamilyKey: string;
    recencyKey?: number;
    evidenceText: string;
    phaseAnchor: PhaseAnchor;
    roles: QueryPlanDocRole[];
    isRuleDocTitle: boolean;
    isProcessNoticeTitle: boolean;
    isOutcomeTitle: boolean;
    isAiSchoolTitle: boolean;
    isOtherProgramTitle: boolean;
    isPreapplyTitle: boolean;
    isTransferTitle: boolean;
    isCandidateListTitle: boolean;
    isReviewResultTitle: boolean;
    isSummerCampTitle: boolean;
    isTuimianTitle: boolean;
    isDoctoralTitle: boolean;
    isMasterOnlyTitle: boolean;
    isSystemNoticeTitle: boolean;
    isRuleDocRole: boolean;
    isRegistrationNoticeRole: boolean;
    isResultNoticeRole: boolean;
    isListNoticeRole: boolean;
    isStageListRole: boolean;
    isAdjustmentNoticeRole: boolean;
    isConstraintRoleDoc: boolean;
    isOperationalRoleDoc: boolean;
    isOutcomeRoleDoc: boolean;
    hasCollegeTitle: boolean;
};

type DocumentRerankEntry = {
    document: PipelineDocumentRecord;
    metadata: DocumentRerankMetadata;
};

type DocumentRerankEntryLookup = {
    entry: DocumentRerankEntry;
    index: number;
};

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

function queryAsksBroadRuleDocLikeTitle(query: string): boolean {
    return /(招生简章|简章|招生章程|章程|实施细则|细则|实施办法|办法|接收办法|录取方案|方案|专业目录|目录)/.test(
        query,
    );
}

function queryNeedsCoverageLikeTitle(query: string): boolean {
    return /分别|以及|并描述|整个流程|申请和录取过程|从预报名到录取|从准备材料到完成面试|条件.*评分|条件.*材料|材料.*时间|申请.*录取过程/.test(
        query,
    );
}

function queryHasPostOutcomeActionCue(query: string): boolean {
    return /体检表|复审表|书面说明|签字|递补|增补|放弃录取|放弃资格/.test(
        query,
    );
}

function queryHasContactChannelCue(query: string): boolean {
    return /联系方式|联系电话|邮箱|邮件|联系学院|联系老师|研究生办公室/.test(
        query,
    );
}

function queryHasResultCommunicationContextCue(query: string): boolean {
    return /拟录取|录取|复试|调剂|结果|公示|名单|监督|申诉|沟通/.test(query);
}

function queryHasCampUpdateChannelCue(query: string): boolean {
    return /更新/.test(query) && /联系方式|联系电话|邮箱|邮件|通信渠道/.test(query);
}

function queryHasCampOperationalCue(query: string): boolean {
    return /报到|营员|入营/.test(query) || queryHasCampUpdateChannelCue(query);
}

function queryHasCampStatisticsCue(query: string): boolean {
    return /人数|总人数|录取人数|男女|男生|女生/.test(query);
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

function normalizeLatestVersionFamilyKey(title: string): string {
    return normalizeTitleDedupKey(title)
        .replace(/（[^）]*）/g, "")
        .replace(/\([^)]*\)/g, "")
        .replace(/第?[一二三四1234]批/g, "")
        .replace(/上半年|下半年/g, "")
        .replace(/第?[一二三四1234]轮/g, "");
}

function resolveDocumentRecencyKey(
    document: Pick<PipelineDocumentRecord, "publish_time" | "ot_title">,
): number | undefined {
    const rawCandidates = [document.publish_time, document.ot_title];
    for (const rawValue of rawCandidates) {
        const raw = (rawValue || "").trim();
        if (!raw) {
            continue;
        }

        const fullDateMatch = raw.match(
            /(20\d{2})[.\-/年](\d{1,2})[.\-/月](\d{1,2})/,
        );
        if (fullDateMatch) {
            const year = Number(fullDateMatch[1]);
            const month = Number(fullDateMatch[2]);
            const day = Number(fullDateMatch[3]);
            return year * 372 + month * 31 + day;
        }

        const yearMonthMatch = raw.match(/(20\d{2})[.\-/年](\d{1,2})[.\-/月]?/);
        if (yearMonthMatch) {
            const year = Number(yearMonthMatch[1]);
            const month = Number(yearMonthMatch[2]);
            return year * 372 + month * 31 + 1;
        }

        const yearMatch = raw.match(/(20\d{2})年?/);
        if (yearMatch) {
            const year = Number(yearMatch[1]);
            return year * 372 + 1;
        }
    }

    return undefined;
}

function buildDocumentRerankMetadata(
    document: PipelineDocumentRecord,
): DocumentRerankMetadata {
    const title = document.ot_title || "";
    const normalizedTitle = normalizePatternText(title);
    const roles = inferDocumentRolesFromTitle(title);
    const isDoctoralTitle = TITLE_DOCTORAL_PATTERN.test(normalizedTitle);
    const isTuimianTitle = TITLE_TUIMIAN_PATTERN.test(normalizedTitle);
    const isRuleDocRole = roles.includes("rule_doc");
    const isRegistrationNoticeRole = roles.includes("registration_notice");
    const isResultNoticeRole = roles.includes("result_notice");
    const isListNoticeRole = roles.includes("list_notice");
    const isStageListRole = roles.includes("stage_list");
    const isAdjustmentNoticeRole = roles.includes("adjustment_notice");

    return {
        normalizedTitle,
        titleDedupKey: normalizeTitleDedupKey(title),
        latestVersionFamilyKey: normalizeLatestVersionFamilyKey(title),
        recencyKey: resolveDocumentRecencyKey(document),
        evidenceText: getDocumentEvidenceText(document),
        phaseAnchor: extractPhaseAnchor(title),
        roles,
        isRuleDocTitle: TITLE_RULE_DOC_PATTERN.test(normalizedTitle),
        isProcessNoticeTitle: TITLE_PROCESS_NOTICE_PATTERN.test(normalizedTitle),
        isOutcomeTitle: TITLE_OUTCOME_PATTERN.test(normalizedTitle),
        isAiSchoolTitle: TITLE_AI_SCHOOL_PATTERN.test(normalizedTitle),
        isOtherProgramTitle: TITLE_OTHER_PROGRAM_PATTERN.test(normalizedTitle),
        isPreapplyTitle: TITLE_PREAPPLY_PATTERN.test(normalizedTitle),
        isTransferTitle: TITLE_TRANSFER_PATTERN.test(normalizedTitle),
        isCandidateListTitle: TITLE_CANDIDATE_LIST_PATTERN.test(normalizedTitle),
        isReviewResultTitle: TITLE_REVIEW_RESULT_PATTERN.test(normalizedTitle),
        isSummerCampTitle: TITLE_SUMMER_CAMP_PATTERN.test(normalizedTitle),
        isTuimianTitle,
        isDoctoralTitle,
        isMasterOnlyTitle:
            TITLE_MASTER_PATTERN.test(normalizedTitle) &&
            !isDoctoralTitle &&
            !isTuimianTitle,
        isSystemNoticeTitle: TITLE_SYSTEM_NOTICE_PATTERN.test(normalizedTitle),
        isRuleDocRole,
        isRegistrationNoticeRole,
        isResultNoticeRole,
        isListNoticeRole,
        isStageListRole,
        isAdjustmentNoticeRole,
        isConstraintRoleDoc: isRuleDocRole || isRegistrationNoticeRole,
        isOperationalRoleDoc:
            isRegistrationNoticeRole || isStageListRole || isAdjustmentNoticeRole,
        isOutcomeRoleDoc: isResultNoticeRole || isListNoticeRole,
        hasCollegeTitle: /学院/.test(normalizedTitle),
    };
}

function buildDocumentRerankEntries(
    documents: PipelineDocumentRecord[],
): DocumentRerankEntry[] {
    return documents.map((document) => ({
        document,
        metadata: buildDocumentRerankMetadata(document),
    }));
}

function queryWantsLatestVersion(query: string): boolean {
    return /现在|最新|最近|最近一次|目前|当前/.test(query);
}

type LatestVersionFamilyStat = {
    count: number;
    latestRecencyKey?: number;
};

function getDocumentDisplayScore(document: PipelineDocumentRecord): number {
    return document.displayScore ?? document.coarseScore ?? document.score ?? 0;
}

function getDocumentCoarseScore(document: PipelineDocumentRecord): number {
    return document.coarseScore ?? document.score ?? getDocumentDisplayScore(document);
}

function sortDocumentRerankEntriesByDisplayScore(
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    return [...entries].sort(
        (left, right) =>
            getDocumentDisplayScore(right.document) -
            getDocumentDisplayScore(left.document),
    );
}

function updateDocumentScores(
    document: PipelineDocumentRecord,
    displayDelta: number,
    coarseDelta?: number,
): PipelineDocumentRecord {
    const nextDisplayScore = getDocumentDisplayScore(document) + displayDelta;
    return {
        ...document,
        score: nextDisplayScore,
        coarseScore:
            coarseDelta === undefined
                ? document.coarseScore
                : getDocumentCoarseScore(document) + coarseDelta,
        displayScore: nextDisplayScore,
    };
}

function updateDocumentRerankEntryScores(
    entry: DocumentRerankEntry,
    displayDelta: number,
    coarseDelta?: number,
): DocumentRerankEntry {
    return {
        document: updateDocumentScores(entry.document, displayDelta, coarseDelta),
        metadata: entry.metadata,
    };
}

function getDocumentsFromRerankEntries(
    entries: DocumentRerankEntry[],
): PipelineDocumentRecord[] {
    return entries.map((entry) => entry.document);
}

function buildDocumentRerankEntryLookup(
    entries: DocumentRerankEntry[],
): Map<string, DocumentRerankEntryLookup> {
    const lookup = new Map<string, DocumentRerankEntryLookup>();
    entries.forEach((entry, index) => {
        const otid = entry.document.otid;
        if (!otid || lookup.has(otid)) {
            return;
        }
        lookup.set(otid, { entry, index });
    });
    return lookup;
}

function buildLatestVersionFamilyStats(
    entries: DocumentRerankEntry[],
): Map<string, LatestVersionFamilyStat> {
    const familyStats = new Map<string, LatestVersionFamilyStat>();

    entries.forEach((entry) => {
        const { latestVersionFamilyKey, recencyKey } = entry.metadata;
        const familyKey = latestVersionFamilyKey;
        if (!familyKey) {
            return;
        }

        const existing = familyStats.get(familyKey) || { count: 0 };
        existing.count += 1;
        if (
            recencyKey !== undefined &&
            (existing.latestRecencyKey === undefined ||
                recencyKey > existing.latestRecencyKey)
        ) {
            existing.latestRecencyKey = recencyKey;
        }
        familyStats.set(familyKey, existing);
    });

    return familyStats;
}

function computeLatestVersionDocDelta(params: {
    entry: DocumentRerankEntry;
    familyStats: Map<string, LatestVersionFamilyStat>;
    querySignals: DocumentRerankQuerySignals;
}): number {
    const { entry, familyStats, querySignals } = params;
    const { metadata } = entry;
    const familyKey = metadata.latestVersionFamilyKey;
    const familyStat = familyKey ? familyStats.get(familyKey) : undefined;
    const recencyKey = metadata.recencyKey;
    const roles = metadata.roles;
    let delta = 0;

    if (familyStat && familyStat.count >= 2) {
        if (
            recencyKey !== undefined &&
            familyStat.latestRecencyKey !== undefined
        ) {
            const gapMonths = (familyStat.latestRecencyKey - recencyKey) / 31;
            if (gapMonths <= 0) {
                delta += 0.92;
                if (
                    querySignals.roleSensitiveLatestVersion &&
                    (roles.includes("rule_doc") ||
                        roles.includes("registration_notice") ||
                        roles.includes("stage_list"))
                ) {
                    delta += 0.12;
                }
            } else {
                delta -= Math.min(1.05, 0.2 + gapMonths * 0.08);
            }
        } else {
            delta -= 0.18;
        }
    }

    if (
        !querySignals.asksOutcomeLikeTitle &&
        querySignals.roleSensitiveLatestVersion &&
        (roles.includes("result_notice") || roles.includes("list_notice"))
    ) {
        delta -= 0.18;
    }

    return delta;
}

function queryIsCompressedKeywordLike(query: string): boolean {
    const normalized = normalizePatternText(query);
    return (
        normalized.length <= 12 &&
        /20\d{2}/.test(normalized) &&
        !/[，。；！？,.!?]/.test(query)
    );
}

function applyCompressedQueryDisplayGuardToEntries(
    querySignals: DocumentRerankQuerySignals,
    baselineEntries: DocumentRerankEntry[],
    rerankedEntries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    if (
        !querySignals.isCompressedKeywordQuery ||
        baselineEntries.length === 0 ||
        rerankedEntries.length === 0
    ) {
        return rerankedEntries;
    }

    const baselineTop = baselineEntries[0];
    const rerankedTop = rerankedEntries[0];
    const baselineTopOtid = baselineTop.document.otid;
    const rerankedTopOtid = rerankedTop.document.otid;
    if (
        !baselineTopOtid ||
        !rerankedTopOtid ||
        baselineTopOtid === rerankedTopOtid
    ) {
        return rerankedEntries;
    }

    const baselineTopScore = getDocumentDisplayScore(baselineTop.document);
    const baselineLookup = buildDocumentRerankEntryLookup(baselineEntries);
    const rerankedTopBaseline = baselineLookup.get(rerankedTopOtid)?.entry;
    const rerankedTopBaselineScore = rerankedTopBaseline
        ? getDocumentDisplayScore(rerankedTopBaseline.document)
        : Number.NEGATIVE_INFINITY;

    if (baselineTopScore + 0.06 < rerankedTopBaselineScore) {
        return rerankedEntries;
    }

    const rerankedLookup = buildDocumentRerankEntryLookup(rerankedEntries);
    const preservedBaselineLookup = rerankedLookup.get(baselineTopOtid);
    if (!preservedBaselineLookup) {
        return rerankedEntries;
    }

    const boostedTopScore = getDocumentDisplayScore(rerankedTop.document) + 0.001;
    const reordered = [...rerankedEntries];
    const [preservedBaseline] = reordered.splice(preservedBaselineLookup.index, 1);
    reordered.unshift({
        document: {
            ...preservedBaseline.document,
            score: boostedTopScore,
            displayScore: boostedTopScore,
        },
        metadata: preservedBaseline.metadata,
    });
    return reordered;
}

function computeQueryPlanDocRoleDelta(
    roles: QueryPlanDocRole[],
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
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    if (!normalizedTitle) {
        return 0;
    }

    let delta = 0;

    if (!querySignals.asksOutcomeLikeTitle && metadata.isOutcomeTitle) {
        delta -= 0.95;
    }
    if (
        (querySignals.asksProcedureLikeTitle ||
            querySignals.asksRequirementLikeTitle) &&
        metadata.isRuleDocTitle
    ) {
        delta +=
            querySignals.asksProcedureLikeTitle &&
            querySignals.asksRequirementLikeTitle
                ? 0.95
                : 0.75;
    }
    if (querySignals.asksProcedureLikeTitle && metadata.isProcessNoticeTitle) {
        delta += 0.55;
    }

    if (querySignals.mentionsAiSchool) {
        if (metadata.isAiSchoolTitle) {
            delta += 0.45;
        } else if (metadata.isOtherProgramTitle) {
            delta -= 0.9;
        } else if (metadata.hasCollegeTitle) {
            delta -= 0.55;
        } else {
            delta -= 0.2;
        }
    }

    if (querySignals.mentionsDoctoral) {
        if (metadata.isMasterOnlyTitle) {
            delta -= 0.95;
        } else if (metadata.isDoctoralTitle) {
            delta += 0.2;
        }
    }

    if (querySignals.mentionsTuimian) {
        if (metadata.isTuimianTitle) {
            delta += 0.45;
        }
        if (metadata.isDoctoralTitle && !metadata.isTuimianTitle) {
            delta -= 0.6;
        }
    }

    if (querySignals.mentionsSummerCamp) {
        if (metadata.isSummerCampTitle) {
            delta += 0.35;
        }
        if (querySignals.asksCampExecutionDetail) {
            if (/入营通知/.test(normalizedTitle)) {
                delta += 0.95;
            }
            if (/活动报名通知|报名通知/.test(normalizedTitle)) {
                delta -= 0.75;
            }
        }
        if (
            !querySignals.asksEventDateLikeTitle &&
            /活动报名通知|报名通知/.test(normalizedTitle)
        ) {
            delta += 0.45;
        }
        if (
            !querySignals.asksOutcomeLikeTitle &&
            !querySignals.asksEventDateLikeTitle &&
            /入营通知/.test(normalizedTitle)
        ) {
            delta -= 0.45;
        }
        if (
            querySignals.asksEventDateLikeTitle &&
            /入营通知|活动通知/.test(normalizedTitle)
        ) {
            delta += 0.95;
        }
        if (
            querySignals.asksEventDateLikeTitle &&
            !querySignals.mentionsRegistration &&
            /活动报名通知|报名通知/.test(normalizedTitle)
        ) {
            delta -= 0.45;
        }
        if (
            !metadata.isSummerCampTitle &&
            (metadata.isDoctoralTitle ||
                metadata.isTuimianTitle ||
                metadata.isMasterOnlyTitle)
        ) {
            delta -= 0.65;
        }
    }

    if (querySignals.mentionsTuimian && /接收办法|工作方案/.test(normalizedTitle)) {
        delta += 0.45;
    }

    if (
        querySignals.mentionsDoctoral &&
        /实施办法|招生简章|综合考核通知/.test(normalizedTitle)
    ) {
        delta += 0.25;
    }

    if (querySignals.asksPolicyOverviewLikeTitle) {
        if (/招生简章|接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.55;
        }
        if (metadata.isPreapplyTitle) {
            delta -= 0.45;
        }
        if (metadata.isTransferTitle && !querySignals.mentionsTransfer) {
            delta -= 0.65;
        }
        if (metadata.isReviewResultTitle && !querySignals.asksOutcomeLikeTitle) {
            delta -= 0.55;
        }
    }

    if (querySignals.asksSystemTimelineLikeTitle) {
        if (/接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.45;
        }
        if (metadata.isPreapplyTitle && querySignals.asksSystemOperationLike) {
            delta -= 0.2;
        }
        if (metadata.isSystemNoticeTitle) {
            delta -= 0.55;
        }
    } else if (
        querySignals.asksTimelineNodeLike &&
        /报名通知|综合考核通知|复试通知/.test(normalizedTitle)
    ) {
        delta += 0.4;
    }

    if (querySignals.asksPostOutcomeAdmission) {
        if (/招生简章|实施办法/.test(normalizedTitle)) {
            delta += 0.4;
        }
        if (metadata.isReviewResultTitle) {
            delta -= 0.55;
        }
    }

    if (querySignals.asksMaterialReviewTiming) {
        if (metadata.isCandidateListTitle) {
            delta += 0.8;
        }
        if (/综合考核结果/.test(normalizedTitle)) {
            delta -= 0.35;
        }
    }

    if (querySignals.asksPostOutcomeOperationalDetail) {
        if (/复试结果|拟录取|增补拟录取|结果公示|名单公示/.test(normalizedTitle)) {
            delta += 1.25;
        }
        if (/增补拟录取|递补录取/.test(normalizedTitle)) {
            delta += 0.95;
        }
        if (/复试录取方案|调剂复试通知|招生简章|实施办法/.test(normalizedTitle)) {
            delta -= 1.15;
        }
    }

    if (
        querySignals.isCompressedKeywordQuery &&
        !querySignals.asksBroadRuleDocLikeTitle
    ) {
        if (!querySignals.asksCompressedConstraintLike && metadata.isRuleDocRole) {
            delta -= querySignals.asksCompressedOutcomeLike ? 0.95 : 0.55;
        }
        if (metadata.isSystemNoticeTitle) {
            delta -= 1.1;
        }
        if (metadata.isOtherProgramTitle) {
            delta -= 1.2;
        }
        if (
            querySignals.asksCompressedConstraintLike &&
            metadata.isConstraintRoleDoc
        ) {
            delta += querySignals.asksCompressedOutcomeLike ? 0.58 : 0.74;
        }
        if (
            querySignals.asksCompressedNoticeLike &&
            (metadata.isOperationalRoleDoc || metadata.isProcessNoticeTitle)
        ) {
            delta += querySignals.asksCompressedConstraintLike ? 0.34 : 0.48;
        }
        if (querySignals.asksCompressedOutcomeLike) {
            if (metadata.isOutcomeRoleDoc || metadata.isReviewResultTitle) {
                delta += querySignals.asksCompressedConstraintLike ? 0.28 : 0.56;
            }
            if (metadata.isStageListRole || metadata.isCandidateListTitle) {
                delta += querySignals.asksCompressedConstraintLike ? 0.18 : 0.32;
            }
        }
        if (
            querySignals.hasCompressedIntentCue &&
            !querySignals.asksCompressedConstraintLike &&
            metadata.isOperationalRoleDoc &&
            !metadata.isOutcomeRoleDoc
        ) {
            delta += 0.16;
        }
        if (
            querySignals.mentionsTuimian &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isTuimianTitle &&
                (metadata.isConstraintRoleDoc ||
                    metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc)
            ) {
                delta += 0.72;
            }
            if (metadata.isDoctoralTitle && !metadata.isTuimianTitle) {
                delta -= 0.88;
            }
        }
        if (
            querySignals.mentionsDoctoral &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isDoctoralTitle &&
                (metadata.isConstraintRoleDoc ||
                    metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc)
            ) {
                delta += 0.64;
            }
            if (metadata.isMasterOnlyTitle) {
                delta -= 0.82;
            }
        }
        if (
            querySignals.mentionsSummerCamp &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isSummerCampTitle &&
                (metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc ||
                    metadata.isStageListRole)
            ) {
                delta += 0.6;
            }
            if (
                !metadata.isSummerCampTitle &&
                (metadata.isDoctoralTitle ||
                    metadata.isTuimianTitle ||
                    metadata.isMasterOnlyTitle)
            ) {
                delta -= 0.58;
            }
        }
        if (
            querySignals.mentionsTransfer &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isTransferTitle &&
                (metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc ||
                    metadata.isAdjustmentNoticeRole)
            ) {
                delta += 0.58;
            }
        }
    }

    return delta;
}

function computeCoverageDocDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { requestedAspects } = querySignals;
    if (requestedAspects.length < 2) {
        return 0;
    }

    const evidenceText = entry.metadata.evidenceText;
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
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    if (!querySignals.hasExplicitPhaseAnchor) {
        return 0;
    }

    const queryPhase = querySignals.queryPhase;
    if (!entry.metadata.normalizedTitle) {
        return -0.15;
    }

    const articlePhase = entry.metadata.phaseAnchor;
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

function buildDocumentRerankQuerySignals(params: {
    query: string;
    queryPlan?: QueryPlan;
    preferLatestWithinTopic: boolean;
}): DocumentRerankQuerySignals {
    const { query, queryPlan, preferLatestWithinTopic } = params;
    const normalizedQuery = queryPlan?.normalizedQuery ?? normalizePatternText(query);
    const asksOutcomeLikeTitle = queryAsksOutcomeLikeTitle(normalizedQuery);
    const asksProcedureLikeTitle = queryAsksProcedureLikeTitle(normalizedQuery);
    const asksRequirementLikeTitle = queryAsksRequirementLikeTitle(normalizedQuery);
    const asksEventDateLikeTitle = queryAsksEventDateLikeTitle(normalizedQuery);
    const asksPolicyOverviewLikeTitle =
        queryAsksPolicyOverviewLikeTitle(normalizedQuery);
    const asksSystemTimelineLikeTitle =
        queryAsksSystemTimelineLikeTitle(normalizedQuery);
    const asksBroadRuleDocLikeTitle =
        queryAsksBroadRuleDocLikeTitle(normalizedQuery);
    const isCompressedKeywordQuery = queryIsCompressedKeywordLike(normalizedQuery);
    const mentionsAiSchool = /人工智能学院|AI学院/.test(normalizedQuery);
    const mentionsDoctoral = /博士/.test(normalizedQuery);
    const mentionsTuimian = /推免|推荐免试/.test(normalizedQuery);
    const mentionsSummerCamp = /夏令营/.test(normalizedQuery);
    const mentionsTransfer = /调剂/.test(normalizedQuery);
    const asksPostOutcomeAdmission =
        /通过考核后|还会被录取吗|确保.*录取|被.*录取/.test(normalizedQuery);
    const asksMaterialReviewTiming =
        /材料审核.*公示|通过材料审核/.test(normalizedQuery);
    const asksCampAudienceLike = /营员|入营|名单|报到/.test(normalizedQuery);
    const asksCompressedNoticeLike =
        /通知|时间安排|安排|细节|要点/.test(normalizedQuery);
    const asksCompressedOutcomeLike =
        /名单|结果|公示|复试|综合考核|调剂/.test(normalizedQuery);
    const asksCompressedConstraintLike =
        /条件|资格|要求|细节|要点/.test(normalizedQuery);
    const mentionsMasterOrGraduate = /硕士|研究生/.test(normalizedQuery);
    const hasCompressedThemeCue =
        mentionsTuimian ||
        mentionsDoctoral ||
        mentionsSummerCamp ||
        mentionsTransfer ||
        mentionsMasterOrGraduate;
    const hasCompressedIntentCue =
        asksCompressedConstraintLike ||
        asksCompressedNoticeLike ||
        asksCompressedOutcomeLike;
    const asksTimelineNodeLike =
        /关键时间节点|时间节点|截止日期|截止时间/.test(normalizedQuery);
    const asksSystemOperationLike = /录取过程|系统操作/.test(normalizedQuery);
    const mentionsRegistration = /报名/.test(normalizedQuery);
    const requestedAspects = QUERY_ASPECT_RULES.filter((rule) =>
        rule.query.test(normalizedQuery),
    );
    const queryPhase = extractPhaseAnchor(query);
    const hasExplicitQueryPhaseAnchor = hasExplicitPhaseAnchor(queryPhase);

    return {
        normalizedQuery,
        asksOutcomeLikeTitle,
        asksProcedureLikeTitle,
        asksRequirementLikeTitle,
        asksEventDateLikeTitle,
        asksPolicyOverviewLikeTitle,
        asksSystemTimelineLikeTitle,
        asksBroadRuleDocLikeTitle,
        isCompressedKeywordQuery,
        mentionsAiSchool,
        mentionsDoctoral,
        mentionsTuimian,
        mentionsSummerCamp,
        mentionsTransfer,
        asksPostOutcomeAdmission,
        asksMaterialReviewTiming,
        asksPostOutcomeOperationalDetail:
            queryHasPostOutcomeActionCue(normalizedQuery) ||
            (queryHasContactChannelCue(normalizedQuery) &&
                queryHasResultCommunicationContextCue(normalizedQuery)),
        asksCampExecutionDetail:
            queryHasCampOperationalCue(normalizedQuery) ||
            (queryHasCampStatisticsCue(normalizedQuery) && asksCampAudienceLike),
        asksCompressedNoticeLike,
        asksCompressedOutcomeLike,
        asksCompressedConstraintLike,
        hasCompressedThemeCue,
        hasCompressedIntentCue,
        asksTimelineNodeLike,
        asksSystemOperationLike,
        mentionsRegistration,
        requestedAspects,
        queryPhase,
        hasExplicitPhaseAnchor: hasExplicitQueryPhaseAnchor,
        wantsLatestVersion: queryWantsLatestVersion(normalizedQuery),
        roleSensitiveLatestVersion:
            asksProcedureLikeTitle ||
            asksRequirementLikeTitle ||
            asksSystemTimelineLikeTitle ||
            asksPolicyOverviewLikeTitle,
        wantsCoverageDiversity:
            (queryPlan?.asksCoverageLike ?? false) ||
            queryNeedsCoverageLikeTitle(normalizedQuery),
        phaseAnchorWeight:
            PHASE_ANCHOR_DOC_WEIGHT * (queryPlan?.phaseAnchorWeightScale ?? 1),
        titleIntentWeight:
            TITLE_INTENT_DOC_WEIGHT * (queryPlan?.titleIntentWeightScale ?? 1),
        coverageWeight:
            TITLE_COVERAGE_DOC_WEIGHT * (queryPlan?.coverageWeightScale ?? 1),
        latestVersionWeight:
            LATEST_VERSION_DOC_WEIGHT * (preferLatestWithinTopic ? 1.1 : 1),
    };
}

function applyPhaseAnchorBoostToDocuments(
    querySignals: DocumentRerankQuerySignals,
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    return sortDocumentRerankEntriesByDisplayScore(
        entries
            .map((entry) => {
                const delta =
                    computePhaseAnchorDocDelta(querySignals, entry) *
                    querySignals.phaseAnchorWeight;
                return updateDocumentRerankEntryScores(entry, delta, delta);
            }),
    );
}

function applyCoverageTitleDiversity(
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    const remaining = [...entries];
    const selected: DocumentRerankEntry[] = [];
    const seenTitleKeys = new Map<string, number>();

    while (remaining.length > 0) {
        let bestIndex = 0;
        let bestAdjustedScore = Number.NEGATIVE_INFINITY;

        remaining.forEach((candidate, index) => {
            const baseScore = getDocumentDisplayScore(candidate.document);
            const titleKey = candidate.metadata.titleDedupKey;
            const seenCount = titleKey ? (seenTitleKeys.get(titleKey) ?? 0) : 0;
            const adjustedScore =
                baseScore - seenCount * TITLE_DIVERSITY_DUPLICATE_PENALTY;

            if (adjustedScore > bestAdjustedScore) {
                bestAdjustedScore = adjustedScore;
                bestIndex = index;
            }
        });

        const chosen = remaining.splice(bestIndex, 1)[0];
        const titleKey = chosen.metadata.titleDedupKey;
        if (titleKey) {
            seenTitleKeys.set(titleKey, (seenTitleKeys.get(titleKey) ?? 0) + 1);
        }
        selected.push({
            document: {
                ...chosen.document,
                score: bestAdjustedScore,
                displayScore: bestAdjustedScore,
            },
            metadata: chosen.metadata,
        });
    }

    return selected;
}

function rerankAnswerDocuments(params: {
    query: string;
    documents: PipelineDocumentRecord[];
    queryPlan?: QueryPlan;
    enablePhaseAnchorBoost: boolean;
    applyTitleAdjustments: boolean;
    preferLatestWithinTopic: boolean;
}): PipelineDocumentRecord[] {
    const {
        query,
        documents,
        queryPlan,
        enablePhaseAnchorBoost,
        applyTitleAdjustments,
        preferLatestWithinTopic,
    } = params;
    const querySignals = buildDocumentRerankQuerySignals({
        query,
        queryPlan,
        preferLatestWithinTopic,
    });
    const documentEntries = buildDocumentRerankEntries(documents);
    const phaseAdjustedEntries = enablePhaseAnchorBoost
        ? applyPhaseAnchorBoostToDocuments(querySignals, documentEntries)
        : documentEntries;

    if (!applyTitleAdjustments) {
        const guardedEntries = applyCompressedQueryDisplayGuardToEntries(
            querySignals,
            phaseAdjustedEntries,
            phaseAdjustedEntries,
        );
        return getDocumentsFromRerankEntries(guardedEntries);
    }

    const shouldApplyLatestVersionBoost =
        querySignals.wantsLatestVersion && phaseAdjustedEntries.length > 1;
    const latestVersionFamilyStats = shouldApplyLatestVersionBoost
        ? buildLatestVersionFamilyStats(phaseAdjustedEntries)
        : undefined;
    const rerankedEntries = sortDocumentRerankEntriesByDisplayScore(
        phaseAdjustedEntries
            .map((entry) => {
                const titleDelta =
                    (computeTitleIntentDocDelta(querySignals, entry) +
                        (queryPlan
                            ? computeQueryPlanDocRoleDelta(
                                  entry.metadata.roles,
                                  queryPlan,
                              )
                            : 0)) *
                    querySignals.titleIntentWeight;
                const latestDelta =
                    shouldApplyLatestVersionBoost && latestVersionFamilyStats
                        ? computeLatestVersionDocDelta({
                              entry,
                              familyStats: latestVersionFamilyStats,
                              querySignals,
                          }) * querySignals.latestVersionWeight
                        : 0;
                const coverageDelta =
                    computeCoverageDocDelta(querySignals, entry) *
                    querySignals.coverageWeight;

                return updateDocumentRerankEntryScores(
                    entry,
                    titleDelta + latestDelta + coverageDelta,
                    titleDelta,
                );
            }),
    );
    const displayEntries = querySignals.wantsCoverageDiversity
        ? applyCoverageTitleDiversity(rerankedEntries)
        : rerankedEntries;
    const guardedEntries = applyCompressedQueryDisplayGuardToEntries(
        querySignals,
        phaseAdjustedEntries,
        displayEntries,
    );

    return getDocumentsFromRerankEntries(guardedEntries);
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
    const plannerQueryPlan = plannerEnabled ? queryContext.queryPlan : undefined;
    const isCompressedKeywordQuery = queryIsCompressedKeywordLike(query);
    const compressedQueryFetchDelta = isCompressedKeywordQuery ? 18 : 0;
    const shouldApplyTitleAdjustments =
        preset.display.useYearPhaseTitleAdjustment;
    const fetchMatchLimit = resolveDynamicFetchLimit(
        preset.display.fetchMatchLimit,
        (plannerQueryPlan?.fetchMatchLimitDelta ?? 0) +
            compressedQueryFetchDelta,
        isCompressedKeywordQuery ? 48 : 28,
    );
    const fetchWeakMatchLimit = resolveDynamicFetchLimit(
        preset.display.fetchWeakMatchLimit,
        plannerQueryPlan?.fetchWeakMatchLimitDelta ?? 0,
        18,
    );
    const shouldFetchAnswerResults = retrievalDecision.behavior === "answer";
    const shouldFetchWeakResults =
        retrievalDecision.behavior === "reject" &&
        searchOutput.rejection?.reason === "low_topic_coverage";
    const matchIds =
        shouldFetchAnswerResults
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

    if (fetchIds.length > 0) {
        onStatus?.("正在请求原文数据...");
        const fetchStartedAt = nowMs();
        const documents = await documentLoader({
            query,
            otids: fetchIds,
        });
        fetchMs = nowMs() - fetchStartedAt;
        fetchedDocumentCount = documents.length;

        if (shouldFetchAnswerResults) {
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
            results = rerankAnswerDocuments({
                query,
                documents: directDocuments,
                queryPlan: plannerQueryPlan,
                enablePhaseAnchorBoost: preset.retrieval.enablePhaseAnchorBoost,
                applyTitleAdjustments: shouldApplyTitleAdjustments,
                preferLatestWithinTopic:
                    searchOutput.responseDecision?.preferLatestWithinTopic ?? false,
            });
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
        finalDecision: retrievalDecision,
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
            queryPlan: plannerQueryPlan,
        },
    };
}

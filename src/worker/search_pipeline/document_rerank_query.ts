import type { QueryPlan } from "../query_planner.ts";
import {
    extractPhaseAnchor,
    hasExplicitPhaseAnchor,
    LATEST_VERSION_DOC_WEIGHT,
    PHASE_ANCHOR_DOC_WEIGHT,
    QUERY_ASPECT_RULES,
    normalizePatternText,
    queryWantsLatestVersion,
    TITLE_COVERAGE_DOC_WEIGHT,
    TITLE_INTENT_DOC_WEIGHT,
    type PhaseAnchor,
    type QueryAspectRule,
} from "./document_rerank_shared.ts";

export type DocumentRerankQuerySignals = {
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

export function queryIsCompressedKeywordLike(query: string): boolean {
    const normalized = normalizePatternText(query);
    return (
        normalized.length <= 12 &&
        /20\d{2}/.test(normalized) &&
        !/[，。；！？,.!?]/.test(query)
    );
}

export function buildDocumentRerankQuerySignals(params: {
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

import type { QueryPlan, QueryPlanDocRole } from "../query_planner.ts";
import { getAdaptiveRerankPlan } from "../rerank_helpers.ts";
import {
    buildDocumentRerankEntryLookup,
    getDocumentDisplayScore,
    getDocumentCoarseScore,
    sortDocumentRerankEntriesByDisplayScore,
    TITLE_DIVERSITY_DUPLICATE_PENALTY,
    updateDocumentRerankEntryScores,
    type DocumentRerankEntry,
    type LatestVersionFamilyStat,
} from "./document_rerank_shared.ts";
import type { DocumentRerankQuerySignals } from "./document_rerank_query.ts";

export type TitleIntentConfusionGate = {
    positiveScale: number;
    negativeScale: number;
    top1Top2Gap: number | null;
    rerankDocCount: number;
    sameTopicDensity: number;
    sameFamilyDensity: number;
};

export function computeLatestVersionDocDelta(params: {
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

export function applyCompressedQueryDisplayGuardToEntries(
    querySignals: DocumentRerankQuerySignals,
    baselineEntries: DocumentRerankEntry[],
    rerankedEntries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    // 极短关键词查询对规则扰动非常敏感。
    // 这里保留粗排第一名的兜底权，避免展示排序被标题修正规则过度放大。
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

export function applyYearlessSameFamilyFreshnessGuardToEntries(
    querySignals: DocumentRerankQuerySignals,
    baselineEntries: DocumentRerankEntry[],
    rerankedEntries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    // 对未显式给出年份的 query，如果 coarse top 与 rerank top 只是同一家族跨年份通知互换，
    // 则不允许展示层仅凭更丰富的结构证据把较新的 baseline top 换成更旧版本。
    if (
        querySignals.queryYears.length > 0 ||
        querySignals.wantsLatestVersion ||
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

    const baselineFamilyKey = baselineTop.metadata.latestVersionFamilyKey;
    const rerankedFamilyKey = rerankedTop.metadata.latestVersionFamilyKey;
    if (
        !baselineFamilyKey ||
        !rerankedFamilyKey ||
        baselineFamilyKey !== rerankedFamilyKey
    ) {
        return rerankedEntries;
    }

    const baselineRecencyKey = baselineTop.metadata.recencyKey;
    const rerankedRecencyKey = rerankedTop.metadata.recencyKey;
    if (
        baselineRecencyKey === undefined ||
        rerankedRecencyKey === undefined ||
        baselineRecencyKey <= rerankedRecencyKey
    ) {
        return rerankedEntries;
    }

    if (baselineRecencyKey - rerankedRecencyKey < 31) {
        return rerankedEntries;
    }

    const baselineTopScore = getDocumentDisplayScore(baselineTop.document);
    const baselineLookup = buildDocumentRerankEntryLookup(baselineEntries);
    const rerankedTopBaseline = baselineLookup.get(rerankedTopOtid)?.entry;
    const rerankedTopBaselineScore = rerankedTopBaseline
        ? getDocumentDisplayScore(rerankedTopBaseline.document)
        : Number.NEGATIVE_INFINITY;
    if (baselineTopScore + 0.08 < rerankedTopBaselineScore) {
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

export function computeQueryPlanDocRoleDelta(
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

type TitleIntentDocDeltaOptions = {
    enableStructuredKpRoleEvidenceAdjustments?: boolean;
    enableLexicalTitleIntentAdjustments?: boolean;
    enableLexicalTitleTypeAdjustments?: boolean;
    enableLexicalScenarioAdjustments?: boolean;
    enableThemeSpecificAdjustments?: boolean;
    enableDoctoralThemeAdjustments?: boolean;
    enableTuimianThemeAdjustments?: boolean;
    enableSummerCampThemeAdjustments?: boolean;
    enableTransferThemeAdjustments?: boolean;
    enableCompressedKeywordAdjustments?: boolean;
};

const RAW_THEME_FALLBACK_WEIGHT_WITH_STRUCTURED_SIGNAL = 0.45;
const STRUCTURED_OUTCOME_EVENT_TYPES = ["录取公示", "推免资格公示"];
const STRUCTURED_PROCEDURE_EVENT_TYPES = [
    "报名通知",
    "考试安排",
    "材料提交",
    "复试通知",
    "推免实施办法",
];
const STRUCTURED_REQUIREMENT_EVENT_TYPES = [
    "招生章程",
    "推免实施办法",
    "资格要求",
    "材料提交",
];
const STRUCTURED_TIMELINE_EVENT_TYPES = [
    "报名通知",
    "考试安排",
    "复试通知",
];
const KP_ROLE_REQUIREMENT_TAGS = [
    "condition",
    "materials",
    "email",
    "deadline",
];
const KP_ROLE_PROCEDURE_TAGS = [
    "procedure",
    "application_stage",
    "schedule",
];
const KP_ROLE_TIMELINE_TAGS = [
    "arrival",
    "deadline",
    "announcement_period",
    "time_expression",
    "schedule",
    "location",
];
const KP_ROLE_POST_OUTCOME_TAGS = ["post_outcome"];

function hasStructuredOverlap(a: string[], b: string[]): boolean {
    if (a.length === 0 || b.length === 0) {
        return false;
    }
    return a.some((item) => b.includes(item));
}

function clampScale(value: number, min = 0.28, max = 1): number {
    return Math.min(max, Math.max(min, value));
}

function computeSameTopicDensity(entries: DocumentRerankEntry[]): number {
    if (entries.length === 0) {
        return 0;
    }

    const topicCounts = new Map<string, number>();
    entries.forEach((entry) => {
        Array.from(new Set(entry.metadata.structuredTopicIds)).forEach((topicId) => {
            topicCounts.set(topicId, (topicCounts.get(topicId) || 0) + 1);
        });
    });

    const maxTopicCount = Math.max(0, ...topicCounts.values());
    return maxTopicCount / entries.length;
}

function computeSameFamilyDensity(entries: DocumentRerankEntry[]): number {
    if (entries.length === 0) {
        return 0;
    }

    const familyCounts = new Map<string, number>();
    entries.forEach((entry) => {
        const familyKey = entry.metadata.latestVersionFamilyKey;
        if (!familyKey) {
            return;
        }
        familyCounts.set(familyKey, (familyCounts.get(familyKey) || 0) + 1);
    });

    const maxFamilyCount = Math.max(0, ...familyCounts.values());
    return maxFamilyCount / entries.length;
}

export function buildTitleIntentConfusionGate(
    querySignals: DocumentRerankQuerySignals,
    entries: DocumentRerankEntry[],
): TitleIntentConfusionGate {
    const sortedEntries = [...entries].sort(
        (left, right) =>
            getDocumentCoarseScore(right.document) -
            getDocumentCoarseScore(left.document),
    );
    const adaptivePlan = getAdaptiveRerankPlan(
        sortedEntries.map((entry) => ({
            coarseScore: getDocumentCoarseScore(entry.document),
        })),
    );
    const topWindowSize = Math.min(
        sortedEntries.length,
        Math.max(3, adaptivePlan.rerankDocCount),
    );
    const topWindow = sortedEntries.slice(0, topWindowSize);
    const sameTopicDensity = computeSameTopicDensity(topWindow);
    const sameFamilyDensity = computeSameFamilyDensity(topWindow);
    const top1Top2Gap = adaptivePlan.top1Top2Gap;
    const gap = top1Top2Gap ?? Number.POSITIVE_INFINITY;
    const broadFlowOrPolicyQuery =
        querySignals.asksFlowOverviewLike ||
        querySignals.asksPolicyOverviewLikeTitle;

    let positiveScale = 1;
    if (adaptivePlan.reason === "top1_clear_lead") {
        positiveScale *= broadFlowOrPolicyQuery
            ? 0.18
            : querySignals.wantsCoverageDiversity
              ? 0.34
              : 0.28;
    } else if (gap > 0.15) {
        positiveScale *= broadFlowOrPolicyQuery ? 0.42 : 0.56;
    } else if (gap > 0.1) {
        positiveScale *= broadFlowOrPolicyQuery ? 0.6 : 0.72;
    } else if (gap > 0.06 && adaptivePlan.rerankDocCount <= 2) {
        positiveScale *= 0.88;
    }

    if (
        !querySignals.wantsLatestVersion &&
        !querySignals.roleSensitiveLatestVersion &&
        sameFamilyDensity >= 0.67
    ) {
        positiveScale *= 0.78;
    }

    if (
        broadFlowOrPolicyQuery &&
        sameTopicDensity >= 0.67 &&
        gap > 0.1
    ) {
        positiveScale *= 0.72;
    }

    const clampedPositiveScale = clampScale(positiveScale, 0.18);
    const negativeScale = clampScale(0.38 + clampedPositiveScale * 0.62, 0.38);

    return {
        positiveScale: clampedPositiveScale,
        negativeScale,
        top1Top2Gap: adaptivePlan.top1Top2Gap,
        rerankDocCount: adaptivePlan.rerankDocCount,
        sameTopicDensity,
        sameFamilyDensity,
    };
}

export function applyTitleIntentConfusionGate(
    delta: number,
    gate?: TitleIntentConfusionGate,
): number {
    if (!gate || delta === 0) {
        return delta;
    }
    return delta > 0
        ? delta * gate.positiveScale
        : delta * gate.negativeScale;
}

function resolveDocumentYearFromRecencyKey(
    recencyKey?: number,
): number | undefined {
    if (!Number.isFinite(recencyKey)) {
        return undefined;
    }
    return Math.floor(Number(recencyKey) / 372);
}

function hasExplicitStructuredMismatch(
    queryValues: string[],
    documentValues: string[],
): boolean {
    return (
        queryValues.length > 0 &&
        documentValues.length > 0 &&
        !hasStructuredOverlap(queryValues, documentValues)
    );
}

function computeStructuredEvidenceConsistencyGate(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): {
    positiveScale: number;
    mismatchPenalty: number;
} {
    const { metadata } = entry;
    let positiveScale = 1;
    let mismatchPenalty = 0;

    const queryYears = querySignals.queryYears;
    const documentYear = resolveDocumentYearFromRecencyKey(metadata.recencyKey);
    const yearMismatch =
        queryYears.length > 0 &&
        documentYear !== undefined &&
        !queryYears.includes(documentYear);
    const degreeMismatch = hasExplicitStructuredMismatch(
        querySignals.queryDegreeLevels,
        metadata.structuredDegreeLevels,
    );
    const topicMismatch = hasExplicitStructuredMismatch(
        querySignals.queryTopicIds,
        metadata.structuredTopicIds,
    );
    const intentMismatch = hasExplicitStructuredMismatch(
        querySignals.queryIntentIds,
        metadata.structuredIntentIds,
    );
    const eventMismatch = hasExplicitStructuredMismatch(
        querySignals.queryEventTypes,
        metadata.structuredEventTypes,
    );

    if (yearMismatch) {
        positiveScale *= 0.22;
        mismatchPenalty -= 0.16;
    }
    if (degreeMismatch) {
        positiveScale *= 0.34;
        mismatchPenalty -= 0.12;
    }
    if (topicMismatch) {
        positiveScale *= 0.72;
        mismatchPenalty -= 0.05;
    }
    if (intentMismatch) {
        positiveScale *= 0.76;
        mismatchPenalty -= 0.04;
    }
    if (eventMismatch) {
        positiveScale *= 0.82;
        mismatchPenalty -= 0.03;
    }

    const hasStructuredQueryTheme =
        querySignals.queryIntentIds.length > 0 ||
        querySignals.queryDegreeLevels.length > 0 ||
        querySignals.queryEventTypes.length > 0 ||
        querySignals.queryTopicIds.length > 0;
    const hasAnyStructuredThemeMatch =
        hasStructuredOverlap(querySignals.queryIntentIds, metadata.structuredIntentIds) ||
        hasStructuredOverlap(
            querySignals.queryDegreeLevels,
            metadata.structuredDegreeLevels,
        ) ||
        hasStructuredOverlap(querySignals.queryEventTypes, metadata.structuredEventTypes) ||
        hasStructuredOverlap(querySignals.queryTopicIds, metadata.structuredTopicIds);
    const documentHasStructuredTheme =
        metadata.structuredIntentIds.length > 0 ||
        metadata.structuredDegreeLevels.length > 0 ||
        metadata.structuredEventTypes.length > 0 ||
        metadata.structuredTopicIds.length > 0;
    if (
        hasStructuredQueryTheme &&
        documentHasStructuredTheme &&
        !hasAnyStructuredThemeMatch
    ) {
        positiveScale *= 0.74;
        mismatchPenalty -= 0.05;
    }

    return {
        positiveScale,
        mismatchPenalty,
    };
}

function computeStructuredThemeConsistencyDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    let delta = 0;

    const hasStructuredQueryTheme =
        querySignals.queryIntentIds.length > 0 ||
        querySignals.queryDegreeLevels.length > 0 ||
        querySignals.queryEventTypes.length > 0;
    if (!hasStructuredQueryTheme) {
        return 0;
    }

    if (querySignals.queryIntentIds.length > 0) {
        if (
            hasStructuredOverlap(
                querySignals.queryIntentIds,
                metadata.structuredIntentIds,
            )
        ) {
            delta += 0.34;
        } else if (metadata.structuredIntentIds.length > 0) {
            delta -= 0.26;
        }
    }

    if (querySignals.queryDegreeLevels.length > 0) {
        if (
            hasStructuredOverlap(
                querySignals.queryDegreeLevels,
                metadata.structuredDegreeLevels,
            )
        ) {
            delta += 0.26;
        } else if (metadata.structuredDegreeLevels.length > 0) {
            delta -= 0.34;
        }
    }

    if (querySignals.queryEventTypes.length > 0) {
        if (
            hasStructuredOverlap(
                querySignals.queryEventTypes,
                metadata.structuredEventTypes,
            )
        ) {
            delta += 0.24;
        } else if (metadata.structuredEventTypes.length > 0) {
            delta -= 0.22;
        }
    }

    if (querySignals.queryTopicIds.length > 0) {
        if (
            hasStructuredOverlap(
                querySignals.queryTopicIds,
                metadata.structuredTopicIds,
            )
        ) {
            delta += 0.14;
        } else if (metadata.structuredTopicIds.length > 0) {
            delta -= 0.1;
        }
    }

    if (
        querySignals.queryIntentIds.length > 0 &&
        querySignals.queryDegreeLevels.length > 0 &&
        hasStructuredOverlap(querySignals.queryIntentIds, metadata.structuredIntentIds) &&
        hasStructuredOverlap(
            querySignals.queryDegreeLevels,
            metadata.structuredDegreeLevels,
        )
    ) {
        delta += 0.12;
    }

    return delta;
}

function computeStructuredDocTypeDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const eventTypes = metadata.structuredEventTypes;
    if (eventTypes.length === 0) {
        return 0;
    }

    const isOutcomeDoc = hasStructuredOverlap(
        STRUCTURED_OUTCOME_EVENT_TYPES,
        eventTypes,
    );
    const isProcedureDoc = hasStructuredOverlap(
        STRUCTURED_PROCEDURE_EVENT_TYPES,
        eventTypes,
    );
    const isRequirementDoc = hasStructuredOverlap(
        STRUCTURED_REQUIREMENT_EVENT_TYPES,
        eventTypes,
    );
    const isTimelineDoc = hasStructuredOverlap(
        STRUCTURED_TIMELINE_EVENT_TYPES,
        eventTypes,
    );

    let delta = 0;

    if (querySignals.asksOutcomeLikeTitle) {
        if (isOutcomeDoc) {
            delta += 0.26;
        } else if (isRequirementDoc) {
            delta -= 0.14;
        }
    }

    if (querySignals.asksProcedureLikeTitle) {
        if (isProcedureDoc) {
            delta += 0.22;
        }
        if (!querySignals.asksOutcomeLikeTitle && isOutcomeDoc) {
            delta -= 0.16;
        }
    }

    if (
        querySignals.asksRequirementLikeTitle ||
        querySignals.asksPolicyOverviewLikeTitle ||
        querySignals.asksSystemTimelineLikeTitle
    ) {
        if (isRequirementDoc) {
            delta += 0.2;
        }
        if (!querySignals.asksOutcomeLikeTitle && isOutcomeDoc) {
            delta -= 0.14;
        }
    }

    if (querySignals.asksTimelineNodeLike && isTimelineDoc) {
        delta += 0.1;
    }

    return delta;
}

function computeStructuredExecutionDetailDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    if (!querySignals.asksCampExecutionDetail) {
        return 0;
    }

    const hasStructuredIntentMatch = hasStructuredOverlap(
        querySignals.queryIntentIds,
        metadata.structuredIntentIds,
    );
    const hasStructuredEventMatch = hasStructuredOverlap(
        querySignals.queryEventTypes,
        metadata.structuredEventTypes,
    );
    if (!hasStructuredIntentMatch && !hasStructuredEventMatch) {
        return 0;
    }

    let delta = 0;
    // 当查询已经命中结构化意图，并且在问“报到/营员/后续更新”这类执行细节时，
    // 更应优先承载执行安排的结果/入围通知，而不是早期报名通知。
    if (metadata.isResultNoticeRole && !metadata.isRegistrationNoticeRole) {
        delta += 0.46;
    }
    if (metadata.isRegistrationNoticeRole && !metadata.isResultNoticeRole) {
        delta -= 0.34;
    }
    if (metadata.isRuleDocRole) {
        delta -= 0.14;
    }

    return delta;
}

function computeStructuredKpRoleEvidenceDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const bestTags = metadata.bestKpRoleTags;
    const evidenceTags = metadata.evidenceTopRoleTags;
    if (bestTags.length === 0 && evidenceTags.length === 0) {
        return 0;
    }

    const bestHasRequirement = hasStructuredOverlap(
        KP_ROLE_REQUIREMENT_TAGS,
        bestTags,
    );
    const evidenceHasRequirement = hasStructuredOverlap(
        KP_ROLE_REQUIREMENT_TAGS,
        evidenceTags,
    );
    const bestHasProcedure = hasStructuredOverlap(
        KP_ROLE_PROCEDURE_TAGS,
        bestTags,
    );
    const evidenceHasProcedure = hasStructuredOverlap(
        KP_ROLE_PROCEDURE_TAGS,
        evidenceTags,
    );
    const bestHasTimeline = hasStructuredOverlap(
        KP_ROLE_TIMELINE_TAGS,
        bestTags,
    );
    const evidenceHasTimeline = hasStructuredOverlap(
        KP_ROLE_TIMELINE_TAGS,
        evidenceTags,
    );
    const bestHasPostOutcome = hasStructuredOverlap(
        KP_ROLE_POST_OUTCOME_TAGS,
        bestTags,
    );
    const evidenceHasPostOutcome = hasStructuredOverlap(
        KP_ROLE_POST_OUTCOME_TAGS,
        evidenceTags,
    );
    const hasConstraintEvidence = evidenceHasRequirement;
    const hasOperationalEvidence =
        evidenceHasProcedure || evidenceHasTimeline || evidenceHasPostOutcome;
    const requirementEvidenceCount =
        metadata.kpEvidenceGroupCounts.requirement || 0;
    const procedureEvidenceCount = metadata.kpEvidenceGroupCounts.procedure || 0;
    const timelineEvidenceCount = metadata.kpEvidenceGroupCounts.timeline || 0;

    let delta = 0;

    if (
        querySignals.asksRequirementLikeTitle ||
        querySignals.asksPolicyOverviewLikeTitle
    ) {
        if (bestHasRequirement) {
            delta += 0.2;
        } else if (evidenceHasRequirement) {
            delta += 0.12;
        } else if (
            !querySignals.asksProcedureLikeTitle &&
            !querySignals.asksSystemTimelineLikeTitle &&
            hasOperationalEvidence
        ) {
            delta -= 0.08;
        }
    }

    if (querySignals.asksProcedureLikeTitle) {
        if (bestHasProcedure) {
            delta += 0.18;
        } else if (evidenceHasProcedure) {
            delta += 0.1;
        }
        if (bestHasTimeline) {
            delta += 0.08;
        } else if (evidenceHasTimeline) {
            delta += 0.04;
        }
    }

    if (
        querySignals.asksTimelineNodeLike ||
        querySignals.asksSystemTimelineLikeTitle
    ) {
        if (bestHasTimeline) {
            delta += 0.14;
        } else if (evidenceHasTimeline) {
            delta += 0.08;
        }
    }

    if (querySignals.asksPostOutcomeOperationalDetail) {
        if (bestHasPostOutcome) {
            delta += 0.22;
        } else if (evidenceHasPostOutcome) {
            delta += 0.12;
        }
    }

    if (
        !querySignals.asksOutcomeLikeTitle &&
        metadata.isOutcomeRoleDoc &&
        !hasConstraintEvidence &&
        !hasOperationalEvidence
    ) {
        delta -= 0.12;
    }

    if (querySignals.asksPostOutcomeAdmission) {
        const requirementLeadsOperationalEvidence =
            requirementEvidenceCount >= procedureEvidenceCount &&
            requirementEvidenceCount >= timelineEvidenceCount;
        const timelineOverrunsConstraintEvidence =
            timelineEvidenceCount > requirementEvidenceCount;
        if (metadata.isRuleDocRole) {
            delta += 0.2;
        }
        if (bestHasRequirement) {
            delta += 0.24;
        } else if (evidenceHasRequirement) {
            delta += 0.1;
        }
        if (bestHasProcedure) {
            delta += 0.08;
        } else if (evidenceHasProcedure) {
            delta += 0.04;
        }
        if (metadata.isRuleDocRole && requirementLeadsOperationalEvidence) {
            delta += 0.08;
        }
        if (
            metadata.isOutcomeRoleDoc &&
            !bestHasRequirement &&
            !metadata.isRuleDocRole
        ) {
            delta -= 0.32;
        }
        if (
            metadata.isOutcomeRoleDoc &&
            timelineOverrunsConstraintEvidence &&
            !bestHasRequirement
        ) {
            delta -= 0.08;
        }
        if (metadata.isListNoticeRole && !bestHasRequirement) {
            delta -= 0.06;
        }
    }

    const { positiveScale, mismatchPenalty } =
        computeStructuredEvidenceConsistencyGate(querySignals, entry);
    if (delta > 0) {
        delta *= positiveScale;
    }

    return delta + mismatchPenalty;
}

function shouldApplyCoarseLexicalTitleType(
    querySignals: DocumentRerankQuerySignals,
): boolean {
    if (querySignals.isCompressedKeywordQuery) {
        return false;
    }

    if (
        querySignals.asksTimelineNodeLike ||
        querySignals.asksEventDateLikeTitle ||
        querySignals.asksSystemTimelineLikeTitle ||
        querySignals.asksMaterialReviewTiming ||
        querySignals.asksPostOutcomeOperationalDetail ||
        querySignals.asksCampExecutionDetail
    ) {
        return false;
    }

    if (
        querySignals.asksPolicyOverviewLikeTitle ||
        querySignals.asksFlowOverviewLike ||
        querySignals.asksPostOutcomeAdmission
    ) {
        return true;
    }

    if (!querySignals.asksProcedureLikeTitle) {
        return false;
    }

    return !/截止|时间|日期|哪天|何时|什么时候|条件|资格|要求|材料|提交|细节|要点|电话|邮箱|联系方式|地点|多少/.test(
        querySignals.normalizedQuery,
    );
}

function computeLexicalTitleTypeDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    if (!shouldApplyCoarseLexicalTitleType(querySignals)) {
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
    // 对未显式指向某个学院/项目的 broad 流程或政策问题，
    // 轻微压低学院级标题，避免 coarse title-type 把 generic 查询推向局部学院通知。
    if (
        !querySignals.mentionsCollegeEntity &&
        metadata.hasCollegeTitle &&
        (querySignals.asksPolicyOverviewLikeTitle ||
            querySignals.asksFlowOverviewLike)
    ) {
        delta -= 0.18;
    }

    return delta;
}

function computeLexicalScenarioTitleDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    let delta = 0;

    if (querySignals.asksPolicyOverviewLikeTitle) {
        if (/招生简章|接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.55;
        }
        if (metadata.isPreapplyTitle) {
            delta -= 0.45;
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

    return delta;
}

function computeLexicalTitleIntentDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
    options: {
        enableLexicalTitleTypeAdjustments?: boolean;
        enableLexicalScenarioAdjustments?: boolean;
    } = {},
): number {
    const {
        enableLexicalTitleTypeAdjustments = true,
        enableLexicalScenarioAdjustments = true,
    } = options;

    let delta = 0;
    if (enableLexicalTitleTypeAdjustments) {
        delta += computeLexicalTitleTypeDelta(querySignals, entry);
    }
    if (enableLexicalScenarioAdjustments) {
        delta += computeLexicalScenarioTitleDelta(querySignals, entry);
    }
    return delta;
}

function computeDoctoralThemeTitleDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    let delta = 0;

    if (querySignals.mentionsDoctoral) {
        if (metadata.isMasterOnlyTitle) {
            delta -= 0.95;
        } else if (metadata.isDoctoralTitle) {
            delta += 0.2;
        }
    }

    if (
        querySignals.mentionsDoctoral &&
        /实施办法|招生简章|综合考核通知/.test(normalizedTitle)
    ) {
        delta += 0.25;
    }

    return delta;
}

function computeTuimianThemeTitleDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    let delta = 0;

    if (querySignals.mentionsTuimian) {
        if (metadata.isTuimianTitle) {
            delta += 0.45;
        }
        if (metadata.isDoctoralTitle && !metadata.isTuimianTitle) {
            delta -= 0.6;
        }
    }

    if (
        querySignals.mentionsTuimian &&
        /接收办法|工作方案/.test(normalizedTitle)
    ) {
        delta += 0.45;
    }

    return delta;
}

function computeSummerCampThemeTitleDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    let delta = 0;

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

    return delta;
}

function computeTransferThemeTitleDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    let delta = 0;

    if (
        querySignals.asksPolicyOverviewLikeTitle &&
        metadata.isTransferTitle &&
        !querySignals.mentionsTransfer
    ) {
        delta -= 0.65;
    }

    return delta;
}

function computeCompressedKeywordTitleDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    let delta = 0;

    if (
        !querySignals.isCompressedKeywordQuery ||
        querySignals.asksBroadRuleDocLikeTitle
    ) {
        return 0;
    }

    if (!querySignals.asksCompressedConstraintLike && metadata.isRuleDocRole) {
        delta -= querySignals.asksCompressedOutcomeLike ? 0.95 : 0.55;
    }
    if (metadata.isSystemNoticeTitle) {
        delta -= 1.1;
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

    return delta;
}

export function computeTitleIntentDocDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
    options: TitleIntentDocDeltaOptions = {},
): number {
    // 这是展示重排里最重的一层打分：
    // 按查询意图判断当前标题更像“条件/流程/结果/系统通知”中的哪一类，
    // 可选地再叠加主题型和压缩短问型修正。
    const {
        enableStructuredKpRoleEvidenceAdjustments = true,
        enableLexicalTitleIntentAdjustments = true,
        enableLexicalTitleTypeAdjustments = true,
        enableLexicalScenarioAdjustments = true,
        enableThemeSpecificAdjustments = true,
        enableDoctoralThemeAdjustments = true,
        enableTuimianThemeAdjustments = true,
        enableSummerCampThemeAdjustments = true,
        enableTransferThemeAdjustments = true,
        enableCompressedKeywordAdjustments = true,
    } = options;
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    if (!normalizedTitle) {
        return 0;
    }

    let delta = 0;

    if (enableLexicalTitleIntentAdjustments) {
        delta += computeLexicalTitleIntentDelta(querySignals, entry, {
            enableLexicalTitleTypeAdjustments,
            enableLexicalScenarioAdjustments,
        });
    }
    const structuredThemeDelta = computeStructuredThemeConsistencyDelta(
        querySignals,
        entry,
    );
    delta += structuredThemeDelta;
    delta += computeStructuredDocTypeDelta(querySignals, entry);
    if (enableStructuredKpRoleEvidenceAdjustments) {
        delta += computeStructuredKpRoleEvidenceDelta(querySignals, entry);
    }
    delta += computeStructuredExecutionDetailDelta(querySignals, entry);

    if (enableThemeSpecificAdjustments) {
        const rawThemeDelta =
            (enableDoctoralThemeAdjustments
                ? computeDoctoralThemeTitleDelta(querySignals, entry)
                : 0) +
            (enableTuimianThemeAdjustments
                ? computeTuimianThemeTitleDelta(querySignals, entry)
                : 0) +
            (enableSummerCampThemeAdjustments
                ? computeSummerCampThemeTitleDelta(querySignals, entry)
                : 0) +
            (enableTransferThemeAdjustments
                ? computeTransferThemeTitleDelta(querySignals, entry)
                : 0);
        const hasStructuredThemeSignal =
            querySignals.queryIntentIds.length > 0 ||
            querySignals.queryDegreeLevels.length > 0 ||
            querySignals.queryEventTypes.length > 0;
        delta +=
            hasStructuredThemeSignal && rawThemeDelta !== 0
                ? rawThemeDelta *
                  RAW_THEME_FALLBACK_WEIGHT_WITH_STRUCTURED_SIGNAL
                : rawThemeDelta;
    }

    if (enableCompressedKeywordAdjustments) {
        delta += computeCompressedKeywordTitleDelta(querySignals, entry);
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

export function applyPhaseAnchorBoostToDocuments(
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

export function applyCoverageTitleDiversity(
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    // 覆盖型问题不追求单篇文档最高分，而是希望前几篇互补。
    // 这里按标题家族做轻量去重，把重复公告往后压。
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

export function computeCoverageWeightedDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    return computeCoverageDocDelta(querySignals, entry) * querySignals.coverageWeight;
}

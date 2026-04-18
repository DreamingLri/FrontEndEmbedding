import {
    applyScoreToAggregatedDocScores,
    createAggregatedDocScores,
    mergeAggregatedDocMetadata,
    type AggregatedDocScores,
    type KPCandidate,
} from "./aggregated_doc_scores.ts";
import type { QueryPlan } from "./query_planner.ts";
import {
    BM25_B,
    BM25_K1,
    BOUNDARY_REJECT_SCORE_THRESHOLD,
    BROAD_LATEST_SCOPE_CUE_PATTERN,
    clamp01,
    CURRENT_PROCESS_EVENT_TYPES,
    CURRENT_PROCESS_TIMESTAMP_BOOST_BASE,
    DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    DEFAULT_KP_ROLE_DOC_WEIGHT,
    DEFAULT_Q_CONFUSION_WEIGHT,
    DEFAULT_WEIGHTS,
    dedupe,
    EVENT_TYPE_MISMATCH_PENALTY,
    HARD_REJECT_SCORE_THRESHOLD,
    hasOnlyOutOfScopeTopics,
    hasPostOutcomeConditionCue,
    INTENT_CONFLICTS,
    INTENT_RULE_MAP,
    LATEST_POLICY_TIMESTAMP_BOOST_BASE,
    LATEST_YEAR_BOOST_BASE,
    QUERY_SCOPE_SPECIFICITY_TERM_SET,
    resolveDocOtid,
    resolveMetadataTopicIds,
    RRF_K,
    RRF_RANK_LIMIT,
    STRONG_EVIDENCE_ROLE_TAGS,
    WEAK_EVIDENCE_ROLE_TAGS,
    withQueryTokenCount,
    dotProduct,
    selectTopLocalIndices,
    type BM25Stats,
    type EvidenceSignals,
    type FusionMode,
    type KPAggregationMode,
    type KPRoleRerankMode,
    type LexicalBonusMode,
    type Metadata,
    type ParsedQueryIntent,
    type QConfusionMode,
    type QuerySignals,
    type RejectTier,
    type ResponseDecision,
    type ResponseMode,
    type RetrievalSignals,
    type SearchRankDiagnostics,
    type SearchRejection,
    type SearchRankOutput,
    type SearchResult,
} from "./vector_engine_shared.ts";
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

function computeQConsensusPenaltyMultiplier(params: {
    weightedQ: number;
    weightedKP: number;
    weightedOT: number;
    qConfusionWeight: number;
}): number {
    const { weightedQ, weightedKP, weightedOT, qConfusionWeight } = params;
    if (weightedQ <= 0) {
        return 1;
    }

    const supportStrength = Math.max(weightedKP, weightedOT);
    if (supportStrength >= weightedQ) {
        return 1;
    }

    const supportGapRatio = Math.min(
        1,
        (weightedQ - supportStrength) / Math.max(weightedQ, 1e-6),
    );
    const dualWeakSupportFactor =
        weightedKP <= weightedQ * 0.6 && weightedOT <= weightedQ * 0.6
            ? 1
            : 0.65;

    return Math.max(
        0.35,
        1 - qConfusionWeight * supportGapRatio * dualWeakSupportFactor,
    );
}

function computeQCompetitionPenaltyMap(params: {
    otidMap: Record<string, AggregatedDocScores>;
    qConfusionWeight: number;
}): Map<string, number> {
    const { otidMap, qConfusionWeight } = params;
    const penaltyMap = new Map<string, number>();

    for (const [otid, scores] of Object.entries(otidMap)) {
        const currentQ = scores.max_q;
        if (currentQ <= 0) {
            penaltyMap.set(otid, 1);
            continue;
        }

        const closeThreshold = Math.max(0.03, currentQ * 0.05);
        let closeDocCount = 0;

        for (const [otherOtid, otherScores] of Object.entries(otidMap)) {
            if (otherOtid === otid || otherScores.max_q <= 0) {
                continue;
            }
            if (otherScores.max_q >= currentQ - closeThreshold) {
                closeDocCount += 1;
            }
        }

        const localCrowding =
            scores.q_scores.length > 1 &&
            scores.q_scores[1]! >= currentQ - closeThreshold
                ? 1
                : 0;
        const normalizedCrowding = Math.min(
            1,
            (Math.min(closeDocCount, 4) + localCrowding) / 4,
        );

        penaltyMap.set(
            otid,
            Math.max(
                0.35,
                1 - qConfusionWeight * 0.85 * normalizedCrowding,
            ),
        );
    }

    return penaltyMap;
}

function computeBaseScore(
    scores: AggregatedDocScores,
    weights: typeof DEFAULT_WEIGHTS,
    options?: {
        kpAggregationMode?: KPAggregationMode;
        kpTopN?: number;
        kpTailWeight?: number;
        fusionMode?: FusionMode;
        qConfusionMode?: QConfusionMode;
        qConfusionWeight?: number;
        qCompetitionPenaltyMultiplier?: number;
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
            : kpAggregationMode === "mean" && topKpScores.length > 0
              ? topKpScores.reduce((sum, item) => sum + item, 0) /
                topKpScores.length
              : kpAggregationMode === "sum" && topKpScores.length > 0
                ? topKpScores.reduce((sum, item) => sum + item, 0)
                : topKpScores[0] || 0;

    const weightedQ = scores.max_q * weights.Q;
    const weightedKP = aggregatedKpScore * weights.KP;
    const weightedOT = scores.ot_score * weights.OT;
    const qConfusionMode = options?.qConfusionMode || "off";
    const qConfusionWeight =
        Number.isFinite(options?.qConfusionWeight) &&
        (options?.qConfusionWeight || 0) > 0
            ? Math.min(options!.qConfusionWeight!, 1)
            : DEFAULT_Q_CONFUSION_WEIGHT;
    let qPenaltyMultiplier = 1;

    if (
        qConfusionMode === "consensus" ||
        qConfusionMode === "combined"
    ) {
        qPenaltyMultiplier *= computeQConsensusPenaltyMultiplier({
            weightedQ,
            weightedKP,
            weightedOT,
            qConfusionWeight,
        });
    }

    if (
        qConfusionMode === "competition" ||
        qConfusionMode === "combined"
    ) {
        qPenaltyMultiplier *= options?.qCompetitionPenaltyMultiplier || 1;
    }

    qPenaltyMultiplier = Math.max(0.35, Math.min(1, qPenaltyMultiplier));
    const effectiveWeightedQ = weightedQ * qPenaltyMultiplier;
    const fusionMode = options?.fusionMode || "default";

    if (fusionMode === "max_q_vs_kpot") {
        return Math.max(effectiveWeightedQ, weightedKP + weightedOT);
    }

    const maxComponent = Math.max(effectiveWeightedQ, weightedKP, weightedOT);
    const unionBonus =
        effectiveWeightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;

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

function hasAnyKpRoleEvidence(
    candidates: readonly KPCandidate[],
    tags: readonly string[],
): boolean {
    return candidates.some((candidate) =>
        tags.some((tag) => hasKpRoleTag(candidate, tag)),
    );
}

function countPlannerEvidenceGroups(
    candidates: readonly KPCandidate[],
): number {
    const window = candidates.slice(0, DEFAULT_KP_ROLE_CANDIDATE_LIMIT);
    return [
        hasAnyKpRoleEvidence(window, ["condition", "deadline"]),
        hasAnyKpRoleEvidence(window, ["materials", "email"]),
        hasAnyKpRoleEvidence(window, [
            "procedure",
            "application_stage",
            "schedule",
        ]),
        hasAnyKpRoleEvidence(window, [
            "arrival",
            "deadline",
            "announcement_period",
            "time_expression",
            "schedule",
        ]),
    ].filter(Boolean).length;
}

function applyQueryPlannerRetrievalBoost(
    boost: number,
    queryPlan: QueryPlan | undefined,
    scores: AggregatedDocScores,
): number {
    if (!queryPlan) {
        return boost;
    }

    const window = scores.kp_candidates.slice(
        0,
        DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    );
    let multiplier = 1;

    if (queryPlan.asksCoverageLike) {
        const coveredGroups = countPlannerEvidenceGroups(window);
        if (coveredGroups >= 2) {
            multiplier *= 1 + Math.min(0.08, coveredGroups * 0.025);
        } else if (coveredGroups === 0) {
            multiplier *= 0.98;
        }
    }

    switch (queryPlan.intentType) {
        case "outcome":
            if (hasAnyOverlap(["录取公示", "推免资格公示"], scores.event_types)) {
                multiplier *= 1.06;
            } else if (
                hasAnyOverlap(
                    ["招生章程", "推免实施办法", "资格要求"],
                    scores.event_types,
                )
            ) {
                multiplier *= 0.97;
            }
            break;
        case "procedure":
        case "system_timeline":
            if (
                hasAnyOverlap(
                    ["报名通知", "考试安排", "材料提交", "复试通知", "推免实施办法"],
                    scores.event_types,
                )
            ) {
                multiplier *= 1.04;
            }
            if (
                hasAnyKpRoleEvidence(window, [
                    "procedure",
                    "application_stage",
                    "schedule",
                    "deadline",
                ])
            ) {
                multiplier *= 1.04;
            }
            if (
                !queryPlan.asksOutcomeLike &&
                hasAnyOverlap(["录取公示", "推免资格公示"], scores.event_types)
            ) {
                multiplier *= 0.97;
            }
            break;
        case "requirement":
        case "policy_overview":
            if (
                hasAnyOverlap(
                    ["招生章程", "推免实施办法", "资格要求", "材料提交"],
                    scores.event_types,
                )
            ) {
                multiplier *= 1.04;
            }
            if (
                hasAnyKpRoleEvidence(window, [
                    "condition",
                    "materials",
                    "application_stage",
                    "procedure",
                ])
            ) {
                multiplier *= 1.03;
            }
            break;
        case "time_location":
            if (
                hasAnyKpRoleEvidence(window, [
                    "schedule",
                    "arrival",
                    "deadline",
                    "announcement_period",
                    "time_expression",
                ])
            ) {
                multiplier *= 1.05;
            }
            break;
        case "fact_detail":
        default:
            break;
    }

    if (queryPlan.difficultyTier === "high" && scores.kp_scores.length >= 2) {
        multiplier *= 1.03;
    }

    return boost * Math.max(0.92, Math.min(1.12, multiplier));
}

function collectPlannerCoverageFacets(scores?: AggregatedDocScores): string[] {
    if (!scores) {
        return [];
    }

    const facets = new Set<string>();
    scores.event_types?.slice(0, 4).forEach((eventType) => {
        facets.add(`event:${eventType}`);
    });
    scores.degree_levels?.slice(0, 3).forEach((degreeLevel) => {
        facets.add(`degree:${degreeLevel}`);
    });
    if (scores.target_year !== undefined) {
        facets.add(`year:${scores.target_year}`);
    }

    const window = scores.kp_candidates.slice(
        0,
        DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    );
    if (hasAnyKpRoleEvidence(window, ["condition", "deadline"])) {
        facets.add("role:condition");
    }
    if (hasAnyKpRoleEvidence(window, ["materials", "email"])) {
        facets.add("role:materials");
    }
    if (
        hasAnyKpRoleEvidence(window, [
            "procedure",
            "application_stage",
            "schedule",
        ])
    ) {
        facets.add("role:procedure");
    }
    if (
        hasAnyKpRoleEvidence(window, [
            "arrival",
            "deadline",
            "announcement_period",
            "time_expression",
            "schedule",
        ])
    ) {
        facets.add("role:time");
    }

    return Array.from(facets);
}

function applyQueryPlannerCoverageDiversification(
    ranking: readonly SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
    queryPlan?: QueryPlan,
): SearchResult[] {
    if (
        !queryPlan ||
        !queryPlan.asksCoverageLike ||
        ranking.length <= 2
    ) {
        return [...ranking];
    }

    const windowSize = Math.min(12, ranking.length);
    const selected: SearchResult[] = [ranking[0]!];
    const remaining = ranking.slice(1, windowSize);
    const tail = ranking.slice(windowSize);
    const seenFacets = new Set(
        collectPlannerCoverageFacets(otidMap[selected[0]!.otid]),
    );

    while (remaining.length > 0) {
        let bestIndex = 0;
        let bestAdjustedScore = Number.NEGATIVE_INFINITY;

        remaining.forEach((candidate, index) => {
            const facets = collectPlannerCoverageFacets(otidMap[candidate.otid]);
            const newFacetCount = facets.filter((facet) => !seenFacets.has(facet))
                .length;
            const duplicateFacetCount = facets.length - newFacetCount;
            const adjustedScore =
                candidate.score +
                Math.min(0.18, newFacetCount * 0.045) -
                Math.min(0.08, duplicateFacetCount * 0.012);

            if (adjustedScore > bestAdjustedScore) {
                bestAdjustedScore = adjustedScore;
                bestIndex = index;
            }
        });

        const [chosen] = remaining.splice(bestIndex, 1);
        collectPlannerCoverageFacets(otidMap[chosen!.otid]).forEach((facet) =>
            seenFacets.add(facet),
        );
        selected.push({
            ...chosen!,
            score: bestAdjustedScore,
        });
    }

    return [...selected, ...tail];
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
    queryPlan?: QueryPlan;
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
        queryPlan,
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
    boost = applyQueryPlannerRetrievalBoost(boost, queryPlan, scores);
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

export function extractEvidenceSignals(
    sortedRanking: SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
): EvidenceSignals {
    const topWindow = sortedRanking
        .slice(0, 3)
        .flatMap((item) =>
            (otidMap[item.otid]?.kp_candidates || []).slice(
                0,
                DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
            ),
        );

    const topRoleTags = dedupe(
        topWindow.flatMap((candidate) => candidate.kp_role_tags || []),
    );
    let strongRoleCount = 0;
    let weakRoleCount = 0;

    topRoleTags.forEach((tag) => {
        if (STRONG_EVIDENCE_ROLE_TAGS.has(tag)) {
            strongRoleCount += 1;
        } else if (WEAK_EVIDENCE_ROLE_TAGS.has(tag)) {
            weakRoleCount += 1;
        }
    });

    const totalEvidenceRoleCount = strongRoleCount + weakRoleCount;

    return {
        topRoleTags,
        topCandidateCount: topWindow.length,
        strongRoleCount,
        weakRoleCount,
        strongRoleRatio:
            totalEvidenceRoleCount > 0
                ? strongRoleCount / totalEvidenceRoleCount
                : 0,
        weakOnly: strongRoleCount === 0 && weakRoleCount > 0,
        hasOperationalRoleEvidence: hasAnyRoleEvidence(
            topWindow,
            Array.from(STRONG_EVIDENCE_ROLE_TAGS),
        ),
    };
}

export function classifyResponseMode(
    querySignals: QuerySignals,
    retrievalSignals: RetrievalSignals,
    evidenceSignals: EvidenceSignals,
): ResponseDecision {
    const tokenCount = querySignals.tokenCount || 0;
    const invalidInput = tokenCount === 0;

    if (invalidInput) {
        return {
            mode: "reject",
            confidence: 0.98,
            reason: "invalid_input",
            preferLatestWithinTopic: false,
            useWeakMatches: false,
            rejectScore: 1,
            rejectTier: "invalid_input",
        };
    }

    const noStructuredAnchor =
        !querySignals.hasExplicitTopicOrIntent && !querySignals.hasExplicitYear;
    const lowTokenRisk = tokenCount <= 1 ? 1 : tokenCount <= 3 ? 0.45 : 0;
    const shortQueryRisk = clamp01((8 - querySignals.queryLength) / 8);
    const genericNextStepRisk =
        querySignals.hasGenericNextStep && !querySignals.hasExplicitTopicOrIntent
            ? 0.45
            : 0;
    const stateWithoutAnchorRisk =
        querySignals.hasResultState && noStructuredAnchor ? 1 : 0;
    const intentRisk = clamp01(
        0.58 * (noStructuredAnchor ? 1 : 0) +
            0.16 * lowTokenRisk +
            0.1 * genericNextStepRisk +
            0.1 * shortQueryRisk +
            0.06 * stateWithoutAnchorRisk,
    );

    const noCandidatesRisk = retrievalSignals.candidateCount === 0 ? 1 : 0;
    const noLabeledRisk = retrievalSignals.labeledTopicCount === 0 ? 1 : 0;
    const spreadRisk = clamp01(
        (retrievalSignals.distinctTopicCount - 2) / 3,
    );
    const dominanceRisk = clamp01(
        (0.55 - retrievalSignals.dominantTopicRatio) / 0.55,
    );
    const topicDispersionRisk = Math.max(spreadRisk, dominanceRisk);
    const gapRisk = clamp01((0.05 - retrievalSignals.top1Top2Gap) / 0.05);
    const retrievalRisk =
        noCandidatesRisk === 1
            ? 1
            : clamp01(
                  0.5 * noLabeledRisk +
                      0.3 * topicDispersionRisk +
                      0.2 * gapRisk,
              );

    const evidenceRisk = clamp01(
        0.68 * (1 - evidenceSignals.strongRoleRatio) +
            0.22 * (evidenceSignals.hasOperationalRoleEvidence ? 0 : 1) +
            0.1 * (evidenceSignals.weakOnly ? 1 : 0),
    );

    const genericProcessSafetyBonus =
        querySignals.hasGenericNextStep &&
        (querySignals.hasStrongDetailAnchor || querySignals.hasResultState)
            ? 0.09
            : 0;
    const postOutcomeOperationalSafetyBonus =
        querySignals.hasResultState && querySignals.hasPostOutcomeOperationalCue
            ? querySignals.hasMultiSlotConstraintCue
                ? 0.14
                : 0.1
            : 0;

    const rejectScore = clamp01(
        0.4 * intentRisk +
            0.25 * retrievalRisk +
            0.35 * evidenceRisk -
            (querySignals.hasExplicitTopicOrIntent ? 0.08 : 0) -
            (querySignals.hasExplicitYear ? 0.03 : 0) -
            genericProcessSafetyBonus -
            postOutcomeOperationalSafetyBonus,
    );

    let mode: ResponseMode = "answer";
    let rejectTier: RejectTier | null = null;
    if (rejectScore >= HARD_REJECT_SCORE_THRESHOLD) {
        mode = "reject";
        rejectTier = "hard_reject";
    } else if (rejectScore >= BOUNDARY_REJECT_SCORE_THRESHOLD) {
        mode = "reject";
        rejectTier = "boundary_uncertain";
    }

    const reasonParts: string[] = [];
    if (noStructuredAnchor) {
        reasonParts.push("no_structured_anchor");
    }
    if (evidenceSignals.weakOnly) {
        reasonParts.push("weak_only_role_evidence");
    } else if (!evidenceSignals.hasOperationalRoleEvidence) {
        reasonParts.push("no_operational_evidence");
    }
    if (noLabeledRisk === 1) {
        reasonParts.push("no_labeled_topic_support");
    } else if (topicDispersionRisk >= 0.45) {
        reasonParts.push("topic_dispersion");
    }
    if (reasonParts.length === 0) {
        reasonParts.push(
            mode === "reject" ? "reject_score_guard" : "answer_score_pass",
        );
    }

    let confidence: number;
    if (mode === "reject" && rejectTier === "hard_reject") {
        confidence = Math.max(
            0.74,
            Math.min(
                0.98,
                0.74 +
                    (rejectScore - HARD_REJECT_SCORE_THRESHOLD) /
                        (1 - HARD_REJECT_SCORE_THRESHOLD) *
                        0.18,
            ),
        );
    } else if (mode === "reject") {
        confidence = Math.max(
            0.58,
            Math.min(
                0.79,
                0.58 +
                    (rejectScore - BOUNDARY_REJECT_SCORE_THRESHOLD) /
                        (HARD_REJECT_SCORE_THRESHOLD -
                            BOUNDARY_REJECT_SCORE_THRESHOLD) *
                        0.18,
            ),
        );
    } else {
        confidence = Math.max(
            0.56,
            Math.min(
                0.92,
                0.92 -
                    (rejectScore / BOUNDARY_REJECT_SCORE_THRESHOLD) * 0.24,
            ),
        );
    }

    return {
        mode,
        confidence,
        reason: reasonParts.slice(0, 3).join("+"),
        preferLatestWithinTopic:
            querySignals.hasLatestPolicyState && !querySignals.hasExplicitYear,
        useWeakMatches: rejectTier === "hard_reject",
        rejectScore,
        rejectTier,
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
    topHybridLimit?: number;
    kpAggregationMode?: KPAggregationMode;
    kpTopN?: number;
    kpTailWeight?: number;
    fusionMode?: FusionMode;
    lexicalBonusMode?: LexicalBonusMode;
    qLexicalMultiplier?: number;
    kpLexicalMultiplier?: number;
    otLexicalMultiplier?: number;
    denseScoreOverrides?: ReadonlyMap<string, number>;
    denseRrfWeight?: number;
    sparseRrfWeight?: number;
    kpRoleRerankMode?: KPRoleRerankMode;
    kpRoleDocWeight?: number;
    otDenseScoreOverrides?: ReadonlyMap<string, number>;
    qConfusionMode?: QConfusionMode;
    qConfusionWeight?: number;
    enableExplicitYearFilter?: boolean;
    queryPlan?: QueryPlan;
    enableQueryPlanner?: boolean;
    minimalMode?: boolean;
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
        topHybridLimit = 1000,
        kpAggregationMode = "max",
        kpTopN = 3,
        kpTailWeight = 0.35,
        fusionMode = "default",
        lexicalBonusMode = "sum",
        qLexicalMultiplier = 1.5,
        kpLexicalMultiplier = 1.2,
        otLexicalMultiplier = 1.0,
        denseScoreOverrides,
        denseRrfWeight = 100,
        sparseRrfWeight = 120,
        kpRoleRerankMode = "off",
        kpRoleDocWeight = DEFAULT_KP_ROLE_DOC_WEIGHT,
        otDenseScoreOverrides,
        qConfusionMode = "off",
        qConfusionWeight = DEFAULT_Q_CONFUSION_WEIGHT,
        enableExplicitYearFilter,
        queryPlan,
        enableQueryPlanner = false,
        minimalMode = false,
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
    const safeDenseRrfWeight = Number.isFinite(denseRrfWeight)
        ? denseRrfWeight
        : 100;
    const safeSparseRrfWeight = Number.isFinite(sparseRrfWeight)
        ? sparseRrfWeight
        : 120;
    const safeEnableExplicitYearFilter =
        typeof enableExplicitYearFilter === "boolean"
            ? enableExplicitYearFilter
            : !minimalMode;

    const activeCandidateIndices =
        candidateIndices && candidateIndices.length > 0 ? candidateIndices : undefined;
    const candidateCount = activeCandidateIndices
        ? activeCandidateIndices.length
        : metadata.length;
    const denseScores = new Float32Array(candidateCount);
    const sparseScores = new Float32Array(candidateCount);
    const lexicalBonusMap = new Map<string, number>();
    const yearHitMap = new Map<string, boolean>();
    const docScopeSpecificityStatsMap = new Map<string, ScopeSpecificityStats>();
    const queryYearWordIdSet =
        queryYearWordIds && queryYearWordIds.length > 0
            ? new Set(queryYearWordIds)
            : undefined;

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

                if (queryYearWordIdSet?.has(wordId)) {
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
    }

    const rrfRankLimit = Math.min(RRF_RANK_LIMIT, candidateCount);
    const denseTopLocalIndices = selectTopLocalIndices(
        denseScores,
        rrfRankLimit,
    );
    const rrfScores = new Map<Metadata, number>();

    for (let rank = 0; rank < denseTopLocalIndices.length; rank++) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[denseTopLocalIndices[rank] as number]
            : (denseTopLocalIndices[rank] as number);
        const meta = metadata[metaIndex];
        rrfScores.set(meta, (1 / (rank + RRF_K)) * safeDenseRrfWeight);
    }

    if (querySparse) {
        const sparseTopLocalIndices = selectTopLocalIndices(
            sparseScores,
            rrfRankLimit,
            {
                minimumScoreExclusive: 0,
            },
        );
        for (let rank = 0; rank < sparseTopLocalIndices.length; rank++) {
            const localIndex = sparseTopLocalIndices[rank] as number;
            const metaIndex = activeCandidateIndices
                ? activeCandidateIndices[localIndex]
                : localIndex;
            const meta = metadata[metaIndex];
            const current = rrfScores.get(meta) || 0;
            rrfScores.set(
                meta,
                current + (1 / (rank + RRF_K)) * safeSparseRrfWeight,
            );
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

    const decisionRanking: SearchResult[] = [];
    const outputRanking: SearchResult[] = [];
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
    const effectiveQConfusionMode: QConfusionMode =
        qConfusionMode === "consensus_no_year"
            ? intentContext.hasExplicitYear
                ? "off"
                : "consensus"
            : qConfusionMode;
    const qCompetitionPenaltyMap =
        effectiveQConfusionMode === "competition" ||
        effectiveQConfusionMode === "combined"
            ? computeQCompetitionPenaltyMap({
                  otidMap,
                  qConfusionWeight:
                      Number.isFinite(qConfusionWeight) && qConfusionWeight > 0
                          ? Math.min(qConfusionWeight, 1)
                          : DEFAULT_Q_CONFUSION_WEIGHT,
              })
            : undefined;

    for (const [otid, scores] of Object.entries(otidMap)) {
        const signals = getDocQuerySignals(
            otid,
            scores,
            intentContext,
            yearHitMap,
        );

        if (safeEnableExplicitYearFilter && shouldSkipForExplicitYear(scores, intentContext, signals)) {
            continue;
        }

        const decisionScore = computeBaseScore(scores, weights, {
            kpAggregationMode,
            kpTopN,
            kpTailWeight,
            fusionMode,
            qConfusionMode: "off",
            qConfusionWeight,
        });
        const outputScore = computeBaseScore(scores, weights, {
            kpAggregationMode,
            kpTopN,
            kpTailWeight,
            fusionMode,
            qConfusionMode: effectiveQConfusionMode,
            qConfusionWeight,
            qCompetitionPenaltyMultiplier: qCompetitionPenaltyMap?.get(otid),
        });
        const kpRoleSelection = minimalMode
            ? {
                  bestKpid: scores.best_kpid,
                  orderedCandidates: scores.kp_candidates,
                  docScoreDelta: 0,
              }
            : rerankKpCandidatesByRole({
                  kpCandidates: scores.kp_candidates,
                  bestKpid: scores.best_kpid,
                  rawQuery: queryIntent?.rawQuery || "",
                  queryScopeHint,
                  mode: kpRoleRerankMode,
              });
        const boost = minimalMode
            ? 1
            : computeBoostMultiplier({
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
                  queryPlan: enableQueryPlanner ? queryPlan : undefined,
              });
        const baseDocScoreDelta =
            kpRoleSelection.docScoreDelta * kpRoleDocWeight;
        const rankingItem = {
            otid,
            score: 0,
            best_kpid: kpRoleSelection.bestKpid,
            kp_candidates: kpRoleSelection.orderedCandidates.slice(0, 5),
        };

        decisionRanking.push({
            ...rankingItem,
            score: decisionScore * boost + baseDocScoreDelta,
        });
        outputRanking.push({
            ...rankingItem,
            score: outputScore * boost + baseDocScoreDelta,
        });
    }

    const sortedDecisionRanking = decisionRanking.sort(
        (a, b) => b.score - a.score,
    );
    const rawOutputRanking =
        effectiveQConfusionMode === "off"
            ? sortedDecisionRanking
            : outputRanking.sort((a, b) => b.score - a.score);
    const sortedOutputRanking =
        enableQueryPlanner && queryPlan
            ? applyQueryPlannerCoverageDiversification(
                  rawOutputRanking,
                  otidMap,
                  queryPlan,
              )
            : rawOutputRanking;
    const defaultQuerySignals: QuerySignals = {
        hasExplicitTopicOrIntent: false,
        hasExplicitYear: false,
        hasHistoricalHint: false,
        hasStrongDetailAnchor: false,
        hasEntryLikeAnchor: false,
        hasResultState: false,
        hasLatestPolicyState: false,
        hasGenericNextStep: false,
        hasPostOutcomeOperationalCue: false,
        hasMultiSlotConstraintCue: false,
        queryLength: queryIntent?.rawQuery.length || 0,
        tokenCount: 0,
    };
    const querySignals = withQueryTokenCount(
        queryIntent?.signals || defaultQuerySignals,
        querySparse,
    );
    const retrievalSignals = extractRetrievalSignals(
        sortedDecisionRanking,
        otidMap,
    );
    const evidenceSignals = extractEvidenceSignals(
        sortedDecisionRanking,
        otidMap,
    );
    const responseDecision = classifyResponseMode(
        querySignals,
        retrievalSignals,
        evidenceSignals,
    );

    const explicitOutOfScopeOnly =
        (queryIntent?.intentIds.length || 0) === 0 &&
        hasOnlyOutOfScopeTopics(queryIntent?.topicIds || []);

    const diagnostics: SearchRankDiagnostics = {
        querySignals,
        retrievalSignals,
        evidenceSignals,
        explicitOutOfScopeOnly,
        inDomainEvidenceRejectLabel: null,
    };

    if (explicitOutOfScopeOnly) {
        return {
            matches: [],
            weakMatches: sortedDecisionRanking.slice(0, 5),
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
                rejectScore: Math.max(
                    responseDecision.rejectScore || 0,
                    HARD_REJECT_SCORE_THRESHOLD,
                ),
                rejectTier: "hard_reject",
            },
            diagnostics,
        };
    }

    if (responseDecision.mode === "reject") {
        const rejectionReason: SearchRejection["reason"] =
            responseDecision.rejectTier === "invalid_input"
                ? "invalid_input"
                : responseDecision.rejectTier === "hard_reject"
                  ? "low_topic_coverage"
                  : "low_consistency";
        return {
            matches: [],
            weakMatches: responseDecision.useWeakMatches
                ? sortedDecisionRanking.slice(0, 5)
                : [],
            rejection: {
                reason: rejectionReason,
                topicIds: queryIntent?.topicIds || [],
            },
            responseDecision,
            diagnostics,
        };
    }

    return {
        matches: sortedOutputRanking.slice(0, 100),
        weakMatches: [],
        responseDecision,
        diagnostics,
    };
}

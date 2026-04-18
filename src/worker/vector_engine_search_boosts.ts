import {
    type AggregatedDocScores,
    type KPCandidate,
} from "./aggregated_doc_scores.ts";
import type { QueryPlan } from "./query_planner.ts";
import {
    CURRENT_PROCESS_EVENT_TYPES,
    CURRENT_PROCESS_TIMESTAMP_BOOST_BASE,
    DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    EVENT_TYPE_MISMATCH_PENALTY,
    hasPostOutcomeConditionCue,
    LATEST_POLICY_TIMESTAMP_BOOST_BASE,
    LATEST_YEAR_BOOST_BASE,
    dedupe,
    type KPRoleRerankMode,
    type SearchResult,
} from "./vector_engine_shared.ts";
import {
    type DocQuerySignals,
    getCoverageComparableTopicIds,
    getDocQuerySignals,
    getMatchedSpecificityTf,
    hasAnyOverlap,
    hasIntentConflict,
    hasIntentMatch,
    type QueryIntentContext,
    type ScopeSpecificityStats,
} from "./vector_engine_search_context.ts";
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

export function rerankKpCandidatesByRole(params: {
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

export function hasAnyRoleEvidence(
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

export function applyQueryPlannerCoverageDiversification(
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

export function computeBoostMultiplier(params: {
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

import type { AggregatedDocScores } from "../aggregated_doc_scores.ts";
import type { QueryPlan } from "../query_planner.ts";
import {
    CURRENT_PROCESS_EVENT_TYPES,
    CURRENT_PROCESS_TIMESTAMP_BOOST_BASE,
    DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    EVENT_TYPE_MISMATCH_PENALTY,
    LATEST_POLICY_TIMESTAMP_BOOST_BASE,
    LATEST_YEAR_BOOST_BASE,
    dedupe,
} from "./shared.ts";
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
} from "./search_context.ts";
import {
    deriveQueryRoleSignals,
    hasAnyRoleEvidence,
    hasKpRoleTag,
} from "./search_role_rerank.ts";
import { applyQueryPlannerRetrievalBoost } from "./search_planner.ts";
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

export function computeBoostMultiplier(params: {
    otid: string;
    scores: AggregatedDocScores;
    lexicalBonusMap: Map<string, number>;
    enableLexicalBonusBoost?: boolean;
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
        enableLexicalBonusBoost = true,
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
    if (enableLexicalBonusBoost) {
        boost = applyLexicalBonusBoost(boost, lexicalBonus);
    }
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

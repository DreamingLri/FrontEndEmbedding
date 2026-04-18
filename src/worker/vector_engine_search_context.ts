import type { AggregatedDocScores } from "./aggregated_doc_scores.ts";
import {
    BROAD_LATEST_SCOPE_CUE_PATTERN,
    DEFAULT_Q_CONFUSION_WEIGHT,
    DEFAULT_WEIGHTS,
    dedupe,
    hasPostOutcomeConditionCue,
    INTENT_CONFLICTS,
    INTENT_RULE_MAP,
    QUERY_SCOPE_SPECIFICITY_TERM_SET,
    type FusionMode,
    type KPAggregationMode,
    type ParsedQueryIntent,
    type QConfusionMode,
} from "./vector_engine_shared.ts";
export function hasIntentConflict(
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

export function hasIntentMatch(
    queryIntentIds: string[],
    docIntentIds?: string[],
): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) =>
        docIntentIds.includes(queryIntentId),
    );
}

export function hasAnyOverlap(a: string[], b?: string[]): boolean {
    if (!b || b.length === 0) return false;
    return a.some((item) => b.includes(item));
}

export function getCoverageComparableTopicIds(doc: {
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

export type QueryIntentContext = {
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

export type DocQuerySignals = {
    hasStructuredYearMatch: boolean;
    hasLexicalYearMatch: boolean;
    hasPublishYearMatch: boolean;
    hasSuspiciousStructuredYear: boolean;
    docPublishYear?: number;
    hasStructuredMonthMatch: boolean;
    docMonth?: number;
};

export type ScopeSpecificityStats = {
    termTf: Record<string, number>;
    totalTf: number;
};

export function getMatchedSpecificityTf(
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

export function createQueryIntentContext(
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

export function getDocQuerySignals(
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

export function shouldSkipForExplicitYear(
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

export function computeQCompetitionPenaltyMap(params: {
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

export function computeBaseScore(
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

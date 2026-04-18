import {
    applyScoreToAggregatedDocScores,
    createAggregatedDocScores,
    mergeAggregatedDocMetadata,
    type AggregatedDocScores,
} from "./aggregated_doc_scores.ts";
import type { QueryPlan } from "./query_planner.ts";
import {
    BM25_B,
    BM25_K1,
    DEFAULT_KP_ROLE_DOC_WEIGHT,
    DEFAULT_Q_CONFUSION_WEIGHT,
    DEFAULT_WEIGHTS,
    HARD_REJECT_SCORE_THRESHOLD,
    hasOnlyOutOfScopeTopics,
    resolveDocOtid,
    resolveMetadataTopicIds,
    RRF_K,
    RRF_RANK_LIMIT,
    withQueryTokenCount,
    dotProduct,
    selectTopLocalIndices,
    type BM25Stats,
    type FusionMode,
    type KPAggregationMode,
    type KPRoleRerankMode,
    type LexicalBonusMode,
    type Metadata,
    type ParsedQueryIntent,
    type QConfusionMode,
    type QuerySignals,
    type SearchRankDiagnostics,
    type SearchRejection,
    type SearchRankOutput,
    type SearchResult,
} from "./vector_engine_shared.ts";
import {
    computeBaseScore,
    computeQCompetitionPenaltyMap,
    createQueryIntentContext,
    getDocQuerySignals,
    getMatchedSpecificityTf,
    type ScopeSpecificityStats,
    shouldSkipForExplicitYear,
} from "./vector_engine_search_context.ts";
import {
    applyQueryPlannerCoverageDiversification,
    computeBoostMultiplier,
    rerankKpCandidatesByRole,
} from "./vector_engine_search_boosts.ts";
import {
    classifyResponseMode,
    extractEvidenceSignals,
    extractRetrievalSignals,
} from "./vector_engine_search_decision.ts";

export {
    classifyResponseMode,
    extractEvidenceSignals,
    extractRetrievalSignals,
} from "./vector_engine_search_decision.ts";
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

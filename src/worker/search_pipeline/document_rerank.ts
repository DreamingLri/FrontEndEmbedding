import type { QueryPlan } from "../query_planner.ts";
import type { PipelineDocumentRecord } from "./types.ts";
import {
    buildDocumentRerankEntries,
    buildLatestVersionFamilyStats,
    getDocumentsFromRerankEntries,
    sortDocumentRerankEntriesByDisplayScore,
    updateDocumentRerankEntryScores,
} from "./document_rerank_shared.ts";
import { buildDocumentRerankQuerySignals } from "./document_rerank_query.ts";
import {
    applyCompressedQueryDisplayGuardToEntries,
    applyCoverageTitleDiversity,
    applyPhaseAnchorBoostToDocuments,
    computeCoverageWeightedDelta,
    computeLatestVersionDocDelta,
    computeQueryPlanDocRoleDelta,
    computeTitleIntentDocDelta,
} from "./document_rerank_scoring.ts";

export { queryIsCompressedKeywordLike } from "./document_rerank_query.ts";
export {
    buildPipelineDocumentLookup,
    collectUniqueFetchOtids,
    mergeCoarseMatchesWithDocumentLookup,
    resolveDynamicFetchLimit,
    selectLimitedCoarseMatches,
} from "./document_rerank_lookup.ts";

export function rerankAnswerDocuments(params: {
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
                const coverageDelta = computeCoverageWeightedDelta(
                    querySignals,
                    entry,
                );

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

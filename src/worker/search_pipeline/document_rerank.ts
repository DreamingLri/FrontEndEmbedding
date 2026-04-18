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
    // 文档重排主线先把 query 信号和文档元数据标准化，
    // 后面所有 title/phase/latest-version 规则都只围绕这两个结构计算。
    const documentEntries = buildDocumentRerankEntries(documents);
    const phaseAdjustedEntries = enablePhaseAnchorBoost
        ? applyPhaseAnchorBoostToDocuments(querySignals, documentEntries)
        : documentEntries;

    if (!applyTitleAdjustments) {
        // 关闭标题修正时仍保留压缩关键词保护，
        // 防止短 query 被少量规则噪声把展示第一名直接拉偏。
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
    // 这里是真正文档展示打分：
    // title intent 决定“这篇文档像不像用户要的那类通知”，
    // latest version 处理同一家族的新旧版本，
    // coverage 则补偿多要素问题对覆盖面的需求。
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
    // 当 query 明确要求流程/条件/材料等覆盖面时，再做一次标题级去重，
    // 避免前几名被同一类公告反复占满。
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

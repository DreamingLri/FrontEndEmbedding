import {
    searchAndRank,
    type SearchRankOutput,
} from "../vector_engine.ts";
import {
    buildPipelineDocumentLookup,
    collectUniqueFetchOtids,
    mergeCoarseMatchesWithDocumentLookup,
    queryIsCompressedKeywordLike,
    rerankAnswerDocuments,
    resolveDynamicFetchLimit,
    selectLimitedCoarseMatches,
} from "./document_rerank.ts";
import {
    CANONICAL_PIPELINE_PRESET,
    clonePipelinePreset,
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    FRONTEND_RESEARCH_SYNC_QUERY_PLANNER_PIPELINE_PRESET,
    MINIMAL_BASELINE_PIPELINE_PRESET,
    PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    PIPELINE_PRESET_REGISTRY,
    PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    resolvePipelinePresetByName,
} from "./presets.ts";
import {
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
} from "./query_context.ts";
import type {
    PipelineDecision,
    PipelineDocumentRecord,
    PipelineDocumentLoader,
    RetrievalStageResult,
    SearchPipelineResult,
    SearchPipelineRuntimeParams,
} from "./types.ts";

export type {
    PipelineBehavior,
    PipelineCoarseMatch,
    PipelineDecision,
    PipelineDocumentLoader,
    PipelineDocumentRecord,
    PipelinePreset,
    PipelineTermMaps,
    PipelineTrace,
    RetrievalStageResult,
    SearchPipelineQueryContext,
    SearchPipelineResult,
} from "./types.ts";
export type { PipelinePresetName } from "./presets.ts";
export {
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    CANONICAL_PIPELINE_PRESET,
    clonePipelinePreset,
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    FRONTEND_RESEARCH_SYNC_QUERY_PLANNER_PIPELINE_PRESET,
    MINIMAL_BASELINE_PIPELINE_PRESET,
    PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    PIPELINE_PRESET_REGISTRY,
    PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    resolvePipelinePresetByName,
};

function nowMs(): number {
    return typeof performance !== "undefined" && typeof performance.now === "function"
        ? performance.now()
        : Date.now();
}

function buildPipelineDecision(searchOutput: SearchRankOutput): PipelineDecision {
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

export function executeRetrievalStage(
    params: SearchPipelineRuntimeParams,
): RetrievalStageResult {
    const {
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
    // 第一阶段只做“召回 + 排序 + 拒答前诊断”，不触碰正文抓取。
    // 这样检索质量和后续展示逻辑可以独立分析，也方便实验复现。
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
        scopeSpecificityWordIdToTerm: termMaps?.scopeSpecificityWordIdToTerm,
        weights: preset.retrieval.weights,
        topHybridLimit: preset.retrieval.topHybridLimit,
        kpAggregationMode: preset.retrieval.kpAggregationMode,
        kpTopN: preset.retrieval.kpTopN,
        kpTailWeight: preset.retrieval.kpTailWeight,
        lexicalBonusMode: preset.retrieval.lexicalBonusMode,
        enableLexicalBonusBoost: preset.retrieval.enableLexicalBonusBoost,
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
        retrievalDecision: buildPipelineDecision(searchOutput),
        candidateCount: queryContext.candidateIndices?.length ?? metadata.length,
        searchMs: nowMs() - startedAt,
    };
}

export async function executeSearchPipeline(
    params: SearchPipelineRuntimeParams & {
        documentLoader: PipelineDocumentLoader;
        onStatus?: (message: string) => void;
    },
): Promise<SearchPipelineResult> {
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
    const displayRoleAlignmentQueryPlan =
        preset.display.enableStructuredQueryPlanDocRoleAdjustments
            ? queryContext.queryPlan
            : plannerQueryPlan;
    const isCompressedKeywordQuery = queryIsCompressedKeywordLike(query);
    const compressedQueryFetchDelta = isCompressedKeywordQuery ? 18 : 0;
    const shouldApplyTitleAdjustments =
        preset.display.useYearPhaseTitleAdjustment;
    // 抓取上限不是固定常数，而是由 query planner 和压缩关键词保护共同调节。
    // 目的是在“召回不稳”时多抓一些候选，但不让常规请求无上限膨胀。
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
    const answerCoarseMatches = shouldFetchAnswerResults
        ? selectLimitedCoarseMatches(searchOutput.matches, fetchMatchLimit)
        : [];
    const weakCoarseMatches = shouldFetchWeakResults
        ? selectLimitedCoarseMatches(
              searchOutput.weakMatches,
              fetchWeakMatchLimit,
          )
        : [];
    const fetchIds = collectUniqueFetchOtids(
        answerCoarseMatches,
        weakCoarseMatches,
    );

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
        const fetchedDocumentLookup = buildPipelineDocumentLookup(documents);

        if (answerCoarseMatches.length > 0) {
            // answer 分支会把粗排结果映射回正文，并做标题/阶段/新旧版本等展示重排，
            // 保证最终给用户看的顺序与纯向量粗排不同。
            const directDocuments = mergeCoarseMatchesWithDocumentLookup(
                fetchedDocumentLookup,
                answerCoarseMatches,
            );
            results = rerankAnswerDocuments({
                query,
                queryIntent: queryContext.queryIntent,
                documents: directDocuments,
                queryPlan: plannerQueryPlan,
                roleAlignmentQueryPlan: displayRoleAlignmentQueryPlan,
                enablePhaseAnchorBoost: preset.retrieval.enablePhaseAnchorBoost,
                applyTitleAdjustments: shouldApplyTitleAdjustments,
                enableTitleIntentConfusionGate:
                    preset.display.enableTitleIntentConfusionGate,
                enableStructuredKpRoleEvidenceAdjustments:
                    preset.display.enableStructuredKpRoleEvidenceAdjustments,
                enableLexicalTitleIntentAdjustments:
                    preset.display.enableLexicalTitleIntentAdjustments,
                enableLexicalTitleTypeAdjustments:
                    preset.display.enableLexicalTitleTypeAdjustments,
                enableThemeSpecificTitleAdjustments:
                    preset.display.enableThemeSpecificTitleAdjustments,
                enableDoctoralThemeTitleAdjustments:
                    preset.display.enableDoctoralThemeTitleAdjustments,
                enableTuimianThemeTitleAdjustments:
                    preset.display.enableTuimianThemeTitleAdjustments,
                enableSummerCampThemeTitleAdjustments:
                    preset.display.enableSummerCampThemeTitleAdjustments,
                enableTransferThemeTitleAdjustments:
                    preset.display.enableTransferThemeTitleAdjustments,
                enableCompressedKeywordTitleAdjustments:
                    preset.display.enableCompressedKeywordTitleAdjustments,
                preferLatestWithinTopic:
                    searchOutput.responseDecision?.preferLatestWithinTopic ?? false,
            });
        }

        if (weakCoarseMatches.length > 0) {
            // weakResults 仅保留给 reject/边界场景兜底，不做 answer 版重排。
            weakResults = mergeCoarseMatchesWithDocumentLookup(
                fetchedDocumentLookup,
                weakCoarseMatches,
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


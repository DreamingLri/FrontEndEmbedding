import type { FeatureExtractionPipeline } from "@huggingface/transformers";

import { fmmTokenize } from "./fmm_tokenize.ts";
import {
    runDirectAnswerDisplayStage,
    type DirectAnswerRescueTrace,
    type DocumentRerankStats,
} from "./direct_answer_display.ts";
import {
    getQuerySparse,
    parseQueryIntent,
    QUERY_SCOPE_SPECIFICITY_TERMS,
    searchAndRank,
    type BM25Stats,
    type KPAggregationMode,
    type KPRoleRerankMode,
    type LexicalBonusMode,
    type Metadata,
    type ParsedQueryIntent,
    type QuerySignals,
    type RejectTier,
    type RetrievalSignals,
    type ResponseDecision,
    type ResponseMode,
    type SearchRankOutput,
    type SearchRejection,
} from "./vector_engine.ts";
import {
    getCandidateIndicesForQuery,
    type TopicPartitionIndex,
} from "./topic_partition.ts";

export type PipelineBehavior = "answer" | "reject";

export type PipelinePreset = {
    name: string;
    retrieval: {
        weights: {
            Q: number;
            KP: number;
            OT: number;
        };
        topHybridLimit: number;
        kpAggregationMode: KPAggregationMode;
        kpTopN: number;
        kpTailWeight: number;
        lexicalBonusMode: LexicalBonusMode;
        kpRoleRerankMode: KPRoleRerankMode;
        kpRoleDocWeight: number;
        useQueryExpansion: boolean;
        useTopicPartition: boolean;
        enableExplicitYearFilter: boolean;
        enablePhaseAnchorBoost: boolean;
        minimalMode: boolean;
    };
    display: {
        rejectThreshold: number;
        rerankBlendAlpha: number;
        bestSentenceThreshold: number;
        fetchMatchLimit: number;
        fetchWeakMatchLimit: number;
        useYearPhaseTitleAdjustment: boolean;
    };
};

const DEFAULT_DISPLAY_CONFIG: PipelinePreset["display"] = {
    rejectThreshold: 0.4,
    rerankBlendAlpha: 0.15,
    bestSentenceThreshold: 0.4,
    fetchMatchLimit: 15,
    fetchWeakMatchLimit: 10,
    useYearPhaseTitleAdjustment: false,
};

export const PAPER_FROZEN_MAIN_PIPELINE_PRESET: PipelinePreset = {
    name: "paper_frozen_main_v1",
    retrieval: {
        weights: {
            Q: 0,
            KP: 0.28571428571428575,
            OT: 0.7142857142857143,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
        useQueryExpansion: true,
        useTopicPartition: true,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: false,
        minimalMode: false,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PRODUCT_CANONICAL_FULL_PIPELINE_PRESET: PipelinePreset = {
    name: "product_canonical_full_v1",
    retrieval: {
        weights: {
            Q: 0.3333333333333333,
            KP: 0.13333333333333333,
            OT: 0.5333333333333333,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
        useQueryExpansion: true,
        useTopicPartition: true,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: false,
        minimalMode: false,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PAPER_TAIL_TOP3_W020_PIPELINE_PRESET: PipelinePreset = {
    name: "paper_tail_top3_w020_v1",
    retrieval: {
        ...PAPER_FROZEN_MAIN_PIPELINE_PRESET.retrieval,
        kpAggregationMode: "max_plus_topn",
        kpTopN: 3,
        kpTailWeight: 0.2,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET: PipelinePreset = {
    name: "product_tail_top3_w020_v1",
    retrieval: {
        ...PRODUCT_CANONICAL_FULL_PIPELINE_PRESET.retrieval,
        kpAggregationMode: "max_plus_topn",
        kpTopN: 3,
        kpTailWeight: 0.2,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const MINIMAL_BASELINE_PIPELINE_PRESET: PipelinePreset = {
    name: "minimal_q_kp_ot_v1",
    retrieval: {
        weights: {
            Q: 0.3333333333333333,
            KP: 0.3333333333333333,
            OT: 0.3333333333333333,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "off",
        kpRoleDocWeight: 0,
        useQueryExpansion: false,
        useTopicPartition: false,
        enableExplicitYearFilter: false,
        enablePhaseAnchorBoost: false,
        minimalMode: true,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET: PipelinePreset = {
    name: "frontend_research_sync_v1",
    retrieval: {
        ...MINIMAL_BASELINE_PIPELINE_PRESET.retrieval,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: true,
    },
    display: {
        ...DEFAULT_DISPLAY_CONFIG,
        fetchMatchLimit: 20,
        fetchWeakMatchLimit: 12,
        useYearPhaseTitleAdjustment: true,
    },
};

export const PIPELINE_PRESET_REGISTRY = {
    paper_frozen_main_v1: PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    product_canonical_full_v1: PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    paper_tail_top3_w020_v1: PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    product_tail_top3_w020_v1: PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    minimal_q_kp_ot_v1: MINIMAL_BASELINE_PIPELINE_PRESET,
    frontend_research_sync_v1: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
} as const;

export type PipelinePresetName = keyof typeof PIPELINE_PRESET_REGISTRY;

export const CANONICAL_PIPELINE_PRESET =
    PRODUCT_CANONICAL_FULL_PIPELINE_PRESET;

export function clonePipelinePreset(preset: PipelinePreset): PipelinePreset {
    return {
        ...preset,
        retrieval: {
            ...preset.retrieval,
            weights: { ...preset.retrieval.weights },
        },
        display: { ...preset.display },
    };
}

export function resolvePipelinePresetByName(
    presetName?: string,
): PipelinePreset {
    if (!presetName) {
        return clonePipelinePreset(CANONICAL_PIPELINE_PRESET);
    }

    const preset = PIPELINE_PRESET_REGISTRY[
        presetName as PipelinePresetName
    ];
    return clonePipelinePreset(preset || CANONICAL_PIPELINE_PRESET);
}

export type PipelineTermMaps = {
    scopeSpecificityWordIdToTerm: Map<number, string>;
};

export type SearchPipelineQueryContext = {
    query: string;
    queryIntent: ParsedQueryIntent;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryYearWordIds: number[];
    candidateIndices?: number[];
};

export type PipelineDocumentRecord = {
    id?: string;
    otid?: string;
    ot_title?: string;
    ot_text?: string;
    link?: string;
    publish_time?: string;
    bestSentence?: string;
    bestPoint?: string;
    best_kpid?: string;
    kps?: Array<{ kpid?: string; kp_text?: string }>;
    score?: number;
    coarseScore?: number;
    displayScore?: number;
    rerankScore?: number;
    confidenceScore?: number;
    snippetScore?: number;
};

export type PipelineDecision = {
    behavior: PipelineBehavior;
    rawMode: ResponseMode;
    confidence: number;
    reason: string;
    preferLatestWithinTopic: boolean;
    useWeakMatches: boolean;
    rejectionReason:
        | SearchRejection["reason"]
        | "display_threshold"
        | null;
    displayRejected: boolean;
    rejectScore?: number;
    rejectTier?: RejectTier | null;
};

export type PipelineTrace = {
    totalMs: number;
    searchMs: number;
    fetchMs: number;
    rerankMs: number;
    candidateCount: number;
    partitionUsed: boolean;
    partitionCandidateCount?: number;
    matchCount: number;
    weakMatchCount: number;
    fetchedDocumentCount: number;
    rerankedDocCount: number;
    chunksScored: number;
    rerankWindowReason?: string;
    maxChunksPerDoc?: number;
    chunkPlanReason?: string;
    initialTopConfidence?: number | null;
    topConfidence?: number | null;
    rejectionThreshold: number;
    querySignals?: QuerySignals;
    retrievalSignals?: RetrievalSignals;
    directAnswerRescue?: DirectAnswerRescueTrace;
};

export type RetrievalStageResult = {
    queryContext: SearchPipelineQueryContext;
    searchOutput: SearchRankOutput;
    retrievalDecision: PipelineDecision;
    candidateCount: number;
    searchMs: number;
};

export type SearchPipelineResult = {
    query: string;
    presetName: string;
    queryContext: SearchPipelineQueryContext;
    searchOutput: SearchRankOutput;
    responseDecision?: ResponseDecision;
    retrievalDecision: PipelineDecision;
    finalDecision: PipelineDecision;
    rejection?: SearchRejection;
    results: PipelineDocumentRecord[];
    weakResults: PipelineDocumentRecord[];
    trace: PipelineTrace;
};

export type PipelineDocumentLoader = (params: {
    query: string;
    otids: string[];
}) => Promise<PipelineDocumentRecord[]>;

type DocumentRerankResult = {
    documents: PipelineDocumentRecord[];
    stats: DocumentRerankStats;
};

function clamp01(value: number): number {
    return Math.min(1, Math.max(0, value));
}

function nowMs(): number {
    if (typeof performance !== "undefined" && performance.now) {
        return performance.now();
    }
    return Date.now();
}

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
}

const QUERY_EXPANSION_RULES: Array<{
    pattern: RegExp;
    terms: string[];
    intentTerms?: string[];
}> = [
    {
        pattern: /现场确认/,
        terms: ["网上确认"],
        intentTerms: ["网上确认"],
    },
];

function buildExpandedQueryWords(
    query: string,
    vocabMap: Map<string, number>,
): string[] {
    const baseWords = fmmTokenize(query, vocabMap);
    const expandedWords = QUERY_EXPANSION_RULES.flatMap((rule) => {
        if (!rule.pattern.test(query)) {
            return [];
        }
        return rule.terms.flatMap((term) => fmmTokenize(term, vocabMap));
    });

    return dedupe([...baseWords, ...expandedWords]);
}

function buildExpandedIntentQuery(query: string): string {
    const expandedTerms = dedupe(
        QUERY_EXPANSION_RULES.flatMap((rule) =>
            rule.pattern.test(query) ? rule.intentTerms || [] : [],
        ),
    );
    if (expandedTerms.length === 0) {
        return query;
    }
    return `${query} ${expandedTerms.join(" ")}`;
}

export function buildPipelineTermMaps(
    vocabMap: Map<string, number>,
): PipelineTermMaps {
    const scopeSpecificityWordIdToTerm = new Map<number, string>();
    QUERY_SCOPE_SPECIFICITY_TERMS.forEach((term) => {
        const wordId = vocabMap.get(term);
        if (wordId !== undefined) {
            scopeSpecificityWordIdToTerm.set(wordId, term);
        }
    });

    return {
        scopeSpecificityWordIdToTerm,
    };
}

export function buildSearchPipelineQueryContext(
    query: string,
    vocabMap: Map<string, number>,
    topicPartitionIndex: TopicPartitionIndex,
    preset: PipelinePreset = CANONICAL_PIPELINE_PRESET,
): SearchPipelineQueryContext {
    const expandedIntentQuery = preset.retrieval.useQueryExpansion
        ? buildExpandedIntentQuery(query)
        : query;
    const parsedQueryIntent = parseQueryIntent(expandedIntentQuery);
    const queryIntent =
        expandedIntentQuery === query
            ? parsedQueryIntent
            : {
                  ...parsedQueryIntent,
                  rawQuery: query,
              };
    const candidateIndices = preset.retrieval.useTopicPartition
        ? getCandidateIndicesForQuery(queryIntent, topicPartitionIndex)
        : undefined;
    const queryWords = preset.retrieval.useQueryExpansion
        ? buildExpandedQueryWords(query, vocabMap)
        : Array.from(new Set(fmmTokenize(query, vocabMap)));
    const querySparse = getQuerySparse(queryWords, vocabMap);
    const queryYearWordIds = queryIntent.years
        .map(String)
        .map((year) => vocabMap.get(year))
        .filter((item): item is number => item !== undefined);

    return {
        query,
        queryIntent,
        queryWords,
        querySparse,
        queryYearWordIds,
        candidateIndices,
    };
}

function buildPipelineDecision(params: {
    query: string;
    queryIntent: ParsedQueryIntent;
    searchOutput: SearchRankOutput;
}): PipelineDecision {
    const { searchOutput } = params;
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
        displayRejected: false,
        rejectScore: searchOutput.responseDecision?.rejectScore,
        rejectTier: searchOutput.responseDecision?.rejectTier ?? null,
    };
}

function deriveDisplayRejectScore(
    topConfidence: number | null,
    rejectThreshold: number,
): number {
    if (topConfidence === null) {
        return 0.62;
    }
    const gapRatio = clamp01(
        (rejectThreshold - topConfidence) / Math.max(rejectThreshold, 0.001),
    );
    return clamp01(0.5 + gapRatio * 0.18);
}

type PhaseAnchor = {
    half?: "上半年" | "下半年";
    batch?: string;
    stages: string[];
};

const PHASE_ANCHOR_DOC_WEIGHT = 0.35;
const PHASE_STAGE_RULES: Array<{ stage: string; pattern: RegExp }> = [
    { stage: "预报名", pattern: /预报名/ },
    { stage: "报名通知", pattern: /报名通知|网上报名/ },
    { stage: "工作方案", pattern: /工作方案/ },
    { stage: "接收办法", pattern: /接收办法/ },
    { stage: "实施办法", pattern: /实施办法/ },
    { stage: "录取方案", pattern: /录取方案/ },
    { stage: "招生简章", pattern: /招生简章/ },
    { stage: "招生章程", pattern: /招生章程/ },
    { stage: "综合考核", pattern: /综合考核/ },
    { stage: "复试", pattern: /复试/ },
    { stage: "调剂", pattern: /调剂/ },
] as const;

function normalizeBatchToken(token: string): string | undefined {
    switch (token) {
        case "1":
        case "一":
            return "1";
        case "2":
        case "二":
            return "2";
        case "3":
        case "三":
            return "3";
        case "4":
        case "四":
            return "4";
        default:
            return undefined;
    }
}

function extractPhaseAnchor(text: string): PhaseAnchor {
    const normalized = text.replace(/\s+/g, "");
    const halfMatch = normalized.match(/上半年|下半年/);
    const batchMatch = normalized.match(/第?\s*([一二三四1234])\s*批/);

    return {
        half:
            halfMatch?.[0] === "上半年" || halfMatch?.[0] === "下半年"
                ? halfMatch[0]
                : undefined,
        batch: batchMatch?.[1]
            ? normalizeBatchToken(batchMatch[1])
            : undefined,
        stages: PHASE_STAGE_RULES.filter((rule) => rule.pattern.test(normalized)).map(
            (rule) => rule.stage,
        ),
    };
}

function hasExplicitPhaseAnchor(anchor: PhaseAnchor): boolean {
    return Boolean(anchor.half || anchor.batch || anchor.stages.length > 0);
}

function computePhaseAnchorDocDelta(
    query: string,
    document: PipelineDocumentRecord,
): number {
    const queryPhase = extractPhaseAnchor(query);
    if (!hasExplicitPhaseAnchor(queryPhase)) {
        return 0;
    }

    const title = (document.ot_title || "").replace(/\s+/g, "");
    if (!title) {
        return -0.15;
    }

    const articlePhase = extractPhaseAnchor(title);
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

function applyPhaseAnchorBoostToDocuments(
    query: string,
    documents: PipelineDocumentRecord[],
): PipelineDocumentRecord[] {
    return [...documents]
        .map((document) => {
            const baseScore =
                document.displayScore ?? document.coarseScore ?? document.score ?? 0;
            const delta = computePhaseAnchorDocDelta(query, document);
            const nextScore = baseScore + delta * PHASE_ANCHOR_DOC_WEIGHT;

            return {
                ...document,
                score: nextScore,
                coarseScore:
                    (document.coarseScore ?? document.score ?? baseScore) +
                    delta * PHASE_ANCHOR_DOC_WEIGHT,
                displayScore: nextScore,
            };
        })
        .sort(
            (left, right) =>
                (right.displayScore ?? right.score ?? 0) -
                (left.displayScore ?? left.score ?? 0),
        );
}

export function mergeCoarseMatchesIntoDocuments(
    documents: PipelineDocumentRecord[],
    coarseMatches: Array<{ otid: string; score: number; best_kpid?: string }>,
): PipelineDocumentRecord[] {
    const documentMap = new Map(
        documents.map((doc) => [doc.otid || doc.id || "", doc]),
    );

    return coarseMatches
        .map((match) => {
            const doc = documentMap.get(match.otid);
            if (!doc) {
                return null;
            }

            return {
                ...doc,
                score: match.score ?? doc.score,
                coarseScore: match.score ?? doc.coarseScore ?? doc.score,
                displayScore: match.score ?? doc.displayScore ?? doc.score,
                best_kpid: match.best_kpid ?? doc.best_kpid,
            };
        })
        .filter(Boolean) as PipelineDocumentRecord[];
}

export function executeRetrievalStage(params: {
    query: string;
    queryVector: Float32Array;
    queryContext: SearchPipelineQueryContext;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    termMaps?: PipelineTermMaps;
    preset?: PipelinePreset;
}): RetrievalStageResult {
    const {
        query,
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
        scopeSpecificityWordIdToTerm:
            termMaps?.scopeSpecificityWordIdToTerm,
        weights: preset.retrieval.weights,
        topHybridLimit: preset.retrieval.topHybridLimit,
        kpAggregationMode: preset.retrieval.kpAggregationMode,
        kpTopN: preset.retrieval.kpTopN,
        kpTailWeight: preset.retrieval.kpTailWeight,
        lexicalBonusMode: preset.retrieval.lexicalBonusMode,
        kpRoleRerankMode: preset.retrieval.kpRoleRerankMode,
        kpRoleDocWeight: preset.retrieval.kpRoleDocWeight,
        enableExplicitYearFilter: preset.retrieval.enableExplicitYearFilter,
        minimalMode: preset.retrieval.minimalMode,
    });

    return {
        queryContext,
        searchOutput,
        retrievalDecision: buildPipelineDecision({
            query,
            queryIntent: queryContext.queryIntent,
            searchOutput,
        }),
        candidateCount: queryContext.candidateIndices?.length ?? metadata.length,
        searchMs: nowMs() - startedAt,
    };
}

export async function executeSearchPipeline(params: {
    query: string;
    queryVector: Float32Array;
    queryContext: SearchPipelineQueryContext;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    extractor: FeatureExtractionPipeline;
    documentLoader: PipelineDocumentLoader;
    termMaps?: PipelineTermMaps;
    preset?: PipelinePreset;
    onStatus?: (message: string) => void;
}): Promise<SearchPipelineResult> {
    const {
        query,
        queryVector,
        queryContext,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        extractor,
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
    const shouldFetchWeakResults =
        retrievalDecision.behavior === "reject" &&
        searchOutput.rejection?.reason === "low_topic_coverage";
    const matchIds =
        retrievalDecision.behavior === "answer"
            ? searchOutput.matches
                  .slice(0, preset.display.fetchMatchLimit)
                  .map((item) => item.otid)
            : [];
    const weakMatchIds = shouldFetchWeakResults
        ? searchOutput.weakMatches
              .slice(0, preset.display.fetchWeakMatchLimit)
              .map((item) => item.otid)
        : [];
    const fetchIds = dedupe([...matchIds, ...weakMatchIds]);

    let fetchMs = 0;
    let rerankMs = 0;
    let fetchedDocumentCount = 0;
    let results: PipelineDocumentRecord[] = [];
    let weakResults: PipelineDocumentRecord[] = [];
    let finalDecision: PipelineDecision = retrievalDecision;
    let rerankStats: DocumentRerankResult["stats"] = {
        rerankedDocCount: 0,
        chunksScored: 0,
        topConfidence: null,
    };
    let initialTopConfidence: number | null = null;
    let directAnswerRescue: DirectAnswerRescueTrace | undefined;

    if (fetchIds.length > 0) {
        onStatus?.("正在请求原文数据...");
        const fetchStartedAt = nowMs();
        const documents = await documentLoader({
            query,
            otids: fetchIds,
        });
        fetchMs = nowMs() - fetchStartedAt;
        fetchedDocumentCount = documents.length;

        if (retrievalDecision.behavior === "answer") {
            const directDocuments = mergeCoarseMatchesIntoDocuments(
                documents,
                searchOutput.matches
                    .slice(0, preset.display.fetchMatchLimit)
                    .map((item) => ({
                        otid: item.otid,
                        score: item.score,
                        best_kpid: item.best_kpid,
                    })),
            );
            const phaseAdjustedDocuments =
                preset.retrieval.enablePhaseAnchorBoost
                    ? applyPhaseAnchorBoostToDocuments(query, directDocuments)
                    : directDocuments;

            onStatus?.("正在重排并提炼可信原话...");
            const rerankStartedAt = nowMs();
            const displayStage = await runDirectAnswerDisplayStage({
                query,
                queryVector,
                documents: phaseAdjustedDocuments,
                extractor,
                dimensions,
                preset,
                querySignals: searchOutput.diagnostics?.querySignals,
                retrievalSignals: searchOutput.diagnostics?.retrievalSignals,
            });
            rerankMs = nowMs() - rerankStartedAt;
            rerankStats = displayStage.stats;
            initialTopConfidence = displayStage.initialTopConfidence;
            directAnswerRescue = displayStage.directAnswerRescue;

            const displayRejected = displayStage.displayRejected;

            finalDecision = displayRejected
                ? {
                      ...retrievalDecision,
                      behavior: "reject",
                      rejectionReason: "display_threshold",
                      reason: "display_threshold_reject",
                      displayRejected: true,
                      useWeakMatches: false,
                      rejectScore: Math.max(
                          retrievalDecision.rejectScore ?? 0,
                          deriveDisplayRejectScore(
                              displayStage.finalTopConfidence,
                              preset.display.rejectThreshold,
                          ),
                      ),
                      rejectTier: "boundary_uncertain",
                  }
                : retrievalDecision;

            results = displayRejected ? [] : displayStage.documents;
        }

        if (shouldFetchWeakResults) {
            weakResults = mergeCoarseMatchesIntoDocuments(
                documents,
                searchOutput.weakMatches
                    .slice(0, preset.display.fetchWeakMatchLimit)
                    .map((item) => ({
                        otid: item.otid,
                        score: item.score,
                        best_kpid: item.best_kpid,
                    })),
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
        finalDecision,
        rejection: searchOutput.rejection,
        results,
        weakResults,
        trace: {
            totalMs: nowMs() - pipelineStartedAt,
            searchMs: retrievalStage.searchMs,
            fetchMs,
            rerankMs,
            candidateCount: retrievalStage.candidateCount,
            partitionUsed: Boolean(queryContext.candidateIndices),
            partitionCandidateCount: queryContext.candidateIndices?.length,
            matchCount: searchOutput.matches.length,
            weakMatchCount: searchOutput.weakMatches.length,
            fetchedDocumentCount,
            rerankedDocCount: rerankStats.rerankedDocCount,
            chunksScored: rerankStats.chunksScored,
            rerankWindowReason: rerankStats.windowReason,
            maxChunksPerDoc: rerankStats.maxChunksPerDoc,
            chunkPlanReason: rerankStats.chunkPlanReason,
            initialTopConfidence,
            topConfidence: rerankStats.topConfidence,
            rejectionThreshold: preset.display.rejectThreshold,
            querySignals: searchOutput.diagnostics?.querySignals,
            retrievalSignals: searchOutput.diagnostics?.retrievalSignals,
            directAnswerRescue,
        },
    };
}

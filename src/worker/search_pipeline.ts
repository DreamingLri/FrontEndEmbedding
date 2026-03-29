import type { FeatureExtractionPipeline } from "@huggingface/transformers";

import { fmmTokenize } from "./fmm_tokenize";
import {
    getAdaptiveChunkPlan,
    getAdaptiveRerankPlan,
    normalizeMinMax,
    normalizeSnippetScore,
    splitIntoSemanticChunks,
} from "./rerank_helpers";
import {
    DIRECT_ANSWER_EVIDENCE_TERMS,
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
    type ResponseDecision,
    type ResponseMode,
    type SearchRankOutput,
    type SearchRejection,
} from "./vector_engine";
import {
    getCandidateIndicesForQuery,
    type TopicPartitionIndex,
} from "./topic_partition";

export type PipelineBehavior =
    | "direct_answer"
    | "clarify"
    | "route_to_entry"
    | "reject";

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
    };
    display: {
        rejectThreshold: number;
        rerankBlendAlpha: number;
        bestSentenceThreshold: number;
        fetchMatchLimit: number;
        fetchWeakMatchLimit: number;
    };
};

export const CANONICAL_PIPELINE_PRESET: PipelinePreset = {
    name: "canonical_full_v1",
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
    },
    display: {
        rejectThreshold: 0.4,
        rerankBlendAlpha: 0.15,
        bestSentenceThreshold: 0.4,
        fetchMatchLimit: 15,
        fetchWeakMatchLimit: 10,
    },
};

export type PipelineTermMaps = {
    scopeSpecificityWordIdToTerm: Map<number, string>;
    directAnswerEvidenceWordIdToTerm: Map<number, string>;
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
    entryTopic?: string;
    preferLatestWithinTopic: boolean;
    useWeakMatches: boolean;
    rejectionReason:
        | SearchRejection["reason"]
        | "display_threshold"
        | null;
    displayRejected: boolean;
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
    topConfidence?: number | null;
    rejectionThreshold: number;
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
    stats: {
        rerankedDocCount: number;
        chunksScored: number;
        windowReason?: string;
        maxChunksPerDoc?: number;
        chunkPlanReason?: string;
        topConfidence?: number | null;
    };
};

const ROUTE_ENTRY_TOPIC_BY_PATTERN: Array<{
    pattern: RegExp;
    entryTopic: string;
}> = [
    {
        pattern: /录取|拟录取|考上|录取结果|拿到.*录取/,
        entryTopic: "新生录取后手续总入口",
    },
    {
        pattern: /新生|入学前|正式入学|入学以后|报到前/,
        entryTopic: "新生入学总入口",
    },
];

function nowMs(): number {
    if (typeof performance !== "undefined" && performance.now) {
        return performance.now();
    }
    return Date.now();
}

function dedupe(items: string[]): string[] {
    return Array.from(new Set(items));
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

    const directAnswerEvidenceWordIdToTerm = new Map<number, string>();
    DIRECT_ANSWER_EVIDENCE_TERMS.forEach((term) => {
        const wordId = vocabMap.get(term);
        if (wordId !== undefined) {
            directAnswerEvidenceWordIdToTerm.set(wordId, term);
        }
    });

    return {
        scopeSpecificityWordIdToTerm,
        directAnswerEvidenceWordIdToTerm,
    };
}

export function buildSearchPipelineQueryContext(
    query: string,
    vocabMap: Map<string, number>,
    topicPartitionIndex: TopicPartitionIndex,
): SearchPipelineQueryContext {
    const queryIntent = parseQueryIntent(query);
    const candidateIndices = getCandidateIndicesForQuery(
        queryIntent,
        topicPartitionIndex,
    );
    const queryWords = dedupe(fmmTokenize(query, vocabMap));
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

function inferEntryTopic(query: string): string | undefined {
    for (const item of ROUTE_ENTRY_TOPIC_BY_PATTERN) {
        if (item.pattern.test(query)) {
            return item.entryTopic;
        }
    }
    return undefined;
}

function inferClarifyOrRouteBehavior(
    query: string,
    queryIntent: ParsedQueryIntent,
): PipelineBehavior {
    const normalizedQuery = query.replace(/\s+/g, "");
    const hasPendingOfferCue = /拟录取/.test(normalizedQuery);
    const hasOnboardingCue =
        /新生|入学|录取|考上|录取结果/.test(normalizedQuery) ||
        /拿到.*录取/.test(normalizedQuery);
    const hasClarifyCue =
        /审核|初审|资格审核|材料|补交|申请|获批|过审|通过了|通知我通过|学校通知我通过/.test(
            normalizedQuery,
        );

    if (hasPendingOfferCue) {
        return "clarify";
    }

    if (
        /新生|入学前|正式入学/.test(normalizedQuery) &&
        queryIntent.signals.hasGenericNextStep
    ) {
        return "route_to_entry";
    }

    if (
        hasOnboardingCue &&
        queryIntent.signals.hasGenericNextStep &&
        !hasClarifyCue &&
        !queryIntent.signals.hasStrongDetailAnchor
    ) {
        return "route_to_entry";
    }

    return "clarify";
}

function buildPipelineDecision(params: {
    query: string;
    queryIntent: ParsedQueryIntent;
    searchOutput: SearchRankOutput;
}): PipelineDecision {
    const { query, queryIntent, searchOutput } = params;
    const rawMode =
        searchOutput.responseDecision?.mode ||
        (searchOutput.rejection ? "reject" : "direct_answer");
    const behavior =
        rawMode === "clarify_or_route"
            ? inferClarifyOrRouteBehavior(query, queryIntent)
            : rawMode === "reject"
              ? "reject"
              : "direct_answer";

    return {
        behavior,
        rawMode,
        confidence: searchOutput.responseDecision?.confidence ?? 0.62,
        reason:
            searchOutput.responseDecision?.reason ||
            searchOutput.rejection?.reason ||
            "scored_pipeline_behavior",
        entryTopic:
            behavior === "route_to_entry" ? inferEntryTopic(query) : undefined,
        preferLatestWithinTopic:
            searchOutput.responseDecision?.preferLatestWithinTopic ?? false,
        useWeakMatches:
            behavior !== "direct_answer" &&
            (searchOutput.responseDecision?.useWeakMatches ??
                searchOutput.weakMatches.length > 0),
        rejectionReason: searchOutput.rejection?.reason || null,
        displayRejected: false,
    };
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

async function rerankDocuments(params: {
    query: string;
    queryVector: Float32Array;
    documents: PipelineDocumentRecord[];
    extractor: FeatureExtractionPipeline;
    dimensions: number;
    preset: PipelinePreset;
}): Promise<DocumentRerankResult> {
    const { query, queryVector, documents, extractor, dimensions, preset } = params;
    const results = documents.map((doc) => {
        let defaultPoint = "暂无要点";
        if (doc.best_kpid && Array.isArray(doc.kps)) {
            const hitKp = doc.kps.find((kp) => kp.kpid === doc.best_kpid);
            if (hitKp?.kp_text) {
                defaultPoint = hitKp.kp_text;
            }
        }

        return {
            ...doc,
            coarseScore: doc.coarseScore ?? doc.score ?? 0,
            displayScore: doc.displayScore ?? doc.score ?? 0,
            rerankScore: 0,
            snippetScore: 0,
            confidenceScore: 0,
            bestPoint: defaultPoint,
            bestSentence: "",
        };
    });

    const rerankPlan = getAdaptiveRerankPlan(results);
    const rerankDocCount = rerankPlan.rerankDocCount;
    const chunkPlan = getAdaptiveChunkPlan(query, rerankDocCount);
    const rerankDocs = results.slice(0, rerankDocCount);
    const batchChunks: Array<{ text: string; docIdx: number }> = [];

    for (let index = 0; index < rerankDocs.length; index += 1) {
        const textChunks = splitIntoSemanticChunks(
            rerankDocs[index].ot_text || "",
            150,
            chunkPlan.maxChunksPerDoc,
        );

        textChunks.forEach((chunk) => {
            const normalizedChunk = (chunk || "").trim();
            if (normalizedChunk) {
                batchChunks.push({
                    text: normalizedChunk,
                    docIdx: index,
                });
            }
        });
    }

    if (batchChunks.length > 0) {
        const batchTexts = batchChunks.map((item) => item.text);
        const batchOutputs = await extractor(batchTexts, {
            pooling: "mean",
            normalize: true,
            truncation: true,
            max_length: 512,
        } as any);

        const pureData = (batchOutputs.data as Float32Array).subarray(
            0,
            batchChunks.length * dimensions,
        );
        const rawDocumentScores = new Float32Array(rerankDocs.length).fill(-1);
        const documentBestSentence = new Array<string>(rerankDocs.length).fill("");

        for (let chunkIndex = 0; chunkIndex < batchChunks.length; chunkIndex += 1) {
            const chunkVec = pureData.subarray(
                chunkIndex * dimensions,
                (chunkIndex + 1) * dimensions,
            );
            let score = 0;
            for (let dimensionIndex = 0; dimensionIndex < dimensions; dimensionIndex += 1) {
                score += queryVector[dimensionIndex] * chunkVec[dimensionIndex];
            }

            const docIdx = batchChunks[chunkIndex].docIdx;
            if (score > rawDocumentScores[docIdx]) {
                rawDocumentScores[docIdx] = score;
                documentBestSentence[docIdx] = batchChunks[chunkIndex].text;
            }
        }

        const coarseNorm = normalizeMinMax(
            rerankDocs.map((doc) => doc.coarseScore ?? 0),
        );

        for (let index = 0; index < rerankDocs.length; index += 1) {
            const normalizedSnippetScore = normalizeSnippetScore(
                rawDocumentScores[index],
            );
            const blendedScore =
                preset.display.rerankBlendAlpha * coarseNorm[index] +
                (1 - preset.display.rerankBlendAlpha) * normalizedSnippetScore;

            rerankDocs[index].snippetScore = normalizedSnippetScore;
            rerankDocs[index].confidenceScore = blendedScore;
            rerankDocs[index].rerankScore = blendedScore;
            rerankDocs[index].displayScore = blendedScore;

            if (
                normalizedSnippetScore > preset.display.bestSentenceThreshold &&
                documentBestSentence[index]
            ) {
                rerankDocs[index].bestSentence = documentBestSentence[index];
            }
        }

        rerankDocs.sort((a, b) => {
            const scoreDiff = (b.rerankScore ?? 0) - (a.rerankScore ?? 0);
            if (Math.abs(scoreDiff) > 1e-9) {
                return scoreDiff;
            }
            return (b.coarseScore ?? 0) - (a.coarseScore ?? 0);
        });
    } else {
        rerankDocs.forEach((doc) => {
            doc.displayScore = 0;
            doc.rerankScore = 0;
            doc.snippetScore = 0;
            doc.confidenceScore = 0;
        });
    }

    const documentsWithRerank = rerankDocs.concat(results.slice(rerankDocCount));
    const topConfidence =
        documentsWithRerank[0]?.confidenceScore ??
        documentsWithRerank[0]?.rerankScore ??
        null;

    return {
        documents: documentsWithRerank,
        stats: {
            rerankedDocCount: rerankDocCount,
            chunksScored: batchChunks.length,
            windowReason: rerankPlan.reason,
            maxChunksPerDoc: chunkPlan.maxChunksPerDoc,
            chunkPlanReason: chunkPlan.reason,
            topConfidence,
        },
    };
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
        directAnswerEvidenceWordIdToTerm:
            termMaps?.directAnswerEvidenceWordIdToTerm,
        weights: preset.retrieval.weights,
        topHybridLimit: preset.retrieval.topHybridLimit,
        kpAggregationMode: preset.retrieval.kpAggregationMode,
        kpTopN: preset.retrieval.kpTopN,
        kpTailWeight: preset.retrieval.kpTailWeight,
        lexicalBonusMode: preset.retrieval.lexicalBonusMode,
        kpRoleRerankMode: preset.retrieval.kpRoleRerankMode,
        kpRoleDocWeight: preset.retrieval.kpRoleDocWeight,
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
        retrievalDecision.behavior === "clarify" ||
        retrievalDecision.behavior === "route_to_entry" ||
        searchOutput.rejection?.reason === "low_topic_coverage";
    const matchIds =
        retrievalDecision.behavior === "direct_answer"
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

    if (fetchIds.length > 0) {
        onStatus?.("正在请求原文数据...");
        const fetchStartedAt = nowMs();
        const documents = await documentLoader({
            query,
            otids: fetchIds,
        });
        fetchMs = nowMs() - fetchStartedAt;
        fetchedDocumentCount = documents.length;

        if (retrievalDecision.behavior === "direct_answer") {
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

            onStatus?.("正在重排并提炼可信原话...");
            const rerankStartedAt = nowMs();
            const rerankResult = await rerankDocuments({
                query,
                queryVector,
                documents: directDocuments,
                extractor,
                dimensions,
                preset,
            });
            rerankMs = nowMs() - rerankStartedAt;
            rerankStats = rerankResult.stats;

            const topConfidence = rerankResult.stats.topConfidence ?? -999;
            const displayRejected =
                rerankResult.documents.length > 0 &&
                topConfidence < preset.display.rejectThreshold;

            finalDecision = displayRejected
                ? {
                      ...retrievalDecision,
                      behavior: "reject",
                      rejectionReason: "display_threshold",
                      reason: "display_threshold_reject",
                      displayRejected: true,
                      useWeakMatches: false,
                  }
                : retrievalDecision;

            results = displayRejected ? [] : rerankResult.documents;
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
            topConfidence: rerankStats.topConfidence,
            rejectionThreshold: preset.display.rejectThreshold,
        },
    };
}

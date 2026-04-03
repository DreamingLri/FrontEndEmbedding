import type { FeatureExtractionPipeline } from "@huggingface/transformers";

import {
    getAdaptiveChunkPlan,
    getAdaptiveRerankPlan,
    normalizeMinMax,
    normalizeSnippetScore,
    splitIntoSemanticChunks,
} from "./rerank_helpers.ts";
import type {
    PipelineDocumentRecord,
    PipelinePreset,
} from "./search_pipeline.ts";
import type { QuerySignals, RetrievalSignals } from "./vector_engine.ts";

const NEAR_TIE_COARSE_WINDOW = 0.025;
const DIRECT_ANSWER_RESCUE_MARGIN = 0.12;
const MIN_RESCUE_CHUNK_LIMIT = 6;
const MAX_RESCUE_CHUNK_LIMIT = 14;
const MAX_RESCUE_RERANK_DOC_COUNT = 6;
const RESCUE_ACCEPT_DELTA = 0.005;

export type DocumentRerankStats = {
    rerankedDocCount: number;
    chunksScored: number;
    windowReason?: string;
    maxChunksPerDoc?: number;
    chunkPlanReason?: string;
    topConfidence?: number | null;
    blendAlpha?: number;
};

export type DirectAnswerRescueTrace = {
    attempted: boolean;
    accepted: boolean;
    succeeded: boolean;
    reason?: string;
    initialTopConfidence?: number | null;
    rescueTopConfidence?: number | null;
    initialRerankDocCount?: number;
    rescueRerankDocCount?: number;
    initialMaxChunksPerDoc?: number;
    rescueMaxChunksPerDoc?: number;
    initialBlendAlpha?: number;
    rescueBlendAlpha?: number;
};

export type DirectAnswerDisplayResult = {
    documents: PipelineDocumentRecord[];
    stats: DocumentRerankStats;
    initialTopConfidence: number | null;
    finalTopConfidence: number | null;
    displayRejected: boolean;
    directAnswerRescue: DirectAnswerRescueTrace;
};

type RerankOverrides = {
    rerankDocCount?: number;
    maxChunksPerDoc?: number;
    blendAlpha?: number;
    overrideReason?: string;
};

function clamp01(value: number): number {
    return Math.min(1, Math.max(0, value));
}

function inferQueryDisambiguationFloor(query: string): {
    rerankDocCount: number;
    reason?: string;
} {
    const normalizedQuery = query.replace(/\s+/g, "");

    if (/考什么|考哪些|考试内容|考试科目|科目|题型/.test(normalizedQuery)) {
        return {
            rerankDocCount: 3,
            reason: "exam_content_disambiguation",
        };
    }

    if (
        /(广州国家实验室|广州实验室|鹏城国家实验室|鹏城)/.test(
            normalizedQuery,
        ) &&
        /联合培养|博士/.test(normalizedQuery)
    ) {
        return {
            rerankDocCount: 3,
            reason: "lab_entity_disambiguation",
        };
    }

    if (
        /临床医学博士/.test(normalizedQuery) &&
        /同等学力|答辩/.test(normalizedQuery)
    ) {
        return {
            rerankDocCount: 3,
            reason: "clinical_doctoral_defense_disambiguation",
        };
    }

    return {
        rerankDocCount: 0,
    };
}

function computeTitleAdjustment(query: string, docTitle?: string): number {
    const normalizedQuery = query.replace(/\s+/g, "");
    const normalizedTitle = (docTitle || "").replace(/\s+/g, "");

    if (!normalizedTitle) {
        return 0;
    }

    let adjustment = 0;

    if (/考什么|考哪些|考试内容|考试科目|科目|题型/.test(normalizedQuery)) {
        if (/章程|专业目录/.test(normalizedTitle)) {
            adjustment += 0.14;
        }
        if (/成绩查询|合格成绩要求|复试/.test(normalizedTitle)) {
            adjustment -= 0.2;
        }
    }

    if (/(广州国家实验室|广州实验室)/.test(normalizedQuery)) {
        if (/(广州国家实验室|广州实验室)/.test(normalizedTitle)) {
            adjustment += 0.16;
        }
        if (/鹏城/.test(normalizedTitle)) {
            adjustment -= 0.22;
        }
    }

    if (/(鹏城国家实验室|鹏城)/.test(normalizedQuery)) {
        if (/鹏城/.test(normalizedTitle)) {
            adjustment += 0.16;
        }
        if (/(广州国家实验室|广州实验室)/.test(normalizedTitle)) {
            adjustment -= 0.18;
        }
    }

    if (/临床医学博士/.test(normalizedQuery)) {
        if (/临床医学博士/.test(normalizedTitle)) {
            adjustment += 0.16;
        }
        if (/硕士学位/.test(normalizedTitle)) {
            adjustment -= 0.24;
        } else if (/博士/.test(normalizedTitle)) {
            adjustment += 0.08;
        }
    }

    if (/同等学力/.test(normalizedQuery) && /同等学力/.test(normalizedTitle)) {
        adjustment += 0.05;
    }

    return Math.max(-0.28, Math.min(0.22, adjustment));
}

function rerankDocuments(params: {
    query: string;
    queryVector: Float32Array;
    documents: PipelineDocumentRecord[];
    extractor: FeatureExtractionPipeline;
    dimensions: number;
    preset: PipelinePreset;
    overrides?: RerankOverrides;
}): Promise<{ documents: PipelineDocumentRecord[]; stats: DocumentRerankStats }> {
    return (async () => {
        const { query, queryVector, documents, extractor, dimensions, preset, overrides } = params;
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

        const adaptiveRerankPlan = getAdaptiveRerankPlan(results);
        const disambiguationFloor = inferQueryDisambiguationFloor(query);
        const rerankDocCount = Math.min(
            results.length,
            Math.max(
                0,
                disambiguationFloor.rerankDocCount,
                overrides?.rerankDocCount ?? adaptiveRerankPlan.rerankDocCount,
            ),
        );
        const adaptiveChunkPlan = getAdaptiveChunkPlan(query, rerankDocCount);
        const maxChunksPerDoc =
            overrides?.maxChunksPerDoc ?? adaptiveChunkPlan.maxChunksPerDoc;
        const blendAlpha =
            overrides?.blendAlpha ?? preset.display.rerankBlendAlpha;
        const rerankDocs = results.slice(0, rerankDocCount);
        const batchChunks: Array<{ text: string; docIdx: number }> = [];

        for (let index = 0; index < rerankDocs.length; index += 1) {
            const textChunks = splitIntoSemanticChunks(
                rerankDocs[index].ot_text || "",
                150,
                maxChunksPerDoc,
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
                    blendAlpha * coarseNorm[index] +
                    (1 - blendAlpha) * normalizedSnippetScore;
                const titleAdjustment = computeTitleAdjustment(
                    query,
                    rerankDocs[index].ot_title,
                );
                const finalScore = clamp01(blendedScore + titleAdjustment);

                rerankDocs[index].snippetScore = normalizedSnippetScore;
                rerankDocs[index].confidenceScore = finalScore;
                rerankDocs[index].rerankScore = finalScore;
                rerankDocs[index].displayScore = finalScore;

                if (
                    normalizedSnippetScore > preset.display.bestSentenceThreshold &&
                    documentBestSentence[index]
                ) {
                    rerankDocs[index].bestSentence = documentBestSentence[index];
                }
            }

            rerankDocs.sort((a, b) => {
                const scoreDiff = (b.rerankScore ?? 0) - (a.rerankScore ?? 0);
                if (Math.abs(scoreDiff) <= NEAR_TIE_COARSE_WINDOW) {
                    return (b.coarseScore ?? 0) - (a.coarseScore ?? 0);
                }
                return scoreDiff;
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
                windowReason:
                    overrides?.overrideReason ||
                    [adaptiveRerankPlan.reason, disambiguationFloor.reason]
                        .filter(Boolean)
                        .join("+"),
                maxChunksPerDoc,
                chunkPlanReason: adaptiveChunkPlan.reason,
                topConfidence,
                blendAlpha,
            },
        };
    })();
}

function buildDirectAnswerRescuePlan(params: {
    documentCount: number;
    initialTopConfidence: number;
    rejectThreshold: number;
    initialStats: DocumentRerankStats;
    querySignals?: QuerySignals;
    retrievalSignals?: RetrievalSignals;
}): {
    rerankDocCount?: number;
    maxChunksPerDoc?: number;
    blendAlpha?: number;
    reason: string;
} | null {
    const {
        documentCount,
        initialTopConfidence,
        rejectThreshold,
        initialStats,
        querySignals,
        retrievalSignals,
    } = params;

    if (initialTopConfidence >= rejectThreshold) {
        return null;
    }

    const thresholdGap = rejectThreshold - initialTopConfidence;
    const strongAnchors =
        querySignals?.hasExplicitTopicOrIntent ||
        querySignals?.hasStrongDetailAnchor ||
        querySignals?.hasExplicitYear;
    const stableRetrieval =
        (retrievalSignals?.top1Top2Gap ?? 0) >= 0.04 ||
        ((retrievalSignals?.distinctTopicCount ?? 99) <= 2 &&
            (retrievalSignals?.dominantTopicRatio ?? 0) >= 0.45);

    if (!strongAnchors && !stableRetrieval) {
        return null;
    }

    if (thresholdGap > DIRECT_ANSWER_RESCUE_MARGIN) {
        return null;
    }

    const queryLength = querySignals?.queryLength ?? 999;
    const isShortQuery = queryLength <= 14;
    const isMediumQuery = queryLength <= 28;
    const veryStableRetrieval =
        (retrievalSignals?.top1Top2Gap ?? 0) >= 1.0 &&
        (retrievalSignals?.dominantTopicRatio ?? 0) >= 0.75;
    const docGrowth = querySignals?.hasStrongDetailAnchor
          ? 6
          : veryStableRetrieval
            ? 4
            : 3;
    const nextRerankDocCount =
        documentCount > (initialStats.rerankedDocCount ?? 0)
            ? Math.min(
                  documentCount,
                  Math.max(
                      Math.min(
                          MAX_RESCUE_RERANK_DOC_COUNT,
                          (initialStats.rerankedDocCount ?? 0) +
                              docGrowth,
                      ),
                      3,
                  ),
              )
            : undefined;
    const chunkGrowth =
        querySignals?.hasStrongDetailAnchor || queryLength > 24
            ? 5
            : veryStableRetrieval
              ? 6
              : 4;
    const nextMaxChunksPerDoc = Math.min(
        MAX_RESCUE_CHUNK_LIMIT,
        Math.max(
            (initialStats.maxChunksPerDoc ?? 0) + chunkGrowth,
            MIN_RESCUE_CHUNK_LIMIT,
        ),
    );
    let rescueBlendAlpha = initialStats.blendAlpha ?? 0.15;
    if (thresholdGap <= 0.03 && stableRetrieval) {
        rescueBlendAlpha = Math.max(rescueBlendAlpha, 0.24);
    }
    if (isMediumQuery && strongAnchors && stableRetrieval) {
        rescueBlendAlpha = Math.max(rescueBlendAlpha, 0.28);
    }
    if (isShortQuery && strongAnchors && veryStableRetrieval) {
        rescueBlendAlpha = Math.max(rescueBlendAlpha, 0.36);
    }

    if (
        nextRerankDocCount === undefined &&
        nextMaxChunksPerDoc <= (initialStats.maxChunksPerDoc ?? 0) &&
        rescueBlendAlpha <= (initialStats.blendAlpha ?? 0)
    ) {
        return null;
    }

    const reasonParts: string[] = [];
    if (thresholdGap <= 0.06) reasonParts.push("near_threshold");
    if (strongAnchors) reasonParts.push("strong_anchor");
    if (stableRetrieval) reasonParts.push("stable_retrieval");

    return {
        rerankDocCount: nextRerankDocCount,
        maxChunksPerDoc: nextMaxChunksPerDoc,
        blendAlpha: rescueBlendAlpha,
        reason: reasonParts.join("+") || "direct_answer_rescue",
    };
}

export async function runDirectAnswerDisplayStage(params: {
    query: string;
    queryVector: Float32Array;
    documents: PipelineDocumentRecord[];
    extractor: FeatureExtractionPipeline;
    dimensions: number;
    preset: PipelinePreset;
    querySignals?: QuerySignals;
    retrievalSignals?: RetrievalSignals;
}): Promise<DirectAnswerDisplayResult> {
    const {
        query,
        queryVector,
        documents,
        extractor,
        dimensions,
        preset,
        querySignals,
        retrievalSignals,
    } = params;

    const initialResult = await rerankDocuments({
        query,
        queryVector,
        documents,
        extractor,
        dimensions,
        preset,
    });
    const initialTopConfidence = initialResult.stats.topConfidence ?? null;
    let selectedResult = initialResult;

    const rescueTrace: DirectAnswerRescueTrace = {
        attempted: false,
        accepted: false,
        succeeded: false,
        initialTopConfidence,
        initialRerankDocCount: initialResult.stats.rerankedDocCount,
        initialMaxChunksPerDoc: initialResult.stats.maxChunksPerDoc,
        initialBlendAlpha: initialResult.stats.blendAlpha,
    };

    const rescuePlan =
        initialTopConfidence !== null
            ? buildDirectAnswerRescuePlan({
                  documentCount: documents.length,
                  initialTopConfidence,
                  rejectThreshold: preset.display.rejectThreshold,
                  initialStats: initialResult.stats,
                  querySignals,
                  retrievalSignals,
              })
            : null;

    if (rescuePlan) {
        rescueTrace.attempted = true;
        rescueTrace.reason = rescuePlan.reason;
        rescueTrace.rescueRerankDocCount = rescuePlan.rerankDocCount;
        rescueTrace.rescueMaxChunksPerDoc = rescuePlan.maxChunksPerDoc;
        rescueTrace.rescueBlendAlpha = rescuePlan.blendAlpha;

        const rescueResult = await rerankDocuments({
            query,
            queryVector,
            documents,
            extractor,
            dimensions,
            preset,
            overrides: {
                rerankDocCount: rescuePlan.rerankDocCount,
                maxChunksPerDoc: rescuePlan.maxChunksPerDoc,
                blendAlpha: rescuePlan.blendAlpha,
                overrideReason: `rescue:${rescuePlan.reason}`,
            },
        });
        rescueTrace.rescueTopConfidence = rescueResult.stats.topConfidence ?? null;

        if (
            (rescueResult.stats.topConfidence ?? -999) >=
                preset.display.rejectThreshold ||
            (rescueResult.stats.topConfidence ?? -999) >=
                (initialTopConfidence ?? -999) + RESCUE_ACCEPT_DELTA
        ) {
            selectedResult = rescueResult;
            rescueTrace.accepted = true;
        }
    }

    const finalTopConfidence = selectedResult.stats.topConfidence ?? null;
    const displayRejected =
        selectedResult.documents.length > 0 &&
        (finalTopConfidence ?? -999) < preset.display.rejectThreshold;

    rescueTrace.succeeded =
        rescueTrace.attempted &&
        !displayRejected &&
        (initialTopConfidence ?? -999) < preset.display.rejectThreshold;

    return {
        documents: selectedResult.documents,
        stats: selectedResult.stats,
        initialTopConfidence,
        finalTopConfidence,
        displayRejected,
        directAnswerRescue: rescueTrace,
    };
}

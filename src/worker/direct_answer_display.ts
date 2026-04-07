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

function extractExactQueryDate(
    query: string,
): { year: number; month: number; day: number } | null {
    const match = query.match(/(20\d{2})年(\d{1,2})月(\d{1,2})日/);
    if (!match) {
        return null;
    }

    return {
        year: Number(match[1]),
        month: Number(match[2]),
        day: Number(match[3]),
    };
}

function parsePublishDate(
    publishTime?: string,
): { year: number; month: number; day: number } | null {
    if (!publishTime) {
        return null;
    }

    const match = publishTime.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) {
        return null;
    }

    return {
        year: Number(match[1]),
        month: Number(match[2]),
        day: Number(match[3]),
    };
}

type PhaseAnchor = {
    half?: "上半年" | "下半年";
    batch?: string;
    stages: string[];
};

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

function extractYears(text: string): number[] {
    return Array.from(
        new Set((text.match(/20\d{2}/g) || []).map((value) => Number(value))),
    );
}

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
    const halfMatch = text.match(/上半年|下半年/);
    const batchMatch = text.match(/第?\s*([一二三四1234])\s*批/);

    return {
        half:
            halfMatch?.[0] === "上半年" || halfMatch?.[0] === "下半年"
                ? halfMatch[0]
                : undefined,
        batch: batchMatch?.[1]
            ? normalizeBatchToken(batchMatch[1])
            : undefined,
        stages: PHASE_STAGE_RULES.filter((rule) => rule.pattern.test(text)).map(
            (rule) => rule.stage,
        ),
    };
}

function hasExplicitPhaseAnchor(anchor: PhaseAnchor): boolean {
    return Boolean(anchor.half || anchor.batch || anchor.stages.length > 0);
}

function computeYearPhaseTitleAdjustment(
    query: string,
    docTitle?: string,
    publishTime?: string,
): number {
    const normalizedQuery = query.replace(/\s+/g, "");
    const normalizedTitle = (docTitle || "").replace(/\s+/g, "");

    if (!normalizedTitle) {
        return 0;
    }

    let adjustment = 0;
    const queryYears = extractYears(normalizedQuery);
    const titleYears = extractYears(normalizedTitle);
    const publishDate = parsePublishDate(publishTime);
    const docYears = Array.from(
        new Set([
            ...titleYears,
            ...(publishDate ? [publishDate.year] : []),
        ]),
    );

    if (queryYears.length > 0) {
        if (docYears.length === 0) {
            adjustment -= 0.06;
        } else if (queryYears.some((year) => docYears.includes(year))) {
            adjustment += 0.12;
        } else {
            adjustment -= 0.18;
        }
    }

    const queryPhase = extractPhaseAnchor(normalizedQuery);
    const titlePhase = extractPhaseAnchor(normalizedTitle);
    if (!hasExplicitPhaseAnchor(queryPhase)) {
        return adjustment;
    }

    if (queryPhase.half) {
        if (titlePhase.half === queryPhase.half) {
            adjustment += 0.08;
        } else if (titlePhase.half) {
            adjustment -= 0.12;
        }
    }

    if (queryPhase.batch) {
        if (titlePhase.batch === queryPhase.batch) {
            adjustment += 0.12;
        } else if (titlePhase.batch) {
            adjustment -= 0.16;
        }
    }

    if (queryPhase.stages.length > 0) {
        const hasExactStage = queryPhase.stages.some((stage) =>
            titlePhase.stages.includes(stage),
        );
        if (hasExactStage) {
            adjustment += 0.08;
        } else if (titlePhase.stages.length > 0) {
            adjustment -= 0.12;
        }
    }

    return adjustment;
}

function extractDegreeLevels(text: string): string[] {
    const levels: string[] = [];
    if (text.includes("本科")) levels.push("本科");
    if (text.includes("硕士")) levels.push("硕士");
    if (text.includes("博士") || text.includes("直博")) levels.push("博士");
    return levels;
}

function hasAnyDegreeOverlap(queryLevels: string[], titleLevels: string[]): boolean {
    return queryLevels.some((level) => titleLevels.includes(level));
}

function computeTitleAdjustment(
    query: string,
    docTitle?: string,
    publishTime?: string,
    useYearPhaseTitleAdjustment = false,
): number {
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

    const queryDegreeLevels = extractDegreeLevels(normalizedQuery);
    const titleDegreeLevels = extractDegreeLevels(normalizedTitle);
    if (queryDegreeLevels.length > 0 && titleDegreeLevels.length > 0) {
        if (hasAnyDegreeOverlap(queryDegreeLevels, titleDegreeLevels)) {
            adjustment += 0.08;
        } else {
            adjustment -= 0.14;
        }
    }

    const exactQueryDate = extractExactQueryDate(normalizedQuery);
    const docPublishDate = parsePublishDate(publishTime);
    if (exactQueryDate && docPublishDate) {
        const exactMatch =
            exactQueryDate.year === docPublishDate.year &&
            exactQueryDate.month === docPublishDate.month &&
            exactQueryDate.day === docPublishDate.day;
        if (exactMatch) {
            adjustment += 0.14;
        } else if (
            exactQueryDate.year === docPublishDate.year &&
            exactQueryDate.month === docPublishDate.month
        ) {
            adjustment -= 0.12;
        }
    }

    const asksScheduleOrProcedure =
        /(安排|流程|步骤|时间|时段|哪天|几分钟|地点|报到|何时|什么时候|怎么|如何|方式)/.test(
            normalizedQuery,
        );
    const titleLooksOperational =
        /(安排|流程|步骤|办法|系统|校对|报到|考核安排)/.test(normalizedTitle);
    const titleLooksOutcomeLike =
        /(名单|公示|结果|标准|入围|录取名单|综合成绩)/.test(normalizedTitle);
    if (asksScheduleOrProcedure) {
        if (titleLooksOperational) {
            adjustment += 0.1;
        }
        if (titleLooksOutcomeLike) {
            adjustment -= 0.16;
        }
    }

    if (useYearPhaseTitleAdjustment) {
        adjustment += computeYearPhaseTitleAdjustment(
            normalizedQuery,
            normalizedTitle,
            publishTime,
        );
    }

    return Math.max(-0.4, Math.min(0.32, adjustment));
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
                    rerankDocs[index].publish_time,
                    preset.display.useYearPhaseTitleAdjustment,
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
        querySignals: _querySignals,
        retrievalSignals: _retrievalSignals,
    } = params;

    // The frontend pipeline is now single-stage reject only.
    // Display stage keeps pure document rerank behavior and no longer
    // performs secondary reject or rescue decisions.
    const rerankResult = await rerankDocuments({
        query,
        queryVector,
        documents,
        extractor,
        dimensions,
        preset,
    });
    const topConfidence = rerankResult.stats.topConfidence ?? null;

    return {
        documents: rerankResult.documents,
        stats: rerankResult.stats,
        initialTopConfidence: topConfidence,
        finalTopConfidence: topConfidence,
        displayRejected: false,
        directAnswerRescue: {
            attempted: false,
            accepted: false,
            succeeded: false,
            initialTopConfidence: topConfidence,
        },
    };
}

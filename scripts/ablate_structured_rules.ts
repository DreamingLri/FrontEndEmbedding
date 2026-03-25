import * as fs from "fs";
import * as path from "path";
import {
    env,
    pipeline,
    type FeatureExtractionPipeline,
} from "@huggingface/transformers";

import {
    DEFAULT_WEIGHTS,
    RRF_K,
    buildBM25Stats,
    dotProduct,
    getQuerySparse,
    parseQueryIntent,
    resolveMetadataTopicIds,
    INTENT_VECTOR_TABLE,
    type BM25Stats,
    type IntentVectorItem,
    type Metadata,
    type ParsedQueryIntent,
} from "../src/worker/vector_engine.ts";
import {
    applyScoreToAggregatedDocScores,
    createAggregatedDocScores,
    mergeAggregatedDocMetadata,
    type AggregatedDocScores,
} from "../src/worker/aggregated_doc_scores.ts";

type DatasetCase = {
    query: string;
    expected_otid: string;
    query_type?: string;
    dataset: string;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryIntent: ParsedQueryIntent;
    queryWords: string[];
    queryYearWordIds: number[];
};

type QueryState = {
    testCase: DatasetCase;
    queryIntent: ParsedQueryIntent;
    queryYearWordIds: number[];
    otidMap: Record<string, AggregatedDocScores>;
    lexicalBonusMap: Map<string, number>;
    yearHitMap: Map<string, boolean>;
    latestTargetYear?: number;
};

type AblationFlags = {
    explicitYearFilter: boolean;
    yearPenalty: boolean;
    intentPrefilter: boolean;
    intentMatchConflictBoost: boolean;
    intentSpecializationRules: boolean;
    degreeRules: boolean;
    eventRules: boolean;
    latestTimestampDecay: boolean;
    latestYearBoost: boolean;
    nonAdmissionPenaltyMultiplier: number;
    preferredEventBoostMultiplier: number;
    genericEventMatchBoostMultiplier: number;
    genericEventMismatchPenaltyMultiplier: number;
    latestYearBoostBase: number;
};

type ModeDefinition = {
    label: string;
    flags: AblationFlags;
};

type Metrics = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type ModeResult = {
    label: string;
    metricsByDataset: Record<string, Metrics>;
    combined: Metrics;
};

type Report = {
    generatedAt: string;
    datasetSizes: Record<string, number>;
    modes: ModeResult[];
};

const MODEL_NAME = "DMetaSoul/Dmeta-embedding-zh-small";
const METADATA_FILE = "public/data/frontend_metadata_dmeta_small.json";
const VECTOR_FILE = "public/data/frontend_vectors_dmeta_small.bin";
const DATASETS = [
    "../Backend/test/test_dataset_v3/test_dataset_standard.json",
    "../Backend/test/test_dataset_v3/test_dataset_short_keyword.json",
    "../Backend/test/test_dataset_v3/test_dataset_situational.json",
] as const;
const QUERY_EMBED_BATCH_SIZE = 16;
const BM25_K1 = 1.2;
const BM25_B = 0.4;
const SECONDS_IN_DAY = 86400;
const LATEST_YEAR_BOOST_BASE = 0.98;
const LATEST_TIMESTAMP_DECAY = 0.0012;

const BASE_FLAGS: AblationFlags = {
    explicitYearFilter: true,
    yearPenalty: true,
    intentPrefilter: true,
    intentMatchConflictBoost: true,
    intentSpecializationRules: true,
    degreeRules: true,
    eventRules: true,
    latestTimestampDecay: false,
    latestYearBoost: true,
    nonAdmissionPenaltyMultiplier: 0.12,
    preferredEventBoostMultiplier: 1.24,
    genericEventMatchBoostMultiplier: 1.1,
    genericEventMismatchPenaltyMultiplier: 0.8,
    latestYearBoostBase: 0.98,
};

function defineMode(
    label: string,
    overrides: Partial<AblationFlags>,
): ModeDefinition {
    return {
        label,
        flags: {
            ...BASE_FLAGS,
            ...overrides,
        },
    };
}

const MODE_DEFINITIONS: ModeDefinition[] = [
    defineMode("baseline_current_rules", {}),
    defineMode("no_intent_prefilter", {
        intentPrefilter: false,
    }),
    defineMode("no_intent_match_conflict", {
        intentMatchConflictBoost: false,
    }),
    defineMode("no_intent_specialization", {
        intentSpecializationRules: false,
    }),
    defineMode("no_degree_rules", {
        degreeRules: false,
    }),
    defineMode("no_event_rules", {
        eventRules: false,
    }),
    defineMode("no_generic_event_match_boost", {
        genericEventMatchBoostMultiplier: 1,
    }),
    defineMode("no_generic_event_mismatch_penalty", {
        genericEventMismatchPenaltyMultiplier: 1,
    }),
    defineMode("soften_generic_event_mismatch_080", {
        genericEventMismatchPenaltyMultiplier: 0.8,
    }),
    defineMode("no_preferred_event_boost", {
        preferredEventBoostMultiplier: 1,
    }),
    defineMode("no_non_admission_penalty", {
        nonAdmissionPenaltyMultiplier: 1,
    }),
    defineMode("no_latest_year_boost", {
        latestYearBoost: false,
    }),
    defineMode("soften_latest_year_boost_090", {
        latestYearBoostBase: 0.9,
    }),
    defineMode("soften_latest_year_boost_095", {
        latestYearBoostBase: 0.95,
    }),
    defineMode("soften_latest_year_boost_098", {
        latestYearBoostBase: 0.98,
    }),
    defineMode("soften_latest_year_boost_100", {
        latestYearBoostBase: 1,
    }),
    defineMode("no_year_rules", {
        explicitYearFilter: false,
        yearPenalty: false,
    }),
    defineMode("all_structured_rules_off", {
        intentPrefilter: false,
        intentMatchConflictBoost: false,
        intentSpecializationRules: false,
        degreeRules: false,
        eventRules: false,
        latestYearBoost: false,
        nonAdmissionPenaltyMultiplier: 1,
        preferredEventBoostMultiplier: 1,
        genericEventMatchBoostMultiplier: 1,
        genericEventMismatchPenaltyMultiplier: 1,
        latestYearBoostBase: 1,
    }),
] as const;

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = path.resolve(process.cwd(), "../Backend/models");

let extractor: FeatureExtractionPipeline | null = null;
let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let dimensions = 768;
let bm25Stats: BM25Stats | null = null;

const INTENT_CONFLICTS: Record<string, readonly string[]> = Object.fromEntries(
    INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item.negative_intents]),
);
const INTENT_RULE_MAP = new Map<string, IntentVectorItem>(
    INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item]),
);

function loadDataset(datasetPath: string): DatasetCase[] {
    const absolutePath = path.resolve(process.cwd(), datasetPath);
    const raw = JSON.parse(fs.readFileSync(absolutePath, "utf-8"));
    const dataset = path.basename(datasetPath, ".json");
    return raw.map((item: Omit<DatasetCase, "dataset">) => ({
        ...item,
        dataset,
    }));
}

function dedupe<T>(items: T[]): T[] {
    return Array.from(new Set(items));
}

function fmmTokenize(text: string): string[] {
    const tokens: string[] = [];
    let index = 0;
    while (index < text.length) {
        let matched = false;
        const maxLen = Math.min(10, text.length - index);
        for (let len = maxLen; len > 0; len--) {
            const word = text.substring(index, index + len);
            if (vocabMap.has(word)) {
                tokens.push(word);
                index += len;
                matched = true;
                break;
            }
        }
        if (!matched) index += 1;
    }
    return tokens;
}

function hasAnyOverlap(a: string[], b?: string[]): boolean {
    if (!b || b.length === 0) return false;
    return a.some((item) => b.includes(item));
}

function hasIntentMatch(
    queryIntentIds: string[],
    docIntentIds?: string[],
): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) =>
        docIntentIds.includes(queryIntentId),
    );
}

function hasIntentConflict(
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

function getPreferredEventTypes(intentIds: string[]): string[] {
    return dedupe(
        intentIds.flatMap(
            (intentId) =>
                INTENT_RULE_MAP.get(intentId)?.preferred_event_types || [],
        ),
    );
}

function getRelatedIntentTypes(intentIds: string[]): string[] {
    return dedupe(
        intentIds.flatMap(
            (intentId) => INTENT_RULE_MAP.get(intentId)?.related_intents || [],
        ),
    );
}

function rankOf(
    matches: readonly { otid: string }[],
    expectedOtid: string,
): number {
    const rankIndex = matches.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

async function loadEngine() {
    console.log("Loading metadata, vectors, and model...");
    const metadataPath = path.resolve(process.cwd(), METADATA_FILE);
    const vectorPath = path.resolve(process.cwd(), VECTOR_FILE);
    const metadataPayload = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));

    metadataList = Array.isArray(metadataPayload.data)
        ? metadataPayload.data
        : metadataPayload;

    const vocabList: string[] = metadataPayload.vocab || [];
    vocabMap.clear();
    vocabList.forEach((word, index) => vocabMap.set(word, index));

    const vectorBuffer = fs.readFileSync(vectorPath);
    vectorMatrix = new Int8Array(
        vectorBuffer.buffer,
        vectorBuffer.byteOffset,
        vectorBuffer.byteLength,
    );

    if (metadataList.length > 0 && vectorMatrix.length > 0) {
        dimensions = Math.round(vectorMatrix.length / metadataList.length);
    }

    bm25Stats = buildBM25Stats(metadataList);
    extractor = await pipeline("feature-extraction", MODEL_NAME, {
        dtype: "q8",
        device: "cpu",
    });

    console.log(
        `Loaded ${metadataList.length} vectors, dimensions=${dimensions}`,
    );
}

async function embedQueries(queries: string[]): Promise<Float32Array[]> {
    if (!extractor) throw new Error("Extractor not initialized");

    const vectors: Float32Array[] = [];
    for (
        let start = 0;
        start < queries.length;
        start += QUERY_EMBED_BATCH_SIZE
    ) {
        const batch = queries.slice(start, start + QUERY_EMBED_BATCH_SIZE);
        const output = await extractor(batch, {
            pooling: "mean",
            normalize: true,
            truncation: true,
            max_length: 512,
        } as any);

        const data = output.data as Float32Array;
        for (let index = 0; index < batch.length; index++) {
            const begin = index * dimensions;
            const end = begin + dimensions;
            vectors.push(new Float32Array(data.slice(begin, end)));
        }

        const done = Math.min(start + batch.length, queries.length);
        console.log(`Embedded ${done} / ${queries.length} queries`);
    }

    return vectors;
}

async function buildQueryCache(
    testCases: DatasetCase[],
): Promise<QueryCacheItem[]> {
    console.log(`Precomputing ${testCases.length} queries...`);
    const queryVectors = await embedQueries(
        testCases.map((item) => item.query),
    );
    return testCases.map((testCase, index) => {
        const queryIntent = parseQueryIntent(testCase.query);
        const queryWords = dedupe([
            ...fmmTokenize(testCase.query),
            ...queryIntent.normalizedTerms,
        ]);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        return {
            testCase,
            queryVector: queryVectors[index],
            queryIntent,
            queryWords,
            queryYearWordIds,
        };
    });
}

function precomputeQueryState(item: QueryCacheItem): QueryState {
    if (!vectorMatrix || !bm25Stats) {
        throw new Error("Engine not initialized");
    }

    const querySparse = getQuerySparse(item.queryWords, vocabMap);
    const n = metadataList.length;
    const denseScores = new Float32Array(n);
    const sparseScores = new Float32Array(n);
    const denseOrder = new Int32Array(n);
    const sparseOrder = new Int32Array(n);
    const lexicalBonusMap = new Map<string, number>();
    const yearHitMap = new Map<string, boolean>();

    for (let i = 0; i < n; i++) {
        const meta = metadataList[i];

        let dense = dotProduct(
            item.queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null) {
            dense *= meta.scale;
        }
        denseScores[i] = dense;
        denseOrder[i] = i;

        let sparse = 0;
        if (meta.sparse && meta.sparse.length > 0) {
            const dl = bm25Stats.docLengths[i];
            const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);

            for (let j = 0; j < meta.sparse.length; j += 2) {
                const wordId = meta.sparse[j];
                const tf = meta.sparse[j + 1];

                if (item.queryYearWordIds.includes(wordId)) {
                    const otid =
                        meta.type === "OT" ? meta.id : meta.parent_otid;
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
                const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
                let currentBonus = lexicalBonusMap.get(otid) || 0;
                if (meta.type === "Q") currentBonus += sparse * 1.5;
                else if (meta.type === "KP") currentBonus += sparse * 1.2;
                else currentBonus += sparse;
                lexicalBonusMap.set(otid, currentBonus);
            }
        }
        sparseScores[i] = sparse;
        sparseOrder[i] = i;
    }

    denseOrder.sort((a, b) => denseScores[b] - denseScores[a]);
    const rrfScores = new Map<Metadata, number>();

    for (let rank = 0; rank < Math.min(4000, n); rank++) {
        const meta = metadataList[denseOrder[rank]];
        rrfScores.set(meta, (1 / (rank + RRF_K)) * 100);
    }

    sparseOrder.sort((a, b) => sparseScores[b] - sparseScores[a]);
    for (let rank = 0; rank < Math.min(4000, n); rank++) {
        const index = sparseOrder[rank];
        if (sparseScores[index] === 0) break;
        const meta = metadataList[index];
        const current = rrfScores.get(meta) || 0;
        rrfScores.set(meta, current + (1.2 / (rank + RRF_K)) * 100);
    }

    const topHybrid = Array.from(rrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 1000);

    const otidMap: Record<string, AggregatedDocScores> = {};
    for (const [meta, score] of topHybrid) {
        const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
        const topicIds = resolveMetadataTopicIds(meta);
        if (!otidMap[otid]) {
            otidMap[otid] = createAggregatedDocScores(meta, topicIds);
        }

        mergeAggregatedDocMetadata(otidMap[otid], meta, topicIds);
        applyScoreToAggregatedDocScores(otidMap[otid], meta, score);
    }

    const candidateTargetYears = Object.values(otidMap)
        .map((scores) => scores.target_year)
        .filter((year): year is number => typeof year === "number");
    const latestTargetYear =
        candidateTargetYears.length > 0
            ? Math.max(...candidateTargetYears)
            : undefined;

    return {
        testCase: item.testCase,
        queryIntent: item.queryIntent,
        queryYearWordIds: item.queryYearWordIds,
        otidMap,
        lexicalBonusMap,
        yearHitMap,
        latestTargetYear,
    };
}

function rankQueryState(
    state: QueryState,
    flags: AblationFlags,
): { otid: string }[] {
    const finalRanking: { otid: string; score: number }[] = [];

    for (const [otid, scores] of Object.entries(state.otidMap)) {
        const explicitYears = state.queryIntent.years || [];
        const hasExplicitYear = explicitYears.length > 0;
        const hasStructuredYearMatch =
            hasExplicitYear &&
            scores.target_year !== undefined &&
            explicitYears.includes(scores.target_year);
        const hasLexicalYearMatch = state.yearHitMap.get(otid) === true;
        const isHighConfidenceSingleIntent =
            state.queryIntent.confidence >= 1 &&
            state.queryIntent.intentIds.length === 1;
        const primaryIntentId = isHighConfidenceSingleIntent
            ? state.queryIntent.intentIds[0]
            : undefined;
        const allowedIntentIds = primaryIntentId
            ? dedupe([
                  primaryIntentId,
                  ...getRelatedIntentTypes([primaryIntentId]),
              ])
            : [];
        const preferredEventTypes = getPreferredEventTypes(
            state.queryIntent.intentIds,
        );

        if (flags.explicitYearFilter && hasExplicitYear) {
            if (scores.target_year !== undefined && !hasStructuredYearMatch) {
                continue;
            }
            if (scores.target_year === undefined && !hasLexicalYearMatch) {
                continue;
            }
        }

        if (
            flags.intentPrefilter &&
            isHighConfidenceSingleIntent &&
            primaryIntentId
        ) {
            const docIntentIds = scores.intent_ids || [];
            const docHasIntentLabels = docIntentIds.length > 0;
            const docMatchesAllowedIntent = allowedIntentIds.some((intentId) =>
                docIntentIds.includes(intentId),
            );
            const docHasPreferredEventType =
                preferredEventTypes.length > 0 &&
                hasAnyOverlap(preferredEventTypes, scores.event_types);

            if (docHasIntentLabels && !docMatchesAllowedIntent) {
                continue;
            }

            if (!docHasIntentLabels && !docHasPreferredEventType) {
                continue;
            }
        }

        const weightedQ = scores.max_q * DEFAULT_WEIGHTS.Q;
        const weightedKP = scores.max_kp * DEFAULT_WEIGHTS.KP;
        const weightedOT = scores.ot_score * DEFAULT_WEIGHTS.OT;
        const maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
        const unionBonus =
            weightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;

        let finalScore = maxComponent + unionBonus;

        if (
            flags.latestTimestampDecay &&
            state.queryIntent.preferLatest &&
            scores.timestamp
        ) {
            const now = Date.now() / 1000;
            const daysDiff = (now - scores.timestamp) / SECONDS_IN_DAY;
            if (daysDiff > 0) {
                finalScore *= Math.exp(-LATEST_TIMESTAMP_DECAY * daysDiff);
            }
        }

        let boost = 1.0;
        const lexicalBonus = state.lexicalBonusMap.get(otid) || 0;
        if (lexicalBonus > 0) {
            boost *= 1 + Math.log1p(lexicalBonus) / 4;
        }

        if (flags.yearPenalty && state.queryYearWordIds.length > 0) {
            if (
                scores.target_year !== undefined &&
                state.queryIntent.years.length > 0
            ) {
                if (!state.queryIntent.years.includes(scores.target_year)) {
                    boost *= 0.01;
                }
            } else if (!hasStructuredYearMatch && !hasLexicalYearMatch) {
                boost *= 0.12;
            }
        }

        if (state.queryIntent.intentIds.length > 0) {
            if (flags.intentMatchConflictBoost) {
                if (
                    hasIntentMatch(
                        state.queryIntent.intentIds,
                        scores.intent_ids,
                    )
                ) {
                    boost *= 1.25;
                } else if (
                    hasIntentConflict(
                        state.queryIntent.intentIds,
                        scores.intent_ids,
                    )
                ) {
                    boost *= 0.18;
                }
            }

            if (flags.intentSpecializationRules) {
                if (
                    state.queryIntent.intentIds.includes(
                        "master_recommend_exemption",
                    ) &&
                    scores.intent_ids?.includes("ug_recommend_admission")
                ) {
                    boost *= 0.05;
                }

                if (
                    isHighConfidenceSingleIntent &&
                    state.queryIntent.intentIds[0] ===
                        "master_recommend_exemption"
                ) {
                    if (scores.intent_ids?.includes("master_unified_exam")) {
                        boost *= 0.06;
                    }
                    if (scores.intent_ids?.includes("master_adjustment")) {
                        boost *= 0.08;
                    }
                    if (
                        !state.queryIntent.degreeLevels.includes("博士") &&
                        (scores.intent_ids?.includes("phd_apply_assessment") ||
                            scores.intent_ids?.includes("phd_general_exam"))
                    ) {
                        boost *= 0.04;
                    }
                }

                if (scores.event_types?.includes("非招生通知")) {
                    boost *= flags.nonAdmissionPenaltyMultiplier;
                } else if (
                    preferredEventTypes.length > 0 &&
                    hasAnyOverlap(preferredEventTypes, scores.event_types)
                ) {
                    boost *= flags.preferredEventBoostMultiplier;
                }
            }
        }

        if (flags.degreeRules && state.queryIntent.degreeLevels.length > 0) {
            if (
                hasAnyOverlap(
                    state.queryIntent.degreeLevels,
                    scores.degree_levels,
                )
            ) {
                boost *= 1.1;
            } else if ((scores.degree_levels?.length || 0) > 0) {
                boost *= 0.45;
            }
        }

        if (flags.eventRules && state.queryIntent.eventTypes.length > 0) {
            if (
                hasAnyOverlap(state.queryIntent.eventTypes, scores.event_types)
            ) {
                boost *= flags.genericEventMatchBoostMultiplier;
            } else if ((scores.event_types?.length || 0) > 0) {
                boost *= flags.genericEventMismatchPenaltyMultiplier;
            }
        }

        if (
            flags.latestYearBoost &&
            state.queryIntent.preferLatest &&
            state.latestTargetYear !== undefined &&
            scores.target_year !== undefined
        ) {
            const yearGap = Math.max(
                0,
                state.latestTargetYear - scores.target_year,
            );
            boost *= Math.pow(flags.latestYearBoostBase, yearGap);
        }

        finalRanking.push({
            otid,
            score: finalScore * boost,
        });
    }

    return finalRanking.sort((a, b) => b.score - a.score).slice(0, 100);
}

function buildMetrics(
    mode: ModeDefinition,
    queryStates: readonly QueryState[],
): ModeResult {
    const metricsByDataset: Record<string, Metrics> = {};
    const seeds: Record<
        string,
        {
            total: number;
            hitAt1: number;
            hitAt3: number;
            hitAt5: number;
            reciprocalRankSum: number;
        }
    > = {};
    const combinedSeed = {
        total: 0,
        hitAt1: 0,
        hitAt3: 0,
        hitAt5: 0,
        reciprocalRankSum: 0,
    };

    queryStates.forEach((state) => {
        const matches = rankQueryState(state, mode.flags);
        const rank = rankOf(matches, state.testCase.expected_otid);
        const seed =
            seeds[state.testCase.dataset] ||
            (seeds[state.testCase.dataset] = {
                total: 0,
                hitAt1: 0,
                hitAt3: 0,
                hitAt5: 0,
                reciprocalRankSum: 0,
            });

        const targets = [seed, combinedSeed];
        targets.forEach((target) => {
            target.total += 1;
            if (rank === 1) target.hitAt1 += 1;
            if (rank <= 3) target.hitAt3 += 1;
            if (rank <= 5) target.hitAt5 += 1;
            if (Number.isFinite(rank)) {
                target.reciprocalRankSum += 1 / rank;
            }
        });
    });

    Object.entries(seeds).forEach(([dataset, seed]) => {
        metricsByDataset[dataset] = {
            total: seed.total,
            hitAt1: (seed.hitAt1 / seed.total) * 100,
            hitAt3: (seed.hitAt3 / seed.total) * 100,
            hitAt5: (seed.hitAt5 / seed.total) * 100,
            mrr: seed.reciprocalRankSum / seed.total,
        };
    });

    return {
        label: mode.label,
        metricsByDataset,
        combined: {
            total: combinedSeed.total,
            hitAt1: (combinedSeed.hitAt1 / combinedSeed.total) * 100,
            hitAt3: (combinedSeed.hitAt3 / combinedSeed.total) * 100,
            hitAt5: (combinedSeed.hitAt5 / combinedSeed.total) * 100,
            mrr: combinedSeed.reciprocalRankSum / combinedSeed.total,
        },
    };
}

function printSummary(result: ModeResult) {
    console.log(
        `${result.label}: Hit@1=${result.combined.hitAt1.toFixed(2)}% | Hit@3=${result.combined.hitAt3.toFixed(2)}% | Hit@5=${result.combined.hitAt5.toFixed(2)}% | MRR=${result.combined.mrr.toFixed(4)}`,
    );
}

async function main() {
    const testCases = DATASETS.flatMap(loadDataset);
    const datasetSizes = testCases.reduce<Record<string, number>>(
        (acc, item) => {
            acc[item.dataset] = (acc[item.dataset] || 0) + 1;
            return acc;
        },
        {},
    );

    await loadEngine();
    const queryCache = await buildQueryCache(testCases);

    console.log("Precomputing shared query states...");
    const queryStates = queryCache.map((item, index) => {
        const state = precomputeQueryState(item);
        if ((index + 1) % 50 === 0) {
            console.log(`Prepared ${index + 1} / ${queryCache.length} queries`);
        }
        return state;
    });

    const modeResults = MODE_DEFINITIONS.map((mode) => {
        const result = buildMetrics(mode, queryStates);
        printSummary(result);
        return result;
    });

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetSizes,
        modes: modeResults,
    };

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
    }

    const outputPath = path.resolve(
        resultsDir,
        `structured_ablation_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`Report saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

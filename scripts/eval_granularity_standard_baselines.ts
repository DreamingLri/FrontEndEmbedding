import * as fs from "fs";
import * as path from "path";
import { performance } from "perf_hooks";

import {
    RRF_K,
    buildBM25Stats,
    dotProduct,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type BM25Stats,
    type Metadata,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import {
    FRONTEND_MODEL_NAME,
    loadDataset,
    resolveGranularityDatasetTarget,
    type EvalDatasetCase,
    type GranularityDatasetTargetKey,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET } from "../src/worker/search_pipeline.ts";
import {
    buildQueryPlan,
    inferDocumentRolesFromTitle,
    type QueryPlan,
} from "../src/worker/query_planner.ts";
import { resolveBackendArticlesFile } from "./kb_version_paths.ts";
import {
    buildStandardBaselinesResultFileName,
    resolveNamedDatasetProfile,
} from "./result_naming.ts";

type DatasetCase = EvalDatasetCase & {
    id?: string;
};

type DatasetTargetKey = GranularityDatasetTargetKey;

type DatasetTarget = {
    key: DatasetTargetKey;
    label: string;
    datasetFile: string;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryIntent: ReturnType<typeof parseQueryIntent>;
    queryPlan: QueryPlan;
    queryYearWordIds: number[];
    queryPhaseAnchor: PhaseAnchor;
};

type RankedDoc = {
    otid: string;
    score: number;
    best_kpid?: string;
};

type PerCaseResult = {
    id?: string;
    query: string;
    expected_otid: string;
    elapsedMs: number;
    rank: number | null;
    hitAt1: boolean;
    reciprocalRank: number;
    top1Otid?: string;
    topMatches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
};

type MetricSummary = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type MetricInterval = {
    lower: number;
    upper: number;
};

type ModelFamily =
    | "sparse"
    | "dense"
    | "structured";

type ModelReport = {
    label: string;
    family: ModelFamily;
    description: string;
    metrics: MetricSummary;
    timing: {
        avgMs: number;
        p50Ms: number;
        p95Ms: number;
        maxMs: number;
        totalMs: number;
    };
    hitAt1Interval95: MetricInterval;
    mrrInterval95: MetricInterval;
    perCase: PerCaseResult[];
};

type PairwiseMetricReport = {
    observedDelta: number;
    interval95: MetricInterval;
    randomizationPValue: number;
};

type PairwiseComparison = {
    challenger: string;
    baseline: string;
    hitAt1: PairwiseMetricReport;
    mrr: PairwiseMetricReport;
};

type DatasetReport = {
    datasetKey: DatasetTargetKey;
    datasetLabel: string;
    datasetAlias?: string;
    datasetDisplayName?: string;
    datasetFile: string;
    caseCount: number;
    bestStandardBaseline: string;
    models: ModelReport[];
    pairwiseComparisons: PairwiseComparison[];
};

type Report = {
    generatedAt: string;
    embeddingModel: string;
    bootstrapIterations: number;
    randomizationIterations: number;
    timingNote: string;
    runtimePresetRetrieval: {
        qConfusionMode: string;
        qConfusionWeight: number;
    };
    datasets: DatasetReport[];
};

type CorpusView = {
    otMetadata: Metadata[];
    kpMetadata: Metadata[];
    otBm25Stats: BM25Stats;
    kpBm25Stats: BM25Stats;
    otIndices: number[];
    kpIndices: number[];
};

type ScoredMeta = {
    meta: Metadata;
    score: number;
};

type PhaseLabel =
    | "half_1"
    | "half_2"
    | "batch_1"
    | "batch_2"
    | "batch_3"
    | "batch_4"
    | "pre_apply"
    | "apply_notice"
    | "rule_plan"
    | "brochure"
    | "assessment"
    | "retest"
    | "adjustment";

type PhaseAnchor = {
    half?: Extract<PhaseLabel, "half_1" | "half_2">;
    batch?: Extract<PhaseLabel, "batch_1" | "batch_2" | "batch_3" | "batch_4">;
    stages: PhaseLabel[];
};

type NormalizedOtidEvalTarget = {
    mode: NonNullable<DatasetCase["otid_eval_mode"]> | "single_expected";
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

const BM25_K1 = 1.2;
const BM25_B = 0.4;
const CURRENT_TIMESTAMP = 0;
const PHASE_ANCHOR_DOC_WEIGHT = 0.35;
const TOP_MATCH_COUNT = Number.parseInt(
    process.env.SUASK_STANDARD_BASELINE_TOP_MATCHES || "",
    10,
);
const BOOTSTRAP_ITERATIONS = Number.parseInt(
    process.env.SUASK_BOOTSTRAP_ITERATIONS || "",
    10,
);
const RANDOMIZATION_ITERATIONS = Number.parseInt(
    process.env.SUASK_RANDOMIZATION_ITERATIONS || "",
    10,
);
const RNG_SEED = Number.parseInt(process.env.SUASK_RANDOM_SEED || "", 10);
const INCLUDE_QUERY_PLANNER_MODEL =
    process.env.SUASK_INCLUDE_QUERY_PLANNER_MODEL === "1";
const INCLUDE_STRUCTURE_RISK_MODEL =
    process.env.SUASK_INCLUDE_STRUCTURE_RISK_MODEL === "1";

const SAFE_TOP_MATCH_COUNT =
    Number.isFinite(TOP_MATCH_COUNT) && TOP_MATCH_COUNT > 0 ? TOP_MATCH_COUNT : 5;
const SAFE_BOOTSTRAP_ITERATIONS =
    Number.isFinite(BOOTSTRAP_ITERATIONS) && BOOTSTRAP_ITERATIONS >= 1000
        ? BOOTSTRAP_ITERATIONS
        : 10000;
const SAFE_RANDOMIZATION_ITERATIONS =
    Number.isFinite(RANDOMIZATION_ITERATIONS) && RANDOMIZATION_ITERATIONS >= 1000
        ? RANDOMIZATION_ITERATIONS
        : 10000;
const SAFE_RANDOM_SEED =
    Number.isFinite(RNG_SEED) && RNG_SEED > 0 ? RNG_SEED : 20260331;

function resolveDatasetTarget(key: DatasetTargetKey): DatasetTarget | null {
    try {
        const target = resolveGranularityDatasetTarget(key);
        return {
            key,
            label: target.label,
            datasetFile: target.datasetFile,
        };
    } catch {
        return null;
    }
}

const AVAILABLE_DATASET_TARGETS = (
    [
        resolveDatasetTarget("main_bench_120"),
        resolveDatasetTarget("in_domain_holdout_50"),
        resolveDatasetTarget("external_ood_50"),
        resolveDatasetTarget("external_ood_holdout_30"),
        resolveDatasetTarget("external_ood_hard_30"),
        resolveDatasetTarget("hard_ood_v2_diag_top30"),
    ] as Array<DatasetTarget | null>
).filter((item): item is DatasetTarget => Boolean(item));

const OPTIONAL_DATASET_TARGETS = (
    [
        resolveDatasetTarget("structure_dev_40"),
        resolveDatasetTarget("ladder_main_balanced_80"),
        resolveDatasetTarget("ladder_generalization_hard_60"),
        resolveDatasetTarget("ladder_structure_stress_40"),
        resolveDatasetTarget("ladder_main_balanced_120"),
        resolveDatasetTarget("ladder_generalization_hard_80"),
        resolveDatasetTarget("ladder_structure_stress_60"),
        resolveDatasetTarget("ladder_main_balanced_150"),
        resolveDatasetTarget("ladder_generalization_hard_100"),
        resolveDatasetTarget("ladder_structure_stress_80"),
    ] as Array<DatasetTarget | null>
).filter((item): item is DatasetTarget => Boolean(item));

const DATASET_TARGETS = Object.fromEntries(
    [...AVAILABLE_DATASET_TARGETS, ...OPTIONAL_DATASET_TARGETS].map((item) => [
        item.key,
        item,
    ]),
) as Partial<Record<DatasetTargetKey, DatasetTarget>>;

const DATASET_TARGET_ORDER = parseDatasetTargets(
    process.env.SUASK_STANDARD_BASELINE_DATASETS ||
        AVAILABLE_DATASET_TARGETS.map((item) => item.key).join(","),
);

const STRUCTURED_KP_OT_WEIGHTS = {
    Q: 0,
    KP: 0.28571428571428575,
    OT: 0.7142857142857143,
};

const STRUCTURED_Q_KP_OT_WEIGHTS = {
    ...FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.weights,
};

const STRUCTURED_FULL_DOC_AWARE_WEIGHTS = {
    ...STRUCTURED_KP_OT_WEIGHTS,
};
const ARTICLE_TEXTS_FILE = resolveBackendArticlesFile();

let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let dimensions = 768;
let corpusView: CorpusView | null = null;
let allMetadataBm25Stats: BM25Stats | null = null;
let topicPartitionIndex: Awaited<
    ReturnType<typeof loadFrontendEvalEngine>
>["topicPartitionIndex"] | null = null;
let extractor: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["extractor"] | null =
    null;
let articlePhaseAnchorMap = new Map<string, PhaseAnchor>();
let articleTitleMap = new Map<string, string>();

function parseDatasetTargets(raw: string): DatasetTarget[] {
    const requested = raw
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean) as DatasetTargetKey[];

    const resolved = requested
        .map((key) => DATASET_TARGETS[key])
        .filter((item): item is DatasetTarget => Boolean(item));

    return resolved.length > 0 ? resolved : AVAILABLE_DATASET_TARGETS;
}

function hashString(input: string): number {
    let hash = 2166136261;
    for (let index = 0; index < input.length; index++) {
        hash ^= input.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function createSeededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state += 0x6d2b79f5;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function percentile(values: number[], q: number): number {
    if (values.length === 0) {
        return 0;
    }
    if (values.length === 1) {
        return values[0];
    }

    const position = (values.length - 1) * q;
    const lowerIndex = Math.floor(position);
    const upperIndex = Math.ceil(position);
    if (lowerIndex === upperIndex) {
        return values[lowerIndex];
    }
    const weight = position - lowerIndex;
    return (
        values[lowerIndex] * (1 - weight) + values[upperIndex] * weight
    );
}

function sortNumeric(values: number[]): number[] {
    return values.sort((a, b) => a - b);
}

function buildMetricSummary(perCase: readonly PerCaseResult[]): MetricSummary {
    const total = perCase.length || 1;
    const hitAt1Count = perCase.filter((item) => item.hitAt1).length;
    const hitAt3Count = perCase.filter(
        (item) => item.rank !== null && item.rank <= 3,
    ).length;
    const hitAt5Count = perCase.filter(
        (item) => item.rank !== null && item.rank <= 5,
    ).length;
    const reciprocalRankSum = perCase.reduce(
        (acc, item) => acc + item.reciprocalRank,
        0,
    );

    return {
        total: perCase.length,
        hitAt1: (hitAt1Count / total) * 100,
        hitAt3: (hitAt3Count / total) * 100,
        hitAt5: (hitAt5Count / total) * 100,
        mrr: reciprocalRankSum / total,
    };
}

function buildTimingSummary(perCase: readonly PerCaseResult[]) {
    const elapsedValues = perCase.map((item) => item.elapsedMs);
    const sortedValues = sortNumeric([...elapsedValues]);
    const totalMs = elapsedValues.reduce((sum, value) => sum + value, 0);
    return {
        avgMs: perCase.length > 0 ? totalMs / perCase.length : 0,
        p50Ms: percentile(sortedValues, 0.5),
        p95Ms: percentile(sortedValues, 0.95),
        maxMs: sortedValues.length > 0 ? sortedValues[sortedValues.length - 1] : 0,
        totalMs,
    };
}

function bootstrapMetricInterval(
    values: readonly number[],
    iterations: number,
    seedKey: string,
    scale = 1,
): MetricInterval {
    if (values.length === 0) {
        return { lower: 0, upper: 0 };
    }

    const random = createSeededRandom(SAFE_RANDOM_SEED ^ hashString(seedKey));
    const samples: number[] = [];

    for (let iteration = 0; iteration < iterations; iteration++) {
        let sum = 0;
        for (let index = 0; index < values.length; index++) {
            const pick = Math.floor(random() * values.length);
            sum += values[pick];
        }
        samples.push((sum / values.length) * scale);
    }

    const sorted = sortNumeric(samples);
    return {
        lower: percentile(sorted, 0.025),
        upper: percentile(sorted, 0.975),
    };
}

function pairedBootstrapDelta(
    challengerValues: readonly number[],
    baselineValues: readonly number[],
    iterations: number,
    seedKey: string,
    scale = 1,
): PairwiseMetricReport {
    if (challengerValues.length !== baselineValues.length) {
        throw new Error("Paired bootstrap requires equal-length arrays.");
    }

    const observedDelta =
        ((mean(challengerValues) - mean(baselineValues)) * scale) || 0;
    const random = createSeededRandom(SAFE_RANDOM_SEED ^ hashString(seedKey));
    const samples: number[] = [];
    const n = challengerValues.length;

    for (let iteration = 0; iteration < iterations; iteration++) {
        let challengerSum = 0;
        let baselineSum = 0;
        for (let index = 0; index < n; index++) {
            const pick = Math.floor(random() * n);
            challengerSum += challengerValues[pick];
            baselineSum += baselineValues[pick];
        }
        samples.push(((challengerSum - baselineSum) / n) * scale);
    }

    const sorted = sortNumeric(samples);
    return {
        observedDelta,
        interval95: {
            lower: percentile(sorted, 0.025),
            upper: percentile(sorted, 0.975),
        },
        randomizationPValue: approximateRandomizationPValue(
            challengerValues,
            baselineValues,
            observedDelta / scale,
            iterations,
            `${seedKey}:randomization`,
        ),
    };
}

function approximateRandomizationPValue(
    challengerValues: readonly number[],
    baselineValues: readonly number[],
    observedDeltaUnscaled: number,
    iterations: number,
    seedKey: string,
): number {
    if (challengerValues.length !== baselineValues.length) {
        throw new Error("Approximate randomization requires equal-length arrays.");
    }

    const random = createSeededRandom(SAFE_RANDOM_SEED ^ hashString(seedKey));
    const n = challengerValues.length;
    let exceedCount = 0;

    for (let iteration = 0; iteration < iterations; iteration++) {
        let challengerSum = 0;
        let baselineSum = 0;
        for (let index = 0; index < n; index++) {
            if (random() < 0.5) {
                challengerSum += challengerValues[index];
                baselineSum += baselineValues[index];
            } else {
                challengerSum += baselineValues[index];
                baselineSum += challengerValues[index];
            }
        }

        const shuffledDelta = challengerSum / n - baselineSum / n;
        if (Math.abs(shuffledDelta) >= Math.abs(observedDeltaUnscaled)) {
            exceedCount += 1;
        }
    }

    return (exceedCount + 1) / (iterations + 1);
}

function mean(values: readonly number[]): number {
    if (values.length === 0) {
        return 0;
    }
    return values.reduce((acc, item) => acc + item, 0) / values.length;
}

function compareMetrics(a: MetricSummary, b: MetricSummary): number {
    if (a.hitAt1 !== b.hitAt1) {
        return b.hitAt1 - a.hitAt1;
    }
    if (a.mrr !== b.mrr) {
        return b.mrr - a.mrr;
    }
    if (a.hitAt5 !== b.hitAt5) {
        return b.hitAt5 - a.hitAt5;
    }
    return b.hitAt3 - a.hitAt3;
}

function hasRegex(text: string, pattern: RegExp): boolean {
    pattern.lastIndex = 0;
    return pattern.test(text);
}

function dedupePhaseStages(stages: PhaseLabel[]): PhaseLabel[] {
    return Array.from(new Set(stages));
}

function extractPhaseAnchor(text: string): PhaseAnchor {
    const normalized = text.replace(/\s+/g, "");
    let half: PhaseAnchor["half"];
    let batch: PhaseAnchor["batch"];
    const stages: PhaseLabel[] = [];

    if (hasRegex(normalized, /上半年/)) {
        half = "half_1";
    } else if (hasRegex(normalized, /下半年/)) {
        half = "half_2";
    }

    if (hasRegex(normalized, /第(?:一|1)批/)) {
        batch = "batch_1";
    } else if (hasRegex(normalized, /第(?:二|2)批/)) {
        batch = "batch_2";
    } else if (hasRegex(normalized, /第(?:三|3)批/)) {
        batch = "batch_3";
    } else if (hasRegex(normalized, /第(?:四|4)批/)) {
        batch = "batch_4";
    }

    if (hasRegex(normalized, /预报名/)) {
        stages.push("pre_apply");
    } else if (
        hasRegex(normalized, /(报名通知|报名公告|申请通知|申请公告|报名方式|申请方式)/)
    ) {
        stages.push("apply_notice");
    }

    if (hasRegex(normalized, /(工作方案|接收办法|实施办法|录取方案)/)) {
        stages.push("rule_plan");
    }

    if (hasRegex(normalized, /(招生简章|简章|招生章程|章程)/)) {
        stages.push("brochure");
    }

    if (hasRegex(normalized, /综合考核/)) {
        stages.push("assessment");
    }

    if (hasRegex(normalized, /复试/)) {
        stages.push("retest");
    }

    if (hasRegex(normalized, /调剂/)) {
        stages.push("adjustment");
    }

    return {
        half,
        batch,
        stages: dedupePhaseStages(stages),
    };
}

function hasExplicitPhaseAnchor(anchor: PhaseAnchor): boolean {
    return Boolean(anchor.half || anchor.batch || anchor.stages.length > 0);
}

function loadArticlePhaseAnchors(): Map<string, PhaseAnchor> {
    const absolutePath = path.resolve(process.cwd(), ARTICLE_TEXTS_FILE);
    if (!fs.existsSync(absolutePath)) {
        return new Map<string, PhaseAnchor>();
    }

    const raw = JSON.parse(
        fs.readFileSync(absolutePath, "utf-8"),
    ) as Array<{
        otid?: string;
        ot_title?: string;
    }>;

    const result = new Map<string, PhaseAnchor>();
    raw.forEach((item) => {
        if (!item.otid) {
            return;
        }
        result.set(item.otid, extractPhaseAnchor(item.ot_title || ""));
    });
    return result;
}

function loadArticleTitles(): Map<string, string> {
    const absolutePath = path.resolve(process.cwd(), ARTICLE_TEXTS_FILE);
    if (!fs.existsSync(absolutePath)) {
        return new Map<string, string>();
    }

    const raw = JSON.parse(
        fs.readFileSync(absolutePath, "utf-8"),
    ) as Array<{
        otid?: string;
        ot_title?: string;
    }>;

    const result = new Map<string, string>();
    raw.forEach((item) => {
        if (!item.otid) {
            return;
        }
        result.set(item.otid, item.ot_title || "");
    });
    return result;
}

function computePhaseAnchorDocDelta(
    queryPhase: PhaseAnchor,
    articlePhase?: PhaseAnchor,
): number {
    if (!hasExplicitPhaseAnchor(queryPhase)) {
        return 0;
    }

    if (!articlePhase) {
        return -0.15;
    }

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

function parseRequiredOtidGroups(groups?: string[][]): string[][] {
    if (!Array.isArray(groups)) {
        return [];
    }

    return groups
        .map((group) =>
            Array.isArray(group)
                ? Array.from(
                      new Set(
                          group.filter(
                              (item): item is string =>
                                  typeof item === "string" && item.length > 0,
                          ),
                      ),
                  )
                : [],
        )
        .filter((group) => group.length > 0);
}

function resolveOtidEvalTarget(testCase: DatasetCase): NormalizedOtidEvalTarget {
    const explicitRequiredGroups = parseRequiredOtidGroups(
        testCase.required_otid_groups,
    );
    const acceptableOtids = Array.from(
        new Set(
            [
                testCase.expected_otid,
                ...(Array.isArray(testCase.acceptable_otids)
                    ? testCase.acceptable_otids
                    : []),
            ].filter((item): item is string => typeof item === "string" && item.length > 0),
        ),
    );

    const inferredMode =
        testCase.otid_eval_mode ||
        (explicitRequiredGroups.length > 0
            ? "required_otid_groups"
            : acceptableOtids.length > 1
              ? "acceptable_otids"
              : "single_expected");

    const minGroupsCandidate = Number.isFinite(testCase.min_otid_groups_to_cover)
        ? Math.max(1, Number(testCase.min_otid_groups_to_cover))
        : explicitRequiredGroups.length > 0
          ? explicitRequiredGroups.length
          : acceptableOtids.length > 0
            ? 1
            : 0;

    return {
        mode: inferredMode,
        acceptableOtids,
        requiredOtidGroups:
            inferredMode === "required_otid_groups" ? explicitRequiredGroups : [],
        minGroupsToCover:
            inferredMode === "required_otid_groups"
                ? Math.min(
                      minGroupsCandidate,
                      Math.max(explicitRequiredGroups.length, 1),
                  )
                : minGroupsCandidate,
    };
}

function getBestRankForOtidSet(
    ranking: readonly RankedDoc[],
    acceptableOtids: readonly string[],
): number {
    if (acceptableOtids.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const acceptableOtidSet = new Set(acceptableOtids);
    const rankIndex = ranking.findIndex((item) => acceptableOtidSet.has(item.otid));
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function getCoverageDepthForRequiredOtidGroups(
    ranking: readonly RankedDoc[],
    requiredGroups: readonly string[][],
    minGroupsToCover: number,
): number {
    if (requiredGroups.length === 0 || minGroupsToCover <= 0) {
        return Number.POSITIVE_INFINITY;
    }

    const covered = new Set<number>();
    for (let index = 0; index < ranking.length; index++) {
        const otid = ranking[index]?.otid;
        if (!otid) {
            continue;
        }

        requiredGroups.forEach((group, groupIndex) => {
            if (!covered.has(groupIndex) && group.includes(otid)) {
                covered.add(groupIndex);
            }
        });

        if (covered.size >= minGroupsToCover) {
            return index + 1;
        }
    }

    return Number.POSITIVE_INFINITY;
}

function rerankStructuredMatchesForDocumentMetrics(
    query: QueryCacheItem,
    ranking: readonly RankedDoc[],
): RankedDoc[] {
    if (!FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.enablePhaseAnchorBoost) {
        return [...ranking];
    }
    if (!hasExplicitPhaseAnchor(query.queryPhaseAnchor)) {
        return [...ranking];
    }

    return ranking
        .map((item) => ({
            ...item,
            score:
                item.score +
                computePhaseAnchorDocDelta(
                    query.queryPhaseAnchor,
                    articlePhaseAnchorMap.get(item.otid),
                ) *
                    PHASE_ANCHOR_DOC_WEIGHT,
        }))
        .sort((a, b) => b.score - a.score);
}

function resolveCorpusView(metadata: readonly Metadata[]): CorpusView {
    const otMetadata = metadata.filter((item) => item.type === "OT");
    const kpMetadata = metadata.filter((item) => item.type === "KP");
    const otIndices: number[] = [];
    const kpIndices: number[] = [];

    metadata.forEach((item, index) => {
        if (item.type === "OT") {
            otIndices.push(index);
        }
        if (item.type === "KP") {
            kpIndices.push(index);
        }
    });

    return {
        otMetadata,
        kpMetadata,
        otBm25Stats: buildBM25Stats(otMetadata),
        kpBm25Stats: buildBM25Stats(kpMetadata),
        otIndices,
        kpIndices,
    };
}

function dotVectorPair(queryVector: Float32Array, candidateVector: Float32Array): number {
    let score = 0;
    for (let index = 0; index < dimensions; index++) {
        score += queryVector[index] * candidateVector[index];
    }
    return score;
}

async function loadEngine() {
    const engine = await loadFrontendEvalEngine();
    vocabMap = engine.vocabMap;
    metadataList = engine.metadataList;
    vectorMatrix = engine.vectorMatrix;
    dimensions = engine.dimensions;
    topicPartitionIndex = engine.topicPartitionIndex;
    extractor = engine.extractor;
    corpusView = resolveCorpusView(metadataList);
    allMetadataBm25Stats = engine.bm25Stats;
    articlePhaseAnchorMap = loadArticlePhaseAnchors();
    articleTitleMap = loadArticleTitles();
}

async function buildQueryCache(testCases: DatasetCase[]): Promise<QueryCacheItem[]> {
    if (!extractor) {
        throw new Error("Extractor not initialized.");
    }

    const queryVectors = await embedFrontendQueries(
        extractor,
        testCases.map((item) => item.query),
        dimensions,
    );

    return testCases.map((testCase, index) => {
        const referenceQueryText = testCase.source_query?.trim() || testCase.query;
        const queryIntent = parseQueryIntent(referenceQueryText);
        const queryPlan = buildQueryPlan(referenceQueryText, queryIntent);
        const queryWords = Array.from(
            new Set(fmmTokenize(testCase.query, vocabMap)),
        );
        const querySparse = getQuerySparse(queryWords, vocabMap);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        return {
            testCase,
            queryVector: queryVectors[index],
            queryWords,
            querySparse,
            queryIntent,
            queryPlan,
            queryYearWordIds,
            queryPhaseAnchor: extractPhaseAnchor(referenceQueryText),
        };
    });
}

function bm25Score(
    querySparse: Record<number, number>,
    meta: Metadata,
    stats: BM25Stats,
    metaIndex: number,
): number {
    if (!meta.sparse || meta.sparse.length === 0) {
        return 0;
    }

    let sparse = 0;
    const dl = stats.docLengths[metaIndex];
    const safeDl = Math.max(dl, stats.avgdl * 0.25);

    for (let entryIndex = 0; entryIndex < meta.sparse.length; entryIndex += 2) {
        const wordId = meta.sparse[entryIndex] as number;
        const tf = meta.sparse[entryIndex + 1] as number;
        const qWeight = querySparse[wordId];
        if (!qWeight) {
            continue;
        }

        const idf = stats.idfMap.get(wordId) || 0;
        const numerator = tf * (BM25_K1 + 1);
        const denominator =
            tf + BM25_K1 * (1 - BM25_B + BM25_B * (safeDl / stats.avgdl));
        sparse += qWeight * idf * (numerator / denominator);
    }

    return sparse;
}

function scoreDirectMetadata(
    query: QueryCacheItem,
    metadata: readonly Metadata[],
    stats: BM25Stats,
): {
    dense: ScoredMeta[];
    sparse: ScoredMeta[];
} {
    if (!vectorMatrix) {
        throw new Error("Vector matrix not initialized.");
    }

    const dense: ScoredMeta[] = [];
    const sparse: ScoredMeta[] = [];

    metadata.forEach((meta, index) => {
        let denseScore = dotProduct(
            query.queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null) {
            denseScore *= meta.scale;
        }
        dense.push({ meta, score: denseScore });
        sparse.push({
            meta,
            score: bm25Score(query.querySparse, meta, stats, index),
        });
    });

    dense.sort((a, b) => b.score - a.score);
    sparse.sort((a, b) => b.score - a.score);

    return { dense, sparse };
}

function buildDirectRanking(scored: readonly ScoredMeta[]): RankedDoc[] {
    return scored.map((item) => ({
        otid: item.meta.id,
        score: item.score,
    }));
}

function buildStructuredRanking(
    query: QueryCacheItem,
    weights: { Q: number; KP: number; OT: number },
    options?: { enableQueryPlanner?: boolean },
): RankedDoc[] {
    if (!vectorMatrix || !corpusView || !allMetadataBm25Stats) {
        throw new Error("Structured ranking dependencies not initialized.");
    }

    const allowedTypes = (Object.entries(weights) as Array<
        [keyof typeof weights, number]
    >)
        .filter(([, weight]) => weight > 0)
        .map(([type]) => type);
    const filteredMetadata =
        allowedTypes.length === 3
            ? metadataList
            : metadataList.filter((item) =>
                  allowedTypes.includes(item.type),
              );
    const filteredBm25Stats =
        filteredMetadata === metadataList
            ? allMetadataBm25Stats
            : buildBM25Stats(filteredMetadata);

    const result = searchAndRank({
        queryVector: query.queryVector,
        querySparse: query.querySparse,
        queryYearWordIds: query.queryYearWordIds,
        queryIntent: query.queryIntent,
        queryScopeHint: query.testCase.query_scope,
        metadata: filteredMetadata,
        vectorMatrix,
        dimensions,
        currentTimestamp: CURRENT_TIMESTAMP,
        bm25Stats: filteredBm25Stats,
        weights,
        topHybridLimit: 1000,
        kpAggregationMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpAggregationMode,
        kpTopN: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTopN,
        kpTailWeight: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTailWeight,
        lexicalBonusMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.lexicalBonusMode,
        kpRoleRerankMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpRoleRerankMode,
        kpRoleDocWeight:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpRoleDocWeight,
        qConfusionMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionMode,
        qConfusionWeight:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionWeight,
        enableExplicitYearFilter:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.enableExplicitYearFilter,
        queryPlan: query.queryPlan,
        enableQueryPlanner: Boolean(options?.enableQueryPlanner),
        minimalMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.minimalMode,
    });

    return rerankStructuredMatchesForDocumentMetrics(
        query,
        result.matches.map((item) => ({
            otid: item.otid,
            score: item.score,
            best_kpid: item.best_kpid,
        })),
    );
}

function isFullDocumentBoundaryQuery(query: QueryCacheItem): boolean {
    const normalizedQuery = query.queryPlan.normalizedQuery;

    if (
        query.queryPlan.asksOutcomeLike ||
        query.queryPlan.intentType === "fact_detail" ||
        query.queryPlan.intentType === "time_location"
    ) {
        return false;
    }

    if (
        /整体|概述|总体|全流程|整个流程|完整流程|主要步骤|主要流程|主要要求|关键要求|整体政策|政策和关键要求|从.+到/.test(
            normalizedQuery,
        )
    ) {
        return true;
    }

    return (
        query.queryPlan.asksCoverageLike &&
        (query.queryPlan.intentType === "procedure" ||
            query.queryPlan.intentType === "policy_overview" ||
            query.queryPlan.intentType === "system_timeline")
    );
}

function resolveStructureRiskAwareWeights(
    query: QueryCacheItem,
): { Q: number; KP: number; OT: number } {
    return isFullDocumentBoundaryQuery(query)
        ? STRUCTURED_FULL_DOC_AWARE_WEIGHTS
        : STRUCTURED_Q_KP_OT_WEIGHTS;
}

const ENTITY_ANCHOR_PATTERNS = [
    /中山大学/g,
    /人工智能学院/g,
    /软件工程学院/g,
    /广州实验室/g,
    /广东省综合评价/g,
    /综合评价/g,
    /同等学力/g,
    /临床医学博士/g,
    /在职临床医师/g,
    /港澳台/g,
    /少干计划/g,
    /骨干计划/g,
    /强基计划/g,
    /报考点/g,
    /推免/g,
    /夏令营/g,
    /博士/g,
    /硕士/g,
    /调剂/g,
    /复试/g,
    /录取/g,
    /资格认定/g,
    /论文答辩/g,
];

function extractEntityAnchors(text: string): string[] {
    const anchors = new Set<string>();
    const normalized = text.replace(/\s+/g, "");
    ENTITY_ANCHOR_PATTERNS.forEach((pattern) => {
        for (const match of normalized.matchAll(pattern)) {
            anchors.add(match[0]);
        }
    });
    return [...anchors];
}

function computeTitleEntityCoverage(query: QueryCacheItem, otid?: string): number {
    if (!otid) {
        return 0;
    }
    const queryAnchors = extractEntityAnchors(
        query.testCase.source_query?.trim() || query.testCase.query || "",
    );
    if (queryAnchors.length === 0) {
        return 0;
    }
    const title = articleTitleMap.get(otid) || "";
    const normalizedTitle = title.replace(/\s+/g, "");
    return queryAnchors.filter((anchor) => normalizedTitle.includes(anchor)).length;
}

function violatesAvoidedRole(query: QueryCacheItem, otid?: string): boolean {
    if (!otid || query.queryPlan.avoidedDocRoles.length === 0) {
        return false;
    }
    const title = articleTitleMap.get(otid) || "";
    const roles = inferDocumentRolesFromTitle(title);
    return roles.some((role) => query.queryPlan.avoidedDocRoles.includes(role));
}

function shouldAdoptFullDocAwareRanking(
    query: QueryCacheItem,
    fullRanking: readonly RankedDoc[],
    fullDocAwareRanking: readonly RankedDoc[],
): boolean {
    if (!isFullDocumentBoundaryQuery(query)) {
        return false;
    }

    if (
        !/我|本人|作为|是一名|如果我|想申请|想报|准备报|准备申请/.test(
            query.queryPlan.normalizedQuery,
        )
    ) {
        return false;
    }

    const fullTop = fullRanking[0]?.otid;
    const candidateTop = fullDocAwareRanking[0]?.otid;
    if (!candidateTop || candidateTop === fullTop) {
        return false;
    }

    if (violatesAvoidedRole(query, candidateTop) && !violatesAvoidedRole(query, fullTop)) {
        return false;
    }

    const fullCoverage = computeTitleEntityCoverage(query, fullTop);
    const candidateCoverage = computeTitleEntityCoverage(query, candidateTop);
    if (candidateCoverage < fullCoverage) {
        return false;
    }

    return true;
}

function buildStructureRiskAwareRanking(query: QueryCacheItem): RankedDoc[] {
    const fullRanking = buildStructuredRanking(query, STRUCTURED_Q_KP_OT_WEIGHTS);
    const fullDocAwareRanking = buildStructuredRanking(
        query,
        resolveStructureRiskAwareWeights(query),
    );
    return shouldAdoptFullDocAwareRanking(query, fullRanking, fullDocAwareRanking)
        ? fullDocAwareRanking
        : fullRanking;
}

function toPerCaseResult(
    query: QueryCacheItem,
    ranking: readonly RankedDoc[],
    elapsedMs: number,
): PerCaseResult {
    const otidEvalTarget = resolveOtidEvalTarget(query.testCase);
    const rankValue =
        otidEvalTarget.mode === "required_otid_groups"
            ? getCoverageDepthForRequiredOtidGroups(
                  ranking,
                  otidEvalTarget.requiredOtidGroups,
                  otidEvalTarget.minGroupsToCover,
              )
            : getBestRankForOtidSet(ranking, otidEvalTarget.acceptableOtids);
    const rank = Number.isFinite(rankValue) ? rankValue : null;
    return {
        id: query.testCase.id,
        query: query.testCase.query,
        expected_otid: query.testCase.expected_otid,
        elapsedMs,
        rank,
        hitAt1: rank === 1,
        reciprocalRank: rank ? 1 / rank : 0,
        top1Otid: ranking[0]?.otid,
        topMatches: ranking.slice(0, SAFE_TOP_MATCH_COUNT).map((item, index) => ({
            rank: index + 1,
            otid: item.otid,
            score: item.score,
            best_kpid: item.best_kpid,
        })),
    };
}

function buildTimedPerCaseResults(
    queryCache: readonly QueryCacheItem[],
    buildRanking: (query: QueryCacheItem) => RankedDoc[],
): PerCaseResult[] {
    return queryCache.map((query) => {
        const startedAt = performance.now();
        const ranking = buildRanking(query);
        const elapsedMs = performance.now() - startedAt;
        return toPerCaseResult(query, ranking, elapsedMs);
    });
}

function buildModelReport(
    label: string,
    family: ModelFamily,
    description: string,
    perCase: PerCaseResult[],
): ModelReport {
    const metrics = buildMetricSummary(perCase);
    const hitAt1Values = perCase.map((item) => (item.hitAt1 ? 1 : 0));
    const reciprocalRankValues = perCase.map((item) => item.reciprocalRank);

    return {
        label,
        family,
        description,
        metrics,
        timing: buildTimingSummary(perCase),
        hitAt1Interval95: bootstrapMetricInterval(
            hitAt1Values,
            SAFE_BOOTSTRAP_ITERATIONS,
            `${label}:hit1`,
            100,
        ),
        mrrInterval95: bootstrapMetricInterval(
            reciprocalRankValues,
            SAFE_BOOTSTRAP_ITERATIONS,
            `${label}:mrr`,
        ),
        perCase,
    };
}

function buildPairwiseComparisons(modelReports: readonly ModelReport[]): PairwiseComparison[] {
    const structuredModels = modelReports.filter(
        (item) => item.family === "structured",
    );
    const standardModels = modelReports.filter(
        (item) => item.family !== "structured",
    );
    const comparisons: PairwiseComparison[] = [];

    structuredModels.forEach((challenger) => {
        standardModels.forEach((baseline) => {
            const challengerHitAt1 = challenger.perCase.map((item) =>
                item.hitAt1 ? 1 : 0,
            );
            const baselineHitAt1 = baseline.perCase.map((item) =>
                item.hitAt1 ? 1 : 0,
            );
            const challengerMrr = challenger.perCase.map(
                (item) => item.reciprocalRank,
            );
            const baselineMrr = baseline.perCase.map(
                (item) => item.reciprocalRank,
            );

            comparisons.push({
                challenger: challenger.label,
                baseline: baseline.label,
                hitAt1: pairedBootstrapDelta(
                    challengerHitAt1,
                    baselineHitAt1,
                    SAFE_BOOTSTRAP_ITERATIONS,
                    `${challenger.label}:${baseline.label}:hit1`,
                    100,
                ),
                mrr: pairedBootstrapDelta(
                    challengerMrr,
                    baselineMrr,
                    SAFE_BOOTSTRAP_ITERATIONS,
                    `${challenger.label}:${baseline.label}:mrr`,
                ),
            });
        });
    });

    return comparisons;
}

function buildDatasetReport(
    target: DatasetTarget,
    queryCache: readonly QueryCacheItem[],
): DatasetReport {
    if (!corpusView) {
        throw new Error("Corpus view not initialized.");
    }

    const bm25OtPerCase = buildTimedPerCaseResults(queryCache, (query) => {
        const scored = scoreDirectMetadata(query, corpusView.otMetadata, corpusView.otBm25Stats);
        return buildDirectRanking(scored.sparse);
    });

    const denseOtPerCase = buildTimedPerCaseResults(queryCache, (query) => {
        const scored = scoreDirectMetadata(query, corpusView.otMetadata, corpusView.otBm25Stats);
        return buildDirectRanking(scored.dense);
    });

    const structuredKpOtPerCase = buildTimedPerCaseResults(queryCache, (query) =>
        buildStructuredRanking(query, STRUCTURED_KP_OT_WEIGHTS),
    );

    const structuredFullPerCase = buildTimedPerCaseResults(queryCache, (query) =>
        buildStructuredRanking(
            query,
            STRUCTURED_Q_KP_OT_WEIGHTS,
        ),
    );
    const structuredFullPlannerPerCase = INCLUDE_QUERY_PLANNER_MODEL
        ? buildTimedPerCaseResults(queryCache, (query) =>
              buildStructuredRanking(query, STRUCTURED_Q_KP_OT_WEIGHTS, {
                  enableQueryPlanner: true,
              }),
          )
        : undefined;
    const structuredRiskAwarePerCase = INCLUDE_STRUCTURE_RISK_MODEL
        ? buildTimedPerCaseResults(queryCache, (query) =>
              buildStructureRiskAwareRanking(query),
          )
        : undefined;

    const datasetProfile = resolveNamedDatasetProfile(target.datasetFile);

    const models: ModelReport[] = [
        buildModelReport(
            "BM25-OT",
            "sparse",
            "经典稀疏检索：仅在 OT 文档上执行 BM25。",
            bm25OtPerCase,
        ),
        buildModelReport(
            "Dense-OT",
            "dense",
            "语义检索：仅在 OT 文档上执行向量相似度排序。",
            denseOtPerCase,
        ),
        buildModelReport(
            "Structured-KP+OT",
            "structured",
            "论文冻结主组合：Q=0, KP=0.2857, OT=0.7143。",
            structuredKpOtPerCase,
        ),
        buildModelReport(
            "Structured-Q+KP+OT",
            "structured",
            "前端 runtime 对齐主线：Q=0.3333, KP=0.3333, OT=0.3333。",
            structuredFullPerCase,
        ),
        ...(structuredFullPlannerPerCase
            ? [
                  buildModelReport(
                      "Structured-Q+KP+OT+Planner",
                      "structured" as const,
                      "实验模型：在 Structured-Q+KP+OT 上启用 retrieval-stage query planner。",
                      structuredFullPlannerPerCase,
                  ),
              ]
            : []),
        ...(structuredRiskAwarePerCase
            ? [
                  buildModelReport(
                      "Structured-Q+KP+OT+RiskAware",
                      "structured" as const,
                      "实验模型：运行时识别完整文档边界类 query，降低 Q 权重并提高 KP/OT 权重。",
                      structuredRiskAwarePerCase,
                  ),
              ]
            : []),
    ];

    const bestStandardBaseline = [...models]
        .filter((item) => item.family !== "structured")
        .sort((a, b) => compareMetrics(a.metrics, b.metrics))[0]?.label;

    if (!bestStandardBaseline) {
        throw new Error("Unable to resolve best standard baseline.");
    }

    return {
        datasetKey: target.key,
        datasetLabel: target.label,
        datasetAlias: datasetProfile.alias,
        datasetDisplayName: datasetProfile.displayName,
        datasetFile: target.datasetFile,
        caseCount: queryCache.length,
        bestStandardBaseline,
        models,
        pairwiseComparisons: buildPairwiseComparisons(models),
    };
}

function printDatasetSummary(report: DatasetReport) {
    console.log(
        `\n[${report.datasetLabel}] best standard baseline: ${report.bestStandardBaseline}`,
    );
    report.models.forEach((model) => {
        console.log(
            `${model.label}: Hit@1=${model.metrics.hitAt1.toFixed(2)}% [${model.hitAt1Interval95.lower.toFixed(2)}, ${model.hitAt1Interval95.upper.toFixed(2)}] | MRR=${model.metrics.mrr.toFixed(4)} [${model.mrrInterval95.lower.toFixed(4)}, ${model.mrrInterval95.upper.toFixed(4)}] | avg=${model.timing.avgMs.toFixed(2)}ms | p50=${model.timing.p50Ms.toFixed(2)}ms | p95=${model.timing.p95Ms.toFixed(2)}ms`,
        );
    });

    report.pairwiseComparisons
        .filter((item) => item.baseline === report.bestStandardBaseline)
        .forEach((item) => {
            console.log(
                `${item.challenger} vs ${item.baseline}: ΔHit@1=${item.hitAt1.observedDelta.toFixed(2)}pp [${item.hitAt1.interval95.lower.toFixed(2)}, ${item.hitAt1.interval95.upper.toFixed(2)}], p=${item.hitAt1.randomizationPValue.toFixed(4)} | ΔMRR=${item.mrr.observedDelta.toFixed(4)} [${item.mrr.interval95.lower.toFixed(4)}, ${item.mrr.interval95.upper.toFixed(4)}], p=${item.mrr.randomizationPValue.toFixed(4)}`,
            );
        });
}

async function main() {
    if (DATASET_TARGET_ORDER.length === 0) {
        throw new Error(
            "未解析到可用的 granularity 数据集目标，请先生成主集或 holdout 新文件。",
        );
    }

    console.log("Loading frontend eval engine...");
    await loadEngine();

    const datasetReports: DatasetReport[] = [];
    for (const target of DATASET_TARGET_ORDER) {
        console.log(`\nEvaluating ${target.label}...`);
        const datasetCases = loadDataset(target.datasetFile, {
            datasetLabel: target.key,
        }) as DatasetCase[];
        console.log(`Loaded ${datasetCases.length} cases from ${target.datasetFile}`);
        const queryCache = await buildQueryCache(datasetCases);
        const datasetReport = buildDatasetReport(target, queryCache);
        datasetReports.push(datasetReport);
        printDatasetSummary(datasetReport);
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        embeddingModel: FRONTEND_MODEL_NAME,
        bootstrapIterations: SAFE_BOOTSTRAP_ITERATIONS,
        randomizationIterations: SAFE_RANDOMIZATION_ITERATIONS,
        timingNote:
            "耗时统计仅覆盖每条 query 的检索/排序阶段，不含 query embedding、模型加载与结果写盘。",
        runtimePresetRetrieval: {
            qConfusionMode:
                FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionMode,
            qConfusionWeight:
                FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionWeight,
        },
        datasets: datasetReports,
    };

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(resultsDir, { recursive: true });
    const outputPath = path.resolve(
        resultsDir,
        buildStandardBaselinesResultFileName(Date.now()),
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`\nReport saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

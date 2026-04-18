import * as fs from "fs";
import * as path from "path";

import {
    buildBM25Stats,
    type FusionMode,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type BM25Stats,
    type KPAggregationMode,
    type KPRoleRerankMode,
    type LexicalBonusMode,
    type Metadata,
    type ParsedQueryIntent,
    type QConfusionMode,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import { FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET } from "../src/worker/search_pipeline.ts";
import {
    ACTIVE_MAIN_DB_VERSION,
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    loadDatasetSources,
    resolveEvalDatasetConfig,
    type EvalDatasetCase,
    type EvalDatasetSource,
    type GranularityDatasetTargetKey,
    type KpEvalMode,
    type OtidEvalMode,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import {
    resolveBackendArticlesFile,
    resolveBackendKnowledgePointsFile,
} from "./kb_version_paths.ts";
import {
    buildGranularityResultFileName,
    resolveNamedDatasetProfile,
} from "./result_naming.ts";
import { updateCurrentResultRegistry } from "./result_registry.ts";

type DatasetCase = EvalDatasetCase;
type GranularityType = "Q" | "KP" | "OT";
type KPCandidateRerankMode = "none" | "heuristic" | "feature_heuristic";
type DocPostRerankMode = "none" | "kp_heuristic_delta" | "time_anchor";
type WeightConfig = {
    Q: number;
    KP: number;
    OT: number;
};

type MetadataWithParentPkid = Metadata & {
    parent_pkid?: string;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    effectiveQueryText: string;
    referenceQueryText: string;
    queryVector: Float32Array;
    queryIntent: ParsedQueryIntent;
    queryMonths: number[];
    queryPhaseAnchor: PhaseAnchor;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryYearWordIds: number[];
    denseScoreOverrides?: ReadonlyMap<string, number>;
};

type Metrics = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type KpidMetrics = {
    applicable: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
    docHitAt1Total: number;
    docHitAt1CorrectKpid: number;
    docHitAt1WrongKpid: number;
    docHitAt5Total: number;
    docHitAt5CorrectKpid: number;
    docHitAt5WrongKpid: number;
};

type SupportCoverageMetrics = {
    applicable: number;
    docHitAt1Total: number;
    docHitAt5Total: number;
    docHitAt1FullCoverAt3: number;
    docHitAt1FullCoverAt5: number;
    docHitAt5FullCoverAt5: number;
    partialCoverAt5: number;
    avgCoveredGroupsAt5: number;
    avgCoverageRatioAt5: number;
};

type MetricsBundle = {
    metricsByDataset: Record<string, Metrics>;
    combined: Metrics;
    kpidMetricsByDataset: Record<string, KpidMetrics>;
    kpidCombined: KpidMetrics;
    supportCoverageMetricsByDataset: Record<string, SupportCoverageMetrics>;
    supportCoverageCombined: SupportCoverageMetrics;
};

type ComboDefinition = {
    label: string;
    allowedTypes: GranularityType[];
};

type WeightCandidateSummary = {
    weights: WeightConfig;
    combined: Metrics;
};

type GroupMetricsReport = {
    total: number;
    uniform: Metrics;
    tunedCombined: Metrics;
    kpidUniform: KpidMetrics;
    kpidTunedCombined: KpidMetrics;
    supportCoverageUniform: SupportCoverageMetrics;
    supportCoverageTunedCombined: SupportCoverageMetrics;
};

type NormalizedKpEvalTarget = {
    mode: KpEvalMode;
    acceptableAnchorKpids: string[];
    requiredKpidGroups: string[][];
    minGroupsToCover: number;
};

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

type RerankedKpCandidate = {
    kpid: string;
    rawScore: number;
    rerankedScore: number;
};

type SupportCoverageState = {
    coverageDepth: number;
    coveredGroupsTop3: number;
    coveredGroupsTop5: number;
    coverageRatioTop5: number;
    totalGroups: number;
    minGroupsToCover: number;
};

type OtidCoverageState = {
    coverageDepth: number;
    coveredGroupsTop3: number;
    coveredGroupsTop5: number;
    coverageRatioTop5: number;
    totalGroups: number;
    minGroupsToCover: number;
};

type CaseDetail = {
    query: string;
    expected_otid: string;
    otid_eval_mode?: OtidEvalMode;
    acceptable_otids?: string[];
    required_otid_groups?: string[][];
    min_otid_groups_to_cover?: number;
    expected_kpid?: string;
    kp_eval_mode?: KpEvalMode;
    acceptable_anchor_kpids?: string[];
    required_kpid_groups?: string[][];
    min_groups_to_cover?: number;
    dataset: string;
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    support_pattern?: string;
    granularity_sensitive?: boolean;
    theme_family?: string;
    source_dataset?: string;
    source_seed_id?: string;
    challenge_tags?: string[];
    notes?: string;
    docRank: number | null;
    kpidRank: number | null;
    supportCoverageDepth: number | null;
    docCoverageDepth: number | null;
    docGroupsCoveredTop5: number;
    docGroupsRequired: number;
    docFullCoverTop5: boolean;
    docPartialCoverTop5: boolean;
    supportGroupsCoveredTop5: number;
    supportGroupsRequired: number;
    supportFullCoverTop5: boolean;
    supportPartialCoverTop5: boolean;
    docHitAt1: boolean;
    docHitAt5: boolean;
    kpidHitAt1: boolean;
    kpidHitAt5: boolean;
    failure_risk: string;
    failure_reasons: string[];
    topDocMatchesSource?: "matches" | "weak_matches" | "none";
    topDocMatches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
    topKpidMatches: Array<{
        rank: number;
        otid: string;
        best_kpid?: string;
    }>;
    expectedDocTopKpCandidates: Array<{
        rank: number;
        kpid: string;
        rawScore: number;
        rerankedScore: number;
    }>;
};

type ComboReport = {
    label: string;
    allowedTypes: GranularityType[];
    metadataCount: number;
    metadataTypeCounts: Record<GranularityType, number>;
    uniform: {
        weights: WeightConfig;
        combined: Metrics;
        kpidCombined: KpidMetrics;
        supportCoverageCombined: SupportCoverageMetrics;
        metricsByDataset: Record<string, Metrics>;
    };
    tuned: {
        selectionMode: "legacy_tune_holdout" | "single_frozen_set";
        candidateCount: number;
        bestWeights: WeightConfig;
        tuneCombined: Metrics;
        kpidTuneCombined: KpidMetrics;
        supportCoverageTuneCombined: SupportCoverageMetrics;
        holdoutCombined: Metrics;
        kpidHoldoutCombined: KpidMetrics;
        supportCoverageHoldoutCombined: SupportCoverageMetrics;
        combinedCombined: Metrics;
        kpidCombinedCombined: KpidMetrics;
        supportCoverageCombinedCombined: SupportCoverageMetrics;
        topTuneCandidates: WeightCandidateSummary[];
    };
    groupBreakdowns: {
        supportPattern: Record<string, GroupMetricsReport>;
        preferredGranularity: Record<string, GroupMetricsReport>;
        queryType: Record<string, GroupMetricsReport>;
    };
    caseDetails?: CaseDetail[];
    caseDetailsWeightMode?: "uniform" | "tuned";
};

type Report = {
    generatedAt: string;
    mainDbVersion: string;
    experimentTrack?: "default" | "minimal_first" | "frontend_runtime";
    pipelinePresetName?: string;
    minimalBaselineMode?: boolean;
    minimalAddYear?: boolean;
    minimalAddPhase?: boolean;
    minimalAddAspect?: boolean;
    minimalDisableDocMulti?: boolean;
    fusionMode?: FusionMode;
    fixedComboWeights?: Record<string, WeightConfig>;
    otDenseMode: string;
    qConfusionMode?: QConfusionMode;
    qConfusionWeight?: number;
    queryStyleMode?: string;
    kpStyleMode?: string;
    qPerDocCap?: number;
    kpPerDocCap?: number;
    lexicalTypeMultipliers?: {
        Q: number;
        KP: number;
        OT: number;
    };
    otDenseSlidingWindowConfig?: {
        windowSize: number;
        overlap: number;
        stride: number;
        batchSize: number;
        docCount: number;
        windowCount: number;
        missingDocCount: number;
    };
    datasetVersion: string;
    datasetMode: "split" | "single_file" | "named_group";
    datasetKey: string;
    datasetLabel: string;
    datasetAlias?: string;
    datasetDisplayName?: string;
    datasetGroups: Array<{
        key: string;
        label: string;
        role: string;
        sourceCount: number;
        resolvedFromFallback?: boolean;
    }>;
    topHybridLimit: number;
    kpAggregationMode: KPAggregationMode;
    kpTopN: number;
    kpTailWeight: number;
    lexicalBonusMode: LexicalBonusMode;
    onlineKpRoleRerankMode: KPRoleRerankMode;
    onlineKpRoleDocWeight: number;
    kpCandidateRerankMode: KPCandidateRerankMode;
    docPostRerankMode: DocPostRerankMode;
    docPostRerankWeight: number;
    limitPerDataset?: number;
    weightSteps: number[];
    datasetSizes: {
        tune: Record<string, number>;
        holdout: Record<string, number>;
        combined: Record<string, number>;
    };
    globalMetadataTypeCounts: Record<GranularityType, number>;
    combos: ComboReport[];
};

function parseCliDatasetTargetKey():
    | GranularityDatasetTargetKey
    | undefined {
    const args = process.argv.slice(2);
    for (let index = 0; index < args.length; index += 1) {
        const current = args[index];
        if (current === "--dataset") {
            const next = args[index + 1];
            if (next) {
                return next as GranularityDatasetTargetKey;
            }
        }
        if (current.startsWith("--dataset=")) {
            const [, value] = current.split("=", 2);
            if (value) {
                return value as GranularityDatasetTargetKey;
            }
        }
    }

    const positional = args.find((item) => !item.startsWith("--"));
    if (positional) {
        return positional as GranularityDatasetTargetKey;
    }

    return undefined;
}

const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || "granularity";
const DATASET_FILE = process.env.SUASK_EVAL_DATASET_FILE;
const DATASET_TARGET_KEY = (
    process.env.SUASK_EVAL_DATASET_TARGET_KEY ||
    process.env.SUASK_EVAL_DATASET_TARGET ||
    parseCliDatasetTargetKey()
) as GranularityDatasetTargetKey | undefined;
const SINGLE_FILE_AS_ALL = process.env.SUASK_EVAL_SINGLE_FILE_AS_ALL !== "0";
const DATASET_CONFIG = resolveEvalDatasetConfig({
    datasetVersion: DATASET_VERSION,
    datasetFile: DATASET_FILE,
    singleFileAsAll: SINGLE_FILE_AS_ALL,
    datasetTargetKey: DATASET_TARGET_KEY,
});
const DATASET_PROFILE = resolveNamedDatasetProfile(DATASET_CONFIG.datasetKey);
const LIMIT_PER_DATASET = Number.parseInt(
    process.env.SUASK_EVAL_LIMIT_PER_DATASET || "",
    10,
);
const TOP_HYBRID_LIMIT = Number.parseInt(
    process.env.SUASK_TOP_HYBRID_LIMIT || "",
    10,
);
const KP_AGGREGATION_MODE = (
    process.env.SUASK_KP_AGG_MODE === "max_plus_topn"
        ? "max_plus_topn"
        : process.env.SUASK_KP_AGG_MODE === "mean"
          ? "mean"
          : process.env.SUASK_KP_AGG_MODE === "sum"
            ? "sum"
            : "max"
) as KPAggregationMode;
const LEXICAL_BONUS_MODE = (
    process.env.SUASK_LEXICAL_BONUS_MODE === "max" ? "max" : "sum"
) as LexicalBonusMode;
const ONLINE_KP_ROLE_RERANK_MODE = (
    process.env.SUASK_ONLINE_KP_ROLE_RERANK_MODE === "feature"
        ? "feature"
        : "off"
) as KPRoleRerankMode;
const ONLINE_KP_ROLE_DOC_WEIGHT = Number.parseFloat(
    process.env.SUASK_ONLINE_KP_ROLE_DOC_WEIGHT || "",
);
const KP_CANDIDATE_RERANK_MODE = (
    process.env.SUASK_KP_CANDIDATE_RERANK_MODE === "heuristic"
        ? "heuristic"
        : process.env.SUASK_KP_CANDIDATE_RERANK_MODE === "feature_heuristic"
          ? "feature_heuristic"
        : "none"
) as KPCandidateRerankMode;
const DOC_POST_RERANK_MODE = (
    process.env.SUASK_DOC_POST_RERANK_MODE === "kp_heuristic_delta"
        ? "kp_heuristic_delta"
        : process.env.SUASK_DOC_POST_RERANK_MODE === "time_anchor"
          ? "time_anchor"
        : "none"
) as DocPostRerankMode;
const DOC_POST_RERANK_WEIGHT = Number.parseFloat(
    process.env.SUASK_DOC_POST_RERANK_WEIGHT || "",
);
const EXPORT_BAD_CASES = process.env.SUASK_EXPORT_BAD_CASES === "1";
const BAD_CASE_COMBO = process.env.SUASK_BAD_CASE_COMBO || "KP+OT";
const BAD_CASE_WEIGHT_MODE = (
    process.env.SUASK_BAD_CASE_WEIGHT_MODE === "uniform" ? "uniform" : "tuned"
) as "uniform" | "tuned";
const BAD_CASE_FAILURES_ONLY = process.env.SUASK_BAD_CASE_FAILURES_ONLY !== "0";
const BAD_CASE_TOP_MATCHES = Number.parseInt(
    process.env.SUASK_BAD_CASE_TOP_MATCHES || "",
    10,
);
const SKIP_RESULT_REGISTRY_UPDATE =
    process.env.SUASK_SKIP_RESULT_REGISTRY_UPDATE === "1";
const EXPERIMENT_TRACK = (
    process.env.SUASK_EXPERIMENT_TRACK === "minimal_first"
        ? "minimal_first"
        : process.env.SUASK_EXPERIMENT_TRACK === "frontend_runtime"
          ? "frontend_runtime"
            : process.env.SUASK_EXPERIMENT_TRACK === "default"
              ? "default"
              : "frontend_runtime"
) as "default" | "minimal_first" | "frontend_runtime";
const FORCE_DISABLE_MINIMAL_YEAR =
    process.env.SUASK_FORCE_DISABLE_MINIMAL_YEAR === "1";
const FORCE_DISABLE_MINIMAL_PHASE =
    process.env.SUASK_FORCE_DISABLE_MINIMAL_PHASE === "1";
const MINIMAL_BASELINE_MODE =
    process.env.SUASK_MINIMAL_BASELINE === "1" ||
    EXPERIMENT_TRACK === "minimal_first" ||
    EXPERIMENT_TRACK === "frontend_runtime";
const MINIMAL_ADD_YEAR =
    !FORCE_DISABLE_MINIMAL_YEAR &&
    (process.env.SUASK_MINIMAL_ADD_YEAR === "1" ||
        EXPERIMENT_TRACK === "frontend_runtime" ||
        FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.enableExplicitYearFilter);
const MINIMAL_ADD_PHASE =
    !FORCE_DISABLE_MINIMAL_PHASE &&
    (EXPERIMENT_TRACK === "frontend_runtime" ||
        process.env.SUASK_MINIMAL_ADD_PHASE === "1");
const MINIMAL_ADD_ASPECT = process.env.SUASK_MINIMAL_ADD_ASPECT === "1";
const MINIMAL_DISABLE_DOC_MULTI =
    process.env.SUASK_MINIMAL_DISABLE_DOC_MULTI === "1";
const FUSION_MODE = (
    process.env.SUASK_FUSION_MODE === "max_q_vs_kpot"
        ? "max_q_vs_kpot"
        : "default"
) as FusionMode;
const OT_DENSE_MODE = process.env.SUASK_OT_DENSE_MODE || "original";
const OT_DENSE_WINDOW_SIZE = Number.parseInt(
    process.env.SUASK_OT_DENSE_WINDOW_SIZE || "",
    10,
);
const OT_DENSE_WINDOW_OVERLAP = Number.parseInt(
    process.env.SUASK_OT_DENSE_WINDOW_OVERLAP || "",
    10,
);
const OT_DENSE_WINDOW_BATCH_SIZE = Number.parseInt(
    process.env.SUASK_OT_DENSE_WINDOW_BATCH_SIZE || "",
    10,
);
const KP_TOP_N = Number.parseInt(process.env.SUASK_KP_TOP_N || "", 10);
const KP_TAIL_WEIGHT = Number.parseFloat(
    process.env.SUASK_KP_TAIL_WEIGHT || "",
);
const Q_PER_DOC_CAP = Number.parseInt(process.env.SUASK_Q_PER_DOC_CAP || "", 10);
const KP_PER_DOC_CAP = Number.parseInt(
    process.env.SUASK_KP_PER_DOC_CAP || "",
    10,
);
const Q_LEXICAL_MULTIPLIER = Number.parseFloat(
    process.env.SUASK_Q_LEXICAL_MULTIPLIER || "",
);
const KP_LEXICAL_MULTIPLIER = Number.parseFloat(
    process.env.SUASK_KP_LEXICAL_MULTIPLIER || "",
);
const OT_LEXICAL_MULTIPLIER = Number.parseFloat(
    process.env.SUASK_OT_LEXICAL_MULTIPLIER || "",
);
const QUERY_STYLE_MODE = process.env.SUASK_QUERY_STYLE_MODE || "original";
const KP_STYLE_MODE = process.env.SUASK_KP_STYLE_MODE || "original";
const Q_CONFUSION_MODE = (
    process.env.SUASK_Q_CONFUSION_MODE === "consensus"
        ? "consensus"
        : process.env.SUASK_Q_CONFUSION_MODE === "consensus_no_year"
          ? "consensus_no_year"
        : process.env.SUASK_Q_CONFUSION_MODE === "competition"
          ? "competition"
          : process.env.SUASK_Q_CONFUSION_MODE === "combined"
            ? "combined"
            : "off"
) as QConfusionMode;
const Q_CONFUSION_WEIGHT = Number.parseFloat(
    process.env.SUASK_Q_CONFUSION_WEIGHT || "",
);
const WEIGHT_STEPS = parseWeightSteps(
    process.env.SUASK_WEIGHT_STEPS || "0.2,0.5,0.8",
);
const FIXED_COMBO_WEIGHTS = parseFixedComboWeights(
    process.env.SUASK_FIXED_COMBO_WEIGHTS || "",
);
const TOP_TUNE_CANDIDATE_LIMIT = 5;
const CURRENT_TIMESTAMP = 0;
const SAFE_OT_DENSE_WINDOW_SIZE =
    Number.isFinite(OT_DENSE_WINDOW_SIZE) && OT_DENSE_WINDOW_SIZE >= 64
        ? Math.floor(OT_DENSE_WINDOW_SIZE)
        : 512;
const SAFE_OT_DENSE_WINDOW_OVERLAP =
    Number.isFinite(OT_DENSE_WINDOW_OVERLAP) && OT_DENSE_WINDOW_OVERLAP >= 0
        ? Math.min(
              Math.floor(OT_DENSE_WINDOW_OVERLAP),
              SAFE_OT_DENSE_WINDOW_SIZE - 1,
          )
        : 128;
const SAFE_OT_DENSE_WINDOW_STRIDE = Math.max(
    1,
    SAFE_OT_DENSE_WINDOW_SIZE - SAFE_OT_DENSE_WINDOW_OVERLAP,
);
const SAFE_OT_DENSE_WINDOW_BATCH_SIZE =
    Number.isFinite(OT_DENSE_WINDOW_BATCH_SIZE) && OT_DENSE_WINDOW_BATCH_SIZE > 0
        ? Math.floor(OT_DENSE_WINDOW_BATCH_SIZE)
        : 16;

const COMBOS: ComboDefinition[] = [
    { label: "Q", allowedTypes: ["Q"] },
    { label: "KP", allowedTypes: ["KP"] },
    { label: "OT", allowedTypes: ["OT"] },
    { label: "Q+KP", allowedTypes: ["Q", "KP"] },
    { label: "Q+OT", allowedTypes: ["Q", "OT"] },
    { label: "KP+OT", allowedTypes: ["KP", "OT"] },
    { label: "Q+KP+OT", allowedTypes: ["Q", "KP", "OT"] },
] as const;

let vocabMap = new Map<string, number>();
let metadataList: Metadata[] = [];
let vectorMatrix: Int8Array | null = null;
let dimensions = 768;
let extractor: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["extractor"] | null = null;
let kpTextMap = new Map<string, string>();
let kpFeatureMap = new Map<string, KPFeatureFlags>();
let articleTimeAnchorMap = new Map<string, ArticleTimeAnchor>();
let otDenseSlidingWindowCorpus: OtDenseSlidingWindowCorpus | null = null;
let kpDenseStyleCorpus: KpDenseStyleCorpus | null = null;

type EvalSearchMatch = {
    otid: string;
    best_kpid?: string;
    score: number;
    kp_candidates?: Array<{ kpid: string; score: number }>;
};

type KPFeatureFlags = {
    hasTimeExpression: boolean;
    hasDeadlineCue: boolean;
    hasArrivalCue: boolean;
    hasAnnouncementPeriodCue: boolean;
    hasScheduleCue: boolean;
    hasApplicationCue: boolean;
    hasConditionCue: boolean;
    hasMaterialsCue: boolean;
    hasEmailCue: boolean;
    hasProcedureCue: boolean;
    hasPublishCue: boolean;
    hasBackgroundCue: boolean;
    hasReminderCue: boolean;
    hasPostOutcomeCue: boolean;
    hasDistributionCue: boolean;
    hasThesisCue: boolean;
};

type ArticleTimeAnchor = {
    year?: number;
    month?: number;
    title: string;
    publishTime: string;
    phaseAnchor: PhaseAnchor;
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

type BackendArticleRecord = {
    otid?: string;
    ot_text?: string;
    ot_title?: string;
    publish_time?: string;
};

type OtDenseSlidingWindowCorpus = {
    windowOtids: string[];
    vectors: Float32Array[];
    windowCountByOtid: Map<string, number>;
    missingOtids: string[];
};

type KpDenseStyleCorpus = {
    kpids: string[];
    vectors: Float32Array[];
};

const KP_TEXTS_FILE = resolveBackendKnowledgePointsFile();
const ARTICLE_TEXTS_FILE = resolveBackendArticlesFile();
const QUERY_STOPWORDS = new Set([
    "什么",
    "什么时候",
    "哪些",
    "一下",
    "现在",
    "已经",
    "还是",
    "应该",
    "这种",
    "情况",
    "具体",
    "确认",
    "准备",
    "需要",
    "一下",
]);

function shouldUseOtDenseSlidingWindow(): boolean {
    return !MINIMAL_BASELINE_MODE && OT_DENSE_MODE === "sliding_window_max";
}

function loadBackendArticleRecords(): BackendArticleRecord[] {
    const absolutePath = path.resolve(process.cwd(), ARTICLE_TEXTS_FILE);
    if (!fs.existsSync(absolutePath)) {
        return [];
    }

    return JSON.parse(
        fs.readFileSync(absolutePath, "utf-8"),
    ) as BackendArticleRecord[];
}

async function tokenizeTextWithoutSpecialTokens(text: string): Promise<number[]> {
    if (!extractor) {
        throw new Error("Extractor not initialized");
    }

    const tokenizerOutput = await extractor.tokenizer(text, {
        truncation: false,
        add_special_tokens: false,
    });
    return Array.from(tokenizerOutput.input_ids.data).map((item) => Number(item));
}

function buildOtDenseSlidingWindowTexts(text: string, tokenIds: number[]): string[] {
    if (!extractor) {
        throw new Error("Extractor not initialized");
    }

    if (tokenIds.length <= SAFE_OT_DENSE_WINDOW_SIZE) {
        return text.trim() ? [text] : [];
    }

    const windows: string[] = [];
    for (
        let start = 0;
        start < tokenIds.length;
        start += SAFE_OT_DENSE_WINDOW_STRIDE
    ) {
        const end = Math.min(start + SAFE_OT_DENSE_WINDOW_SIZE, tokenIds.length);
        const windowIds = tokenIds.slice(start, end);
        if (windowIds.length === 0) {
            continue;
        }

        const decoded = extractor.tokenizer.decode(windowIds);
        if (decoded.trim()) {
            windows.push(decoded);
        }

        if (end >= tokenIds.length) {
            break;
        }
    }

    return windows.length > 0 ? windows : (text.trim() ? [text] : []);
}

async function buildOtDenseSlidingWindowCorpus(): Promise<OtDenseSlidingWindowCorpus> {
    if (!extractor) {
        throw new Error("Extractor not initialized");
    }

    const otidSet = new Set(
        metadataList
            .filter((item) => item.type === "OT")
            .map((item) => item.id),
    );
    const articleRecords = loadBackendArticleRecords();
    const windowTexts: string[] = [];
    const windowOtids: string[] = [];
    const windowCountByOtid = new Map<string, number>();

    console.log(
        "Building OT dense sliding-window corpus: " +
            `window=${SAFE_OT_DENSE_WINDOW_SIZE}, ` +
            `overlap=${SAFE_OT_DENSE_WINDOW_OVERLAP}, ` +
            `stride=${SAFE_OT_DENSE_WINDOW_STRIDE}`,
    );

    for (const article of articleRecords) {
        const otid = article.otid;
        const otText = article.ot_text || "";
        if (!otid || !otidSet.has(otid) || !otText.trim()) {
            continue;
        }

        const tokenIds = await tokenizeTextWithoutSpecialTokens(otText);
        const windows = buildOtDenseSlidingWindowTexts(otText, tokenIds);
        if (windows.length === 0) {
            continue;
        }

        windowCountByOtid.set(otid, windows.length);
        windows.forEach((windowText) => {
            windowOtids.push(otid);
            windowTexts.push(windowText);
        });
    }

    const missingOtids = metadataList
        .filter((item) => item.type === "OT")
        .map((item) => item.id)
        .filter((otid) => !windowCountByOtid.has(otid));

    console.log(
        `OT dense sliding-window corpus ready: ${windowCountByOtid.size} docs, ${windowTexts.length} windows` +
            (missingOtids.length > 0 ? `, missing ${missingOtids.length} docs` : ""),
    );

    const vectors = await embedFrontendQueries(extractor, windowTexts, dimensions, {
        batchSize: SAFE_OT_DENSE_WINDOW_BATCH_SIZE,
        onProgress: (done, total) => {
            if (
                done === total ||
                done === Math.min(total, SAFE_OT_DENSE_WINDOW_BATCH_SIZE)
            ) {
                console.log(`OT dense window embedding progress: ${done}/${total}`);
            }
        },
    });

    return {
        windowOtids,
        vectors,
        windowCountByOtid,
        missingOtids,
    };
}

function dotVectorPair(queryVector: Float32Array, candidateVector: Float32Array): number {
    let score = 0;
    for (let index = 0; index < dimensions; index++) {
        score += queryVector[index] * candidateVector[index];
    }
    return score;
}

function buildOtDenseScoreOverrides(
    queryVector: Float32Array,
): ReadonlyMap<string, number> | undefined {
    if (!otDenseSlidingWindowCorpus) {
        return undefined;
    }

    const scoreMap = new Map<string, number>();
    otDenseSlidingWindowCorpus.windowOtids.forEach((otid, index) => {
        const score = dotVectorPair(
            queryVector,
            otDenseSlidingWindowCorpus.vectors[index],
        );
        const current = scoreMap.get(otid);
        if (current === undefined || score > current) {
            scoreMap.set(otid, score);
        }
    });
    return scoreMap;
}

function shouldUseKpDenseStyleCorpus(): boolean {
    return (
        !MINIMAL_BASELINE_MODE &&
        (KP_STYLE_MODE === "question_wrap" || KP_STYLE_MODE === "truncate128")
    );
}

async function buildKpDenseStyleCorpus(): Promise<KpDenseStyleCorpus | null> {
    if (!extractor || !shouldUseKpDenseStyleCorpus()) {
        return null;
    }

    const kpids = metadataList
        .filter((item) => item.type === "KP" && kpTextMap.has(item.id))
        .map((item) => item.id);
    let rewrittenTexts: string[] = [];
    if (KP_STYLE_MODE === "truncate128") {
        for (
            let startIndex = 0;
            startIndex < kpids.length;
            startIndex += DEFAULT_QUERY_EMBED_BATCH_SIZE
        ) {
            const batchKpids = kpids.slice(
                startIndex,
                startIndex + DEFAULT_QUERY_EMBED_BATCH_SIZE,
            );
            const batchTexts = batchKpids.map((kpid) =>
                rewriteKpTextByStyle(kpTextMap.get(kpid) || ""),
            );
            const tokenizerOutput = await extractor.tokenizer(batchTexts, {
                truncation: true,
                max_length: 128,
                padding: true,
            });
            const inputIds = tokenizerOutput.input_ids.tolist() as bigint[][];
            inputIds.forEach((row) => {
                const trimmed = row.filter(
                    (tokenId) => tokenId !== 0n && tokenId !== 101n && tokenId !== 102n,
                );
                const decoded = extractor!.tokenizer
                    .decode(trimmed)
                    .replace(/\[CLS\]|\[SEP\]|\[PAD\]/g, "")
                    .replace(/\s+/g, "")
                    .trim();
                rewrittenTexts.push(decoded);
            });
        }
    } else {
        rewrittenTexts = kpids.map((kpid) =>
            rewriteKpTextByStyle(kpTextMap.get(kpid) || ""),
        );
    }
    const vectors = await embedFrontendQueries(extractor, rewrittenTexts, dimensions, {
        batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
        onProgress: (done, total) => {
            if (done === total || done % 256 === 0) {
                console.log(`KP style corpus embedding progress: ${done}/${total}`);
            }
        },
    });

    return {
        kpids,
        vectors,
    };
}

function buildKpDenseScoreOverrides(
    queryVector: Float32Array,
): ReadonlyMap<string, number> | undefined {
    if (!kpDenseStyleCorpus) {
        return undefined;
    }

    const scoreMap = new Map<string, number>();
    kpDenseStyleCorpus.kpids.forEach((kpid, index) => {
        scoreMap.set(
            kpid,
            dotVectorPair(queryVector, kpDenseStyleCorpus.vectors[index]),
        );
    });
    return scoreMap;
}

function mergeDenseScoreOverrides(
    ...maps: Array<ReadonlyMap<string, number> | undefined>
): ReadonlyMap<string, number> | undefined {
    const merged = new Map<string, number>();
    maps.forEach((map) => {
        map?.forEach((value, key) => {
            merged.set(key, value);
        });
    });
    return merged.size > 0 ? merged : undefined;
}

function formatOtDenseModeSlug(): string {
    if (!shouldUseOtDenseSlidingWindow()) {
        return "otdenseorig";
    }

    return `otdensewin${SAFE_OT_DENSE_WINDOW_SIZE}o${SAFE_OT_DENSE_WINDOW_OVERLAP}`;
}

function getSafeLexicalTypeMultipliers(): {
    Q: number;
    KP: number;
    OT: number;
} {
    return {
        Q: Number.isFinite(Q_LEXICAL_MULTIPLIER) ? Q_LEXICAL_MULTIPLIER : 1.5,
        KP: Number.isFinite(KP_LEXICAL_MULTIPLIER) ? KP_LEXICAL_MULTIPLIER : 1.2,
        OT: Number.isFinite(OT_LEXICAL_MULTIPLIER) ? OT_LEXICAL_MULTIPLIER : 1.0,
    };
}

function formatLexicalTypeMultiplierSlug(): string {
    const multipliers = getSafeLexicalTypeMultipliers();
    const isDefault =
        multipliers.Q === 1.5 &&
        multipliers.KP === 1.2 &&
        multipliers.OT === 1.0;
    if (isDefault) {
        return "";
    }

    const formatPart = (value: number): string =>
        value.toFixed(2).replace(".", "");
    return `_typelex-q${formatPart(multipliers.Q)}-kp${formatPart(multipliers.KP)}-ot${formatPart(multipliers.OT)}`;
}

function formatQueryStyleModeSlug(): string {
    if (QUERY_STYLE_MODE === "userized_v1") {
        return "_qstyleuserv1";
    }
    if (MINIMAL_BASELINE_MODE) {
        return "";
    }
    return QUERY_STYLE_MODE === "declarative" ? "_qstyledecl" : "";
}

function formatKpStyleModeSlug(): string {
    if (MINIMAL_BASELINE_MODE) {
        return "";
    }
    if (KP_STYLE_MODE === "question_wrap") {
        return "_kpstyleqwrap";
    }
    if (KP_STYLE_MODE === "truncate128") {
        return "_kpstyletruncate128";
    }
    return "";
}

function formatExperimentTrackSlug(): string {
    if (EXPERIMENT_TRACK === "minimal_first") {
        return "_track-minadd";
    }
    return "";
}

function formatMinimalAdditionsSlug(): string {
    if (!MINIMAL_BASELINE_MODE) {
        return "";
    }

    const additions: string[] = [];
    if (MINIMAL_ADD_YEAR) {
        additions.push("year");
    }
    if (MINIMAL_ADD_PHASE) {
        additions.push("phase");
    }
    if (MINIMAL_ADD_ASPECT) {
        additions.push("aspect");
    }
    if (MINIMAL_DISABLE_DOC_MULTI) {
        additions.push("nodocmulti");
    }

    return additions.length > 0 ? `_add-${additions.join("-")}` : "";
}

function formatQConfusionModeSlug(): string {
    if (Q_CONFUSION_MODE === "off") {
        return "";
    }
    const weightPart =
        Number.isFinite(Q_CONFUSION_WEIGHT) && Q_CONFUSION_WEIGHT > 0
            ? `-w${Q_CONFUSION_WEIGHT.toFixed(2).replace(".", "")}`
            : "";
    return `_qconf-${Q_CONFUSION_MODE}${weightPart}`;
}

function formatFusionModeSlug(): string {
    if (FUSION_MODE === "default") {
        return "";
    }
    return `_fusion-${FUSION_MODE.replace(/_/g, "-")}`;
}

function formatFixedComboWeightsSlug(): string {
    const entries = Object.entries(FIXED_COMBO_WEIGHTS);
    if (entries.length === 0) {
        return "";
    }

    const parts = entries
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([label, weights]) => {
            const comboSlug = label.toLowerCase().replace(/\+/g, "");
            const q = (weights.Q * 100).toFixed(0).padStart(2, "0");
            const kp = (weights.KP * 100).toFixed(0).padStart(2, "0");
            const ot = (weights.OT * 100).toFixed(0).padStart(2, "0");
            return `${comboSlug}-q${q}-kp${kp}-ot${ot}`;
        });

    return `_fw-${parts.join("__")}`;
}

function parseWeightSteps(raw: string): number[] {
    const values = raw
        .split(",")
        .map((item) => Number(item.trim()))
        .filter((item) => Number.isFinite(item) && item > 0);

    if (values.length === 0) {
        return [0.2, 0.5, 0.8];
    }

    return Array.from(new Set(values)).sort((a, b) => a - b);
}

function parseFixedComboWeights(raw: string): Record<string, WeightConfig> {
    if (!raw.trim()) {
        return {};
    }

    try {
        const parsed = JSON.parse(raw) as Record<
            string,
            Partial<Record<GranularityType, number>>
        >;
        const result: Record<string, WeightConfig> = {};
        Object.entries(parsed).forEach(([comboLabel, weights]) => {
            result[comboLabel] = {
                Q: Number.isFinite(weights.Q) ? Number(weights.Q) : 0,
                KP: Number.isFinite(weights.KP) ? Number(weights.KP) : 0,
                OT: Number.isFinite(weights.OT) ? Number(weights.OT) : 0,
            };
        });
        return result;
    } catch (error) {
        throw new Error(
            `Invalid SUASK_FIXED_COMBO_WEIGHTS JSON: ${
                error instanceof Error ? error.message : String(error)
            }`,
        );
    }
}

function createZeroWeights(): WeightConfig {
    return { Q: 0, KP: 0, OT: 0 };
}

function normalizeWeights(
    allowedTypes: readonly GranularityType[],
    values: readonly number[],
): WeightConfig {
    const result = createZeroWeights();
    const sum = values.reduce((acc, item) => acc + item, 0);
    const safeSum = sum > 0 ? sum : allowedTypes.length;

    allowedTypes.forEach((type, index) => {
        result[type] = values[index] / safeSum;
    });

    return result;
}

function normalizeProvidedWeights(
    allowedTypes: readonly GranularityType[],
    weights: WeightConfig,
): WeightConfig {
    return normalizeWeights(
        allowedTypes,
        allowedTypes.map((type) => {
            if (type === "Q") return weights.Q;
            if (type === "KP") return weights.KP;
            return weights.OT;
        }),
    );
}

function formatWeightKey(weights: WeightConfig): string {
    return [weights.Q, weights.KP, weights.OT]
        .map((item) => item.toFixed(4))
        .join("|");
}

function compareMetrics(a: Metrics, b: Metrics): number {
    if (a.hitAt1 !== b.hitAt1) return b.hitAt1 - a.hitAt1;
    if (a.mrr !== b.mrr) return b.mrr - a.mrr;
    if (a.hitAt5 !== b.hitAt5) return b.hitAt5 - a.hitAt5;
    if (a.hitAt3 !== b.hitAt3) return b.hitAt3 - a.hitAt3;
    return 0;
}

function countByDataset(testCases: readonly DatasetCase[]): Record<string, number> {
    const counts: Record<string, number> = {};
    testCases.forEach((item) => {
        counts[item.dataset] = (counts[item.dataset] || 0) + 1;
    });
    return counts;
}

function countMetadataTypes(
    metadata: readonly Metadata[],
): Record<GranularityType, number> {
    const counts: Record<GranularityType, number> = {
        Q: 0,
        KP: 0,
        OT: 0,
    };

    metadata.forEach((item) => {
        counts[item.type] += 1;
    });
    return counts;
}

function applyQPerDocCap(metadata: readonly Metadata[]): Metadata[] {
    if (!Number.isFinite(Q_PER_DOC_CAP) || Q_PER_DOC_CAP <= 0) {
        return [...metadata];
    }

    const qGroupsByDoc = new Map<
        string,
        Map<string, MetadataWithParentPkid[]>
    >();

    metadata.forEach((item) => {
        if (item.type !== "Q") {
            return;
        }

        const qItem = item as MetadataWithParentPkid;
        const docKey = item.parent_otid || item.id;
        const kpKey = qItem.parent_pkid || `__ungrouped__:${item.id}`;

        if (!qGroupsByDoc.has(docKey)) {
            qGroupsByDoc.set(docKey, new Map<string, MetadataWithParentPkid[]>());
        }
        const groupedByKp = qGroupsByDoc.get(docKey)!;
        if (!groupedByKp.has(kpKey)) {
            groupedByKp.set(kpKey, []);
        }
        groupedByKp.get(kpKey)!.push(qItem);
    });

    const keptQids = new Set<string>();

    qGroupsByDoc.forEach((groupedByKp) => {
        const buckets = Array.from(groupedByKp.values()).map((bucket) =>
            [...bucket].sort((left, right) => left.vector_index - right.vector_index),
        );
        let keptForDoc = 0;
        let depth = 0;

        while (keptForDoc < Q_PER_DOC_CAP) {
            let madeProgress = false;
            for (const bucket of buckets) {
                if (keptForDoc >= Q_PER_DOC_CAP) {
                    break;
                }
                const candidate = bucket[depth];
                if (!candidate) {
                    continue;
                }
                keptQids.add(candidate.id);
                keptForDoc += 1;
                madeProgress = true;
            }
            if (!madeProgress) {
                break;
            }
            depth += 1;
        }
    });

    return metadata.filter((item) => item.type !== "Q" || keptQids.has(item.id));
}

function applyKpPerDocCap(metadata: readonly Metadata[]): Metadata[] {
    if (!Number.isFinite(KP_PER_DOC_CAP) || KP_PER_DOC_CAP <= 0) {
        return [...metadata];
    }

    const kpByDoc = new Map<string, Metadata[]>();

    metadata.forEach((item) => {
        if (item.type !== "KP") {
            return;
        }

        const docKey = item.parent_otid || item.id;
        if (!kpByDoc.has(docKey)) {
            kpByDoc.set(docKey, []);
        }
        kpByDoc.get(docKey)!.push(item);
    });

    const keptKpids = new Set<string>();
    kpByDoc.forEach((items) => {
        [...items]
            .sort((left, right) => left.vector_index - right.vector_index)
            .slice(0, KP_PER_DOC_CAP)
            .forEach((item) => keptKpids.add(item.id));
    });

    return metadata.filter((item) => item.type !== "KP" || keptKpids.has(item.id));
}

function applyPerDocTypeCaps(metadata: readonly Metadata[]): Metadata[] {
    return applyKpPerDocCap(applyQPerDocCap(metadata));
}

function loadDatasets(datasetSources: readonly EvalDatasetSource[]): DatasetCase[] {
    return loadDatasetSources(datasetSources, {
        limitPerSource:
            Number.isFinite(LIMIT_PER_DATASET) && LIMIT_PER_DATASET > 0
                ? LIMIT_PER_DATASET
                : undefined,
    });
}

function loadKnowledgePointTexts(): Map<string, string> {
    const absolutePath = path.resolve(process.cwd(), KP_TEXTS_FILE);
    if (!fs.existsSync(absolutePath)) {
        return new Map<string, string>();
    }

    const raw = JSON.parse(
        fs.readFileSync(absolutePath, "utf-8"),
    ) as Array<{ pkid?: string; kp_text?: string }>;
    const result = new Map<string, string>();
    raw.forEach((item) => {
        if (item.pkid && item.kp_text) {
            result.set(item.pkid, item.kp_text);
        }
    });
    return result;
}

function uniqueNumbers(values: readonly number[]): number[] {
    return Array.from(new Set(values.filter((item) => Number.isFinite(item))));
}

function extractYears(text: string): number[] {
    return uniqueNumbers((text.match(/20\d{2}/g) || []).map(Number));
}

function extractMonths(text: string): number[] {
    return uniqueNumbers(
        Array.from(text.matchAll(/(^|[^\d])(\d{1,2})月/g))
            .map((match) => Number(match[2]))
            .filter((month) => month >= 1 && month <= 12),
    );
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

function parsePublishTimeParts(publishTime: string): {
    year?: number;
    month?: number;
} {
    const match = publishTime.match(/(20\d{2})-(\d{1,2})-(\d{1,2})/);
    if (!match) {
        return {};
    }

    return {
        year: Number(match[1]),
        month: Number(match[2]),
    };
}

function loadArticleTimeAnchors(): Map<string, ArticleTimeAnchor> {
    const absolutePath = path.resolve(process.cwd(), ARTICLE_TEXTS_FILE);
    if (!fs.existsSync(absolutePath)) {
        return new Map<string, ArticleTimeAnchor>();
    }

    const raw = JSON.parse(
        fs.readFileSync(absolutePath, "utf-8"),
    ) as Array<{
        otid?: string;
        ot_title?: string;
        publish_time?: string;
    }>;

    const result = new Map<string, ArticleTimeAnchor>();
    raw.forEach((item) => {
        if (!item.otid) {
            return;
        }

        const title = item.ot_title || "";
        const publishTime = item.publish_time || "";
        const titleYears = extractYears(title);
        const titleMonths = extractMonths(title);
        const publishParts = parsePublishTimeParts(publishTime);

        result.set(item.otid, {
            year: titleYears[0] ?? publishParts.year,
            month: titleMonths[0] ?? publishParts.month,
            title,
            publishTime,
            phaseAnchor: extractPhaseAnchor(title),
        });
    });

    return result;
}

function buildKnowledgePointFeatureFlags(kpText: string): KPFeatureFlags {
    const normalized = stripKpTimestampPrefix(kpText);

    return {
        hasTimeExpression:
            /\d{4}年|\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}|至\d{1,2}月\d{1,2}日/.test(
                normalized,
            ),
        hasDeadlineCue: /截止|截至|截止时间|截止日期/.test(normalized),
        hasArrivalCue: /到账|发放到账|到卡/.test(normalized),
        hasAnnouncementPeriodCue: /公示期|公示时间/.test(normalized),
        hasScheduleCue: /时间安排|安排如下|具体时间|时间为/.test(normalized),
        hasApplicationCue:
            /申请答辩|办理申请答辩手续|申请人需于|提交以下申请材料/.test(
                normalized,
            ),
        hasConditionCue:
            /条件包括|申请条件|需满足|须满足|符合以下条件|满足以下条件/.test(
                normalized,
            ),
        hasMaterialsCue: /材料|扫描件|电子版|原件|复印件/.test(normalized),
        hasEmailCue: /邮箱|mail|发送至|提交至.*邮箱/i.test(normalized),
        hasProcedureCue:
            /步骤包括|审核不通过|补充材料|重新提交|现场审核|再次上传|完成上述步骤|按以下流程|处理方式/.test(
                normalized,
            ),
        hasPublishCue: /发布|予以公示|公布/.test(normalized),
        hasBackgroundCue:
            /根据|按照|现将|现就|有关事项|以下简称|为进一步|为做好|通知如下/.test(
                normalized,
            ),
        hasReminderCue:
            /特别提醒|请注意|务必|资格审查|疫情防控|健康状况|行动轨迹/.test(
                normalized,
            ),
        hasPostOutcomeCue:
            /答辩通过后|通过者|审批|获得学位|领取学位证书|一周内|后续/.test(
                normalized,
            ),
        hasDistributionCue:
            /评审进度|分情况处理|第一批发放|分批发放/.test(normalized),
        hasThesisCue: /论文电子版|论文|学位论文/.test(normalized),
    };
}

function loadKnowledgePointFeatures(
    texts: ReadonlyMap<string, string>,
): Map<string, KPFeatureFlags> {
    const result = new Map<string, KPFeatureFlags>();
    texts.forEach((kpText, kpid) => {
        result.set(kpid, buildKnowledgePointFeatureFlags(kpText));
    });
    return result;
}

async function loadEngine() {
    const engine = await loadFrontendEvalEngine();
    extractor = engine.extractor;
    vocabMap = engine.vocabMap;
    metadataList = engine.metadataList;
    vectorMatrix = engine.vectorMatrix;
    dimensions = engine.dimensions;
    kpTextMap = loadKnowledgePointTexts();
    kpFeatureMap = loadKnowledgePointFeatures(kpTextMap);
    articleTimeAnchorMap = loadArticleTimeAnchors();
    otDenseSlidingWindowCorpus = shouldUseOtDenseSlidingWindow()
        ? await buildOtDenseSlidingWindowCorpus()
        : null;
    kpDenseStyleCorpus = shouldUseKpDenseStyleCorpus()
        ? await buildKpDenseStyleCorpus()
        : null;
}

async function buildQueryCache(
    testCases: DatasetCase[],
): Promise<QueryCacheItem[]> {
    if (!extractor) {
        throw new Error("Extractor not initialized");
    }

    const effectiveQueries = testCases.map((item) =>
        rewriteQueryTextByStyle(item.query),
    );

    const queryVectors = await embedFrontendQueries(
        extractor,
        effectiveQueries,
        dimensions,
        {
            batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
            onProgress: (done, total) => {
                console.log(`Embedded ${done} / ${total} queries`);
            },
        },
    );

    const queryCache: QueryCacheItem[] = [];
    testCases.forEach((testCase, index) => {
        const effectiveQueryText = effectiveQueries[index] || testCase.query;
        const referenceQueryText =
            testCase.source_query?.trim() || testCase.query;
        const queryIntent = parseQueryIntent(referenceQueryText);
        const queryMonths = extractMonths(referenceQueryText);
        const queryPhaseAnchor = extractPhaseAnchor(referenceQueryText);
        const queryWords = Array.from(
            new Set(fmmTokenize(effectiveQueryText, vocabMap)),
        );
        const querySparse = getQuerySparse(queryWords, vocabMap);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => vocabMap.get(year))
            .filter((item): item is number => item !== undefined);
        const denseScoreOverrides = MINIMAL_BASELINE_MODE
            ? undefined
            : mergeDenseScoreOverrides(
                  buildOtDenseScoreOverrides(queryVectors[index]),
                  buildKpDenseScoreOverrides(queryVectors[index]),
              );

        queryCache.push({
            testCase,
            effectiveQueryText,
            referenceQueryText,
            queryVector: queryVectors[index],
            queryIntent,
            queryMonths,
            queryPhaseAnchor,
            queryWords,
            querySparse,
            queryYearWordIds,
            denseScoreOverrides,
        });

        if (
            shouldUseOtDenseSlidingWindow() &&
            (index + 1 === testCases.length || (index + 1) % 16 === 0)
        ) {
            console.log(
                `Computed OT dense sliding overrides: ${index + 1} / ${testCases.length}`,
            );
        }
    });

    return queryCache;
}

function getRank(matches: readonly { otid: string }[], expectedOtid: string): number {
    const rankIndex = matches.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function getBestRankForOtidSet(
    matches: readonly { otid: string }[],
    acceptableOtids: readonly string[],
): number {
    if (acceptableOtids.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const acceptableOtidSet = new Set(acceptableOtids);
    const rankIndex = matches.findIndex((item) => acceptableOtidSet.has(item.otid));
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function getKpidRank(
    matches: readonly { otid: string; best_kpid?: string }[],
    expectedOtid: string,
    expectedKpids: readonly string[],
): number {
    if (expectedKpids.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const expectedKpidSet = new Set(expectedKpids);
    const rankIndex = matches.findIndex(
        (item) =>
            item.otid === expectedOtid &&
            item.best_kpid !== undefined &&
            expectedKpidSet.has(item.best_kpid),
    );
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function parseRequiredKpidGroups(
    groups?: string[][],
): string[][] {
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

function parseRequiredOtidGroups(
    groups?: string[][],
): string[][] {
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
    if (MINIMAL_BASELINE_MODE && MINIMAL_DISABLE_DOC_MULTI) {
        const acceptableOtids = testCase.expected_otid
            ? [testCase.expected_otid]
            : [];
        return {
            mode: "single_expected",
            acceptableOtids,
            requiredOtidGroups: [],
            minGroupsToCover: acceptableOtids.length > 0 ? 1 : 0,
        };
    }

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

    const inferredMode: OtidEvalMode =
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

function computeOtidCoverageState(
    requiredGroups: readonly string[][],
    minGroupsToCover: number,
    matches: readonly { otid: string }[],
): OtidCoverageState {
    if (requiredGroups.length === 0) {
        return {
            coverageDepth: Number.POSITIVE_INFINITY,
            coveredGroupsTop3: 0,
            coveredGroupsTop5: 0,
            coverageRatioTop5: 0,
            totalGroups: 0,
            minGroupsToCover: 0,
        };
    }

    const groupDepths = requiredGroups.map((group) => {
        const groupSet = new Set(group);
        const rankIndex = matches.findIndex((match) => groupSet.has(match.otid));
        return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
    });

    const coveredGroupsTop3 = groupDepths.filter((depth) => depth <= 3).length;
    const coveredGroupsTop5 = groupDepths.filter((depth) => depth <= 5).length;
    const sortedDepths = groupDepths
        .filter((depth) => Number.isFinite(depth))
        .sort((left, right) => left - right);
    const requiredCount = Math.max(
        1,
        Math.min(minGroupsToCover || requiredGroups.length, requiredGroups.length),
    );
    const coverageDepth =
        sortedDepths.length >= requiredCount
            ? sortedDepths[requiredCount - 1]
            : Number.POSITIVE_INFINITY;

    return {
        coverageDepth,
        coveredGroupsTop3,
        coveredGroupsTop5,
        coverageRatioTop5:
            requiredGroups.length > 0
                ? coveredGroupsTop5 / requiredGroups.length
                : 0,
        totalGroups: requiredGroups.length,
        minGroupsToCover: requiredCount,
    };
}

function resolveKpEvalTarget(testCase: DatasetCase): NormalizedKpEvalTarget {
    const inferredGroups = parseRequiredKpidGroups(testCase.required_kpid_groups);
    const supportGroups =
        inferredGroups.length > 0
            ? inferredGroups
            : testCase.support_pattern === "multi_kp" &&
                Array.isArray(testCase.support_kpids) &&
                testCase.support_kpids.length > 1
              ? Array.from(
                    new Set(
                        testCase.support_kpids.filter(
                            (item): item is string =>
                                typeof item === "string" && item.length > 0,
                        ),
                    ),
                ).map((kpid) => [kpid])
              : [];

    const preliminaryAnchorKpids = Array.from(
        new Set(
            [
                ...(testCase.expected_kpid ? [testCase.expected_kpid] : []),
                ...supportGroups.flat(),
            ].filter((item): item is string => typeof item === "string" && item.length > 0),
        ),
    );

    const inferredMode: KpEvalMode =
        testCase.kp_eval_mode ||
        (supportGroups.length > 1
            ? "aspect_coverage"
            : preliminaryAnchorKpids.length > 0
              ? "single_anchor"
              : "ot_only");

    const acceptableAnchorKpids = Array.from(
        new Set(
            (
                inferredMode === "single_anchor"
                    ? [
                          ...(testCase.expected_kpid ? [testCase.expected_kpid] : []),
                          ...(supportGroups.length === 1 ? supportGroups[0] : []),
                      ]
                    : [
                          ...(testCase.expected_kpid ? [testCase.expected_kpid] : []),
                          ...supportGroups.flat(),
                      ]
            ).filter((item): item is string => typeof item === "string" && item.length > 0),
        ),
    );

    const minGroupsCandidate = Number.isFinite(testCase.min_groups_to_cover)
        ? Math.max(1, Number(testCase.min_groups_to_cover))
        : supportGroups.length > 0
          ? supportGroups.length
          : acceptableAnchorKpids.length > 0
            ? 1
            : 0;

    return {
        mode: inferredMode,
        acceptableAnchorKpids,
        requiredKpidGroups:
            inferredMode === "aspect_coverage"
                ? supportGroups.length > 0
                    ? supportGroups
                    : acceptableAnchorKpids.map((kpid) => [kpid])
                : supportGroups,
        minGroupsToCover:
            inferredMode === "aspect_coverage"
                ? Math.min(
                      minGroupsCandidate,
                      supportGroups.length > 0
                          ? supportGroups.length
                          : Math.max(acceptableAnchorKpids.length, 1),
                  )
                : minGroupsCandidate,
    };
}

function stripKpTimestampPrefix(text: string): string {
    return text.replace(/^\[[^\]]+\]\s*/, "");
}

function rewriteQueryToDeclarative(text: string): string {
    let rewritten = text.trim();
    rewritten = rewritten.replace(/[？?]+$/g, "");
    rewritten = rewritten.replace(
        /^(请问一下|请问|我想知道|想知道|我想了解|想了解|我想问一下|我想问|想问一下|想问|咨询一下|咨询|请教一下|请教)\s*/g,
        "",
    );
    rewritten = rewritten.replace(/^如果我/, "");
    rewritten = rewritten.replace(/^我是([^，。]*)[，,]\s*/g, "");
    rewritten = rewritten.replace(/^我(准备|打算|想|要|需要)/, "");
    rewritten = rewritten.replace(/什么时候|何时|什么时间/g, "时间");
    rewritten = rewritten.replace(/在哪(里|儿)?|去哪里/g, "地点");
    rewritten = rewritten.replace(/哪些材料|什么材料/g, "材料要求");
    rewritten = rewritten.replace(/哪些条件|什么条件|条件是什么/g, "条件要求");
    rewritten = rewritten.replace(/分别是什么/g, "具体内容");
    rewritten = rewritten.replace(/有哪些/g, "相关内容");
    rewritten = rewritten.replace(/怎么(办|做|操作|申请|报名|提交|确认)|如何(办|做|操作|申请|报名|提交|确认)/g, "流程");
    rewritten = rewritten.replace(/怎么办/g, "处理方式");
    rewritten = rewritten.replace(/是多少/g, "数额");
    rewritten = rewritten.replace(/吗|呢|呀|么|嘛|吧/g, "");
    rewritten = rewritten.replace(/[，,；;！!]/g, " ");
    rewritten = rewritten.replace(/\s+/g, " ").trim();

    if (!rewritten) {
        return text.trim();
    }
    return rewritten;
}

function rewriteQueryToUserizedV1(text: string): string {
    let rewritten = text.trim();
    rewritten = rewritten.replace(/[？?]+$/g, "");
    rewritten = rewritten.replace(
        /^(请问一下|请问|我想知道|想知道|我想了解|想了解|我想问一下|我想问|想问一下|想问|咨询一下|咨询|请教一下|请教)\s*/g,
        "",
    );
    rewritten = rewritten.replace(/中山大学/g, "");
    rewritten = rewritten.replace(/推荐免试(攻读)?研究生/g, "推免");
    rewritten = rewritten.replace(/全国优秀大学生夏令营/g, "夏令营");
    rewritten = rewritten.replace(/硕士研究生/g, "硕士");
    rewritten = rewritten.replace(/博士研究生/g, "博士");
    rewritten = rewritten.replace(/招生章程|招生简章/g, "招生要求");
    rewritten = rewritten.replace(/招生学科专业目录/g, "专业目录");
    rewritten = rewritten.replace(/报名方式|申请方式|通过什么渠道申请/g, "怎么报名");
    rewritten = rewritten.replace(/申请材料|材料清单|提交材料/g, "要准备什么材料");
    rewritten = rewritten.replace(/资格条件|申请条件|报考条件|录取条件/g, "要什么条件");
    rewritten = rewritten.replace(/时间线|时间安排|时间节点/g, "时间怎么安排");
    rewritten = rewritten.replace(/完整流程|具体流程|流程包括哪些步骤|流程是什么/g, "流程怎么走");
    rewritten = rewritten.replace(/联系方式|咨询电话/g, "找谁");
    rewritten = rewritten.replace(/是否接收调剂/g, "能不能调剂");
    rewritten = rewritten.replace(/综合考核/g, "考核");
    rewritten = rewritten.replace(/复试录取方案|复试录取实施细则/g, "复试要求");
    rewritten = rewritten.replace(/什么时候|何时|什么时间/g, "什么时候");
    rewritten = rewritten.replace(/哪些材料|什么材料/g, "要准备什么材料");
    rewritten = rewritten.replace(/哪些条件|什么条件|条件是什么/g, "要什么条件");
    rewritten = rewritten.replace(/分别是什么|有哪些/g, "都有哪些");
    rewritten = rewritten.replace(
        /怎么(办|做|操作|申请|报名|提交|确认)|如何(办|做|操作|申请|报名|提交|确认)/g,
        "怎么弄",
    );
    rewritten = rewritten.replace(/^如果我/, "");
    rewritten = rewritten.replace(/^我是([^，。]*)[，,]\s*/g, "$1 ");
    rewritten = rewritten.replace(/^我(准备|打算|想|要|需要)/, "");
    rewritten = rewritten.replace(/[；;！!]/g, " ");
    rewritten = rewritten.replace(/[，,]/g, " ");
    rewritten = rewritten.replace(/\s+/g, " ").trim();

    if (!rewritten) {
        return text.trim();
    }
    return rewritten;
}

function rewriteQueryTextByStyle(text: string): string {
    if (QUERY_STYLE_MODE === "userized_v1") {
        return rewriteQueryToUserizedV1(text);
    }
    if (MINIMAL_BASELINE_MODE) {
        return text;
    }
    if (QUERY_STYLE_MODE === "declarative") {
        return rewriteQueryToDeclarative(text);
    }
    return text;
}

function rewriteKpTextToQuestionWrap(kpText: string): string {
    const normalized = stripKpTimestampPrefix(kpText).trim();
    const firstSentence = normalized
        .split(/[。！？]/)
        .map((item) => item.trim())
        .find((item) => item.length > 0) || normalized;
    let focus = firstSentence;
    if (focus.startsWith("根据") && focus.includes("，")) {
        focus = focus.slice(focus.indexOf("，") + 1).trim();
    }
    if (focus.length > 72) {
        focus = `${focus.slice(0, 72)}...`;
    }
    return `关于${focus}，具体要求是什么？`;
}

function rewriteKpTextByStyle(kpText: string): string {
    if (KP_STYLE_MODE === "question_wrap") {
        return rewriteKpTextToQuestionWrap(kpText);
    }
    return stripKpTimestampPrefix(kpText).trim();
}

function hasAnyKeyword(text: string, keywords: readonly string[]): boolean {
    return keywords.some((keyword) => text.includes(keyword));
}

function computeQueryWordBonus(
    queryWords: readonly string[],
    kpText: string,
): number {
    const filteredWords = queryWords.filter(
        (word) =>
            word.length >= 2 &&
            !QUERY_STOPWORDS.has(word) &&
            !/^\d+$/.test(word),
    );
    const hits = filteredWords.filter((word) => kpText.includes(word)).length;
    return Math.min(hits, 4) * 0.18;
}

function computeHeuristicKpBonus(
    item: QueryCacheItem,
    kpText: string,
): number {
    const query = item.referenceQueryText;
    const scope = item.testCase.query_scope || "";
    const normalized = stripKpTimestampPrefix(kpText);
    let bonus = computeQueryWordBonus(item.queryWords, normalized);

    const asksTime =
        /什么时候|何时|哪几天|几号|截止|到账|时间|公示期/.test(query) ||
        scope === "time_location";
    const asksCondition =
        /条件|满足|资格/.test(query) || scope === "eligibility_condition";
    const asksMaterials = /材料|扫描件|电子版|邮箱|mail/i.test(query);
    const asksProcedure = /怎么办|怎么处理|不通过|补交|补充|流程|步骤/.test(query);
    const asksAnnouncementPeriod = /公示期|哪几天/.test(query);

    if (asksTime) {
        if (hasAnyKeyword(normalized, ["到账", "截止", "公示期", "时间安排"])) {
            bonus += 0.9;
        }
        if (/\d{4}年|\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}|至\d{1,2}月\d{1,2}日/.test(normalized)) {
            bonus += 0.45;
        }
    }

    if (asksCondition) {
        if (hasAnyKeyword(normalized, ["条件包括", "申请答辩的条件包括", "需满足", "须满足", "申请条件"])) {
            bonus += 1.1;
        }
        if (hasAnyKeyword(normalized, ["通过者", "审批", "获得学位", "领取学位证书"])) {
            bonus -= 0.7;
        }
    }

    if (asksMaterials) {
        if (hasAnyKeyword(normalized, ["材料", "扫描件", "电子版", "邮箱", "mail"])) {
            bonus += 0.8;
        }
        if (normalized.includes("材料") && /邮箱|mail/i.test(normalized)) {
            bonus += 0.9;
        }
        if (!query.includes("论文") && hasAnyKeyword(normalized, ["论文电子版", "答辩通过后", "一周内"])) {
            bonus -= 1.0;
        }
    }

    if (asksProcedure) {
        if (
            hasAnyKeyword(normalized, [
                "步骤包括",
                "审核不通过",
                "补充材料",
                "重新提交",
                "现场审核",
                "再次上传",
                "完成上述步骤",
            ])
        ) {
            bonus += 1.0;
        }
        if (hasAnyKeyword(normalized, ["特别提醒", "资格审查", "疫情防控", "健康状况", "行动轨迹"])) {
            bonus -= 0.8;
        }
    }

    if (asksAnnouncementPeriod) {
        if (normalized.includes("公示期")) {
            bonus += 1.2;
        }
        if (hasAnyKeyword(normalized, ["发布", "予以公示"]) && !normalized.includes("公示期")) {
            bonus -= 0.6;
        }
    }

    if (/到账/.test(query)) {
        if (normalized.includes("到账")) {
            bonus += 1.0;
        }
        if (hasAnyKeyword(normalized, ["分情况处理", "评审进度", "第一批发放"])) {
            bonus -= 0.5;
        }
    }

    return bonus;
}

function computeFeatureHeuristicKpBonus(
    item: QueryCacheItem,
    features: KPFeatureFlags,
): number {
    const query = item.referenceQueryText;
    const scope = item.testCase.query_scope || "";
    let bonus = 0;

    const asksTime =
        /什么时候|何时|哪几天|几号|截止|到账|时间|公示期/.test(query) ||
        scope === "time_location";
    const asksCondition =
        /条件|满足|资格/.test(query) || scope === "eligibility_condition";
    const asksMaterials = /材料|扫描件|电子版|邮箱|mail/i.test(query);
    const asksProcedure = /怎么办|怎么处理|不通过|补交|补充|流程|步骤/.test(query);
    const asksAnnouncementPeriod = /公示期|哪几天/.test(query);
    const asksApplicationStage =
        /申请|报名|确认|提交/.test(query) &&
        !/通过后|答辩通过|审批后|获得学位/.test(query);

    if (asksTime) {
        if (
            features.hasArrivalCue ||
            features.hasDeadlineCue ||
            features.hasAnnouncementPeriodCue ||
            features.hasScheduleCue
        ) {
            bonus += 0.9;
        }
        if (features.hasTimeExpression) {
            bonus += 0.45;
        }
    }

    if (asksCondition) {
        if (features.hasConditionCue) {
            bonus += 1.1;
        }
        if (features.hasPostOutcomeCue) {
            bonus -= 0.7;
        }
    }

    if (asksMaterials) {
        if (features.hasMaterialsCue) {
            bonus += 0.8;
        }
        if (features.hasMaterialsCue && features.hasEmailCue) {
            bonus += 0.9;
        }
        if (/申请|答辩/.test(query) && features.hasApplicationCue) {
            bonus += 0.9;
        }
        if (
            !query.includes("论文") &&
            (features.hasPostOutcomeCue || features.hasThesisCue)
        ) {
            bonus -= 1.2;
        }
    }

    if (asksApplicationStage) {
        if (features.hasApplicationCue) {
            bonus += 1.1;
        }
        if (features.hasPostOutcomeCue) {
            bonus -= 1.1;
        }
    }

    if (asksProcedure) {
        if (features.hasProcedureCue) {
            bonus += 1.0;
        }
        if (features.hasReminderCue || features.hasBackgroundCue) {
            bonus -= 0.8;
        }
    }

    if (asksAnnouncementPeriod) {
        if (features.hasAnnouncementPeriodCue) {
            bonus += 1.2;
        }
        if (features.hasPublishCue && !features.hasAnnouncementPeriodCue) {
            bonus -= 0.6;
        }
    }

    if (/到账/.test(query)) {
        if (features.hasArrivalCue) {
            bonus += 1.0;
        }
        if (features.hasDistributionCue) {
            bonus -= 0.5;
        }
    }

    return bonus;
}

function getEffectiveKpCandidateRerankMode(
    enableMinimalAspect = false,
): KPCandidateRerankMode {
    if (MINIMAL_BASELINE_MODE && enableMinimalAspect) {
        return "feature_heuristic";
    }
    return KP_CANDIDATE_RERANK_MODE;
}

function getHeuristicKpSelection(
    item: QueryCacheItem,
    match: {
        best_kpid?: string;
        kp_candidates?: Array<{ kpid: string; score: number }>;
    },
): {
    bestKpid?: string;
    rawTopScore: number;
    heuristicTopScore: number;
} {
    const kpCandidates = match.kp_candidates || [];
    if (kpCandidates.length === 0) {
        return {
            bestKpid: match.best_kpid,
            rawTopScore: Number.NEGATIVE_INFINITY,
            heuristicTopScore: Number.NEGATIVE_INFINITY,
        };
    }

    let bestKpid = match.best_kpid;
    let bestScore = Number.NEGATIVE_INFINITY;
    const rawTopScore = kpCandidates[0]?.score ?? Number.NEGATIVE_INFINITY;
    const rerankMode = getEffectiveKpCandidateRerankMode();

    kpCandidates.forEach((candidate) => {
        let rerankedScore = candidate.score;

        if (rerankMode === "heuristic") {
            const kpText = kpTextMap.get(candidate.kpid);
            if (!kpText) {
                if (candidate.score > bestScore) {
                    bestScore = candidate.score;
                    bestKpid = candidate.kpid;
                }
                return;
            }

            rerankedScore += computeHeuristicKpBonus(item, kpText);
        } else if (rerankMode === "feature_heuristic") {
            const kpFeatures = kpFeatureMap.get(candidate.kpid);
            if (!kpFeatures) {
                if (candidate.score > bestScore) {
                    bestScore = candidate.score;
                    bestKpid = candidate.kpid;
                }
                return;
            }

            rerankedScore += computeFeatureHeuristicKpBonus(item, kpFeatures);
        }

        if (rerankedScore > bestScore) {
            bestScore = rerankedScore;
            bestKpid = candidate.kpid;
        }
    });

    return {
        bestKpid,
        rawTopScore,
        heuristicTopScore: bestScore,
    };
}

function rerankBestKpidForMatch(
    item: QueryCacheItem,
    match: {
        best_kpid?: string;
        kp_candidates?: Array<{ kpid: string; score: number }>;
    },
    options?: {
        enableMinimalAspect?: boolean;
    },
): string | undefined {
    const rerankMode = getEffectiveKpCandidateRerankMode(
        options?.enableMinimalAspect,
    );
    if (MINIMAL_BASELINE_MODE && !options?.enableMinimalAspect) {
        return match.best_kpid;
    }
    if (rerankMode === "none") {
        return match.best_kpid;
    }

    return getHeuristicKpSelection(item, match).bestKpid;
}

function getRerankedKpCandidates(
    item: QueryCacheItem,
    match: {
        kp_candidates?: Array<{ kpid: string; score: number }>;
    },
    options?: {
        enableMinimalAspect?: boolean;
    },
): RerankedKpCandidate[] {
    const kpCandidates = match.kp_candidates || [];
    const rerankMode = getEffectiveKpCandidateRerankMode(
        options?.enableMinimalAspect,
    );

    if (MINIMAL_BASELINE_MODE && !options?.enableMinimalAspect) {
        return kpCandidates
            .map((candidate) => ({
                kpid: candidate.kpid,
                rawScore: candidate.score,
                rerankedScore: candidate.score,
            }))
            .sort((left, right) => right.rerankedScore - left.rerankedScore);
    }

    return kpCandidates
        .map((candidate) => {
            let rerankedScore = candidate.score;

            if (rerankMode === "heuristic") {
                const kpText = kpTextMap.get(candidate.kpid);
                if (kpText) {
                    rerankedScore += computeHeuristicKpBonus(item, kpText);
                }
            } else if (rerankMode === "feature_heuristic") {
                const kpFeatures = kpFeatureMap.get(candidate.kpid);
                if (kpFeatures) {
                    rerankedScore += computeFeatureHeuristicKpBonus(item, kpFeatures);
                }
            }

            return {
                kpid: candidate.kpid,
                rawScore: candidate.score,
                rerankedScore,
            };
        })
        .sort((left, right) => right.rerankedScore - left.rerankedScore);
}

function computeCoveredGroupCount(
    requiredGroups: readonly string[][],
    candidates: readonly RerankedKpCandidate[],
    depth: number,
): number {
    if (requiredGroups.length === 0 || depth <= 0) {
        return 0;
    }

    const topKpidSet = new Set(
        candidates.slice(0, depth).map((candidate) => candidate.kpid),
    );

    return requiredGroups.filter((group) =>
        group.some((kpid) => topKpidSet.has(kpid)),
    ).length;
}

function computeSupportCoverageState(
    requiredGroups: readonly string[][],
    minGroupsToCover: number,
    candidates: readonly RerankedKpCandidate[],
): SupportCoverageState {
    if (requiredGroups.length === 0 || minGroupsToCover <= 0) {
        return {
            coverageDepth: Number.POSITIVE_INFINITY,
            coveredGroupsTop3: 0,
            coveredGroupsTop5: 0,
            coverageRatioTop5: 0,
            totalGroups: requiredGroups.length,
            minGroupsToCover,
        };
    }

    const maxDepth = Math.min(candidates.length, 5);
    let coverageDepth = Number.POSITIVE_INFINITY;

    for (let depth = 1; depth <= maxDepth; depth += 1) {
        const covered = computeCoveredGroupCount(requiredGroups, candidates, depth);
        if (covered >= minGroupsToCover) {
            coverageDepth = depth;
            break;
        }
    }

    const coveredGroupsTop3 = computeCoveredGroupCount(requiredGroups, candidates, 3);
    const coveredGroupsTop5 = computeCoveredGroupCount(requiredGroups, candidates, 5);

    return {
        coverageDepth,
        coveredGroupsTop3,
        coveredGroupsTop5,
        coverageRatioTop5:
            requiredGroups.length > 0 ? coveredGroupsTop5 / requiredGroups.length : 0,
        totalGroups: requiredGroups.length,
        minGroupsToCover,
    };
}

function rerankMatchesForDocumentMetrics(
    item: QueryCacheItem,
    matches: readonly EvalSearchMatch[],
): EvalSearchMatch[] {
    const shouldApplyMinimalPhaseAnchor =
        MINIMAL_BASELINE_MODE && MINIMAL_ADD_PHASE;
    if (MINIMAL_BASELINE_MODE && !shouldApplyMinimalPhaseAnchor) {
        return [...matches];
    }
    if (!shouldApplyMinimalPhaseAnchor && DOC_POST_RERANK_MODE === "none") {
        return [...matches];
    }

    const safeWeight =
        Number.isFinite(DOC_POST_RERANK_WEIGHT) && DOC_POST_RERANK_WEIGHT >= 0
            ? DOC_POST_RERANK_WEIGHT
            : 0.35;

    return matches
        .map((match) => {
            let delta = 0;

            if (shouldApplyMinimalPhaseAnchor) {
                delta += computePhaseAnchorDocDelta(item, match);
            }
            if (DOC_POST_RERANK_MODE === "kp_heuristic_delta") {
                const selection = getHeuristicKpSelection(item, match);
                delta =
                    Number.isFinite(selection.rawTopScore) &&
                    Number.isFinite(selection.heuristicTopScore)
                        ? Math.max(
                              0,
                              selection.heuristicTopScore - selection.rawTopScore,
                          )
                        : 0;
            } else if (DOC_POST_RERANK_MODE === "time_anchor") {
                delta = computeTimeAnchorDocDelta(item, match);
            }

            return {
                ...match,
                score: match.score + delta * safeWeight,
            };
        })
        .sort((a, b) => b.score - a.score);
}

function computeTimeAnchorDocDelta(
    item: QueryCacheItem,
    match: EvalSearchMatch,
): number {
    const queryYears = item.queryIntent.years;
    const queryMonths = item.queryMonths;
    if (queryYears.length === 0 && queryMonths.length === 0) {
        return 0;
    }

    const articleAnchor = articleTimeAnchorMap.get(match.otid);
    if (!articleAnchor) {
        return queryYears.length > 0 ? -0.2 : 0;
    }

    let delta = 0;

    if (queryYears.length > 0) {
        if (articleAnchor.year === undefined) {
            delta -= 0.2;
        } else if (queryYears.includes(articleAnchor.year)) {
            delta += 1.0;
        } else {
            delta -= 1.0;
        }
    }

    if (queryMonths.length > 0) {
        if (articleAnchor.month !== undefined) {
            if (queryMonths.includes(articleAnchor.month)) {
                delta += 0.6;
            } else {
                delta -= 0.6;
            }
        }
    }

    return delta;
}

function computePhaseAnchorDocDelta(
    item: QueryCacheItem,
    match: EvalSearchMatch,
): number {
    const queryPhase = item.queryPhaseAnchor;
    const hasExplicitPhase =
        queryPhase.half !== undefined ||
        queryPhase.batch !== undefined ||
        queryPhase.stages.length > 0;
    if (!hasExplicitPhase) {
        return 0;
    }

    const articleAnchor = articleTimeAnchorMap.get(match.otid);
    if (!articleAnchor) {
        return -0.15;
    }

    const articlePhase = articleAnchor.phaseAnchor;
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

    return Math.max(-1.8, Math.min(1.8, delta));
}

function formatDocPostRerankSlug(): string {
    if (DOC_POST_RERANK_MODE === "kp_heuristic_delta") {
        const safeWeight =
            Number.isFinite(DOC_POST_RERANK_WEIGHT) && DOC_POST_RERANK_WEIGHT >= 0
                ? DOC_POST_RERANK_WEIGHT
                : 0.35;
        return `kpdelta-w${safeWeight.toFixed(2).replace(".", "")}`;
    }

    if (DOC_POST_RERANK_MODE === "time_anchor") {
        const safeWeight =
            Number.isFinite(DOC_POST_RERANK_WEIGHT) && DOC_POST_RERANK_WEIGHT >= 0
                ? DOC_POST_RERANK_WEIGHT
                : 0.35;
        return `timeanchor-w${safeWeight.toFixed(2).replace(".", "")}`;
    }

    return "none";
}

function rerankMatchesForKpidMetrics(
    item: QueryCacheItem,
    matches: readonly EvalSearchMatch[],
    options?: {
        enableMinimalAspect?: boolean;
    },
): Array<{ otid: string; best_kpid?: string }> {
    if (MINIMAL_BASELINE_MODE) {
        return matches.map((match) => ({
            otid: match.otid,
            best_kpid: options?.enableMinimalAspect
                ? rerankBestKpidForMatch(item, match, options)
                : match.best_kpid,
        }));
    }
    return matches.map((match) => ({
        otid: match.otid,
        best_kpid: rerankBestKpidForMatch(item, match, options),
    }));
}

function shouldApplyMinimalAspectForCase(
    item: QueryCacheItem,
    docRank: number,
): boolean {
    return (
        MINIMAL_BASELINE_MODE &&
        MINIMAL_ADD_ASPECT &&
        item.testCase.support_pattern === "multi_kp" &&
        Number.isFinite(docRank) &&
        docRank === 1
    );
}

function rankToNullable(rank: number): number | null {
    return Number.isFinite(rank) ? rank : null;
}

function uniqueStrings(values: readonly string[]): string[] {
    return Array.from(new Set(values.filter((item) => item.length > 0)));
}

function buildFailureReasons(
    item: QueryCacheItem,
    docRank: number,
    kpidRank: number,
    otidEvalTarget: NormalizedOtidEvalTarget,
    otidCoverageState: OtidCoverageState,
    kpEvalTarget: NormalizedKpEvalTarget,
    supportCoverageState: SupportCoverageState,
): string[] {
    const reasons: string[] = [];

    if (otidEvalTarget.mode === "required_otid_groups") {
        if (!Number.isFinite(docRank) || docRank > 5) {
            reasons.push("必需文档组未能在前5内完成覆盖，优先表现为多文档召回或排序失败。");
        } else if (docRank > 1) {
            reasons.push("必需文档组进入前5但未能更早完成覆盖，存在多文档排序偏差。");
        }

        if (Number.isFinite(docRank) && docRank <= 5) {
            if (otidCoverageState.coveredGroupsTop5 > 0) {
                reasons.push("文档候选中已覆盖部分必需通知，但仍未形成完整文档链。");
            } else {
                reasons.push("文档候选中尚未覆盖任何必需通知组。");
            }
        }
    } else if (!Number.isFinite(docRank) || docRank > 5) {
        reasons.push("正确文档未进入前5，优先表现为文档级召回或排序失败。");
    } else if (docRank > 1) {
        reasons.push("正确文档进入前5但未到第1名，存在文档级排序偏差。");
    }

    if (kpEvalTarget.acceptableAnchorKpids.length > 0) {
        if (!Number.isFinite(kpidRank) || kpidRank > 5) {
            reasons.push("文档候选中未能稳定选出正确主证据 KP。");
        } else if (kpidRank > 1) {
            reasons.push("正确文档命中后，主证据 KP 选择仍不够准确。");
        }
    }

    if (kpEvalTarget.mode === "aspect_coverage") {
        if (Number.isFinite(docRank) && docRank <= 5) {
            if (
                !Number.isFinite(supportCoverageState.coverageDepth) ||
                supportCoverageState.coverageDepth > 5
            ) {
                if (supportCoverageState.coveredGroupsTop5 > 0) {
                    reasons.push("正确文档命中后，仅覆盖了部分同等重要的证据方面。");
                } else {
                    reasons.push("正确文档命中后，前5个 KP 候选仍未覆盖任何必需证据方面。");
                }
            }
        } else {
            reasons.push("多重点样本当前仍先受文档级召回限制，尚未进入方面覆盖判定阶段。");
        }
    }

    if (item.testCase.support_pattern === "multi_kp") {
        reasons.push("该样本依赖多条 KP 联合支撑，适合继续补 multi_kp 邻近问法。");
    }

    if (item.testCase.support_pattern === "ot_required") {
        reasons.push("该样本更依赖整篇通知上下文，适合补 ot_required 或 OT 邻近样本。");
    }

    return uniqueStrings(reasons);
}

function inferFailureRisk(
    item: QueryCacheItem,
    docRank: number,
    kpidRank: number,
    otidEvalTarget: NormalizedOtidEvalTarget,
    otidCoverageState: OtidCoverageState,
    kpEvalTarget: NormalizedKpEvalTarget,
    supportCoverageState: SupportCoverageState,
): string {
    if (
        otidEvalTarget.mode === "required_otid_groups" &&
        Number.isFinite(docRank) &&
        docRank <= 5 &&
        (!Number.isFinite(otidCoverageState.coverageDepth) ||
            otidCoverageState.coverageDepth > 5)
    ) {
        return otidCoverageState.coveredGroupsTop5 > 0
            ? "partial_multi_doc_coverage"
            : "multi_doc_chain_miss";
    }

    if (
        kpEvalTarget.mode === "aspect_coverage" &&
        Number.isFinite(docRank) &&
        docRank <= 5 &&
        (!Number.isFinite(supportCoverageState.coverageDepth) ||
            supportCoverageState.coverageDepth > 5)
    ) {
        return supportCoverageState.coveredGroupsTop5 > 0
            ? "partial_multi_kp_coverage"
            : "multi_kp_aspect_miss";
    }

    if (
        kpEvalTarget.acceptableAnchorKpids.length > 0 &&
        Number.isFinite(docRank) &&
        docRank <= 5 &&
        (!Number.isFinite(kpidRank) || kpidRank > 1)
    ) {
        return "best_kpid_confusion";
    }

    if (item.testCase.support_pattern === "multi_kp") {
        return "requires_multi_kp";
    }

    if (
        item.testCase.support_pattern === "ot_required" ||
        item.testCase.preferred_granularity === "OT"
    ) {
        return "needs_ot_context";
    }

    if (!Number.isFinite(docRank) || docRank > 5) {
        return "document_miss";
    }

    if (docRank > 1) {
        return "document_rank_bias";
    }

    return "none";
}

function shouldKeepCaseDetail(
    detail: CaseDetail,
): boolean {
    if (!BAD_CASE_FAILURES_ONLY) {
        return true;
    }

    if (!detail.docHitAt1) {
        return true;
    }

    if (detail.acceptable_anchor_kpids?.length && !detail.kpidHitAt1) {
        return true;
    }

    if (detail.kp_eval_mode === "aspect_coverage" && !detail.supportFullCoverTop5) {
        return true;
    }

    return false;
}

function collectCaseDetails(
    queryCache: readonly QueryCacheItem[],
    filteredMetadata: readonly Metadata[],
    bm25Stats: BM25Stats,
    weights: WeightConfig,
): CaseDetail[] {
    if (!vectorMatrix) {
        throw new Error("Vector matrix not initialized");
    }

    const topMatchLimit =
        Number.isFinite(BAD_CASE_TOP_MATCHES) && BAD_CASE_TOP_MATCHES > 0
            ? BAD_CASE_TOP_MATCHES
            : 5;

    const details = queryCache.map((item) => {
        const otidEvalTarget = resolveOtidEvalTarget(item.testCase);
        const kpEvalTarget = resolveKpEvalTarget(item.testCase);
        const result = searchAndRank({
            queryVector: item.queryVector,
            querySparse: item.querySparse,
            queryYearWordIds: item.queryYearWordIds,
            queryIntent: item.queryIntent,
            queryScopeHint: item.testCase.query_scope,
            metadata: filteredMetadata as Metadata[],
            vectorMatrix,
            dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats,
            weights,
            topHybridLimit:
                Number.isFinite(TOP_HYBRID_LIMIT) && TOP_HYBRID_LIMIT > 0
                    ? TOP_HYBRID_LIMIT
                    : undefined,
            kpAggregationMode: KP_AGGREGATION_MODE,
            kpTopN: Number.isFinite(KP_TOP_N) && KP_TOP_N > 0 ? KP_TOP_N : undefined,
            kpTailWeight:
                Number.isFinite(KP_TAIL_WEIGHT) && KP_TAIL_WEIGHT >= 0
                    ? KP_TAIL_WEIGHT
                    : undefined,
            fusionMode: FUSION_MODE,
            lexicalBonusMode: LEXICAL_BONUS_MODE,
            qLexicalMultiplier: Number.isFinite(Q_LEXICAL_MULTIPLIER)
                ? Q_LEXICAL_MULTIPLIER
                : undefined,
            kpLexicalMultiplier: Number.isFinite(KP_LEXICAL_MULTIPLIER)
                ? KP_LEXICAL_MULTIPLIER
                : undefined,
            otLexicalMultiplier: Number.isFinite(OT_LEXICAL_MULTIPLIER)
                ? OT_LEXICAL_MULTIPLIER
                : undefined,
            kpRoleRerankMode: ONLINE_KP_ROLE_RERANK_MODE,
            kpRoleDocWeight:
                Number.isFinite(ONLINE_KP_ROLE_DOC_WEIGHT)
                && ONLINE_KP_ROLE_DOC_WEIGHT >= 0
                    ? ONLINE_KP_ROLE_DOC_WEIGHT
                    : undefined,
            denseScoreOverrides: item.denseScoreOverrides,
            qConfusionMode: Q_CONFUSION_MODE,
            qConfusionWeight: Number.isFinite(Q_CONFUSION_WEIGHT)
                ? Q_CONFUSION_WEIGHT
                : undefined,
            enableExplicitYearFilter: !MINIMAL_BASELINE_MODE || MINIMAL_ADD_YEAR,
            minimalMode: MINIMAL_BASELINE_MODE,
        });
        const docRerankedMatches = rerankMatchesForDocumentMetrics(
            item,
            result.matches,
        );
        const weakDocRerankedMatches =
            docRerankedMatches.length === 0 && result.weakMatches.length > 0
                ? rerankMatchesForDocumentMetrics(item, result.weakMatches)
                : [];
        const exportDocMatches =
            docRerankedMatches.length > 0
                ? docRerankedMatches
                : weakDocRerankedMatches;
        const otidCoverageState =
            otidEvalTarget.mode === "required_otid_groups"
                ? computeOtidCoverageState(
                      otidEvalTarget.requiredOtidGroups,
                      otidEvalTarget.minGroupsToCover,
                      docRerankedMatches,
                  )
                : {
                      coverageDepth: getBestRankForOtidSet(
                          docRerankedMatches,
                          otidEvalTarget.acceptableOtids,
                      ),
                      coveredGroupsTop3: 0,
                      coveredGroupsTop5: 0,
                      coverageRatioTop5: 0,
                      totalGroups: 0,
                      minGroupsToCover: 0,
                  };
        const docRank = otidCoverageState.coverageDepth;
        const enableMinimalAspect = shouldApplyMinimalAspectForCase(
            item,
            docRank,
        );
        const rerankedMatches = rerankMatchesForKpidMetrics(
            item,
            docRerankedMatches,
            { enableMinimalAspect },
        );
        const exportRerankedMatches =
            exportDocMatches === docRerankedMatches
                ? rerankedMatches
                : rerankMatchesForKpidMetrics(item, exportDocMatches, {
                      enableMinimalAspect,
                  });
        const kpidRank = getKpidRank(
            rerankedMatches,
            item.testCase.expected_otid,
            kpEvalTarget.acceptableAnchorKpids,
        );
        const expectedDocMatch = docRerankedMatches.find(
            (match) => match.otid === item.testCase.expected_otid,
        );
        const expectedDocTopKpCandidates = expectedDocMatch
            ? getRerankedKpCandidates(item, expectedDocMatch, {
                  enableMinimalAspect,
              })
            : [];
        const exportExpectedDocMatch =
            expectedDocMatch ||
            exportDocMatches.find(
                (match) => match.otid === item.testCase.expected_otid,
            );
        const exportExpectedDocTopKpCandidates =
            exportExpectedDocMatch && exportExpectedDocMatch !== expectedDocMatch
                ? getRerankedKpCandidates(item, exportExpectedDocMatch, {
                      enableMinimalAspect,
                  })
                : expectedDocTopKpCandidates;
        const supportCoverageState =
            kpEvalTarget.mode === "aspect_coverage"
                ? computeSupportCoverageState(
                      kpEvalTarget.requiredKpidGroups,
                      kpEvalTarget.minGroupsToCover,
                      expectedDocTopKpCandidates,
                  )
                : {
                      coverageDepth: Number.POSITIVE_INFINITY,
                      coveredGroupsTop3: 0,
                      coveredGroupsTop5: 0,
                      coverageRatioTop5: 0,
                      totalGroups: 0,
                      minGroupsToCover: 0,
                  };
        const failureReasons = buildFailureReasons(
            item,
            docRank,
            kpidRank,
            otidEvalTarget,
            otidCoverageState,
            kpEvalTarget,
            supportCoverageState,
        );
        const docFullCoverTop5 =
            otidEvalTarget.mode === "required_otid_groups" &&
            Number.isFinite(otidCoverageState.coverageDepth) &&
            otidCoverageState.coverageDepth <= 5;
        const docPartialCoverTop5 =
            otidEvalTarget.mode === "required_otid_groups" &&
            otidCoverageState.coveredGroupsTop5 > 0;
        const supportFullCoverTop5 =
            Number.isFinite(docRank) &&
            docRank <= 5 &&
            Number.isFinite(supportCoverageState.coverageDepth) &&
            supportCoverageState.coverageDepth <= 5;
        const supportPartialCoverTop5 =
            Number.isFinite(docRank) &&
            docRank <= 5 &&
            supportCoverageState.coveredGroupsTop5 > 0;

        const detail: CaseDetail = {
            query: item.testCase.query,
            expected_otid: item.testCase.expected_otid,
            otid_eval_mode: otidEvalTarget.mode,
            acceptable_otids: otidEvalTarget.acceptableOtids,
            required_otid_groups: otidEvalTarget.requiredOtidGroups,
            min_otid_groups_to_cover: otidEvalTarget.minGroupsToCover,
            expected_kpid: item.testCase.expected_kpid,
            kp_eval_mode: kpEvalTarget.mode,
            acceptable_anchor_kpids: kpEvalTarget.acceptableAnchorKpids,
            required_kpid_groups: kpEvalTarget.requiredKpidGroups,
            min_groups_to_cover: kpEvalTarget.minGroupsToCover,
            dataset: item.testCase.dataset,
            query_type: item.testCase.query_type,
            query_scope: item.testCase.query_scope,
            preferred_granularity: item.testCase.preferred_granularity,
            support_pattern: item.testCase.support_pattern,
            granularity_sensitive: item.testCase.granularity_sensitive,
            theme_family: item.testCase.theme_family,
            source_dataset: item.testCase.source_dataset,
            source_seed_id: item.testCase.source_seed_id,
            challenge_tags: item.testCase.challenge_tags,
            notes: item.testCase.notes,
            docRank: rankToNullable(docRank),
            kpidRank: rankToNullable(kpidRank),
            supportCoverageDepth: rankToNullable(
                supportCoverageState.coverageDepth,
            ),
            docCoverageDepth: rankToNullable(otidCoverageState.coverageDepth),
            docGroupsCoveredTop5: otidCoverageState.coveredGroupsTop5,
            docGroupsRequired: otidCoverageState.totalGroups,
            docFullCoverTop5,
            docPartialCoverTop5,
            supportGroupsCoveredTop5: supportCoverageState.coveredGroupsTop5,
            supportGroupsRequired: supportCoverageState.totalGroups,
            supportFullCoverTop5,
            supportPartialCoverTop5,
            docHitAt1: docRank === 1,
            docHitAt5: docRank <= 5,
            kpidHitAt1: kpidRank === 1,
            kpidHitAt5: kpidRank <= 5,
            failure_risk: inferFailureRisk(
                item,
                docRank,
                kpidRank,
                otidEvalTarget,
                otidCoverageState,
                kpEvalTarget,
                supportCoverageState,
            ),
            failure_reasons: failureReasons,
            topDocMatchesSource:
                docRerankedMatches.length > 0
                    ? "matches"
                    : exportDocMatches.length > 0
                      ? "weak_matches"
                      : "none",
            topDocMatches: exportDocMatches
                .slice(0, topMatchLimit)
                .map((match, index) => ({
                    rank: index + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                })),
            topKpidMatches: exportRerankedMatches
                .slice(0, topMatchLimit)
                .map((match, index) => ({
                    rank: index + 1,
                    otid: match.otid,
                    best_kpid: match.best_kpid,
                })),
            expectedDocTopKpCandidates: exportExpectedDocTopKpCandidates
                .slice(0, topMatchLimit)
                .map((candidate, index) => ({
                    rank: index + 1,
                    kpid: candidate.kpid,
                    rawScore: candidate.rawScore,
                    rerankedScore: candidate.rerankedScore,
                })),
        };

        return detail;
    });

    return details.filter(shouldKeepCaseDetail);
}

function evaluateQueryCache(
    queryCache: readonly QueryCacheItem[],
    filteredMetadata: readonly Metadata[],
    bm25Stats: BM25Stats,
    weights: WeightConfig,
): MetricsBundle {
    if (!vectorMatrix) {
        throw new Error("Vector matrix not initialized");
    }

    const metricsSeed: Record<
        string,
        {
            total: number;
            hitAt1: number;
            hitAt3: number;
            hitAt5: number;
            reciprocalRankSum: number;
        }
    > = {};
    const kpidMetricsSeed: Record<
        string,
        {
            applicable: number;
            hitAt1: number;
            hitAt3: number;
            hitAt5: number;
            reciprocalRankSum: number;
            docHitAt1Total: number;
            docHitAt1CorrectKpid: number;
            docHitAt1WrongKpid: number;
            docHitAt5Total: number;
            docHitAt5CorrectKpid: number;
            docHitAt5WrongKpid: number;
        }
    > = {};
    const supportCoverageMetricsSeed: Record<
        string,
        {
            applicable: number;
            docHitAt1Total: number;
            docHitAt5Total: number;
            docHitAt1FullCoverAt3: number;
            docHitAt1FullCoverAt5: number;
            docHitAt5FullCoverAt5: number;
            partialCoverAt5: number;
            coveredGroupsTop5Sum: number;
            coverageRatioTop5Sum: number;
        }
    > = {};

    const combinedSeed = {
        total: 0,
        hitAt1: 0,
        hitAt3: 0,
        hitAt5: 0,
        reciprocalRankSum: 0,
    };
    const kpidCombinedSeed = {
        applicable: 0,
        hitAt1: 0,
        hitAt3: 0,
        hitAt5: 0,
        reciprocalRankSum: 0,
        docHitAt1Total: 0,
        docHitAt1CorrectKpid: 0,
        docHitAt1WrongKpid: 0,
        docHitAt5Total: 0,
        docHitAt5CorrectKpid: 0,
        docHitAt5WrongKpid: 0,
    };
    const supportCoverageCombinedSeed = {
        applicable: 0,
        docHitAt1Total: 0,
        docHitAt5Total: 0,
        docHitAt1FullCoverAt3: 0,
        docHitAt1FullCoverAt5: 0,
        docHitAt5FullCoverAt5: 0,
        partialCoverAt5: 0,
        coveredGroupsTop5Sum: 0,
        coverageRatioTop5Sum: 0,
    };

    queryCache.forEach((item) => {
        const otidEvalTarget = resolveOtidEvalTarget(item.testCase);
        const kpEvalTarget = resolveKpEvalTarget(item.testCase);
        const result = searchAndRank({
            queryVector: item.queryVector,
            querySparse: item.querySparse,
            queryYearWordIds: item.queryYearWordIds,
            queryIntent: item.queryIntent,
            queryScopeHint: item.testCase.query_scope,
            metadata: filteredMetadata as Metadata[],
            vectorMatrix,
            dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats,
            weights,
            topHybridLimit:
                Number.isFinite(TOP_HYBRID_LIMIT) && TOP_HYBRID_LIMIT > 0
                    ? TOP_HYBRID_LIMIT
                    : undefined,
            kpAggregationMode: KP_AGGREGATION_MODE,
            kpTopN: Number.isFinite(KP_TOP_N) && KP_TOP_N > 0 ? KP_TOP_N : undefined,
            kpTailWeight:
                Number.isFinite(KP_TAIL_WEIGHT) && KP_TAIL_WEIGHT >= 0
                    ? KP_TAIL_WEIGHT
                    : undefined,
            fusionMode: FUSION_MODE,
            lexicalBonusMode: LEXICAL_BONUS_MODE,
            qLexicalMultiplier: Number.isFinite(Q_LEXICAL_MULTIPLIER)
                ? Q_LEXICAL_MULTIPLIER
                : undefined,
            kpLexicalMultiplier: Number.isFinite(KP_LEXICAL_MULTIPLIER)
                ? KP_LEXICAL_MULTIPLIER
                : undefined,
            otLexicalMultiplier: Number.isFinite(OT_LEXICAL_MULTIPLIER)
                ? OT_LEXICAL_MULTIPLIER
                : undefined,
            kpRoleRerankMode: ONLINE_KP_ROLE_RERANK_MODE,
            kpRoleDocWeight:
                Number.isFinite(ONLINE_KP_ROLE_DOC_WEIGHT)
                && ONLINE_KP_ROLE_DOC_WEIGHT >= 0
                    ? ONLINE_KP_ROLE_DOC_WEIGHT
                    : undefined,
            denseScoreOverrides: item.denseScoreOverrides,
            qConfusionMode: Q_CONFUSION_MODE,
            qConfusionWeight: Number.isFinite(Q_CONFUSION_WEIGHT)
                ? Q_CONFUSION_WEIGHT
                : undefined,
            enableExplicitYearFilter: !MINIMAL_BASELINE_MODE || MINIMAL_ADD_YEAR,
            minimalMode: MINIMAL_BASELINE_MODE,
        });
        const docRerankedMatches = rerankMatchesForDocumentMetrics(
            item,
            result.matches,
        );
        const otidCoverageState =
            otidEvalTarget.mode === "required_otid_groups"
                ? computeOtidCoverageState(
                      otidEvalTarget.requiredOtidGroups,
                      otidEvalTarget.minGroupsToCover,
                      docRerankedMatches,
                  )
                : {
                      coverageDepth: getBestRankForOtidSet(
                          docRerankedMatches,
                          otidEvalTarget.acceptableOtids,
                      ),
                      coveredGroupsTop3: 0,
                      coveredGroupsTop5: 0,
                      coverageRatioTop5: 0,
                      totalGroups: 0,
                      minGroupsToCover: 0,
                  };
        const rank = otidCoverageState.coverageDepth;
        const enableMinimalAspect = shouldApplyMinimalAspectForCase(
            item,
            rank,
        );
        const rerankedMatches = rerankMatchesForKpidMetrics(
            item,
            docRerankedMatches,
            { enableMinimalAspect },
        );
        const kpidRank = getKpidRank(
            rerankedMatches,
            item.testCase.expected_otid,
            kpEvalTarget.acceptableAnchorKpids,
        );
        const expectedDocMatch = docRerankedMatches.find(
            (match) => match.otid === item.testCase.expected_otid,
        );
        const expectedDocTopKpCandidates = expectedDocMatch
            ? getRerankedKpCandidates(item, expectedDocMatch, {
                  enableMinimalAspect,
              })
            : [];
        const supportCoverageState =
            kpEvalTarget.mode === "aspect_coverage"
                ? computeSupportCoverageState(
                      kpEvalTarget.requiredKpidGroups,
                      kpEvalTarget.minGroupsToCover,
                      expectedDocTopKpCandidates,
                  )
                : {
                      coverageDepth: Number.POSITIVE_INFINITY,
                      coveredGroupsTop3: 0,
                      coveredGroupsTop5: 0,
                      coverageRatioTop5: 0,
                      totalGroups: 0,
                      minGroupsToCover: 0,
                  };
        const datasetSeed =
            metricsSeed[item.testCase.dataset] ||
            (metricsSeed[item.testCase.dataset] = {
                total: 0,
                hitAt1: 0,
                hitAt3: 0,
                hitAt5: 0,
                reciprocalRankSum: 0,
            });
        const datasetKpidSeed =
            kpidMetricsSeed[item.testCase.dataset] ||
            (kpidMetricsSeed[item.testCase.dataset] = {
                applicable: 0,
                hitAt1: 0,
                hitAt3: 0,
                hitAt5: 0,
                reciprocalRankSum: 0,
                docHitAt1Total: 0,
                docHitAt1CorrectKpid: 0,
                docHitAt1WrongKpid: 0,
                docHitAt5Total: 0,
                docHitAt5CorrectKpid: 0,
                docHitAt5WrongKpid: 0,
            });
        const datasetSupportCoverageSeed =
            supportCoverageMetricsSeed[item.testCase.dataset] ||
            (supportCoverageMetricsSeed[item.testCase.dataset] = {
                applicable: 0,
                docHitAt1Total: 0,
                docHitAt5Total: 0,
                docHitAt1FullCoverAt3: 0,
                docHitAt1FullCoverAt5: 0,
                docHitAt5FullCoverAt5: 0,
                partialCoverAt5: 0,
                coveredGroupsTop5Sum: 0,
                coverageRatioTop5Sum: 0,
            });

        const targets = [datasetSeed, combinedSeed];
        targets.forEach((target) => {
            target.total += 1;
            if (rank === 1) target.hitAt1 += 1;
            if (rank <= 3) target.hitAt3 += 1;
            if (rank <= 5) target.hitAt5 += 1;
            if (Number.isFinite(rank)) {
                target.reciprocalRankSum += 1 / rank;
            }
        });

        if (kpEvalTarget.acceptableAnchorKpids.length > 0) {
            const kpidTargets = [datasetKpidSeed, kpidCombinedSeed];
            kpidTargets.forEach((target) => {
                target.applicable += 1;
                if (kpidRank === 1) target.hitAt1 += 1;
                if (kpidRank <= 3) target.hitAt3 += 1;
                if (kpidRank <= 5) target.hitAt5 += 1;
                if (Number.isFinite(kpidRank)) {
                    target.reciprocalRankSum += 1 / kpidRank;
                }

                if (rank === 1) {
                    target.docHitAt1Total += 1;
                    if (kpidRank === 1) target.docHitAt1CorrectKpid += 1;
                    else target.docHitAt1WrongKpid += 1;
                }

                if (rank <= 5) {
                    target.docHitAt5Total += 1;
                    if (kpidRank <= 5) target.docHitAt5CorrectKpid += 1;
                    else target.docHitAt5WrongKpid += 1;
                }
            });
        }

        if (
            kpEvalTarget.mode === "aspect_coverage" &&
            kpEvalTarget.requiredKpidGroups.length > 0
        ) {
            const supportTargets = [
                datasetSupportCoverageSeed,
                supportCoverageCombinedSeed,
            ];
            supportTargets.forEach((target) => {
                target.applicable += 1;

                const docHitAt1 = rank === 1;
                const docHitAt5 = rank <= 5;
                const fullCoverAt3 =
                    docHitAt1 &&
                    Number.isFinite(supportCoverageState.coverageDepth) &&
                    supportCoverageState.coverageDepth <= 3;
                const fullCoverAt5 =
                    docHitAt5 &&
                    Number.isFinite(supportCoverageState.coverageDepth) &&
                    supportCoverageState.coverageDepth <= 5;
                const partialCoverAt5 =
                    docHitAt5 && supportCoverageState.coveredGroupsTop5 > 0;

                if (docHitAt1) {
                    target.docHitAt1Total += 1;
                    if (fullCoverAt3) target.docHitAt1FullCoverAt3 += 1;
                    if (fullCoverAt5) target.docHitAt1FullCoverAt5 += 1;
                }

                if (docHitAt5) {
                    target.docHitAt5Total += 1;
                    if (fullCoverAt5) target.docHitAt5FullCoverAt5 += 1;
                    if (partialCoverAt5) target.partialCoverAt5 += 1;
                    target.coveredGroupsTop5Sum += supportCoverageState.coveredGroupsTop5;
                    target.coverageRatioTop5Sum +=
                        supportCoverageState.coverageRatioTop5;
                }
            });
        }
    });

    const metricsByDataset: Record<string, Metrics> = {};
    Object.entries(metricsSeed).forEach(([dataset, seed]) => {
        metricsByDataset[dataset] = {
            total: seed.total,
            hitAt1: (seed.hitAt1 / seed.total) * 100,
            hitAt3: (seed.hitAt3 / seed.total) * 100,
            hitAt5: (seed.hitAt5 / seed.total) * 100,
            mrr: seed.reciprocalRankSum / seed.total,
        };
    });
    const kpidMetricsByDataset: Record<string, KpidMetrics> = {};
    Object.entries(kpidMetricsSeed).forEach(([dataset, seed]) => {
        const safeApplicable = seed.applicable || 1;
        kpidMetricsByDataset[dataset] = {
            applicable: seed.applicable,
            hitAt1: seed.applicable ? (seed.hitAt1 / safeApplicable) * 100 : 0,
            hitAt3: seed.applicable ? (seed.hitAt3 / safeApplicable) * 100 : 0,
            hitAt5: seed.applicable ? (seed.hitAt5 / safeApplicable) * 100 : 0,
            mrr: seed.applicable ? seed.reciprocalRankSum / safeApplicable : 0,
            docHitAt1Total: seed.docHitAt1Total,
            docHitAt1CorrectKpid: seed.docHitAt1CorrectKpid,
            docHitAt1WrongKpid: seed.docHitAt1WrongKpid,
            docHitAt5Total: seed.docHitAt5Total,
            docHitAt5CorrectKpid: seed.docHitAt5CorrectKpid,
            docHitAt5WrongKpid: seed.docHitAt5WrongKpid,
        };
    });
    const supportCoverageMetricsByDataset: Record<string, SupportCoverageMetrics> =
        {};
    Object.entries(supportCoverageMetricsSeed).forEach(([dataset, seed]) => {
        const safeApplicable = seed.applicable || 1;
        supportCoverageMetricsByDataset[dataset] = {
            applicable: seed.applicable,
            docHitAt1Total: seed.docHitAt1Total,
            docHitAt5Total: seed.docHitAt5Total,
            docHitAt1FullCoverAt3: seed.applicable
                ? (seed.docHitAt1FullCoverAt3 / safeApplicable) * 100
                : 0,
            docHitAt1FullCoverAt5: seed.applicable
                ? (seed.docHitAt1FullCoverAt5 / safeApplicable) * 100
                : 0,
            docHitAt5FullCoverAt5: seed.applicable
                ? (seed.docHitAt5FullCoverAt5 / safeApplicable) * 100
                : 0,
            partialCoverAt5: seed.applicable
                ? (seed.partialCoverAt5 / safeApplicable) * 100
                : 0,
            avgCoveredGroupsAt5: seed.applicable
                ? seed.coveredGroupsTop5Sum / safeApplicable
                : 0,
            avgCoverageRatioAt5: seed.applicable
                ? seed.coverageRatioTop5Sum / safeApplicable
                : 0,
        };
    });
    const safeCombinedApplicable = kpidCombinedSeed.applicable || 1;
    const safeSupportCombinedApplicable = supportCoverageCombinedSeed.applicable || 1;

    return {
        metricsByDataset,
        combined: {
            total: combinedSeed.total,
            hitAt1: (combinedSeed.hitAt1 / combinedSeed.total) * 100,
            hitAt3: (combinedSeed.hitAt3 / combinedSeed.total) * 100,
            hitAt5: (combinedSeed.hitAt5 / combinedSeed.total) * 100,
            mrr: combinedSeed.reciprocalRankSum / combinedSeed.total,
        },
        kpidMetricsByDataset,
        kpidCombined: {
            applicable: kpidCombinedSeed.applicable,
            hitAt1: kpidCombinedSeed.applicable
                ? (kpidCombinedSeed.hitAt1 / safeCombinedApplicable) * 100
                : 0,
            hitAt3: kpidCombinedSeed.applicable
                ? (kpidCombinedSeed.hitAt3 / safeCombinedApplicable) * 100
                : 0,
            hitAt5: kpidCombinedSeed.applicable
                ? (kpidCombinedSeed.hitAt5 / safeCombinedApplicable) * 100
                : 0,
            mrr: kpidCombinedSeed.applicable
                ? kpidCombinedSeed.reciprocalRankSum / safeCombinedApplicable
                : 0,
            docHitAt1Total: kpidCombinedSeed.docHitAt1Total,
            docHitAt1CorrectKpid: kpidCombinedSeed.docHitAt1CorrectKpid,
            docHitAt1WrongKpid: kpidCombinedSeed.docHitAt1WrongKpid,
            docHitAt5Total: kpidCombinedSeed.docHitAt5Total,
            docHitAt5CorrectKpid: kpidCombinedSeed.docHitAt5CorrectKpid,
            docHitAt5WrongKpid: kpidCombinedSeed.docHitAt5WrongKpid,
        },
        supportCoverageMetricsByDataset,
        supportCoverageCombined: {
            applicable: supportCoverageCombinedSeed.applicable,
            docHitAt1Total: supportCoverageCombinedSeed.docHitAt1Total,
            docHitAt5Total: supportCoverageCombinedSeed.docHitAt5Total,
            docHitAt1FullCoverAt3: supportCoverageCombinedSeed.applicable
                ? (supportCoverageCombinedSeed.docHitAt1FullCoverAt3 /
                      safeSupportCombinedApplicable) *
                  100
                : 0,
            docHitAt1FullCoverAt5: supportCoverageCombinedSeed.applicable
                ? (supportCoverageCombinedSeed.docHitAt1FullCoverAt5 /
                      safeSupportCombinedApplicable) *
                  100
                : 0,
            docHitAt5FullCoverAt5: supportCoverageCombinedSeed.applicable
                ? (supportCoverageCombinedSeed.docHitAt5FullCoverAt5 /
                      safeSupportCombinedApplicable) *
                  100
                : 0,
            partialCoverAt5: supportCoverageCombinedSeed.applicable
                ? (supportCoverageCombinedSeed.partialCoverAt5 /
                      safeSupportCombinedApplicable) *
                  100
                : 0,
            avgCoveredGroupsAt5: supportCoverageCombinedSeed.applicable
                ? supportCoverageCombinedSeed.coveredGroupsTop5Sum /
                  safeSupportCombinedApplicable
                : 0,
            avgCoverageRatioAt5: supportCoverageCombinedSeed.applicable
                ? supportCoverageCombinedSeed.coverageRatioTop5Sum /
                  safeSupportCombinedApplicable
                : 0,
        },
    };
}

function buildGroupBreakdown(
    queryCache: readonly QueryCacheItem[],
    filteredMetadata: readonly Metadata[],
    bm25Stats: BM25Stats,
    uniformWeights: WeightConfig,
    tunedWeights: WeightConfig,
    getGroupKey: (item: QueryCacheItem) => string | undefined,
): Record<string, GroupMetricsReport> {
    const groups = new Map<string, QueryCacheItem[]>();

    queryCache.forEach((item) => {
        const groupKey = getGroupKey(item);
        if (!groupKey) return;

        const bucket = groups.get(groupKey) || [];
        bucket.push(item);
        groups.set(groupKey, bucket);
    });

    const report: Record<string, GroupMetricsReport> = {};
    Array.from(groups.entries())
        .sort(([a], [b]) => a.localeCompare(b))
        .forEach(([groupKey, items]) => {
            const uniformResult = evaluateQueryCache(
                items,
                filteredMetadata,
                bm25Stats,
                uniformWeights,
            );
            const tunedResult = evaluateQueryCache(
                items,
                filteredMetadata,
                bm25Stats,
                tunedWeights,
            );
            report[groupKey] = {
                total: items.length,
                uniform: uniformResult.combined,
                tunedCombined: tunedResult.combined,
                kpidUniform: uniformResult.kpidCombined,
                kpidTunedCombined: tunedResult.kpidCombined,
                supportCoverageUniform: uniformResult.supportCoverageCombined,
                supportCoverageTunedCombined: tunedResult.supportCoverageCombined,
            };
        });

    return report;
}

function generateWeightConfigs(
    allowedTypes: readonly GranularityType[],
): WeightConfig[] {
    if (allowedTypes.length === 1) {
        return [normalizeWeights(allowedTypes, [1])];
    }

    const unique = new Map<string, WeightConfig>();
    const working: number[] = [];

    const visit = (depth: number) => {
        if (depth === allowedTypes.length) {
            const weights = normalizeWeights(allowedTypes, working);
            unique.set(formatWeightKey(weights), weights);
            return;
        }

        WEIGHT_STEPS.forEach((step) => {
            working.push(step);
            visit(depth + 1);
            working.pop();
        });
    };

    visit(0);
    return Array.from(unique.values());
}

function summarizeCombo(
    combo: ComboDefinition,
    tuneCache: readonly QueryCacheItem[],
    holdoutCache: readonly QueryCacheItem[],
    allCache: readonly QueryCacheItem[],
): ComboReport {
    const filteredMetadata = applyPerDocTypeCaps(
        metadataList.filter((item) => combo.allowedTypes.includes(item.type)),
    );
    const bm25Stats = buildBM25Stats(filteredMetadata);
    const metadataTypeCounts = countMetadataTypes(filteredMetadata);
    const uniformWeights = normalizeWeights(
        combo.allowedTypes,
        combo.allowedTypes.map(() => 1),
    );
    const uniformMetrics = evaluateQueryCache(
        allCache,
        filteredMetadata,
        bm25Stats,
        uniformWeights,
    );
    const hasIndependentHoldout = holdoutCache.length > 0;

    const fixedWeights = FIXED_COMBO_WEIGHTS[combo.label];
    const candidateWeights = fixedWeights
        ? [normalizeProvidedWeights(combo.allowedTypes, fixedWeights)]
        : hasIndependentHoldout
          ? generateWeightConfigs(combo.allowedTypes)
          : [uniformWeights];
    const candidates = candidateWeights.map((weights) => ({
        weights,
        result: evaluateQueryCache(tuneCache, filteredMetadata, bm25Stats, weights),
    }));

    candidates.sort((a, b) => compareMetrics(a.result.combined, b.result.combined));
    const bestCandidate = candidates[0];
    const combinedMetrics = evaluateQueryCache(
        allCache,
        filteredMetadata,
        bm25Stats,
        bestCandidate.weights,
    );
    const holdoutMetrics = hasIndependentHoldout
        ? evaluateQueryCache(
              holdoutCache,
              filteredMetadata,
              bm25Stats,
              bestCandidate.weights,
          )
        : combinedMetrics;
    const shouldAttachCaseDetails =
        EXPORT_BAD_CASES && combo.label === BAD_CASE_COMBO;
    const caseDetailsWeights =
        BAD_CASE_WEIGHT_MODE === "uniform"
            ? uniformWeights
            : bestCandidate.weights;
    const caseDetails = shouldAttachCaseDetails
        ? collectCaseDetails(
              allCache,
              filteredMetadata,
              bm25Stats,
              caseDetailsWeights,
          )
        : undefined;

    return {
        label: combo.label,
        allowedTypes: [...combo.allowedTypes],
        metadataCount: filteredMetadata.length,
        metadataTypeCounts,
        uniform: {
            weights: uniformWeights,
            combined: uniformMetrics.combined,
            kpidCombined: uniformMetrics.kpidCombined,
            supportCoverageCombined: uniformMetrics.supportCoverageCombined,
            metricsByDataset: uniformMetrics.metricsByDataset,
        },
        tuned: {
            selectionMode: hasIndependentHoldout
                ? "legacy_tune_holdout"
                : "single_frozen_set",
            candidateCount: candidates.length,
            bestWeights: bestCandidate.weights,
            tuneCombined: bestCandidate.result.combined,
            kpidTuneCombined: bestCandidate.result.kpidCombined,
            supportCoverageTuneCombined: bestCandidate.result.supportCoverageCombined,
            holdoutCombined: holdoutMetrics.combined,
            kpidHoldoutCombined: holdoutMetrics.kpidCombined,
            supportCoverageHoldoutCombined: holdoutMetrics.supportCoverageCombined,
            combinedCombined: combinedMetrics.combined,
            kpidCombinedCombined: combinedMetrics.kpidCombined,
            supportCoverageCombinedCombined: combinedMetrics.supportCoverageCombined,
            topTuneCandidates: candidates
                .slice(0, TOP_TUNE_CANDIDATE_LIMIT)
                .map((item) => ({
                    weights: item.weights,
                    combined: item.result.combined,
                })),
        },
        groupBreakdowns: {
            supportPattern: buildGroupBreakdown(
                allCache,
                filteredMetadata,
                bm25Stats,
                uniformWeights,
                bestCandidate.weights,
                (item) => item.testCase.support_pattern,
            ),
            preferredGranularity: buildGroupBreakdown(
                allCache,
                filteredMetadata,
                bm25Stats,
                uniformWeights,
                bestCandidate.weights,
                (item) => item.testCase.preferred_granularity,
            ),
            queryType: buildGroupBreakdown(
                allCache,
                filteredMetadata,
                bm25Stats,
                uniformWeights,
                bestCandidate.weights,
                (item) => item.testCase.query_type,
            ),
        },
        caseDetails,
        caseDetailsWeightMode: caseDetails ? BAD_CASE_WEIGHT_MODE : undefined,
    };
}

function formatMetricLine(metrics: Metrics): string {
    return [
        `Hit@1=${metrics.hitAt1.toFixed(2)}%`,
        `Hit@3=${metrics.hitAt3.toFixed(2)}%`,
        `Hit@5=${metrics.hitAt5.toFixed(2)}%`,
        `MRR=${metrics.mrr.toFixed(4)}`,
    ].join(" | ");
}

function formatWeights(weights: WeightConfig): string {
    return `Q=${weights.Q.toFixed(2)}, KP=${weights.KP.toFixed(2)}, OT=${weights.OT.toFixed(2)}`;
}

function formatSupportCoverageLine(metrics: SupportCoverageMetrics): string {
    return [
        `applicable=${metrics.applicable}`,
        `doc@1+cover@5=${metrics.docHitAt1FullCoverAt5.toFixed(2)}%`,
        `doc@5+cover@5=${metrics.docHitAt5FullCoverAt5.toFixed(2)}%`,
        `partial@5=${metrics.partialCoverAt5.toFixed(2)}%`,
        `avgCoverRatio@5=${metrics.avgCoverageRatioAt5.toFixed(4)}`,
    ].join(" | ");
}

async function main() {
    console.log("Loading evaluation engine...");
    console.log(`Active main DB version: ${ACTIVE_MAIN_DB_VERSION}`);
    console.log(`Dataset target key: ${DATASET_TARGET_KEY}`);
    console.log(`Experiment track: ${EXPERIMENT_TRACK}`);
    console.log(`OT dense mode: ${OT_DENSE_MODE}`);
    console.log(`Minimal baseline mode: ${MINIMAL_BASELINE_MODE ? "on" : "off"}`);
    console.log(`Minimal add year: ${MINIMAL_ADD_YEAR ? "on" : "off"}`);
    console.log(`Minimal add phase: ${MINIMAL_ADD_PHASE ? "on" : "off"}`);
    console.log(`Minimal add aspect: ${MINIMAL_ADD_ASPECT ? "on" : "off"}`);
    console.log(
        `Minimal disable doc multi: ${MINIMAL_DISABLE_DOC_MULTI ? "on" : "off"}`,
    );
    console.log(`Fusion mode: ${FUSION_MODE}`);
    console.log(
        `Q confusion mode: ${Q_CONFUSION_MODE}${Number.isFinite(Q_CONFUSION_WEIGHT) ? ` (weight=${Q_CONFUSION_WEIGHT})` : ""}`,
    );
    await loadEngine();

    const tuneCases = loadDatasets(DATASET_CONFIG.tuneSources);
    const holdoutCases = loadDatasets(DATASET_CONFIG.holdoutSources);
    const allCases = [...tuneCases, ...holdoutCases];
    const hasIndependentHoldout = holdoutCases.length > 0;
    const usesFrozenDatasetBundle =
        !hasIndependentHoldout && DATASET_CONFIG.groups.length > 1;

    if (hasIndependentHoldout) {
        console.log(
            `Loaded tune=${tuneCases.length}, holdout=${holdoutCases.length}, combined=${allCases.length} cases`,
        );
    } else if (usesFrozenDatasetBundle) {
        console.log(
            `Loaded frozen evaluation bundle ${DATASET_CONFIG.datasetLabel}: total=${allCases.length} cases, groups=${DATASET_CONFIG.groups.length}`,
        );
    } else {
        console.log(
            `Loaded frozen evaluation set ${DATASET_CONFIG.datasetLabel}: total=${allCases.length} cases`,
        );
    }

    const allCache = await buildQueryCache(allCases);
    const tuneCache = allCache.slice(0, tuneCases.length);
    const holdoutCache = allCache.slice(tuneCases.length);

    const report: Report = {
        generatedAt: new Date().toISOString(),
        mainDbVersion: ACTIVE_MAIN_DB_VERSION,
        experimentTrack: EXPERIMENT_TRACK !== "default" ? EXPERIMENT_TRACK : undefined,
        pipelinePresetName:
            EXPERIMENT_TRACK === "frontend_runtime"
                ? FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name
                : undefined,
        minimalBaselineMode: MINIMAL_BASELINE_MODE || undefined,
        minimalAddYear: MINIMAL_ADD_YEAR || undefined,
        minimalAddPhase: MINIMAL_ADD_PHASE || undefined,
        minimalAddAspect: MINIMAL_ADD_ASPECT || undefined,
        minimalDisableDocMulti: MINIMAL_DISABLE_DOC_MULTI || undefined,
        fusionMode: FUSION_MODE !== "default" ? FUSION_MODE : undefined,
        fixedComboWeights:
            Object.keys(FIXED_COMBO_WEIGHTS).length > 0
                ? FIXED_COMBO_WEIGHTS
                : undefined,
        otDenseMode: OT_DENSE_MODE,
        qConfusionMode: Q_CONFUSION_MODE !== "off" ? Q_CONFUSION_MODE : undefined,
        qConfusionWeight:
            Q_CONFUSION_MODE !== "off" && Number.isFinite(Q_CONFUSION_WEIGHT)
                ? Q_CONFUSION_WEIGHT
                : undefined,
        queryStyleMode: QUERY_STYLE_MODE,
        kpStyleMode: KP_STYLE_MODE,
        qPerDocCap:
            Number.isFinite(Q_PER_DOC_CAP) && Q_PER_DOC_CAP > 0
                ? Q_PER_DOC_CAP
                : undefined,
        kpPerDocCap:
            Number.isFinite(KP_PER_DOC_CAP) && KP_PER_DOC_CAP > 0
                ? KP_PER_DOC_CAP
                : undefined,
        lexicalTypeMultipliers: getSafeLexicalTypeMultipliers(),
        otDenseSlidingWindowConfig: shouldUseOtDenseSlidingWindow()
            ? {
                  windowSize: SAFE_OT_DENSE_WINDOW_SIZE,
                  overlap: SAFE_OT_DENSE_WINDOW_OVERLAP,
                  stride: SAFE_OT_DENSE_WINDOW_STRIDE,
                  batchSize: SAFE_OT_DENSE_WINDOW_BATCH_SIZE,
                  docCount:
                      otDenseSlidingWindowCorpus?.windowCountByOtid.size || 0,
                  windowCount:
                      otDenseSlidingWindowCorpus?.windowOtids.length || 0,
                  missingDocCount:
                      otDenseSlidingWindowCorpus?.missingOtids.length || 0,
              }
            : undefined,
        datasetVersion: DATASET_CONFIG.datasetVersion,
        datasetMode: DATASET_CONFIG.datasetMode,
        datasetKey: DATASET_CONFIG.datasetKey,
        datasetLabel: DATASET_CONFIG.datasetLabel,
        datasetAlias: DATASET_PROFILE.alias,
        datasetDisplayName: DATASET_PROFILE.displayName,
        datasetGroups: DATASET_CONFIG.groups.map((group) => ({
            key: group.key,
            label: group.label,
            role: group.role,
            sourceCount: group.sources.length,
            resolvedFromFallback: group.resolvedFromFallback,
        })),
        topHybridLimit:
            Number.isFinite(TOP_HYBRID_LIMIT) && TOP_HYBRID_LIMIT > 0
                ? TOP_HYBRID_LIMIT
                : 1000,
        kpAggregationMode: KP_AGGREGATION_MODE,
        kpTopN: Number.isFinite(KP_TOP_N) && KP_TOP_N > 0 ? KP_TOP_N : 3,
        kpTailWeight:
            Number.isFinite(KP_TAIL_WEIGHT) && KP_TAIL_WEIGHT >= 0
                ? KP_TAIL_WEIGHT
                : 0.35,
        lexicalBonusMode: LEXICAL_BONUS_MODE,
        onlineKpRoleRerankMode: ONLINE_KP_ROLE_RERANK_MODE,
        onlineKpRoleDocWeight:
            Number.isFinite(ONLINE_KP_ROLE_DOC_WEIGHT)
            && ONLINE_KP_ROLE_DOC_WEIGHT >= 0
                ? ONLINE_KP_ROLE_DOC_WEIGHT
                : 0.35,
        kpCandidateRerankMode: KP_CANDIDATE_RERANK_MODE,
        docPostRerankMode: DOC_POST_RERANK_MODE,
        docPostRerankWeight:
            Number.isFinite(DOC_POST_RERANK_WEIGHT) && DOC_POST_RERANK_WEIGHT >= 0
                ? DOC_POST_RERANK_WEIGHT
                : 0.35,
        limitPerDataset:
            Number.isFinite(LIMIT_PER_DATASET) && LIMIT_PER_DATASET > 0
                ? LIMIT_PER_DATASET
                : undefined,
        weightSteps: WEIGHT_STEPS,
        datasetSizes: {
            tune: countByDataset(tuneCases),
            holdout: countByDataset(holdoutCases),
            combined: countByDataset(allCases),
        },
        globalMetadataTypeCounts: countMetadataTypes(metadataList),
        combos: [],
    };

    COMBOS.forEach((combo) => {
        console.log(`\n=== ${combo.label} ===`);
        const comboReport = summarizeCombo(combo, tuneCache, holdoutCache, allCache);
        report.combos.push(comboReport);

        console.log(
            `Uniform  | ${formatWeights(comboReport.uniform.weights)} | ${formatMetricLine(comboReport.uniform.combined)}`,
        );
        if (hasIndependentHoldout) {
            console.log(
                `Best tune| ${formatWeights(comboReport.tuned.bestWeights)} | ${formatMetricLine(comboReport.tuned.tuneCombined)}`,
            );
            console.log(
                `Holdout  | ${formatMetricLine(comboReport.tuned.holdoutCombined)}`,
            );
        } else {
            console.log(
                `Frozen set| ${formatWeights(comboReport.tuned.bestWeights)} | ${formatMetricLine(comboReport.tuned.combinedCombined)}`,
            );
        }
        console.log(
            `Combined | ${formatMetricLine(comboReport.tuned.combinedCombined)}`,
        );
        if (comboReport.tuned.supportCoverageCombinedCombined.applicable > 0) {
            console.log(
                `Aspect  | ${formatSupportCoverageLine(comboReport.tuned.supportCoverageCombinedCombined)}`,
            );
        }
    });

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
    }

    const outputPath = path.join(
        resultsDir,
        buildGranularityResultFileName(DATASET_CONFIG.datasetKey, Date.now()),
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    if (
        !SKIP_RESULT_REGISTRY_UPDATE &&
        (EXPERIMENT_TRACK === "default" ||
            EXPERIMENT_TRACK === "frontend_runtime") &&
        ACTIVE_MAIN_DB_VERSION === "main_v2_plus" &&
        OT_DENSE_MODE === "original"
    ) {
        updateCurrentResultRegistry({
            datasetName: DATASET_CONFIG.datasetKey,
            datasetAlias: DATASET_PROFILE.alias,
            datasetDisplayName: DATASET_PROFILE.displayName,
            datasetFile: DATASET_FILE || DATASET_CONFIG.allSources[0]?.path || "",
            outputPath,
            sourceScript: "eval_granularity_mix.ts",
            note: hasIndependentHoldout
                ? "当前结果基于旧式 tune/holdout 分流口径导出。"
                : usesFrozenDatasetBundle
                  ? `当前结果已切换到主线三冻结集口径 ${DATASET_CONFIG.datasetLabel} 导出。`
                : EXPERIMENT_TRACK === "frontend_runtime"
                  ? `当前结果已对齐前端 runtime preset ${FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name}，基于单冻结集入口 ${DATASET_CONFIG.datasetLabel} 导出。`
                  : `当前结果基于单冻结集入口 ${DATASET_CONFIG.datasetLabel} 导出，不再默认按 tune/holdout 切分。`,
        });
    } else {
        console.log(
            `Skip current result registry update: db=${ACTIVE_MAIN_DB_VERSION}, track=${EXPERIMENT_TRACK}, otDenseMode=${OT_DENSE_MODE}, explicitSkip=${SKIP_RESULT_REGISTRY_UPDATE}`,
        );
    }
    console.log(`\nSaved report to ${outputPath}`);

    if (EXPORT_BAD_CASES) {
        const targetCombo = report.combos.find(
            (item) => item.label === BAD_CASE_COMBO && item.caseDetails,
        );
        if (targetCombo?.caseDetails) {
            const comboSlug = targetCombo.label.toLowerCase().replace(/\+/g, "_");
            const badCasePath = outputPath.replace(
                /\.json$/,
                `_bad_cases_${comboSlug}.json`,
            );
            fs.writeFileSync(
                badCasePath,
                JSON.stringify(
                    {
                        generatedAt: report.generatedAt,
                        datasetVersion: report.datasetVersion,
                        datasetKey: report.datasetKey,
                        comboLabel: targetCombo.label,
                        weightMode: targetCombo.caseDetailsWeightMode,
                        caseCount: targetCombo.caseDetails.length,
                        cases: targetCombo.caseDetails,
                    },
                    null,
                    2,
                ),
                "utf-8",
            );
            console.log(`Saved bad case export to ${badCasePath}`);
        } else {
            console.log(
                `No case details exported. Check SUASK_BAD_CASE_COMBO=${BAD_CASE_COMBO}.`,
            );
        }
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

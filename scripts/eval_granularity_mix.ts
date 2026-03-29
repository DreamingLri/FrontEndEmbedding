import * as fs from "fs";
import * as path from "path";

import {
    buildBM25Stats,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type BM25Stats,
    type KPAggregationMode,
    type KPRoleRerankMode,
    type LexicalBonusMode,
    type Metadata,
    type ParsedQueryIntent,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import {
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    loadDatasetSources,
    resolveEvalDatasetConfig,
    type EvalDatasetCase,
    type EvalDatasetSource,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";

type DatasetCase = EvalDatasetCase;
type GranularityType = "Q" | "KP" | "OT";
type KPCandidateRerankMode = "none" | "heuristic" | "feature_heuristic";
type DocPostRerankMode = "none" | "kp_heuristic_delta" | "time_anchor";
type WeightConfig = {
    Q: number;
    KP: number;
    OT: number;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryIntent: ParsedQueryIntent;
    queryMonths: number[];
    queryWords: string[];
    querySparse: Record<number, number>;
    queryYearWordIds: number[];
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

type MetricsBundle = {
    metricsByDataset: Record<string, Metrics>;
    combined: Metrics;
    kpidMetricsByDataset: Record<string, KpidMetrics>;
    kpidCombined: KpidMetrics;
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
};

type CaseDetail = {
    query: string;
    expected_otid: string;
    expected_kpid?: string;
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
    docHitAt1: boolean;
    docHitAt5: boolean;
    kpidHitAt1: boolean;
    kpidHitAt5: boolean;
    failure_risk: string;
    failure_reasons: string[];
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
        metricsByDataset: Record<string, Metrics>;
    };
    tuned: {
        candidateCount: number;
        bestWeights: WeightConfig;
        tuneCombined: Metrics;
        kpidTuneCombined: KpidMetrics;
        holdoutCombined: Metrics;
        kpidHoldoutCombined: KpidMetrics;
        combinedCombined: Metrics;
        kpidCombinedCombined: KpidMetrics;
        topTuneCandidates: WeightCandidateSummary[];
    };
    groupBreakdowns: {
        supportPattern: Record<string, GroupMetricsReport>;
        preferredGranularity: Record<string, GroupMetricsReport>;
    };
    caseDetails?: CaseDetail[];
    caseDetailsWeightMode?: "uniform" | "tuned";
};

type Report = {
    generatedAt: string;
    datasetVersion: string;
    datasetMode: "split" | "single_file";
    datasetKey: string;
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

const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || "v2";
const DATASET_FILE = process.env.SUASK_EVAL_DATASET_FILE;
const SINGLE_FILE_AS_ALL = process.env.SUASK_EVAL_SINGLE_FILE_AS_ALL === "1";
const DATASET_CONFIG = resolveEvalDatasetConfig({
    datasetVersion: DATASET_VERSION,
    datasetFile: DATASET_FILE,
    singleFileAsAll: SINGLE_FILE_AS_ALL,
});
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
const KP_TOP_N = Number.parseInt(process.env.SUASK_KP_TOP_N || "", 10);
const KP_TAIL_WEIGHT = Number.parseFloat(
    process.env.SUASK_KP_TAIL_WEIGHT || "",
);
const WEIGHT_STEPS = parseWeightSteps(
    process.env.SUASK_WEIGHT_STEPS || "0.2,0.5,0.8",
);
const FIXED_COMBO_WEIGHTS = parseFixedComboWeights(
    process.env.SUASK_FIXED_COMBO_WEIGHTS || "",
);
const TOP_TUNE_CANDIDATE_LIMIT = 5;
const CURRENT_TIMESTAMP = 0;

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
};

const KP_TEXTS_FILE = "../Backend/data/embeddings_v2/backend_knowledge_points.json";
const ARTICLE_TEXTS_FILE = "../Backend/data/embeddings_v2/backend_articles.json";
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
}

async function buildQueryCache(
    testCases: DatasetCase[],
): Promise<QueryCacheItem[]> {
    if (!extractor) {
        throw new Error("Extractor not initialized");
    }

    const queryVectors = await embedFrontendQueries(
        extractor,
        testCases.map((item) => item.query),
        dimensions,
        {
            batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
            onProgress: (done, total) => {
                console.log(`Embedded ${done} / ${total} queries`);
            },
        },
    );

    return testCases.map((testCase, index) => {
        const queryIntent = parseQueryIntent(testCase.query);
        const queryMonths = extractMonths(testCase.query);
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
            queryIntent,
            queryMonths,
            queryWords,
            querySparse,
            queryYearWordIds,
        };
    });
}

function getRank(matches: readonly { otid: string }[], expectedOtid: string): number {
    const rankIndex = matches.findIndex((item) => item.otid === expectedOtid);
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function getKpidRank(
    matches: readonly { otid: string; best_kpid?: string }[],
    expectedOtid: string,
    expectedKpid?: string,
): number {
    if (!expectedKpid) {
        return Number.POSITIVE_INFINITY;
    }

    const rankIndex = matches.findIndex(
        (item) =>
            item.otid === expectedOtid && item.best_kpid === expectedKpid,
    );
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function stripKpTimestampPrefix(text: string): string {
    return text.replace(/^\[[^\]]+\]\s*/, "");
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
    const query = item.testCase.query;
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
    const query = item.testCase.query;
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

    kpCandidates.forEach((candidate) => {
        let rerankedScore = candidate.score;

        if (KP_CANDIDATE_RERANK_MODE === "heuristic") {
            const kpText = kpTextMap.get(candidate.kpid);
            if (!kpText) {
                if (candidate.score > bestScore) {
                    bestScore = candidate.score;
                    bestKpid = candidate.kpid;
                }
                return;
            }

            rerankedScore += computeHeuristicKpBonus(item, kpText);
        } else if (KP_CANDIDATE_RERANK_MODE === "feature_heuristic") {
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
): string | undefined {
    if (KP_CANDIDATE_RERANK_MODE === "none") {
        return match.best_kpid;
    }

    return getHeuristicKpSelection(item, match).bestKpid;
}

function rerankMatchesForDocumentMetrics(
    item: QueryCacheItem,
    matches: readonly EvalSearchMatch[],
): EvalSearchMatch[] {
    if (DOC_POST_RERANK_MODE === "none") {
        return [...matches];
    }

    const safeWeight =
        Number.isFinite(DOC_POST_RERANK_WEIGHT) && DOC_POST_RERANK_WEIGHT >= 0
            ? DOC_POST_RERANK_WEIGHT
            : 0.35;

    return matches
        .map((match) => {
            let delta = 0;

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
): Array<{ otid: string; best_kpid?: string }> {
    return matches.map((match) => ({
        otid: match.otid,
        best_kpid: rerankBestKpidForMatch(item, match),
    }));
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
): string[] {
    const reasons: string[] = [];

    if (!Number.isFinite(docRank) || docRank > 5) {
        reasons.push("正确文档未进入前5，优先表现为文档级召回或排序失败。");
    } else if (docRank > 1) {
        reasons.push("正确文档进入前5但未到第1名，存在文档级排序偏差。");
    }

    if (item.testCase.expected_kpid) {
        if (!Number.isFinite(kpidRank) || kpidRank > 5) {
            reasons.push("文档候选中未能稳定选出正确主证据 KP。");
        } else if (kpidRank > 1) {
            reasons.push("正确文档命中后，主证据 KP 选择仍不够准确。");
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
): string {
    if (
        item.testCase.expected_kpid &&
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

    if (detail.expected_kpid && !detail.kpidHitAt1) {
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
            lexicalBonusMode: LEXICAL_BONUS_MODE,
            kpRoleRerankMode: ONLINE_KP_ROLE_RERANK_MODE,
            kpRoleDocWeight:
                Number.isFinite(ONLINE_KP_ROLE_DOC_WEIGHT)
                && ONLINE_KP_ROLE_DOC_WEIGHT >= 0
                    ? ONLINE_KP_ROLE_DOC_WEIGHT
                    : undefined,
        });
        const docRerankedMatches = rerankMatchesForDocumentMetrics(
            item,
            result.matches,
        );
        const rerankedMatches = rerankMatchesForKpidMetrics(
            item,
            docRerankedMatches,
        );
        const docRank = getRank(
            docRerankedMatches,
            item.testCase.expected_otid,
        );
        const kpidRank = getKpidRank(
            rerankedMatches,
            item.testCase.expected_otid,
            item.testCase.expected_kpid,
        );
        const failureReasons = buildFailureReasons(item, docRank, kpidRank);

        const detail: CaseDetail = {
            query: item.testCase.query,
            expected_otid: item.testCase.expected_otid,
            expected_kpid: item.testCase.expected_kpid,
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
            docHitAt1: docRank === 1,
            docHitAt5: docRank <= 5,
            kpidHitAt1: kpidRank === 1,
            kpidHitAt5: kpidRank <= 5,
            failure_risk: inferFailureRisk(item, docRank, kpidRank),
            failure_reasons: failureReasons,
            topDocMatches: docRerankedMatches
                .slice(0, topMatchLimit)
                .map((match, index) => ({
                    rank: index + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                })),
            topKpidMatches: rerankedMatches
                .slice(0, topMatchLimit)
                .map((match, index) => ({
                    rank: index + 1,
                    otid: match.otid,
                    best_kpid: match.best_kpid,
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

    queryCache.forEach((item) => {
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
            lexicalBonusMode: LEXICAL_BONUS_MODE,
            kpRoleRerankMode: ONLINE_KP_ROLE_RERANK_MODE,
            kpRoleDocWeight:
                Number.isFinite(ONLINE_KP_ROLE_DOC_WEIGHT)
                && ONLINE_KP_ROLE_DOC_WEIGHT >= 0
                    ? ONLINE_KP_ROLE_DOC_WEIGHT
                    : undefined,
        });
        const docRerankedMatches = rerankMatchesForDocumentMetrics(
            item,
            result.matches,
        );
        const rerankedMatches = rerankMatchesForKpidMetrics(
            item,
            docRerankedMatches,
        );
        const rank = getRank(
            docRerankedMatches,
            item.testCase.expected_otid,
        );
        const kpidRank = getKpidRank(
            rerankedMatches,
            item.testCase.expected_otid,
            item.testCase.expected_kpid,
        );
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

        if (item.testCase.expected_kpid) {
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
    const safeCombinedApplicable = kpidCombinedSeed.applicable || 1;

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
    const filteredMetadata = metadataList.filter((item) =>
        combo.allowedTypes.includes(item.type),
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

    const fixedWeights = FIXED_COMBO_WEIGHTS[combo.label];
    const candidateWeights = fixedWeights
        ? [normalizeProvidedWeights(combo.allowedTypes, fixedWeights)]
        : generateWeightConfigs(combo.allowedTypes);
    const candidates = candidateWeights.map((weights) => ({
        weights,
        result: evaluateQueryCache(tuneCache, filteredMetadata, bm25Stats, weights),
    }));

    candidates.sort((a, b) => compareMetrics(a.result.combined, b.result.combined));
    const bestCandidate = candidates[0];
    const holdoutMetrics = evaluateQueryCache(
        holdoutCache,
        filteredMetadata,
        bm25Stats,
        bestCandidate.weights,
    );
    const combinedMetrics = evaluateQueryCache(
        allCache,
        filteredMetadata,
        bm25Stats,
        bestCandidate.weights,
    );
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
            metricsByDataset: uniformMetrics.metricsByDataset,
        },
        tuned: {
            candidateCount: candidates.length,
            bestWeights: bestCandidate.weights,
            tuneCombined: bestCandidate.result.combined,
            kpidTuneCombined: bestCandidate.result.kpidCombined,
            holdoutCombined: holdoutMetrics.combined,
            kpidHoldoutCombined: holdoutMetrics.kpidCombined,
            combinedCombined: combinedMetrics.combined,
            kpidCombinedCombined: combinedMetrics.kpidCombined,
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

async function main() {
    console.log("Loading evaluation engine...");
    await loadEngine();

    const tuneCases = loadDatasets(DATASET_CONFIG.tuneSources);
    const holdoutCases = loadDatasets(DATASET_CONFIG.holdoutSources);
    const allCases = [...tuneCases, ...holdoutCases];

    console.log(
        `Loaded tune=${tuneCases.length}, holdout=${holdoutCases.length}, combined=${allCases.length} cases`,
    );

    const allCache = await buildQueryCache(allCases);
    const tuneCache = allCache.slice(0, tuneCases.length);
    const holdoutCache = allCache.slice(tuneCases.length);

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetVersion: DATASET_CONFIG.datasetVersion,
        datasetMode: DATASET_CONFIG.datasetMode,
        datasetKey: DATASET_CONFIG.datasetKey,
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
        console.log(
            `Best tune| ${formatWeights(comboReport.tuned.bestWeights)} | ${formatMetricLine(comboReport.tuned.tuneCombined)}`,
        );
        console.log(
            `Holdout  | ${formatMetricLine(comboReport.tuned.holdoutCombined)}`,
        );
        console.log(
            `Combined | ${formatMetricLine(comboReport.tuned.combinedCombined)}`,
        );
    });

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
    }

    const outputPath = path.join(
        resultsDir,
        `granularity_mix_${DATASET_CONFIG.datasetKey}_top${Number.isFinite(TOP_HYBRID_LIMIT) && TOP_HYBRID_LIMIT > 0 ? TOP_HYBRID_LIMIT : 1000}_${KP_AGGREGATION_MODE === "max_plus_topn" ? `kpagg-top${Number.isFinite(KP_TOP_N) && KP_TOP_N > 0 ? KP_TOP_N : 3}-w${(Number.isFinite(KP_TAIL_WEIGHT) && KP_TAIL_WEIGHT >= 0 ? KP_TAIL_WEIGHT : 0.35).toFixed(2).replace(".", "")}` : "kpagg-max"}_lex${LEXICAL_BONUS_MODE}_onlinekprole${ONLINE_KP_ROLE_RERANK_MODE}${ONLINE_KP_ROLE_RERANK_MODE === "feature" ? `-w${(Number.isFinite(ONLINE_KP_ROLE_DOC_WEIGHT) && ONLINE_KP_ROLE_DOC_WEIGHT >= 0 ? ONLINE_KP_ROLE_DOC_WEIGHT : 0.35).toFixed(2).replace(".", "")}` : ""}_kprerank${KP_CANDIDATE_RERANK_MODE}_docrerank${formatDocPostRerankSlug()}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
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

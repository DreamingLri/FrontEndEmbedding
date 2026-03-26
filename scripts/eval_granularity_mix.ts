import * as fs from "fs";
import * as path from "path";

import {
    buildBM25Stats,
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type BM25Stats,
    type KPAggregationMode,
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
type KPCandidateRerankMode = "none" | "heuristic";
type WeightConfig = {
    Q: number;
    KP: number;
    OT: number;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryIntent: ParsedQueryIntent;
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
    kpCandidateRerankMode: KPCandidateRerankMode;
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
const DATASET_CONFIG = resolveEvalDatasetConfig({
    datasetVersion: DATASET_VERSION,
    datasetFile: DATASET_FILE,
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
const KP_CANDIDATE_RERANK_MODE = (
    process.env.SUASK_KP_CANDIDATE_RERANK_MODE === "heuristic"
        ? "heuristic"
        : "none"
) as KPCandidateRerankMode;
const KP_TOP_N = Number.parseInt(process.env.SUASK_KP_TOP_N || "", 10);
const KP_TAIL_WEIGHT = Number.parseFloat(
    process.env.SUASK_KP_TAIL_WEIGHT || "",
);
const WEIGHT_STEPS = parseWeightSteps(
    process.env.SUASK_WEIGHT_STEPS || "0.2,0.5,0.8",
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

const KP_TEXTS_FILE = "../Backend/data/embeddings_v2/backend_knowledge_points.json";
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

async function loadEngine() {
    const engine = await loadFrontendEvalEngine();
    extractor = engine.extractor;
    vocabMap = engine.vocabMap;
    metadataList = engine.metadataList;
    vectorMatrix = engine.vectorMatrix;
    dimensions = engine.dimensions;
    kpTextMap = loadKnowledgePointTexts();
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

function rerankBestKpidForMatch(
    item: QueryCacheItem,
    match: { best_kpid?: string; kp_candidates?: Array<{ kpid: string; score: number }> },
): string | undefined {
    const kpCandidates = match.kp_candidates || [];
    if (KP_CANDIDATE_RERANK_MODE !== "heuristic" || kpCandidates.length === 0) {
        return match.best_kpid;
    }

    let bestKpid = match.best_kpid;
    let bestScore = Number.NEGATIVE_INFINITY;

    kpCandidates.forEach((candidate) => {
        const kpText = kpTextMap.get(candidate.kpid);
        if (!kpText) {
            if (candidate.score > bestScore) {
                bestScore = candidate.score;
                bestKpid = candidate.kpid;
            }
            return;
        }

        const rerankedScore =
            candidate.score + computeHeuristicKpBonus(item, kpText);
        if (rerankedScore > bestScore) {
            bestScore = rerankedScore;
            bestKpid = candidate.kpid;
        }
    });

    return bestKpid;
}

function rerankMatchesForKpidMetrics(
    item: QueryCacheItem,
    matches: readonly {
        otid: string;
        best_kpid?: string;
        score: number;
        kp_candidates?: Array<{ kpid: string; score: number }>;
    }[],
): Array<{ otid: string; best_kpid?: string }> {
    return matches.map((match) => ({
        otid: match.otid,
        best_kpid: rerankBestKpidForMatch(item, match),
    }));
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
        });
        const rerankedMatches = rerankMatchesForKpidMetrics(
            item,
            result.matches,
        );

        const rank = getRank(result.matches, item.testCase.expected_otid);
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

    const candidates = generateWeightConfigs(combo.allowedTypes).map((weights) => ({
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
        kpCandidateRerankMode: KP_CANDIDATE_RERANK_MODE,
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
        `granularity_mix_${DATASET_CONFIG.datasetKey}_top${Number.isFinite(TOP_HYBRID_LIMIT) && TOP_HYBRID_LIMIT > 0 ? TOP_HYBRID_LIMIT : 1000}_${KP_AGGREGATION_MODE === "max_plus_topn" ? `kpagg-top${Number.isFinite(KP_TOP_N) && KP_TOP_N > 0 ? KP_TOP_N : 3}-w${(Number.isFinite(KP_TAIL_WEIGHT) && KP_TAIL_WEIGHT >= 0 ? KP_TAIL_WEIGHT : 0.35).toFixed(2).replace(".", "")}` : "kpagg-max"}_lex${LEXICAL_BONUS_MODE}_kprerank${KP_CANDIDATE_RERANK_MODE}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`\nSaved report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

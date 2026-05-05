import * as fs from "fs";
import * as path from "path";

import {
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type Metadata,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import { resolvePipelinePresetByName } from "../src/worker/search_pipeline.ts";
import {
    loadDataset,
    resolveGranularityDatasetTarget,
    type EvalDatasetCase,
    type GranularityDatasetTargetKey,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";

type DatasetCase = EvalDatasetCase;

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryIntent: ReturnType<typeof parseQueryIntent>;
    queryYearWordIds: number[];
};

type WeightConfig = {
    Q: number;
    KP: number;
    OT: number;
};

type MetricSummary = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type WeightReport = {
    rank: number;
    weights: WeightConfig;
    metrics: MetricSummary;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetLabel: string;
    presetName: string;
    stepValues: number[];
    topResults: WeightReport[];
};

const DEFAULT_DATASET_TARGET =
    "in_domain_generalization_100" as GranularityDatasetTargetKey;
const DEFAULT_PRESET_NAME = "frontend_research_sync_v1";
const DEFAULT_STEP_VALUES = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1";
const DEFAULT_TOP_N = 12;
const CURRENT_TIMESTAMP = Date.now() / 1000;

function parseArgs(): {
    datasetFile?: string;
    datasetTarget?: GranularityDatasetTargetKey;
    presetName: string;
    stepValues: number[];
    topN: number;
} {
    const args = process.argv.slice(2);
    let datasetFile = process.env.SUASK_EVAL_DATASET_FILE;
    let datasetTarget = process.env
        .SUASK_EVAL_DATASET_TARGET as GranularityDatasetTargetKey | undefined;
    let presetName =
        process.env.SUASK_PIPELINE_PRESET || DEFAULT_PRESET_NAME;
    let stepValues = parseStepValues(
        process.env.SUASK_WEIGHT_GRID_STEPS || DEFAULT_STEP_VALUES,
    );
    let topN = Number.parseInt(
        process.env.SUASK_WEIGHT_GRID_TOP_N || "",
        10,
    );

    for (let index = 0; index < args.length; index += 1) {
        const current = args[index];
        if (current === "--dataset-file") {
            datasetFile = args[index + 1];
            index += 1;
            continue;
        }
        if (current.startsWith("--dataset-file=")) {
            datasetFile = current.split("=", 2)[1];
            continue;
        }
        if (current === "--dataset") {
            datasetTarget = args[index + 1] as GranularityDatasetTargetKey;
            index += 1;
            continue;
        }
        if (current.startsWith("--dataset=")) {
            datasetTarget = current.split("=", 2)[1] as GranularityDatasetTargetKey;
            continue;
        }
        if (current === "--preset") {
            presetName = args[index + 1] || presetName;
            index += 1;
            continue;
        }
        if (current.startsWith("--preset=")) {
            presetName = current.split("=", 2)[1] || presetName;
            continue;
        }
        if (current === "--steps") {
            stepValues = parseStepValues(args[index + 1] || DEFAULT_STEP_VALUES);
            index += 1;
            continue;
        }
        if (current.startsWith("--steps=")) {
            stepValues = parseStepValues(current.split("=", 2)[1] || DEFAULT_STEP_VALUES);
            continue;
        }
        if (current === "--top") {
            topN = Number.parseInt(args[index + 1] || "", 10);
            index += 1;
            continue;
        }
        if (current.startsWith("--top=")) {
            topN = Number.parseInt(current.split("=", 2)[1] || "", 10);
            continue;
        }
    }

    return {
        datasetFile,
        datasetTarget,
        presetName,
        stepValues,
        topN:
            Number.isFinite(topN) && topN > 0
                ? Math.max(1, Math.min(topN, 100))
                : DEFAULT_TOP_N,
    };
}

function parseStepValues(raw: string): number[] {
    const parsed = raw
        .split(",")
        .map((item) => Number.parseFloat(item.trim()))
        .filter((item) => Number.isFinite(item))
        .map((item) => Math.max(0, Math.min(1, item)));
    const uniqueSorted = Array.from(new Set(parsed)).sort((a, b) => a - b);
    if (uniqueSorted.length === 0) {
        throw new Error(`Invalid step values: "${raw}"`);
    }
    return uniqueSorted;
}

function resolveDatasetSource(params: {
    datasetFile?: string;
    datasetTarget?: GranularityDatasetTargetKey;
}): {
    datasetFile: string;
    datasetLabel: string;
} {
    if (params.datasetFile) {
        const datasetFile = params.datasetFile;
        return {
            datasetFile,
            datasetLabel: path.basename(datasetFile, ".json"),
        };
    }

    const target = resolveGranularityDatasetTarget(
        params.datasetTarget || DEFAULT_DATASET_TARGET,
    );
    return {
        datasetFile: target.datasetFile,
        datasetLabel: target.label,
    };
}

function formatWeightKey(weights: WeightConfig): string {
    return `${weights.Q.toFixed(4)}|${weights.KP.toFixed(4)}|${weights.OT.toFixed(4)}`;
}

function generateWeightConfigs(stepValues: readonly number[]): WeightConfig[] {
    const configs = new Map<string, WeightConfig>();
    for (const qWeight of stepValues) {
        for (const kpWeight of stepValues) {
            const otWeight = Number((1 - qWeight - kpWeight).toFixed(10));
            if (otWeight < 0 || !stepValues.includes(otWeight)) {
                continue;
            }
            const weights = {
                Q: qWeight,
                KP: kpWeight,
                OT: otWeight,
            };
            if (weights.Q + weights.KP + weights.OT <= 0) {
                continue;
            }
            configs.set(formatWeightKey(weights), weights);
        }
    }
    return Array.from(configs.values());
}

function compareMetrics(a: MetricSummary, b: MetricSummary): number {
    if (b.hitAt1 !== a.hitAt1) return b.hitAt1 - a.hitAt1;
    if (b.mrr !== a.mrr) return b.mrr - a.mrr;
    if (b.hitAt3 !== a.hitAt3) return b.hitAt3 - a.hitAt3;
    return b.hitAt5 - a.hitAt5;
}

function buildMetricSummary(ranks: Array<number | null>): MetricSummary {
    const total = ranks.length;
    let hitAt1 = 0;
    let hitAt3 = 0;
    let hitAt5 = 0;
    let reciprocalRankSum = 0;

    ranks.forEach((rank) => {
        if (!rank) {
            return;
        }
        if (rank <= 1) hitAt1 += 1;
        if (rank <= 3) hitAt3 += 1;
        if (rank <= 5) hitAt5 += 1;
        reciprocalRankSum += 1 / rank;
    });

    return {
        total,
        hitAt1: (hitAt1 / total) * 100,
        hitAt3: (hitAt3 / total) * 100,
        hitAt5: (hitAt5 / total) * 100,
        mrr: reciprocalRankSum / total,
    };
}

async function buildQueryCache(testCases: DatasetCase[]): Promise<{
    queryCache: QueryCacheItem[];
    metadataList: Metadata[];
    vectorMatrix: Int8Array;
    dimensions: number;
    bm25Stats: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["bm25Stats"];
}> {
    const engine = await loadFrontendEvalEngine();
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        testCases.map((item) => item.query),
        engine.dimensions,
    );

    const queryCache = testCases.map((testCase, index) => {
        const queryIntent = parseQueryIntent(testCase.query);
        const queryWords = Array.from(
            new Set(fmmTokenize(testCase.query, engine.vocabMap)),
        );
        const querySparse = getQuerySparse(queryWords, engine.vocabMap);
        const queryYearWordIds = queryIntent.years
            .map(String)
            .map((year) => engine.vocabMap.get(year))
            .filter((item): item is number => item !== undefined);

        return {
            testCase,
            queryVector: queryVectors[index],
            queryWords,
            querySparse,
            queryIntent,
            queryYearWordIds,
        };
    });

    return {
        queryCache,
        metadataList: engine.metadataList,
        vectorMatrix: engine.vectorMatrix,
        dimensions: engine.dimensions,
        bm25Stats: engine.bm25Stats,
    };
}

function evaluateWeightSetting(params: {
    queryCache: readonly QueryCacheItem[];
    metadataList: Metadata[];
    vectorMatrix: Int8Array;
    dimensions: number;
    bm25Stats: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["bm25Stats"];
    presetName: string;
    weights: WeightConfig;
}): MetricSummary {
    const preset = resolvePipelinePresetByName(params.presetName);
    const ranks = params.queryCache.map((item) => {
        const result = searchAndRank({
            queryVector: item.queryVector,
            querySparse: item.querySparse,
            queryWords: item.queryWords,
            queryYearWordIds: item.queryYearWordIds,
            queryIntent: item.queryIntent,
            rawQueryText: item.testCase.query,
            queryScopeHint: item.testCase.query_scope,
            metadata: params.metadataList,
            vectorMatrix: params.vectorMatrix,
            dimensions: params.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: params.bm25Stats,
            weights: params.weights,
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
            conditionalKpDownweight: preset.retrieval.conditionalKpDownweight,
            conditionalOtDownweight: preset.retrieval.conditionalOtDownweight,
            enableExplicitYearFilter: preset.retrieval.enableExplicitYearFilter,
            minimalMode: preset.retrieval.minimalMode,
        });

        const rankIndex = result.matches.findIndex(
            (match) => match.otid === item.testCase.expected_otid,
        );
        return rankIndex === -1 ? null : rankIndex + 1;
    });

    return buildMetricSummary(ranks);
}

async function main() {
    const args = parseArgs();
    const datasetSource = resolveDatasetSource({
        datasetFile: args.datasetFile,
        datasetTarget: args.datasetTarget,
    });
    const datasetFile = path.resolve(process.cwd(), datasetSource.datasetFile);
    const testCases = loadDataset(datasetFile, {
        datasetLabel: datasetSource.datasetLabel,
    }) as DatasetCase[];
    const weightConfigs = generateWeightConfigs(args.stepValues);
    const { queryCache, metadataList, vectorMatrix, dimensions, bm25Stats } =
        await buildQueryCache(testCases);

    const reports = weightConfigs.map((weights) => ({
        weights,
        metrics: evaluateWeightSetting({
            queryCache,
            metadataList,
            vectorMatrix,
            dimensions,
            bm25Stats,
            presetName: args.presetName,
            weights,
        }),
    }));

    reports.sort((a, b) => compareMetrics(a.metrics, b.metrics));
    const topResults = reports.slice(0, args.topN).map((item, index) => ({
        rank: index + 1,
        weights: item.weights,
        metrics: item.metrics,
    }));

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetFile,
        datasetLabel: datasetSource.datasetLabel,
        presetName: args.presetName,
        stepValues: args.stepValues,
        topResults,
    };

    const outputDir = path.resolve(process.cwd(), "./scripts/results");
    fs.mkdirSync(outputDir, { recursive: true });
    const outputFile = path.join(
        outputDir,
        `granularity_weight_grid_${path.basename(datasetFile, ".json")}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputFile, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Dataset: ${datasetFile}`);
    console.log(`Preset: ${args.presetName}`);
    console.log(`Cases: ${testCases.length}`);
    console.log(`Grid size: ${weightConfigs.length}`);
    topResults.forEach((item) => {
        console.log(
            `#${item.rank} Q=${item.weights.Q.toFixed(2)} KP=${item.weights.KP.toFixed(2)} OT=${item.weights.OT.toFixed(2)} | Hit@1=${item.metrics.hitAt1.toFixed(2)}% | Hit@3=${item.metrics.hitAt3.toFixed(2)}% | Hit@5=${item.metrics.hitAt5.toFixed(2)}% | MRR=${item.metrics.mrr.toFixed(4)}`,
        );
    });
    console.log(`Saved report: ${outputFile}`);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});

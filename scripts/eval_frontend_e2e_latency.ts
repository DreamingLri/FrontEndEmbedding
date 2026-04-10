import * as fs from "fs";
import * as path from "path";
import { spawnSync } from "child_process";

import {
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
} from "../src/worker/search_pipeline.ts";
import { searchAndRank } from "../src/worker/vector_engine.ts";
import {
    ACTIVE_MAIN_DB_VERSION,
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    FRONTEND_MODEL_NAME,
    loadDatasetSources,
    resolveEvalDatasetConfig,
    type EvalDatasetCase,
    type GranularityDatasetTargetKey,
} from "./eval_shared.ts";
import {
    embedQueries,
    loadFrontendEvalEngine,
    type FrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { resolveNamedDatasetProfile } from "./result_naming.ts";

type TimingSummary = {
    avgMs: number;
    p50Ms: number;
    p95Ms: number;
    minMs: number;
    maxMs: number;
};

type WarmQueryTiming = {
    queryIndex: number;
    preprocessMs: number;
    embedMs: number;
    retrievalMs: number;
    totalMs: number;
};

type ColdOnceTiming = WarmQueryTiming & {
    loadEngineMs: number;
};

type DatasetLatencyReport = {
    datasetKey: string;
    datasetLabel: string;
    datasetAlias: string;
    caseCount: number;
    coldSampleIndices: number[];
    warm: {
        preprocess: TimingSummary;
        embed: TimingSummary;
        retrieval: TimingSummary;
        total: TimingSummary;
    };
    coldStart: {
        sampleCount: number;
        loadEngine: TimingSummary;
        preprocess: TimingSummary;
        embed: TimingSummary;
        retrieval: TimingSummary;
        total: TimingSummary;
    };
};

type Report = {
    generatedAt: string;
    mainDbVersion: string;
    embeddingModel: string;
    pipelinePresetName: string;
    qConfusionMode: string;
    qConfusionWeight: number;
    queryEmbedBatchSize: number;
    coldSampleLimit: number;
    currentTimestamp: number;
    notes: {
        coreLatency: string;
        warmEndToEnd: string;
        coldStart: string;
    };
    warmSessionLoadEngineMs: number;
    datasets: DatasetLatencyReport[];
};

const CURRENT_TIMESTAMP = 0;
const DEFAULT_COLD_SAMPLE_LIMIT = Number.parseInt(
    process.env.SUASK_COLD_SAMPLE_LIMIT || "",
    10,
);
const OFFICIAL_DATASET_KEYS: GranularityDatasetTargetKey[] = [
    "main_bench_120",
    "in_domain_holdout_50",
    "external_ood_50",
];

function nowMs(): number {
    if (typeof performance !== "undefined" && performance.now) {
        return performance.now();
    }
    return Date.now();
}

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

function parseCliFlag(name: string): boolean {
    return process.argv.slice(2).includes(name);
}

function parseCliNumber(name: string): number | undefined {
    const args = process.argv.slice(2);
    for (let index = 0; index < args.length; index += 1) {
        const current = args[index];
        if (current === name) {
            const next = args[index + 1];
            if (!next) {
                return undefined;
            }
            const value = Number.parseInt(next, 10);
            return Number.isFinite(value) ? value : undefined;
        }
        if (current.startsWith(`${name}=`)) {
            const [, rawValue] = current.split("=", 2);
            const value = Number.parseInt(rawValue, 10);
            return Number.isFinite(value) ? value : undefined;
        }
    }

    return undefined;
}

function percentile(values: readonly number[], ratio: number): number {
    if (values.length === 0) {
        return 0;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.max(
        0,
        Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1),
    );
    return sorted[index] || 0;
}

function buildTimingSummary(values: readonly number[]): TimingSummary {
    if (values.length === 0) {
        return {
            avgMs: 0,
            p50Ms: 0,
            p95Ms: 0,
            minMs: 0,
            maxMs: 0,
        };
    }

    const sorted = [...values].sort((a, b) => a - b);
    const total = sorted.reduce((sum, value) => sum + value, 0);
    return {
        avgMs: total / sorted.length,
        p50Ms: percentile(sorted, 0.5),
        p95Ms: percentile(sorted, 0.95),
        minMs: sorted[0] || 0,
        maxMs: sorted[sorted.length - 1] || 0,
    };
}

function resolveDatasetCases(
    datasetKey: GranularityDatasetTargetKey,
): EvalDatasetCase[] {
    const datasetConfig = resolveEvalDatasetConfig({
        datasetVersion: "granularity",
        singleFileAsAll: true,
        datasetTargetKey: datasetKey,
    });
    return loadDatasetSources(datasetConfig.allSources);
}

function buildColdSampleIndices(
    total: number,
    requestedLimit: number,
): number[] {
    if (total <= 0) {
        return [];
    }

    const safeLimit =
        Number.isFinite(requestedLimit) && requestedLimit > 0
            ? Math.min(requestedLimit, total)
            : Math.min(DEFAULT_COLD_SAMPLE_LIMIT || 5, total);
    if (safeLimit >= total) {
        return Array.from({ length: total }, (_, index) => index);
    }

    const picks = new Set<number>();
    for (let index = 0; index < safeLimit; index += 1) {
        const pick = Math.min(
            total - 1,
            Math.floor((index * total) / safeLimit),
        );
        picks.add(pick);
    }

    return Array.from(picks).sort((a, b) => a - b);
}

async function measureWarmQuery(
    engine: FrontendEvalEngine,
    termMaps: ReturnType<typeof buildPipelineTermMaps>,
    query: string,
): Promise<WarmQueryTiming> {
    const preprocessStartedAt = nowMs();
    const queryContext = buildSearchPipelineQueryContext(
        query,
        engine.vocabMap,
        engine.topicPartitionIndex,
        FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    );
    const preprocessMs = nowMs() - preprocessStartedAt;

    const embedStartedAt = nowMs();
    const [queryVector] = await embedQueries(
        engine.extractor,
        [query],
        engine.dimensions,
        { batchSize: 1 },
    );
    const embedMs = nowMs() - embedStartedAt;

    const retrievalStartedAt = nowMs();
    searchAndRank({
        queryVector,
        querySparse: queryContext.querySparse,
        queryWords: queryContext.queryWords,
        queryYearWordIds: queryContext.queryYearWordIds,
        queryIntent: queryContext.queryIntent,
        metadata: engine.metadataList,
        vectorMatrix: engine.vectorMatrix,
        dimensions: engine.dimensions,
        currentTimestamp: CURRENT_TIMESTAMP,
        bm25Stats: engine.bm25Stats,
        candidateIndices: queryContext.candidateIndices,
        scopeSpecificityWordIdToTerm: termMaps.scopeSpecificityWordIdToTerm,
        weights: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.weights,
        topHybridLimit:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.topHybridLimit,
        kpAggregationMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpAggregationMode,
        kpTopN: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTopN,
        kpTailWeight:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.kpTailWeight,
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
        minimalMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.minimalMode,
    });
    const retrievalMs = nowMs() - retrievalStartedAt;

    return {
        queryIndex: -1,
        preprocessMs,
        embedMs,
        retrievalMs,
        totalMs: preprocessMs + embedMs + retrievalMs,
    };
}

async function runColdOnce(): Promise<void> {
    const datasetKey = parseCliDatasetTargetKey();
    const queryIndex = parseCliNumber("--query-index");
    if (!datasetKey || queryIndex === undefined) {
        throw new Error("cold-once 模式需要同时提供 --dataset 与 --query-index。");
    }

    const cases = resolveDatasetCases(datasetKey);
    if (queryIndex < 0 || queryIndex >= cases.length) {
        throw new Error(
            `query-index 越界：${queryIndex}，当前数据集共有 ${cases.length} 条样本。`,
        );
    }

    const query = cases[queryIndex]?.query || "";
    const loadStartedAt = nowMs();
    const engine = await loadFrontendEvalEngine();
    const loadEngineMs = nowMs() - loadStartedAt;
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const timing = await measureWarmQuery(engine, termMaps, query);
    const coldTiming: ColdOnceTiming = {
        ...timing,
        queryIndex,
        loadEngineMs,
        totalMs: loadEngineMs + timing.totalMs,
    };

    process.stdout.write(`${JSON.stringify(coldTiming)}\n`);
}

function getTsxCliPath(): string {
    const cliPath = path.resolve(process.cwd(), "node_modules/tsx/dist/cli.mjs");
    if (!fs.existsSync(cliPath)) {
        throw new Error(`未找到 tsx CLI：${cliPath}`);
    }
    return cliPath;
}

function runColdChild(
    datasetKey: GranularityDatasetTargetKey,
    queryIndex: number,
): ColdOnceTiming {
    const cliPath = getTsxCliPath();
    const scriptPath = path.resolve(process.cwd(), "scripts/eval_frontend_e2e_latency.ts");
    const result = spawnSync(
        process.execPath,
        [
            cliPath,
            scriptPath,
            "--cold-once",
            "--dataset",
            datasetKey,
            "--query-index",
            String(queryIndex),
        ],
        {
            cwd: process.cwd(),
            encoding: "utf-8",
            env: process.env,
            maxBuffer: 10 * 1024 * 1024,
        },
    );

    if (result.status !== 0) {
        throw new Error(
            [
                `cold-start 子进程失败：dataset=${datasetKey}, queryIndex=${queryIndex}`,
                result.stdout?.trim(),
                result.stderr?.trim(),
            ]
                .filter(Boolean)
                .join("\n"),
        );
    }

    const stdout = result.stdout?.trim();
    if (!stdout) {
        throw new Error(
            `cold-start 子进程未返回结果：dataset=${datasetKey}, queryIndex=${queryIndex}`,
        );
    }

    return JSON.parse(stdout) as ColdOnceTiming;
}

async function buildDatasetLatencyReport(
    engine: FrontendEvalEngine,
    datasetKey: GranularityDatasetTargetKey,
    coldSampleLimit: number,
): Promise<DatasetLatencyReport> {
    const cases = resolveDatasetCases(datasetKey);
    const datasetProfile = resolveNamedDatasetProfile(datasetKey);
    const warmTimings: WarmQueryTiming[] = [];
    const termMaps = buildPipelineTermMaps(engine.vocabMap);

    for (let index = 0; index < cases.length; index += 1) {
        const query = cases[index]?.query || "";
        const timing = await measureWarmQuery(engine, termMaps, query);
        warmTimings.push({
            ...timing,
            queryIndex: index,
        });

        if (index + 1 === cases.length || (index + 1) % 16 === 0) {
            console.log(
                `Warm latency ${datasetProfile.displayName}: ${index + 1}/${cases.length}`,
            );
        }
    }

    const coldSampleIndices = buildColdSampleIndices(cases.length, coldSampleLimit);
    const coldTimings = coldSampleIndices.map((queryIndex) =>
        runColdChild(datasetKey, queryIndex),
    );

    return {
        datasetKey,
        datasetLabel: datasetProfile.displayName,
        datasetAlias: datasetProfile.alias,
        caseCount: cases.length,
        coldSampleIndices,
        warm: {
            preprocess: buildTimingSummary(
                warmTimings.map((item) => item.preprocessMs),
            ),
            embed: buildTimingSummary(warmTimings.map((item) => item.embedMs)),
            retrieval: buildTimingSummary(
                warmTimings.map((item) => item.retrievalMs),
            ),
            total: buildTimingSummary(warmTimings.map((item) => item.totalMs)),
        },
        coldStart: {
            sampleCount: coldTimings.length,
            loadEngine: buildTimingSummary(
                coldTimings.map((item) => item.loadEngineMs),
            ),
            preprocess: buildTimingSummary(
                coldTimings.map((item) => item.preprocessMs),
            ),
            embed: buildTimingSummary(coldTimings.map((item) => item.embedMs)),
            retrieval: buildTimingSummary(
                coldTimings.map((item) => item.retrievalMs),
            ),
            total: buildTimingSummary(coldTimings.map((item) => item.totalMs)),
        },
    };
}

async function main() {
    if (parseCliFlag("--cold-once")) {
        await runColdOnce();
        return;
    }

    const coldSampleLimit = Number.isFinite(DEFAULT_COLD_SAMPLE_LIMIT)
        && DEFAULT_COLD_SAMPLE_LIMIT > 0
        ? DEFAULT_COLD_SAMPLE_LIMIT
        : 5;
    const datasetKey = parseCliDatasetTargetKey();
    const targets = datasetKey ? [datasetKey] : OFFICIAL_DATASET_KEYS;

    console.log("Loading frontend eval engine for warm session...");
    console.log(`Active main DB version: ${ACTIVE_MAIN_DB_VERSION}`);
    const warmLoadStartedAt = nowMs();
    const engine = await loadFrontendEvalEngine();
    const warmSessionLoadEngineMs = nowMs() - warmLoadStartedAt;
    console.log(
        `Warm session engine load finished in ${warmSessionLoadEngineMs.toFixed(2)}ms`,
    );

    const datasets: DatasetLatencyReport[] = [];
    for (let index = 0; index < targets.length; index += 1) {
        const target = targets[index]!;
        console.log(`\nEvaluating end-to-end latency for ${target}...`);
        const report = await buildDatasetLatencyReport(
            engine,
            target,
            coldSampleLimit,
        );
        datasets.push(report);

        console.log(
            [
                `[${report.datasetLabel}]`,
                `warm core avg=${report.warm.retrieval.avgMs.toFixed(2)}ms`,
                `warm e2e avg=${report.warm.total.avgMs.toFixed(2)}ms`,
                `cold total avg=${report.coldStart.total.avgMs.toFixed(2)}ms`,
            ].join(" | "),
        );
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        mainDbVersion: ACTIVE_MAIN_DB_VERSION,
        embeddingModel: FRONTEND_MODEL_NAME,
        pipelinePresetName: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name,
        qConfusionMode:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionMode,
        qConfusionWeight:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.qConfusionWeight,
        queryEmbedBatchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
        coldSampleLimit,
        currentTimestamp: CURRENT_TIMESTAMP,
        notes: {
            coreLatency:
                "warm core latency 仅覆盖 query 已完成预处理与 embedding 后的检索/排序主链。",
            warmEndToEnd:
                "warm end-to-end latency 覆盖 query 预处理、embedding 与检索/排序，不含文档抓取、展示重排与结果写盘。",
            coldStart:
                "cold-start latency 基于 fresh process 下的单 query 首次请求，覆盖 engine/model load、query 预处理、embedding 与检索/排序。",
        },
        warmSessionLoadEngineMs,
        datasets,
    };

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(resultsDir, { recursive: true });
    const outputPath = path.resolve(
        resultsDir,
        `frontend_e2e_latency_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`\nSaved report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

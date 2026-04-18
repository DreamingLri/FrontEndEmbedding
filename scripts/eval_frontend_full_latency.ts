import * as fs from "fs";
import * as path from "path";
import { performance } from "perf_hooks";

import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    executeSearchPipeline,
} from "../src/worker/search_pipeline.ts";
import {
    ACTIVE_MAIN_DB_VERSION,
    FRONTEND_METADATA_FILE,
    FRONTEND_VECTOR_FILE,
    loadDatasetSources,
    resolveEvalDatasetConfig,
    type EvalDatasetCase,
    type GranularityDatasetTargetKey,
    type OtidEvalMode,
} from "./eval_shared.ts";
import { loadFrontendEvalEngine } from "./frontend_eval_engine.ts";
import {
    createApiDocumentLoader,
    createLocalDocumentLoader,
} from "./local_document_provider.ts";

type DatasetCase = EvalDatasetCase;

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

type CaseLatencyRecord = {
    id: string;
    query: string;
    dataset: string;
    behavior: "answer" | "reject";
    rank: number | null;
    hitAt1: boolean;
    hitAt5: boolean;
    mrr: number;
    embedMs: number;
    searchMs: number;
    fetchMs: number;
    stageTotalMs: number;
    endToEndMs: number;
    matchCount: number;
    weakMatchCount: number;
    fetchedDocumentCount: number;
    topOtid: string | null;
};

type Summary = {
    totalCases: number;
    answerRate: number;
    rejectRate: number;
    hitAt1: number;
    hitAt5: number;
    mrr: number;
    avgEmbedMs: number;
    p50EmbedMs: number;
    p95EmbedMs: number;
    avgSearchMs: number;
    p50SearchMs: number;
    p95SearchMs: number;
    avgFetchMs: number;
    p50FetchMs: number;
    p95FetchMs: number;
    avgStageTotalMs: number;
    p50StageTotalMs: number;
    p95StageTotalMs: number;
    avgEndToEndMs: number;
    p50EndToEndMs: number;
    p95EndToEndMs: number;
    avgFetchedDocumentCount: number;
};

type Report = {
    generatedAt: string;
    mainDbVersion: string;
    frontendMetadataFile: string;
    frontendVectorFile: string;
    datasetVersion: string;
    datasetMode: string;
    datasetKey: string;
    datasetLabel: string;
    datasetTargetKey?: string;
    presetName: string;
    note: string;
    documentLoaderMode: "local" | "api";
    documentLoaderLabel: string;
    contentApiBaseUrl?: string;
    contentApiPath?: string;
    queryEmbedMode: "per_query";
    fetchMatchLimit: number;
    fetchWeakMatchLimit: number;
    totalCases: number;
    summary: Summary;
    caseReports: CaseLatencyRecord[];
};

const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || "granularity";
const DATASET_FILE = process.env.SUASK_EVAL_DATASET_FILE;
const DATASET_TARGET_KEY = process.env.SUASK_EVAL_DATASET_TARGET as
    | GranularityDatasetTargetKey
    | undefined;
const SINGLE_FILE_AS_ALL = process.env.SUASK_EVAL_SINGLE_FILE_AS_ALL !== "0";
const LIMIT_PER_SOURCE = Number.parseInt(
    process.env.SUASK_EVAL_LIMIT_PER_SOURCE || "",
    10,
);
const DOCUMENT_LOADER_MODE =
    (process.env.SUASK_DOCUMENT_LOADER_MODE || "local").trim().toLowerCase() ===
    "api"
        ? "api"
        : "local";
const CONTENT_API_BASE_URL = process.env.SUASK_CONTENT_API_BASE_URL?.trim();
const CONTENT_API_PATH = process.env.SUASK_CONTENT_API_PATH?.trim();

const DATASET_CONFIG = resolveEvalDatasetConfig({
    datasetVersion: DATASET_VERSION,
    datasetFile: DATASET_FILE,
    singleFileAsAll: SINGLE_FILE_AS_ALL,
    datasetTargetKey: DATASET_TARGET_KEY,
});

function safeAvg(values: number[]): number {
    if (values.length === 0) {
        return 0;
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: number[], ratio: number): number {
    if (values.length === 0) {
        return 0;
    }

    const sorted = [...values].sort((left, right) => left - right);
    const index = Math.min(
        sorted.length - 1,
        Math.max(0, Math.ceil(sorted.length * ratio) - 1),
    );
    return sorted[index];
}

function round4(value: number): number {
    return Number(value.toFixed(4));
}

function round2(value: number): number {
    return Number(value.toFixed(2));
}

async function embedSingleQuery(
    extractor: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["extractor"],
    query: string,
): Promise<Float32Array> {
    const output = await extractor(query, {
        pooling: "mean",
        normalize: true,
        truncation: true,
        max_length: 512,
    } as any);

    return new Float32Array(output.data as Float32Array);
}

function resolveDocumentLoader(): {
    loader: ReturnType<typeof createLocalDocumentLoader>;
    mode: "local" | "api";
    label: string;
    baseUrl?: string;
    apiPath?: string;
} {
    if (DOCUMENT_LOADER_MODE === "api") {
        const resolvedBaseUrl = CONTENT_API_BASE_URL || "http://127.0.0.1:8000";
        const resolvedApiPath = CONTENT_API_PATH || "/api/get_answers";
        return {
            loader: createApiDocumentLoader({
                baseUrl: resolvedBaseUrl,
                path: resolvedApiPath,
            }),
            mode: "api",
            label: `${resolvedBaseUrl}${resolvedApiPath.startsWith("/") ? resolvedApiPath : `/${resolvedApiPath}`}`,
            baseUrl: resolvedBaseUrl,
            apiPath: resolvedApiPath,
        };
    }

    return {
        loader: createLocalDocumentLoader(),
        mode: "local",
        label: "local_document_provider",
    };
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

function getBestRankForOtidSet(
    matches: readonly { otid?: string }[],
    acceptableOtids: readonly string[],
): number {
    if (acceptableOtids.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const acceptableOtidSet = new Set(acceptableOtids);
    const rankIndex = matches.findIndex(
        (item) => item.otid && acceptableOtidSet.has(item.otid),
    );
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
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
            ].filter(
                (item): item is string =>
                    typeof item === "string" && item.length > 0,
            ),
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

function computeCoverageDepth(
    requiredGroups: readonly string[][],
    minGroupsToCover: number,
    matches: readonly { otid?: string }[],
): number {
    if (requiredGroups.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const groupDepths = requiredGroups.map((group) => {
        const groupSet = new Set(group);
        const rankIndex = matches.findIndex(
            (match) => match.otid && groupSet.has(match.otid),
        );
        return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
    });

    const sortedDepths = groupDepths
        .filter((depth) => Number.isFinite(depth))
        .sort((left, right) => left - right);
    const requiredCount = Math.max(
        1,
        Math.min(minGroupsToCover || requiredGroups.length, requiredGroups.length),
    );

    return sortedDepths.length >= requiredCount
        ? sortedDepths[requiredCount - 1]
        : Number.POSITIVE_INFINITY;
}

function getRankForCase(
    matches: readonly { otid?: string }[],
    testCase: DatasetCase,
): number {
    const target = resolveOtidEvalTarget(testCase);
    if (target.mode === "required_otid_groups") {
        return computeCoverageDepth(
            target.requiredOtidGroups,
            target.minGroupsToCover,
            matches,
        );
    }

    return getBestRankForOtidSet(matches, target.acceptableOtids);
}

function rankToNullable(rank: number): number | null {
    return Number.isFinite(rank) ? rank : null;
}

function buildSummary(caseReports: CaseLatencyRecord[]): Summary {
    const totalCases = caseReports.length || 1;
    const answerCount = caseReports.filter(
        (item) => item.behavior === "answer",
    ).length;
    const rejectCount = caseReports.filter(
        (item) => item.behavior === "reject",
    ).length;
    const hitAt1 = caseReports.filter((item) => item.hitAt1).length;
    const hitAt5 = caseReports.filter((item) => item.hitAt5).length;
    const mrr = caseReports.reduce((sum, item) => sum + item.mrr, 0) / totalCases;

    const embedMsList = caseReports.map((item) => item.embedMs);
    const searchMsList = caseReports.map((item) => item.searchMs);
    const fetchMsList = caseReports.map((item) => item.fetchMs);
    const stageTotalMsList = caseReports.map((item) => item.stageTotalMs);
    const endToEndMsList = caseReports.map((item) => item.endToEndMs);

    return {
        totalCases: caseReports.length,
        answerRate: round2((answerCount / totalCases) * 100),
        rejectRate: round2((rejectCount / totalCases) * 100),
        hitAt1: round2((hitAt1 / totalCases) * 100),
        hitAt5: round2((hitAt5 / totalCases) * 100),
        mrr: round4(mrr),
        avgEmbedMs: round4(safeAvg(embedMsList)),
        p50EmbedMs: round4(percentile(embedMsList, 0.5)),
        p95EmbedMs: round4(percentile(embedMsList, 0.95)),
        avgSearchMs: round4(safeAvg(searchMsList)),
        p50SearchMs: round4(percentile(searchMsList, 0.5)),
        p95SearchMs: round4(percentile(searchMsList, 0.95)),
        avgFetchMs: round4(safeAvg(fetchMsList)),
        p50FetchMs: round4(percentile(fetchMsList, 0.5)),
        p95FetchMs: round4(percentile(fetchMsList, 0.95)),
        avgStageTotalMs: round4(safeAvg(stageTotalMsList)),
        p50StageTotalMs: round4(percentile(stageTotalMsList, 0.5)),
        p95StageTotalMs: round4(percentile(stageTotalMsList, 0.95)),
        avgEndToEndMs: round4(safeAvg(endToEndMsList)),
        p50EndToEndMs: round4(percentile(endToEndMsList, 0.5)),
        p95EndToEndMs: round4(percentile(endToEndMsList, 0.95)),
        avgFetchedDocumentCount: round4(
            safeAvg(caseReports.map((item) => item.fetchedDocumentCount)),
        ),
    };
}

async function main() {
    const engine = await loadFrontendEvalEngine();
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoaderConfig = resolveDocumentLoader();
    const documentLoader = documentLoaderConfig.loader;
    const testCases = loadDatasetSources(DATASET_CONFIG.allSources, {
        limitPerSource:
            Number.isFinite(LIMIT_PER_SOURCE) && LIMIT_PER_SOURCE > 0
                ? LIMIT_PER_SOURCE
                : undefined,
    });

    if (testCases.length === 0) {
        throw new Error("当前评测集为空，无法执行完整时延评测。");
    }

    const caseReports: CaseLatencyRecord[] = [];

    for (let index = 0; index < testCases.length; index += 1) {
        const testCase = testCases[index];

        const embedStartedAt = performance.now();
        const queryVector = await embedSingleQuery(engine.extractor, testCase.query);
        const embedMs = performance.now() - embedStartedAt;

        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            engine.vocabMap,
            engine.topicPartitionIndex,
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        );

        const pipelineResult = await executeSearchPipeline({
            query: testCase.query,
            queryVector,
            queryContext,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: Date.now() / 1000,
            bm25Stats: engine.bm25Stats,
            documentLoader,
            termMaps,
            preset: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        });

        const rank = getRankForCase(pipelineResult.results, testCase);

        caseReports.push({
            id: `${DATASET_CONFIG.datasetKey}:${index + 1}`,
            query: testCase.query,
            dataset: testCase.dataset,
            behavior: pipelineResult.finalDecision.behavior,
            rank: rankToNullable(rank),
            hitAt1: rank === 1,
            hitAt5: Number.isFinite(rank) && rank <= 5,
            mrr: Number.isFinite(rank) ? 1 / rank : 0,
            embedMs,
            searchMs: pipelineResult.trace.searchMs,
            fetchMs: pipelineResult.trace.fetchMs,
            stageTotalMs: pipelineResult.trace.totalMs,
            endToEndMs: embedMs + pipelineResult.trace.totalMs,
            matchCount: pipelineResult.trace.matchCount,
            weakMatchCount: pipelineResult.trace.weakMatchCount,
            fetchedDocumentCount: pipelineResult.trace.fetchedDocumentCount,
            topOtid:
                pipelineResult.results[0]?.otid ||
                pipelineResult.weakResults[0]?.otid ||
                null,
        });

        if ((index + 1) % 10 === 0 || index + 1 === testCases.length) {
            console.log(
                `Processed ${index + 1} / ${testCases.length} queries for ${DATASET_CONFIG.datasetKey}`,
            );
        }
    }

    const summary = buildSummary(caseReports);
    const report: Report = {
        generatedAt: new Date().toISOString(),
        mainDbVersion: ACTIVE_MAIN_DB_VERSION,
        frontendMetadataFile: FRONTEND_METADATA_FILE,
        frontendVectorFile: FRONTEND_VECTOR_FILE,
        datasetVersion: DATASET_CONFIG.datasetVersion,
        datasetMode: DATASET_CONFIG.datasetMode,
        datasetKey: DATASET_CONFIG.datasetKey,
        datasetLabel: DATASET_CONFIG.datasetLabel,
        datasetTargetKey: DATASET_TARGET_KEY,
        presetName: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name,
        note:
            "正式完整时延：逐条查询执行 query embedding，并调用当前主链 executeSearchPipeline 统计 retrieval+fetch。当前主链不包含 display rerank、display threshold 或 direct answer rescue；endToEndMs = embedMs + pipeline trace totalMs。" +
            (documentLoaderConfig.mode === "api"
                ? " fetchMs 基于真实 HTTP /api/get_answers 抓文。"
                : " fetchMs 基于本地 JSON 直读，不含 HTTP 往返。"),
        documentLoaderMode: documentLoaderConfig.mode,
        documentLoaderLabel: documentLoaderConfig.label,
        contentApiBaseUrl: documentLoaderConfig.baseUrl,
        contentApiPath: documentLoaderConfig.apiPath,
        queryEmbedMode: "per_query",
        fetchMatchLimit: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.fetchMatchLimit,
        fetchWeakMatchLimit:
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.fetchWeakMatchLimit,
        totalCases: testCases.length,
        summary,
        caseReports,
    };

    const resultDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(resultDir, { recursive: true });
    const outputPath = path.join(
        resultDir,
        `frontend_full_latency_${documentLoaderConfig.mode}_${DATASET_CONFIG.datasetKey}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log("\n===== Frontend Full Latency Summary =====");
    console.log(
        `Dataset: ${DATASET_CONFIG.datasetLabel} (${DATASET_CONFIG.datasetKey})`,
    );
    console.log(
        `Document loader: ${documentLoaderConfig.mode} (${documentLoaderConfig.label})`,
    );
    console.log(
        `behavior    | answerRate=${summary.answerRate.toFixed(2)}% | rejectRate=${summary.rejectRate.toFixed(2)}% | Hit@1=${summary.hitAt1.toFixed(2)}% | Hit@5=${summary.hitAt5.toFixed(2)}% | MRR=${summary.mrr.toFixed(4)}`,
    );
    console.log(
        `latency     | avgEmbedMs=${summary.avgEmbedMs.toFixed(4)} | avgSearchMs=${summary.avgSearchMs.toFixed(4)} | avgFetchMs=${summary.avgFetchMs.toFixed(4)} | avgStageTotalMs=${summary.avgStageTotalMs.toFixed(4)} | avgEndToEndMs=${summary.avgEndToEndMs.toFixed(4)} | p95EndToEndMs=${summary.p95EndToEndMs.toFixed(4)}`,
    );
    console.log(`Report saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

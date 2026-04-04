import * as fs from "fs";
import * as path from "path";
import { performance } from "perf_hooks";

import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    executeRetrievalStage,
    mergeCoarseMatchesIntoDocuments,
} from "../src/worker/search_pipeline.ts";
import { runDirectAnswerDisplayStage } from "../src/worker/direct_answer_display.ts";
import {
    ACTIVE_MAIN_DB_VERSION,
    FRONTEND_METADATA_FILE,
    FRONTEND_VECTOR_FILE,
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    loadDatasetSources,
    resolveEvalDatasetConfig,
    type EvalDatasetCase,
    type GranularityDatasetTargetKey,
    type OtidEvalMode,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";

type DatasetCase = EvalDatasetCase;

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

type ModeName = "coarse_only" | "current_ui";

type CaseLatencyRecord = {
    id: string;
    query: string;
    dataset: string;
    mode: ModeName;
    rank: number | null;
    hitAt1: boolean;
    hitAt5: boolean;
    mrr: number;
    searchMs: number;
    fetchMs: number;
    rerankMs: number;
    totalMs: number;
    matchCount: number;
    fetchedDocumentCount: number;
    rerankedDocCount: number;
    chunksScored: number;
    displayRejected: boolean;
    rescueAttempted: boolean;
    rescueAccepted: boolean;
    topOtid: string | null;
};

type ModeSummary = {
    mode: ModeName;
    totalCases: number;
    hitAt1: number;
    hitAt5: number;
    mrr: number;
    avgSearchMs: number;
    p50SearchMs: number;
    p95SearchMs: number;
    avgFetchMs: number;
    p50FetchMs: number;
    p95FetchMs: number;
    avgRerankMs: number;
    p50RerankMs: number;
    p95RerankMs: number;
    avgTotalMs: number;
    p50TotalMs: number;
    p95TotalMs: number;
    avgRerankedDocCount: number;
    avgChunksScored: number;
    displayRejectRate: number;
    rescueAttemptRate: number;
    rescueAcceptedRate: number;
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
    queryEmbedBatchSize: number;
    fetchMatchLimit: number;
    totalCases: number;
    summaries: ModeSummary[];
    deltas: {
        hitAt1: number;
        hitAt5: number;
        mrr: number;
        avgRerankMs: number;
        avgTotalMs: number;
        p95RerankMs: number;
        p95TotalMs: number;
    };
    caseReports: CaseLatencyRecord[];
};

const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || "granularity";
const DATASET_FILE = process.env.SUASK_EVAL_DATASET_FILE;
const DATASET_TARGET_KEY = process.env.SUASK_EVAL_DATASET_TARGET as
    | GranularityDatasetTargetKey
    | undefined;
const SINGLE_FILE_AS_ALL = process.env.SUASK_EVAL_SINGLE_FILE_AS_ALL !== "0";
const QUERY_EMBED_BATCH_SIZE = Number.parseInt(
    process.env.SUASK_QUERY_EMBED_BATCH_SIZE || "",
    10,
);
const LIMIT_PER_SOURCE = Number.parseInt(
    process.env.SUASK_EVAL_LIMIT_PER_SOURCE || "",
    10,
);

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
    matches: readonly { otid: string }[],
): number {
    if (requiredGroups.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const groupDepths = requiredGroups.map((group) => {
        const groupSet = new Set(group);
        const rankIndex = matches.findIndex((match) => groupSet.has(match.otid));
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
    matches: readonly { otid: string }[],
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

function buildModeSummary(
    mode: ModeName,
    caseReports: CaseLatencyRecord[],
): ModeSummary {
    const totalCases = caseReports.length || 1;
    const hitAt1 = caseReports.filter((item) => item.hitAt1).length;
    const hitAt5 = caseReports.filter((item) => item.hitAt5).length;
    const mrr = caseReports.reduce((sum, item) => sum + item.mrr, 0) / totalCases;

    const searchMsList = caseReports.map((item) => item.searchMs);
    const fetchMsList = caseReports.map((item) => item.fetchMs);
    const rerankMsList = caseReports.map((item) => item.rerankMs);
    const totalMsList = caseReports.map((item) => item.totalMs);

    return {
        mode,
        totalCases: caseReports.length,
        hitAt1: round2((hitAt1 / totalCases) * 100),
        hitAt5: round2((hitAt5 / totalCases) * 100),
        mrr: round4(mrr),
        avgSearchMs: round4(safeAvg(searchMsList)),
        p50SearchMs: round4(percentile(searchMsList, 0.5)),
        p95SearchMs: round4(percentile(searchMsList, 0.95)),
        avgFetchMs: round4(safeAvg(fetchMsList)),
        p50FetchMs: round4(percentile(fetchMsList, 0.5)),
        p95FetchMs: round4(percentile(fetchMsList, 0.95)),
        avgRerankMs: round4(safeAvg(rerankMsList)),
        p50RerankMs: round4(percentile(rerankMsList, 0.5)),
        p95RerankMs: round4(percentile(rerankMsList, 0.95)),
        avgTotalMs: round4(safeAvg(totalMsList)),
        p50TotalMs: round4(percentile(totalMsList, 0.5)),
        p95TotalMs: round4(percentile(totalMsList, 0.95)),
        avgRerankedDocCount: round4(
            safeAvg(caseReports.map((item) => item.rerankedDocCount)),
        ),
        avgChunksScored: round4(
            safeAvg(caseReports.map((item) => item.chunksScored)),
        ),
        displayRejectRate: round2(
            (caseReports.filter((item) => item.displayRejected).length / totalCases) *
                100,
        ),
        rescueAttemptRate: round2(
            (caseReports.filter((item) => item.rescueAttempted).length / totalCases) *
                100,
        ),
        rescueAcceptedRate: round2(
            (caseReports.filter((item) => item.rescueAccepted).length / totalCases) *
                100,
        ),
    };
}

async function main() {
    const engine = await loadFrontendEvalEngine();
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();
    const testCases = loadDatasetSources(DATASET_CONFIG.allSources, {
        limitPerSource:
            Number.isFinite(LIMIT_PER_SOURCE) && LIMIT_PER_SOURCE > 0
                ? LIMIT_PER_SOURCE
                : undefined,
    });

    if (testCases.length === 0) {
        throw new Error("当前评测集为空，无法执行 display rerank A/B。");
    }

    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        testCases.map((item) => item.query),
        engine.dimensions,
        {
            batchSize:
                Number.isFinite(QUERY_EMBED_BATCH_SIZE) && QUERY_EMBED_BATCH_SIZE > 0
                    ? QUERY_EMBED_BATCH_SIZE
                    : DEFAULT_QUERY_EMBED_BATCH_SIZE,
        },
    );

    const coarseOnlyReports: CaseLatencyRecord[] = [];
    const currentUiReports: CaseLatencyRecord[] = [];

    for (let index = 0; index < testCases.length; index += 1) {
        const testCase = testCases[index];
        const queryVector = queryVectors[index];
        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            engine.vocabMap,
            engine.topicPartitionIndex,
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        );

        const retrievalStage = executeRetrievalStage({
            query: testCase.query,
            queryVector,
            queryContext,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: Date.now() / 1000,
            bm25Stats: engine.bm25Stats,
            termMaps,
            preset: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        });

        const coarseMatches = retrievalStage.searchOutput.matches
            .slice(0, FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.fetchMatchLimit)
            .map((item) => ({
                otid: item.otid,
                score: item.score,
                best_kpid: item.best_kpid,
            }));

        const fetchStartedAt = performance.now();
        const fetchedDocuments = await documentLoader({
            query: testCase.query,
            otids: coarseMatches.map((item) => item.otid),
        });
        const fetchMs = performance.now() - fetchStartedAt;
        const directDocuments = mergeCoarseMatchesIntoDocuments(
            fetchedDocuments,
            coarseMatches,
        );

        const coarseRank = getRankForCase(directDocuments, testCase);
        coarseOnlyReports.push({
            id: `${DATASET_CONFIG.datasetKey}:${index + 1}:coarse_only`,
            query: testCase.query,
            dataset: testCase.dataset,
            mode: "coarse_only",
            rank: rankToNullable(coarseRank),
            hitAt1: coarseRank === 1,
            hitAt5: Number.isFinite(coarseRank) && coarseRank <= 5,
            mrr: Number.isFinite(coarseRank) ? 1 / coarseRank : 0,
            searchMs: retrievalStage.searchMs,
            fetchMs,
            rerankMs: 0,
            totalMs: retrievalStage.searchMs + fetchMs,
            matchCount: retrievalStage.searchOutput.matches.length,
            fetchedDocumentCount: directDocuments.length,
            rerankedDocCount: 0,
            chunksScored: 0,
            displayRejected: false,
            rescueAttempted: false,
            rescueAccepted: false,
            topOtid: directDocuments[0]?.otid || null,
        });

        const rerankStartedAt = performance.now();
        const displayStage = await runDirectAnswerDisplayStage({
            query: testCase.query,
            queryVector,
            documents: directDocuments,
            extractor: engine.extractor,
            dimensions: engine.dimensions,
            preset: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
            querySignals: retrievalStage.searchOutput.diagnostics?.querySignals,
            retrievalSignals:
                retrievalStage.searchOutput.diagnostics?.retrievalSignals,
        });
        const rerankMs = performance.now() - rerankStartedAt;
        const currentUiRank = getRankForCase(displayStage.documents, testCase);

        currentUiReports.push({
            id: `${DATASET_CONFIG.datasetKey}:${index + 1}:current_ui`,
            query: testCase.query,
            dataset: testCase.dataset,
            mode: "current_ui",
            rank: rankToNullable(currentUiRank),
            hitAt1: currentUiRank === 1,
            hitAt5: Number.isFinite(currentUiRank) && currentUiRank <= 5,
            mrr: Number.isFinite(currentUiRank) ? 1 / currentUiRank : 0,
            searchMs: retrievalStage.searchMs,
            fetchMs,
            rerankMs,
            totalMs: retrievalStage.searchMs + fetchMs + rerankMs,
            matchCount: retrievalStage.searchOutput.matches.length,
            fetchedDocumentCount: directDocuments.length,
            rerankedDocCount: displayStage.stats.rerankedDocCount,
            chunksScored: displayStage.stats.chunksScored,
            displayRejected: displayStage.displayRejected,
            rescueAttempted: displayStage.directAnswerRescue.attempted,
            rescueAccepted: displayStage.directAnswerRescue.accepted,
            topOtid: displayStage.documents[0]?.otid || null,
        });

        if ((index + 1) % 10 === 0 || index + 1 === testCases.length) {
            console.log(
                `Processed ${index + 1} / ${testCases.length} queries for ${DATASET_CONFIG.datasetKey}`,
            );
        }
    }

    const coarseSummary = buildModeSummary("coarse_only", coarseOnlyReports);
    const currentSummary = buildModeSummary("current_ui", currentUiReports);
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
        note: "真实展示 stage A/B：coarse_only 直接沿用 coarse 排序；current_ui 调用 runDirectAnswerDisplayStage，记录文档命中与 rerank/total 耗时。totalMs 不含 query embedding。",
        queryEmbedBatchSize:
            Number.isFinite(QUERY_EMBED_BATCH_SIZE) && QUERY_EMBED_BATCH_SIZE > 0
                ? QUERY_EMBED_BATCH_SIZE
                : DEFAULT_QUERY_EMBED_BATCH_SIZE,
        fetchMatchLimit: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display.fetchMatchLimit,
        totalCases: testCases.length,
        summaries: [coarseSummary, currentSummary],
        deltas: {
            hitAt1: round2(currentSummary.hitAt1 - coarseSummary.hitAt1),
            hitAt5: round2(currentSummary.hitAt5 - coarseSummary.hitAt5),
            mrr: round4(currentSummary.mrr - coarseSummary.mrr),
            avgRerankMs: round4(currentSummary.avgRerankMs - coarseSummary.avgRerankMs),
            avgTotalMs: round4(currentSummary.avgTotalMs - coarseSummary.avgTotalMs),
            p95RerankMs: round4(currentSummary.p95RerankMs - coarseSummary.p95RerankMs),
            p95TotalMs: round4(currentSummary.p95TotalMs - coarseSummary.p95TotalMs),
        },
        caseReports: [...coarseOnlyReports, ...currentUiReports],
    };

    const resultDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(resultDir, { recursive: true });
    const outputPath = path.join(
        resultDir,
        `display_rerank_ab_${DATASET_CONFIG.datasetKey}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log("\n===== Display Rerank A/B Summary =====");
    console.log(
        `Dataset: ${DATASET_CONFIG.datasetLabel} (${DATASET_CONFIG.datasetKey})`,
    );
    console.log(
        `coarse_only | Hit@1=${coarseSummary.hitAt1.toFixed(2)}% | Hit@5=${coarseSummary.hitAt5.toFixed(2)}% | MRR=${coarseSummary.mrr.toFixed(4)} | avgTotalMs=${coarseSummary.avgTotalMs.toFixed(4)} | p95TotalMs=${coarseSummary.p95TotalMs.toFixed(4)}`,
    );
    console.log(
        `current_ui  | Hit@1=${currentSummary.hitAt1.toFixed(2)}% | Hit@5=${currentSummary.hitAt5.toFixed(2)}% | MRR=${currentSummary.mrr.toFixed(4)} | avgRerankMs=${currentSummary.avgRerankMs.toFixed(4)} | p95RerankMs=${currentSummary.p95RerankMs.toFixed(4)} | avgTotalMs=${currentSummary.avgTotalMs.toFixed(4)} | p95TotalMs=${currentSummary.p95TotalMs.toFixed(4)}`,
    );
    console.log(
        `delta       | Hit@1=${report.deltas.hitAt1.toFixed(2)} | Hit@5=${report.deltas.hitAt5.toFixed(2)} | MRR=${report.deltas.mrr.toFixed(4)} | avgTotalMs=${report.deltas.avgTotalMs.toFixed(4)} | p95TotalMs=${report.deltas.p95TotalMs.toFixed(4)}`,
    );
    console.log(`Report saved to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

import * as fs from "fs";
import * as path from "path";

import {
    getQuerySparse,
    parseQueryIntent,
    searchAndRank,
    type Metadata,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import { FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET } from "../src/worker/search_pipeline.ts";
import { loadDataset, type EvalDatasetCase } from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";

type DatasetCase = EvalDatasetCase & {
    id?: string;
};

type QueryCacheItem = {
    testCase: DatasetCase;
    queryVector: Float32Array;
    queryWords: string[];
    querySparse: Record<number, number>;
    queryIntent: ReturnType<typeof parseQueryIntent>;
    queryYearWordIds: number[];
};

type MetricSummary = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type WeightSetting = {
    label: string;
    denseRrfWeight: number;
    sparseRrfWeight: number;
};

type SettingReport = {
    label: string;
    denseRrfWeight: number;
    sparseRrfWeight: number;
    metrics: MetricSummary;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetLabel: string;
    caseCount: number;
    note: string;
    settings: SettingReport[];
};

const DEFAULT_DATASET_FILE =
    "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_120_reviewed_userized_v1.json";

const DATASET_FILE =
    process.env.SUASK_RRF_SWEEP_DATASET_FILE || DEFAULT_DATASET_FILE;

const DATASET_LABEL =
    process.env.SUASK_RRF_SWEEP_DATASET_LABEL || "legacy_main120_devlike";

const WEIGHT_SETTINGS: WeightSetting[] = [
    { label: "dense100_sparse080", denseRrfWeight: 100, sparseRrfWeight: 80 },
    { label: "dense100_sparse100", denseRrfWeight: 100, sparseRrfWeight: 100 },
    { label: "dense100_sparse120", denseRrfWeight: 100, sparseRrfWeight: 120 },
    { label: "dense100_sparse140", denseRrfWeight: 100, sparseRrfWeight: 140 },
];

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
    bm25Stats: ReturnType<(typeof loadFrontendEvalEngine)> extends () => Promise<infer T>
        ? T["bm25Stats"]
        : never;
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

function evaluateSetting(
    queryCache: readonly QueryCacheItem[],
    metadataList: Metadata[],
    vectorMatrix: Int8Array,
    dimensions: number,
    bm25Stats: Awaited<ReturnType<typeof loadFrontendEvalEngine>>["bm25Stats"],
    setting: WeightSetting,
): SettingReport {
    const ranks = queryCache.map((item) => {
        const result = searchAndRank({
            queryVector: item.queryVector,
            querySparse: item.querySparse,
            queryYearWordIds: item.queryYearWordIds,
            queryIntent: item.queryIntent,
            queryScopeHint: item.testCase.query_scope,
            metadata: metadataList,
            vectorMatrix,
            dimensions,
            currentTimestamp: Date.now() / 1000,
            bm25Stats,
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
            enableExplicitYearFilter:
                FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.enableExplicitYearFilter,
            minimalMode:
                FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval.minimalMode,
            denseRrfWeight: setting.denseRrfWeight,
            sparseRrfWeight: setting.sparseRrfWeight,
        });

        const rankIndex = result.matches.findIndex(
            (match) => match.otid === item.testCase.expected_otid,
        );
        return rankIndex === -1 ? null : rankIndex + 1;
    });

    return {
        label: setting.label,
        denseRrfWeight: setting.denseRrfWeight,
        sparseRrfWeight: setting.sparseRrfWeight,
        metrics: buildMetricSummary(ranks),
    };
}

async function main() {
    const datasetFile = path.resolve(process.cwd(), DATASET_FILE);
    const testCases = loadDataset(datasetFile, {
        datasetLabel: DATASET_LABEL,
    }) as DatasetCase[];

    const { queryCache, metadataList, vectorMatrix, dimensions, bm25Stats } =
        await buildQueryCache(testCases);

    const settings = WEIGHT_SETTINGS.map((setting) =>
        evaluateSetting(
            queryCache,
            metadataList,
            vectorMatrix,
            dimensions,
            bm25Stats,
            setting,
        ),
    );

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetFile,
        datasetLabel: DATASET_LABEL,
        caseCount: testCases.length,
        note: "Historical dev-like granularity set, isolated from the current formal main benchmark.",
        settings,
    };

    const outputDir = path.resolve(process.cwd(), "./scripts/results");
    fs.mkdirSync(outputDir, { recursive: true });
    const outputFile = path.join(
        outputDir,
        `rrf_weight_sweep_${path.basename(datasetFile, ".json")}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputFile, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Dataset: ${datasetFile}`);
    console.log(`Cases: ${testCases.length}`);
    settings.forEach((item) => {
        console.log(
            `${item.label}: Hit@1=${item.metrics.hitAt1.toFixed(2)}% | Hit@3=${item.metrics.hitAt3.toFixed(2)}% | Hit@5=${item.metrics.hitAt5.toFixed(2)}% | MRR=${item.metrics.mrr.toFixed(4)}`,
        );
    });
    console.log(`Saved report: ${outputFile}`);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});

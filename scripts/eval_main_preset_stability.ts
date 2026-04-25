import * as fs from "fs";
import * as path from "path";
import { execFileSync } from "child_process";

import {
    PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    type PipelinePreset,
} from "../src/worker/search_pipeline.ts";
import {
    DEFAULT_GRANULARITY_BENCHMARK_TARGET_KEY,
    DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS,
    resolveGranularityDatasetTarget,
} from "./eval_shared.ts";

type DatasetTarget = {
    name: string;
    label: string;
    file: string;
    isolateForRegistry?: boolean;
};

type PresetProbe = {
    presetName: string;
    label: string;
    comboLabel: "KP+OT" | "Q+KP+OT";
    preset: PipelinePreset;
};

type DatasetMetrics = {
    datasetName: string;
    datasetLabel: string;
    outputPath: string;
    comboLabel: "KP+OT" | "Q+KP+OT";
    docHitAt1: number;
    docMRR: number;
    kpidHitAt1: number;
    kpidMRR: number;
};

type PresetSummary = {
    presetName: string;
    label: string;
    comboLabel: "KP+OT" | "Q+KP+OT";
    retrieval: PipelinePreset["retrieval"];
    datasets: DatasetMetrics[];
    aggregate: {
        benchmarkDocHitAt1?: number;
        benchmarkDocMRR?: number;
        benchmarkKpidHitAt1?: number;
        benchmarkKpidMRR?: number;
        generalizationAverageDocHitAt1: number;
        generalizationAverageDocMRR: number;
        generalizationAverageKpidHitAt1: number;
        generalizationAverageKpidMRR: number;
        generalizationWorstDocHitAt1: number;
        generalizationWorstDocMRR: number;
        generalizationWorstKpidHitAt1: number;
        generalizationWorstKpidMRR: number;
        generalizationDocHitAt1Range: number;
        generalizationKpidHitAt1Range: number;
    };
};

type StabilityReport = {
    generatedAt: string;
    note: string;
    datasets: DatasetTarget[];
    probes: PresetSummary[];
    ranking: Array<{
        rank: number;
        presetName: string;
        label: string;
        comboLabel: "KP+OT" | "Q+KP+OT";
        generalizationWorstDocHitAt1: number;
        generalizationAverageDocHitAt1: number;
        benchmarkDocHitAt1?: number;
        generalizationAverageKpidHitAt1: number;
    }>;
};

const FRONTEND_ROOT = process.cwd();
const RESULTS_DIR = path.resolve(FRONTEND_ROOT, "./scripts/results");
const TEMP_DATASET_DIR = path.resolve(
    FRONTEND_ROOT,
    "../Backend/test/_archive/drafts/preset_stability",
);

function buildDefaultDatasetTargets(): DatasetTarget[] {
    return DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS.flatMap((key) => {
        try {
            const target = resolveGranularityDatasetTarget(key);
            return [
                {
                    name: target.key,
                    label: target.label,
                    file: path.resolve(FRONTEND_ROOT, target.datasetFile),
                    isolateForRegistry: true,
                } satisfies DatasetTarget,
            ];
        } catch {
            return [];
        }
    });
}

const DEFAULT_DATASET_TARGETS: DatasetTarget[] = buildDefaultDatasetTargets();

const DEFAULT_PROBES: PresetProbe[] = [
    {
        presetName: PAPER_FROZEN_MAIN_PIPELINE_PRESET.name,
        label: "论文主配置",
        comboLabel: "KP+OT",
        preset: PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    },
    {
        presetName: PRODUCT_CANONICAL_FULL_PIPELINE_PRESET.name,
        label: "产品旧 canonical 配置",
        comboLabel: "Q+KP+OT",
        preset: PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    },
    {
        presetName: PAPER_TAIL_TOP3_W020_PIPELINE_PRESET.name,
        label: "论文主配置 + top3 tail0.20",
        comboLabel: "KP+OT",
        preset: PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    },
    {
        presetName: PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET.name,
        label: "产品旧 canonical + top3 tail0.20",
        comboLabel: "Q+KP+OT",
        preset: PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    },
];

function parseListEnv(value: string | undefined): string[] | null {
    if (!value) {
        return null;
    }

    const items = value
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0);

    return items.length > 0 ? items : null;
}

function ensureDir(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

function toForwardSlashes(input: string): string {
    return input.replace(/\\/g, "/");
}

function materializeDatasetFile(target: DatasetTarget): string {
    if (!target.isolateForRegistry) {
        return target.file;
    }

    ensureDir(TEMP_DATASET_DIR);
    const suffix = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    const basename = path.basename(target.file, ".json");
    const clonedPath = path.join(
        TEMP_DATASET_DIR,
        `${basename}_preset_stability_${suffix}.json`,
    );
    fs.copyFileSync(target.file, clonedPath);
    return clonedPath;
}

function runGranularityProbe(
    datasetTarget: DatasetTarget,
    probe: PresetProbe,
): DatasetMetrics {
    const datasetFile = materializeDatasetFile(datasetTarget);
    const env = {
        ...process.env,
        SUASK_EVAL_DATASET_FILE: datasetFile,
        SUASK_TOP_HYBRID_LIMIT: String(probe.preset.retrieval.topHybridLimit),
        SUASK_KP_AGG_MODE:
            probe.preset.retrieval.kpAggregationMode === "max_plus_topn"
                ? "max_plus_topn"
                : "max",
        SUASK_KP_TOP_N: String(probe.preset.retrieval.kpTopN),
        SUASK_KP_TAIL_WEIGHT: String(probe.preset.retrieval.kpTailWeight),
        SUASK_LEXICAL_BONUS_MODE: probe.preset.retrieval.lexicalBonusMode,
        SUASK_ONLINE_KP_ROLE_RERANK_MODE:
            probe.preset.retrieval.kpRoleRerankMode,
        SUASK_ONLINE_KP_ROLE_DOC_WEIGHT: String(
            probe.preset.retrieval.kpRoleDocWeight,
        ),
        SUASK_KP_CANDIDATE_RERANK_MODE: "none",
        SUASK_DOC_POST_RERANK_MODE: "none",
        SUASK_FIXED_COMBO_WEIGHTS: JSON.stringify({
            [probe.comboLabel]: probe.preset.retrieval.weights,
        }),
    };

    const stdout =
        process.platform === "win32"
            ? execFileSync(
                  "cmd.exe",
                  ["/d", "/s", "/c", "npm run eval:granularity"],
                  {
                      cwd: FRONTEND_ROOT,
                      env,
                      encoding: "utf-8",
                      stdio: ["ignore", "pipe", "pipe"],
                  },
              )
            : execFileSync("npm", ["run", "eval:granularity"], {
                  cwd: FRONTEND_ROOT,
                  env,
                  encoding: "utf-8",
                  stdio: ["ignore", "pipe", "pipe"],
              });
    const outputMatch = stdout.match(/Saved report to (.+\.json)/);
    if (!outputMatch?.[1]) {
        throw new Error(
            `未能从评测输出中解析结果路径: ${datasetTarget.name} / ${probe.presetName}`,
        );
    }

    const outputPath = outputMatch[1].trim();
    const report = JSON.parse(
        fs.readFileSync(outputPath, "utf-8"),
    ) as {
        combos: Array<{
            label: string;
            tuned: {
                combinedCombined: {
                    hitAt1: number;
                    mrr: number;
                };
                kpidCombinedCombined: {
                    hitAt1: number;
                    mrr: number;
                };
            };
        }>;
    };
    const combo = report.combos.find((item) => item.label === probe.comboLabel);
    if (!combo) {
        throw new Error(
            `结果文件中缺少目标组合 ${probe.comboLabel}: ${outputPath}`,
        );
    }

    return {
        datasetName: datasetTarget.name,
        datasetLabel: datasetTarget.label,
        outputPath: toForwardSlashes(outputPath),
        comboLabel: probe.comboLabel,
        docHitAt1: combo.tuned.combinedCombined.hitAt1,
        docMRR: combo.tuned.combinedCombined.mrr,
        kpidHitAt1: combo.tuned.kpidCombinedCombined.hitAt1,
        kpidMRR: combo.tuned.kpidCombinedCombined.mrr,
    };
}

function mean(values: number[]): number {
    if (values.length === 0) {
        return 0;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeSummary(
    probe: PresetProbe,
    datasets: DatasetMetrics[],
): PresetSummary {
    const benchmark = datasets.find(
        (item) => item.datasetName === DEFAULT_GRANULARITY_BENCHMARK_TARGET_KEY,
    );
    const generalizationSets = datasets.filter(
        (item) => item.datasetName !== DEFAULT_GRANULARITY_BENCHMARK_TARGET_KEY,
    );
    const generalizationDocHitAt1 = generalizationSets.map((item) => item.docHitAt1);
    const generalizationDocMRR = generalizationSets.map((item) => item.docMRR);
    const generalizationKpidHitAt1 = generalizationSets.map(
        (item) => item.kpidHitAt1,
    );
    const generalizationKpidMRR = generalizationSets.map((item) => item.kpidMRR);

    return {
        presetName: probe.presetName,
        label: probe.label,
        comboLabel: probe.comboLabel,
        retrieval: probe.preset.retrieval,
        datasets,
        aggregate: {
            benchmarkDocHitAt1: benchmark?.docHitAt1,
            benchmarkDocMRR: benchmark?.docMRR,
            benchmarkKpidHitAt1: benchmark?.kpidHitAt1,
            benchmarkKpidMRR: benchmark?.kpidMRR,
            generalizationAverageDocHitAt1: mean(generalizationDocHitAt1),
            generalizationAverageDocMRR: mean(generalizationDocMRR),
            generalizationAverageKpidHitAt1: mean(generalizationKpidHitAt1),
            generalizationAverageKpidMRR: mean(generalizationKpidMRR),
            generalizationWorstDocHitAt1:
                generalizationDocHitAt1.length > 0
                    ? Math.min(...generalizationDocHitAt1)
                    : 0,
            generalizationWorstDocMRR:
                generalizationDocMRR.length > 0
                    ? Math.min(...generalizationDocMRR)
                    : 0,
            generalizationWorstKpidHitAt1:
                generalizationKpidHitAt1.length > 0
                    ? Math.min(...generalizationKpidHitAt1)
                    : 0,
            generalizationWorstKpidMRR:
                generalizationKpidMRR.length > 0
                    ? Math.min(...generalizationKpidMRR)
                    : 0,
            generalizationDocHitAt1Range:
                generalizationDocHitAt1.length > 0
                    ? Math.max(...generalizationDocHitAt1) -
                      Math.min(...generalizationDocHitAt1)
                    : 0,
            generalizationKpidHitAt1Range:
                generalizationKpidHitAt1.length > 0
                    ? Math.max(...generalizationKpidHitAt1) -
                      Math.min(...generalizationKpidHitAt1)
                    : 0,
        },
    };
}

function buildRanking(probes: PresetSummary[]): StabilityReport["ranking"] {
    return [...probes]
        .sort((left, right) => {
            if (
                right.aggregate.generalizationWorstDocHitAt1 !==
                left.aggregate.generalizationWorstDocHitAt1
            ) {
                return (
                    right.aggregate.generalizationWorstDocHitAt1 -
                    left.aggregate.generalizationWorstDocHitAt1
                );
            }
            if (
                right.aggregate.generalizationAverageDocHitAt1 !==
                left.aggregate.generalizationAverageDocHitAt1
            ) {
                return (
                    right.aggregate.generalizationAverageDocHitAt1 -
                    left.aggregate.generalizationAverageDocHitAt1
                );
            }
            return (
                (right.aggregate.benchmarkDocHitAt1 || 0) -
                (left.aggregate.benchmarkDocHitAt1 || 0)
            );
        })
        .map((item, index) => ({
            rank: index + 1,
            presetName: item.presetName,
            label: item.label,
            comboLabel: item.comboLabel,
            generalizationWorstDocHitAt1:
                item.aggregate.generalizationWorstDocHitAt1,
            generalizationAverageDocHitAt1:
                item.aggregate.generalizationAverageDocHitAt1,
            benchmarkDocHitAt1: item.aggregate.benchmarkDocHitAt1,
            generalizationAverageKpidHitAt1:
                item.aggregate.generalizationAverageKpidHitAt1,
        }));
}

async function main(): Promise<void> {
    ensureDir(RESULTS_DIR);

    const selectedDatasetNames = parseListEnv(
        process.env.SUASK_STABILITY_DATASET_NAMES,
    );
    const selectedPresetNames = parseListEnv(
        process.env.SUASK_STABILITY_PRESET_NAMES,
    );

    const datasetTargets = DEFAULT_DATASET_TARGETS.filter((item) =>
        selectedDatasetNames ? selectedDatasetNames.includes(item.name) : true,
    );
    const probes = DEFAULT_PROBES.filter((item) =>
        selectedPresetNames ? selectedPresetNames.includes(item.presetName) : true,
    );

    if (datasetTargets.length === 0) {
        throw new Error("未选中任何数据集，请检查 SUASK_STABILITY_DATASET_NAMES。");
    }
    if (probes.length === 0) {
        throw new Error("未选中任何 preset，请检查 SUASK_STABILITY_PRESET_NAMES。");
    }

    const summaries: PresetSummary[] = [];

    for (const probe of probes) {
        console.log(`\n=== ${probe.label} (${probe.presetName}) ===`);
        const datasetMetrics: DatasetMetrics[] = [];

        for (const datasetTarget of datasetTargets) {
            console.log(`Running ${datasetTarget.label} ...`);
            const metrics = runGranularityProbe(datasetTarget, probe);
            datasetMetrics.push(metrics);
            console.log(
                [
                    `${datasetTarget.label}`,
                    `${metrics.comboLabel}`,
                    `doc Hit@1=${metrics.docHitAt1.toFixed(2)}%`,
                    `MRR=${metrics.docMRR.toFixed(4)}`,
                    `kpid Hit@1=${metrics.kpidHitAt1.toFixed(2)}%`,
                    `kpid MRR=${metrics.kpidMRR.toFixed(4)}`,
                ].join(" | "),
            );
        }

        summaries.push(computeSummary(probe, datasetMetrics));
    }

    const report: StabilityReport = {
        generatedAt: new Date().toISOString(),
        note:
            "该报告用于在当前主线三测试集上比较固定主配置的稳定性；排序优先看泛化集最差 Hit@1，再看泛化集平均 Hit@1，最后看主集。",
        datasets: datasetTargets,
        probes: summaries,
        ranking: buildRanking(summaries),
    };

    const outputPath = path.join(
        RESULTS_DIR,
        `preset_stability_granularity_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`\nSaved preset stability report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

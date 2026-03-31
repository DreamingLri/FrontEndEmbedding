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
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";

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
        main106DocHitAt1?: number;
        main106DocMRR?: number;
        main106KpidHitAt1?: number;
        main106KpidMRR?: number;
        holdoutAverageDocHitAt1: number;
        holdoutAverageDocMRR: number;
        holdoutAverageKpidHitAt1: number;
        holdoutAverageKpidMRR: number;
        holdoutWorstDocHitAt1: number;
        holdoutWorstDocMRR: number;
        holdoutWorstKpidHitAt1: number;
        holdoutWorstKpidMRR: number;
        holdoutDocHitAt1Range: number;
        holdoutKpidHitAt1Range: number;
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
        holdoutWorstDocHitAt1: number;
        holdoutAverageDocHitAt1: number;
        main106DocHitAt1?: number;
        holdoutAverageKpidHitAt1: number;
    }>;
};

const FRONTEND_ROOT = process.cwd();
const RESULTS_DIR = path.resolve(FRONTEND_ROOT, "./scripts/results");
const TEMP_DATASET_DIR = path.resolve(
    FRONTEND_ROOT,
    "../Backend/test/_archive/drafts/preset_stability",
);

const DEFAULT_DATASET_TARGETS: DatasetTarget[] = [
    {
        name: "main_106",
        label: "main_106",
        file: path.resolve(
            FRONTEND_ROOT,
            CURRENT_EVAL_DATASET_FILES.granularityMain106,
        ),
        isolateForRegistry: true,
    },
    {
        name: "holdout_v1",
        label: "holdout_v1",
        file: path.resolve(
            FRONTEND_ROOT,
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_holdout_v1_reviewed.json",
        ),
    },
    {
        name: "holdout_v2",
        label: "holdout_v2",
        file: path.resolve(
            FRONTEND_ROOT,
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_holdout_v2_reviewed.json",
        ),
    },
    {
        name: "holdout_v3",
        label: "holdout_v3",
        file: path.resolve(
            FRONTEND_ROOT,
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_holdout_v3_reviewed.json",
        ),
    },
];

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
    const main106 = datasets.find((item) => item.datasetName === "main_106");
    const holdouts = datasets.filter((item) => item.datasetName !== "main_106");
    const holdoutDocHitAt1 = holdouts.map((item) => item.docHitAt1);
    const holdoutDocMRR = holdouts.map((item) => item.docMRR);
    const holdoutKpidHitAt1 = holdouts.map((item) => item.kpidHitAt1);
    const holdoutKpidMRR = holdouts.map((item) => item.kpidMRR);

    return {
        presetName: probe.presetName,
        label: probe.label,
        comboLabel: probe.comboLabel,
        retrieval: probe.preset.retrieval,
        datasets,
        aggregate: {
            main106DocHitAt1: main106?.docHitAt1,
            main106DocMRR: main106?.docMRR,
            main106KpidHitAt1: main106?.kpidHitAt1,
            main106KpidMRR: main106?.kpidMRR,
            holdoutAverageDocHitAt1: mean(holdoutDocHitAt1),
            holdoutAverageDocMRR: mean(holdoutDocMRR),
            holdoutAverageKpidHitAt1: mean(holdoutKpidHitAt1),
            holdoutAverageKpidMRR: mean(holdoutKpidMRR),
            holdoutWorstDocHitAt1:
                holdoutDocHitAt1.length > 0 ? Math.min(...holdoutDocHitAt1) : 0,
            holdoutWorstDocMRR:
                holdoutDocMRR.length > 0 ? Math.min(...holdoutDocMRR) : 0,
            holdoutWorstKpidHitAt1:
                holdoutKpidHitAt1.length > 0
                    ? Math.min(...holdoutKpidHitAt1)
                    : 0,
            holdoutWorstKpidMRR:
                holdoutKpidMRR.length > 0 ? Math.min(...holdoutKpidMRR) : 0,
            holdoutDocHitAt1Range:
                holdoutDocHitAt1.length > 0
                    ? Math.max(...holdoutDocHitAt1) - Math.min(...holdoutDocHitAt1)
                    : 0,
            holdoutKpidHitAt1Range:
                holdoutKpidHitAt1.length > 0
                    ? Math.max(...holdoutKpidHitAt1) -
                      Math.min(...holdoutKpidHitAt1)
                    : 0,
        },
    };
}

function buildRanking(probes: PresetSummary[]): StabilityReport["ranking"] {
    return [...probes]
        .sort((left, right) => {
            if (
                right.aggregate.holdoutWorstDocHitAt1 !==
                left.aggregate.holdoutWorstDocHitAt1
            ) {
                return (
                    right.aggregate.holdoutWorstDocHitAt1 -
                    left.aggregate.holdoutWorstDocHitAt1
                );
            }
            if (
                right.aggregate.holdoutAverageDocHitAt1 !==
                left.aggregate.holdoutAverageDocHitAt1
            ) {
                return (
                    right.aggregate.holdoutAverageDocHitAt1 -
                    left.aggregate.holdoutAverageDocHitAt1
                );
            }
            return (
                (right.aggregate.main106DocHitAt1 || 0) -
                (left.aggregate.main106DocHitAt1 || 0)
            );
        })
        .map((item, index) => ({
            rank: index + 1,
            presetName: item.presetName,
            label: item.label,
            comboLabel: item.comboLabel,
            holdoutWorstDocHitAt1: item.aggregate.holdoutWorstDocHitAt1,
            holdoutAverageDocHitAt1: item.aggregate.holdoutAverageDocHitAt1,
            main106DocHitAt1: item.aggregate.main106DocHitAt1,
            holdoutAverageKpidHitAt1: item.aggregate.holdoutAverageKpidHitAt1,
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
            "该报告用于在 granularity 主线与外部 holdout 上比较固定主配置的稳定性；排序优先看 holdout 最差 Hit@1，再看 holdout 平均 Hit@1，最后看 main_106。",
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

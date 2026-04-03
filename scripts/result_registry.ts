import * as fs from "fs";
import * as path from "path";

export type CurrentResultSlot =
    | "granularity_main_bench_120_current"
    | "granularity_in_domain_holdout_50_current"
    | "granularity_external_ood_50_current"
    | "granularity_external_ood_holdout_30_current"
    | "granularity_main_106_current"
    | "granularity_holdout_v3_current"
    | "answer_reject_current"
    | "platform_reject_kb_absent_v2_dev_current"
    | "platform_reject_kb_absent_v2_holdout_current"
    | "platform_reject_kb_absent_pair_control_v2_holdout_flat_current";

type ResultRegistryEntry = {
    slot: CurrentResultSlot;
    label: string;
    datasetName: string;
    datasetFile: string;
    resultFile: string;
    sourceScript: string;
    managedByScript: boolean;
    updatedAt: string;
    note?: string;
};

type ResultRegistryFile = {
    updatedAt: string;
    entries: Partial<Record<CurrentResultSlot, ResultRegistryEntry>>;
};

const SLOT_LABELS: Record<CurrentResultSlot, string> = {
    granularity_main_bench_120_current: "主方法主结果 `main_bench_120`",
    granularity_in_domain_holdout_50_current:
        "同域泛化结果 `in_domain_holdout_50`",
    granularity_external_ood_50_current: "跨域泛化结果 `external_ood_50`",
    granularity_external_ood_holdout_30_current:
        "内部 hard stress `external_ood_holdout_30`",
    granularity_main_106_current: "主方法主结果 `main_106`",
    granularity_holdout_v3_current: "外部泛化主结果 `holdout_v3`",
    answer_reject_current: "唯一行为结果 `answer_reject_holdout`",
    platform_reject_kb_absent_v2_dev_current:
        "高风险拒答边界结果 `kb_absent_v2_dev`",
    platform_reject_kb_absent_v2_holdout_current:
        "高风险拒答边界结果 `kb_absent_v2_holdout`",
    platform_reject_kb_absent_pair_control_v2_holdout_flat_current:
        "高风险拒答对照结果 `pair_control_v2_holdout_flat`",
};

const SLOT_BY_DATASET_NAME: Partial<Record<string, CurrentResultSlot>> = {
    test_dataset_granularity_main_120_reviewed:
        "granularity_main_bench_120_current",
    test_dataset_granularity_main_120_reviewed_userized_v1:
        "granularity_main_bench_120_current",
    test_dataset_granularity_in_domain_holdout_50_reviewed:
        "granularity_in_domain_holdout_50_current",
    test_dataset_granularity_in_domain_holdout_50_reviewed_userized_v1:
        "granularity_in_domain_holdout_50_current",
    test_dataset_granularity_external_ood_50_reviewed_userized_v1:
        "granularity_external_ood_50_current",
    test_dataset_granularity_external_ood_holdout_30_reviewed:
        "granularity_external_ood_holdout_30_current",
    test_dataset_granularity_main_106_reviewed:
        "granularity_main_106_current",
    test_dataset_answer_reject_v1_holdout_reviewed:
        "answer_reject_current",
    test_dataset_platform_reject_kb_absent_v2_dev_reviewed:
        "platform_reject_kb_absent_v2_dev_current",
    test_dataset_platform_reject_kb_absent_v2_holdout_reviewed:
        "platform_reject_kb_absent_v2_holdout_current",
    test_dataset_platform_reject_kb_absent_pair_control_v2_holdout_flat_reviewed:
        "platform_reject_kb_absent_pair_control_v2_holdout_flat_current",
};

function toForwardSlashes(input: string): string {
    return input.replace(/\\/g, "/");
}

function readCurrentResultsRegistry(registryPath: string): ResultRegistryFile {
    if (!fs.existsSync(registryPath)) {
        return {
            updatedAt: "",
            entries: {},
        };
    }

    try {
        return JSON.parse(
            fs.readFileSync(registryPath, "utf-8"),
        ) as ResultRegistryFile;
    } catch {
        return {
            updatedAt: "",
            entries: {},
        };
    }
}

export function resolveCurrentResultSlot(
    datasetName: string,
): CurrentResultSlot | null {
    return SLOT_BY_DATASET_NAME[datasetName] || null;
}

export function updateCurrentResultRegistry(params: {
    datasetName: string;
    datasetFile: string;
    outputPath: string;
    sourceScript: string;
    note?: string;
}): CurrentResultSlot | null {
    const slot = resolveCurrentResultSlot(params.datasetName);
    if (!slot) {
        return null;
    }

    const registryDir = path.resolve(process.cwd(), "./scripts/results/_registry");
    fs.mkdirSync(registryDir, { recursive: true });

    const registryPath = path.join(registryDir, "current_results.json");
    const currentRegistry = readCurrentResultsRegistry(registryPath);
    const updatedAt = new Date().toISOString();
    const nextEntries = {
        ...currentRegistry.entries,
    } as Record<string, ResultRegistryEntry>;

    delete nextEntries["answer_or_reject_current"];
    delete nextEntries["platform_mixed_daily_v1_2_current"];
    delete nextEntries["route_or_clarify_v2_dev_current"];
    delete nextEntries["route_or_clarify_v2_holdout_current"];

    [
        "answer_or_reject_current.json",
        "platform_mixed_daily_v1_2_current.json",
        "route_or_clarify_v2_dev_current.json",
        "route_or_clarify_v2_holdout_current.json",
    ].forEach((fileName) => {
        const filePath = path.join(registryDir, fileName);
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
        }
    });

    const entry: ResultRegistryEntry = {
        slot,
        label: SLOT_LABELS[slot],
        datasetName: params.datasetName,
        datasetFile: toForwardSlashes(
            path.relative(registryDir, path.resolve(process.cwd(), params.datasetFile)),
        ),
        resultFile: toForwardSlashes(
            path.relative(registryDir, path.resolve(params.outputPath)),
        ),
        sourceScript: toForwardSlashes(
            path.relative(
                registryDir,
                path.resolve(process.cwd(), `./scripts/${params.sourceScript}`),
            ),
        ),
        managedByScript: true,
        updatedAt,
        note: params.note,
    };

    const nextRegistry: ResultRegistryFile = {
        updatedAt,
        entries: {
            ...nextEntries,
            [slot]: entry,
        },
    };

    fs.writeFileSync(registryPath, JSON.stringify(nextRegistry, null, 2), "utf-8");
    fs.writeFileSync(
        path.join(registryDir, `${slot}.json`),
        JSON.stringify(entry, null, 2),
        "utf-8",
    );

    return slot;
}

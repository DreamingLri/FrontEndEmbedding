import * as fs from "fs";
import * as path from "path";

export type CurrentResultSlot =
    | "granularity_main_106_current"
    | "granularity_holdout_v3_current"
    | "platform_mixed_daily_v1_2_current"
    | "route_or_clarify_v2_dev_current"
    | "route_or_clarify_v2_holdout_current"
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
    granularity_main_106_current: "主方法主结果 `main_106`",
    granularity_holdout_v3_current: "外部泛化主结果 `holdout_v3`",
    platform_mixed_daily_v1_2_current: "现实问法主结果 `mixed_daily_v1.2`",
    route_or_clarify_v2_dev_current: "行为分流边界结果 `route_v2_dev`",
    route_or_clarify_v2_holdout_current: "行为分流边界结果 `route_v2_holdout`",
    platform_reject_kb_absent_v2_dev_current:
        "高风险拒答边界结果 `kb_absent_v2_dev`",
    platform_reject_kb_absent_v2_holdout_current:
        "高风险拒答边界结果 `kb_absent_v2_holdout`",
    platform_reject_kb_absent_pair_control_v2_holdout_flat_current:
        "高风险拒答对照结果 `pair_control_v2_holdout_flat`",
};

const SLOT_BY_DATASET_NAME: Partial<Record<string, CurrentResultSlot>> = {
    test_dataset_granularity_main_106_reviewed:
        "granularity_main_106_current",
    test_dataset_route_or_clarify_v2_dev_reviewed:
        "route_or_clarify_v2_dev_current",
    test_dataset_route_or_clarify_v2_holdout_reviewed:
        "route_or_clarify_v2_holdout_current",
    test_dataset_platform_mixed_daily_v1_2_reviewed:
        "platform_mixed_daily_v1_2_current",
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
            ...currentRegistry.entries,
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

import * as fs from "fs";
import * as path from "path";

import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import {
    MAIN_DB_VERSION,
    resolveFrontendMetadataFile,
    resolveFrontendVectorFile,
} from "./kb_version_paths.ts";

export type KpEvalMode = "single_anchor" | "aspect_coverage" | "ot_only";

export type EvalDatasetCase = {
    query: string;
    expected_otid: string;
    expected_kpid?: string;
    support_kpids?: string[];
    kp_eval_mode?: KpEvalMode;
    required_kpid_groups?: string[][];
    min_groups_to_cover?: number;
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    support_pattern?: string;
    granularity_sensitive?: boolean;
    theme_family?: string;
    source_dataset?: string;
    source_seed_id?: string;
    challenge_tags?: string[];
    notes?: string;
    dataset: string;
};

type RawEvalDatasetCase = Omit<EvalDatasetCase, "dataset">;

export type EvalDatasetSource = {
    path: string;
    datasetLabel?: string;
    queryTypes?: string[];
};

export type EvalDatasetGroupRole =
    | "benchmark"
    | "in_domain_holdout"
    | "external_ood_holdout"
    | "legacy_tune"
    | "legacy_holdout"
    | "adhoc";

export type EvalDatasetGroup = {
    key: string;
    label: string;
    role: EvalDatasetGroupRole;
    sources: EvalDatasetSource[];
    resolvedFromFallback?: boolean;
};

export type EvalDatasetConfig = {
    datasetVersion: string;
    datasetMode: "split" | "single_file" | "named_group";
    datasetKey: string;
    datasetLabel: string;
    groups: EvalDatasetGroup[];
    tuneSources: EvalDatasetSource[];
    holdoutSources: EvalDatasetSource[];
    allSources: EvalDatasetSource[];
};

export type GranularityDatasetTargetKey =
    | "main_bench_120"
    | "in_domain_holdout_50"
    | "external_ood_holdout_30";

type GranularityDatasetTargetDefinition = {
    key: GranularityDatasetTargetKey;
    label: string;
    role: EvalDatasetGroupRole;
    primaryPath: string;
    fallbackPath?: string;
};

export type ResolvedGranularityDatasetTarget = {
    key: GranularityDatasetTargetKey;
    label: string;
    role: EvalDatasetGroupRole;
    datasetFile: string;
    datasetKey: string;
    resolvedFromFallback: boolean;
};

export const FRONTEND_MODEL_NAME = "DMetaSoul/Dmeta-embedding-zh-small";
export const ACTIVE_MAIN_DB_VERSION = MAIN_DB_VERSION;
export const FRONTEND_METADATA_FILE = resolveFrontendMetadataFile();
export const FRONTEND_VECTOR_FILE = resolveFrontendVectorFile();
export const DEFAULT_QUERY_EMBED_BATCH_SIZE = 16;

const DEFAULT_GRANULARITY_TARGET_KEY: GranularityDatasetTargetKey =
    "main_bench_120";

const GRANULARITY_DATASET_TARGETS: Record<
    GranularityDatasetTargetKey,
    GranularityDatasetTargetDefinition
> = {
    main_bench_120: {
        key: "main_bench_120",
        label: "MainBench-120",
        role: "benchmark",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityMain120,
        // 仅在 MainBench-120 本体缺失时回退到历史主集，正常测评不应依赖这个分支。
        fallbackPath: CURRENT_EVAL_DATASET_FILES.granularityMain106,
    },
    in_domain_holdout_50: {
        key: "in_domain_holdout_50",
        label: "InDomainHoldout-50",
        role: "in_domain_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityInDomainHoldout50,
        // skeleton_v2 只作为未完成 review 时的兼容回退。
        fallbackPath: CURRENT_EVAL_DATASET_FILES.granularityInDomainHoldout50SkeletonV2,
    },
    external_ood_holdout_30: {
        key: "external_ood_holdout_30",
        label: "ExternalOODHoldout-30",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityExternalOodHoldout30,
    },
};

export function loadDataset(
    datasetPath: string,
    options?: {
        datasetLabel?: string;
        queryTypes?: string[];
    },
): EvalDatasetCase[] {
    const absolutePath = path.resolve(process.cwd(), datasetPath);
    const raw = JSON.parse(
        fs.readFileSync(absolutePath, "utf-8"),
    ) as RawEvalDatasetCase[];
    const dataset = options?.datasetLabel || path.basename(datasetPath, ".json");
    const queryTypes = options?.queryTypes;
    const filtered = queryTypes?.length
        ? raw.filter((item) => item.query_type && queryTypes.includes(item.query_type))
        : raw;

    return filtered.map((item) => ({
        ...item,
        dataset,
    }));
}

export function loadDatasetSources(
    sources: readonly EvalDatasetSource[],
    options?: {
        limitPerSource?: number;
    },
): EvalDatasetCase[] {
    return sources.flatMap((source) => {
        const cases = loadDataset(source.path, {
            datasetLabel: source.datasetLabel,
            queryTypes: source.queryTypes,
        });

        if (
            Number.isFinite(options?.limitPerSource) &&
            (options?.limitPerSource || 0) > 0
        ) {
            return cases.slice(0, options?.limitPerSource);
        }

        return cases;
    });
}

function buildDatasetGroup(
    key: string,
    label: string,
    role: EvalDatasetGroupRole,
    sources: EvalDatasetSource[],
    resolvedFromFallback = false,
): EvalDatasetGroup {
    return {
        key,
        label,
        role,
        sources,
        resolvedFromFallback,
    };
}

function buildSplitSources(datasetDir: string): EvalDatasetConfig {
    const tuneSources: EvalDatasetSource[] = [
        { path: `${datasetDir}/test_dataset_standard.json` },
        { path: `${datasetDir}/test_dataset_short_keyword.json` },
    ];
    const holdoutSources: EvalDatasetSource[] = [
        { path: `${datasetDir}/test_dataset_situational.json` },
    ];
    const datasetKey = path.basename(datasetDir);

    return {
        datasetVersion: path.basename(datasetDir).replace(/^test_dataset_/, ""),
        datasetMode: "split",
        datasetKey,
        datasetLabel: datasetKey,
        groups: [
            buildDatasetGroup("legacy_tune", "legacy_tune", "legacy_tune", tuneSources),
            buildDatasetGroup(
                "legacy_holdout",
                "legacy_holdout",
                "legacy_holdout",
                holdoutSources,
            ),
        ],
        tuneSources,
        holdoutSources,
        allSources: [...tuneSources, ...holdoutSources],
    };
}

function buildSingleFileSources(
    datasetVersion: string,
    datasetFile: string,
): EvalDatasetConfig {
    const datasetKey = path.basename(datasetFile, ".json");
    const tuneSources: EvalDatasetSource[] = [
        {
            path: datasetFile,
            datasetLabel: `${datasetKey}_standard`,
            queryTypes: ["standard"],
        },
        {
            path: datasetFile,
            datasetLabel: `${datasetKey}_short_keyword`,
            queryTypes: ["short_keyword"],
        },
    ];
    const holdoutSources: EvalDatasetSource[] = [
        {
            path: datasetFile,
            datasetLabel: `${datasetKey}_situational`,
            queryTypes: ["situational"],
        },
    ];

    return {
        datasetVersion,
        datasetMode: "single_file",
        datasetKey,
        datasetLabel: datasetKey,
        groups: [
            buildDatasetGroup("legacy_tune", "legacy_tune", "legacy_tune", tuneSources),
            buildDatasetGroup(
                "legacy_holdout",
                "legacy_holdout",
                "legacy_holdout",
                holdoutSources,
            ),
        ],
        tuneSources,
        holdoutSources,
        allSources: [...tuneSources, ...holdoutSources],
    };
}

function buildSingleFrozenFileConfig(
    datasetVersion: string,
    datasetFile: string,
    datasetLabel?: string,
): EvalDatasetConfig {
    const datasetKey = path.basename(datasetFile, ".json");
    const sources: EvalDatasetSource[] = [
        {
            path: datasetFile,
            datasetLabel: datasetKey,
        },
    ];

    return {
        datasetVersion,
        datasetMode: "single_file",
        datasetKey,
        datasetLabel: datasetLabel || datasetKey,
        groups: [
            buildDatasetGroup("evaluation", datasetLabel || datasetKey, "adhoc", sources),
        ],
        tuneSources: sources,
        holdoutSources: [],
        allSources: sources,
    };
}

function buildNamedGroupConfig(
    datasetVersion: string,
    target: ResolvedGranularityDatasetTarget,
): EvalDatasetConfig {
    const sources: EvalDatasetSource[] = [
        {
            path: target.datasetFile,
            datasetLabel: target.datasetKey,
        },
    ];

    return {
        datasetVersion,
        datasetMode: "named_group",
        datasetKey: target.datasetKey,
        datasetLabel: target.label,
        groups: [
            buildDatasetGroup(
                target.key,
                target.label,
                target.role,
                sources,
                target.resolvedFromFallback,
            ),
        ],
        tuneSources: sources,
        holdoutSources: [],
        allSources: sources,
    };
}

function resolveExistingDatasetPath(
    datasetPaths: readonly string[],
): { datasetFile: string; resolvedFromFallback: boolean } | null {
    for (let index = 0; index < datasetPaths.length; index++) {
        const datasetPath = datasetPaths[index];
        const absolutePath = path.resolve(process.cwd(), datasetPath);
        if (fs.existsSync(absolutePath)) {
            return {
                datasetFile: datasetPath,
                resolvedFromFallback: index > 0,
            };
        }
    }

    return null;
}

export function resolveGranularityDatasetTarget(
    key: GranularityDatasetTargetKey,
): ResolvedGranularityDatasetTarget {
    const definition = GRANULARITY_DATASET_TARGETS[key];
    const resolved = resolveExistingDatasetPath(
        definition.fallbackPath
            ? [definition.primaryPath, definition.fallbackPath]
            : [definition.primaryPath],
    );

    if (!resolved) {
        throw new Error(
            `Unable to resolve dataset target "${key}". Checked ${[
                definition.primaryPath,
                definition.fallbackPath,
            ]
                .filter(Boolean)
                .join(" and ")}.`,
        );
    }

    return {
        key: definition.key,
        label: definition.label,
        role: definition.role,
        datasetFile: resolved.datasetFile,
        datasetKey: path.basename(resolved.datasetFile, ".json"),
        resolvedFromFallback: resolved.resolvedFromFallback,
    };
}

export function listAvailableGranularityDatasetTargets(
    keys: readonly GranularityDatasetTargetKey[] = [
        "main_bench_120",
        "in_domain_holdout_50",
        "external_ood_holdout_30",
    ],
): ResolvedGranularityDatasetTarget[] {
    return keys.flatMap((key) => {
        try {
            return [resolveGranularityDatasetTarget(key)];
        } catch {
            return [];
        }
    });
}

export function resolveEvalDatasetConfig(options?: {
    datasetVersion?: string;
    datasetFile?: string;
    singleFileAsAll?: boolean;
    datasetTargetKey?: GranularityDatasetTargetKey;
}): EvalDatasetConfig {
    const datasetVersion = options?.datasetVersion || "v2";
    const explicitDatasetFile = options?.datasetFile;
    const singleFileAsAll = options?.singleFileAsAll || false;

    if (explicitDatasetFile) {
        if (singleFileAsAll) {
            return buildSingleFrozenFileConfig(datasetVersion, explicitDatasetFile);
        }
        return buildSingleFileSources(datasetVersion, explicitDatasetFile);
    }

    if (datasetVersion === "granularity") {
        const target = resolveGranularityDatasetTarget(
            options?.datasetTargetKey || DEFAULT_GRANULARITY_TARGET_KEY,
        );
        return buildNamedGroupConfig(datasetVersion, target);
    }

    const datasetDir = `../Backend/test/test_dataset_${datasetVersion}`;
    const splitPaths = [
        `${datasetDir}/test_dataset_standard.json`,
        `${datasetDir}/test_dataset_short_keyword.json`,
        `${datasetDir}/test_dataset_situational.json`,
    ];
    const hasSplitDatasets = splitPaths.every((item) =>
        fs.existsSync(path.resolve(process.cwd(), item)),
    );

    if (hasSplitDatasets) {
        return buildSplitSources(datasetDir);
    }

    const singleFilePath = `${datasetDir}/test_dataset_${datasetVersion}.json`;
    if (fs.existsSync(path.resolve(process.cwd(), singleFilePath))) {
        if (singleFileAsAll) {
            return buildSingleFrozenFileConfig(datasetVersion, singleFilePath);
        }
        return buildSingleFileSources(datasetVersion, singleFilePath);
    }

    throw new Error(
        `Unable to resolve evaluation dataset for version "${datasetVersion}". Checked split files under ${datasetDir} and single file ${singleFilePath}.`,
    );
}

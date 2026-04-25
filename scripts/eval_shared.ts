import * as fs from "fs";
import * as path from "path";

import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import {
    MAIN_DB_VERSION,
    resolveFrontendMetadataFile,
    resolveFrontendVectorFile,
} from "./kb_version_paths.ts";

export type KpEvalMode = "single_anchor" | "aspect_coverage" | "ot_only";
export type OtidEvalMode =
    | "single_expected"
    | "acceptable_otids"
    | "required_otid_groups";

export type EvalDatasetCase = {
    query: string;
    source_query?: string;
    expected_otid: string;
    otid_eval_mode?: OtidEvalMode;
    acceptable_otids?: string[];
    required_otid_groups?: string[][];
    min_otid_groups_to_cover?: number;
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
    | "in_domain_generalization_100"
    | "blind_ext_ood_100"
    | "in_domain_holdout_50"
    | "ext_ood_blind_60"
    | "matched_ext_ood_60"
    | "hard_ood_blind_30"
    | "legacy_external_ood_hard_30"
    | "external_ood_50"
    | "external_ood_holdout_30"
    | "external_ood_hard_30"
    | "hard_ood_v2_diag_top30"
    | "structure_dev_40"
    | "ladder_main_balanced_80"
    | "ladder_generalization_hard_60"
    | "ladder_structure_stress_40"
    | "ladder_main_balanced_120"
    | "ladder_generalization_hard_80"
    | "ladder_structure_stress_60"
    | "ladder_main_balanced_150"
    | "ladder_generalization_hard_100"
    | "ladder_structure_stress_80"
    | "ladder_cross_doc_coverage_diag_18";

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
export const DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS = [
    "ladder_main_balanced_150",
    "in_domain_generalization_100",
    "blind_ext_ood_100",
] as const;
export const DEFAULT_GRANULARITY_BENCHMARK_TARGET_KEY: GranularityDatasetTargetKey =
    DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS[0];
export const DEFAULT_GRANULARITY_DATASET_KEY =
    "granularity_mainline_150_100_100";
export const DEFAULT_GRANULARITY_DATASET_LABEL =
    "MainBalanced150+InDomain100+BlindExtOOD100";

const DEFAULT_GRANULARITY_TARGET_KEY: GranularityDatasetTargetKey =
    DEFAULT_GRANULARITY_BENCHMARK_TARGET_KEY;

const GRANULARITY_DATASET_TARGETS: Record<
    GranularityDatasetTargetKey,
    GranularityDatasetTargetDefinition
> = {
    main_bench_120: {
        key: "main_bench_120",
        label: "Main",
        role: "benchmark",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityMain120,
    },
    in_domain_generalization_100: {
        key: "in_domain_generalization_100",
        label: "InDomain100",
        role: "in_domain_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityInDomainGeneralization100,
    },
    blind_ext_ood_100: {
        key: "blind_ext_ood_100",
        label: "BlindExtOOD100",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityBlindExtOod100,
    },
    in_domain_holdout_50: {
        key: "in_domain_holdout_50",
        label: "InDomainLegacyAlias",
        role: "in_domain_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityInDomainHoldout50,
    },
    ext_ood_blind_60: {
        key: "ext_ood_blind_60",
        label: "BlindExtOODLegacyAlias",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityExtOodBlind60,
    },
    matched_ext_ood_60: {
        key: "matched_ext_ood_60",
        label: "ExtOOD",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityMatchedExtOod60,
    },
    hard_ood_blind_30: {
        key: "hard_ood_blind_30",
        label: "HardOOD",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityHardOodBlind30,
    },
    legacy_external_ood_hard_30: {
        key: "legacy_external_ood_hard_30",
        label: "LegacyHardOOD30",
        role: "legacy_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLegacyExternalOodHard30,
    },
    external_ood_50: {
        key: "external_ood_50",
        // 兼容旧 key，但当前主线下该入口回到 matched external 结果。
        label: "ExtOOD",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityExternalOod50,
    },
    external_ood_holdout_30: {
        key: "external_ood_holdout_30",
        label: "ExtHard30",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityExternalOodHoldout30,
    },
    external_ood_hard_30: {
        key: "external_ood_hard_30",
        // 兼容旧 key，但当前论文口径下该入口已切到 blind HardOOD。
        label: "HardOOD",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityExternalOodHard30,
    },
    hard_ood_v2_diag_top30: {
        key: "hard_ood_v2_diag_top30",
        label: "HardOODv2Diag",
        role: "external_ood_holdout",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityHardOodV2DiagTop30,
    },
    structure_dev_40: {
        key: "structure_dev_40",
        label: "StructureDev40",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityStructureDev40,
    },
    ladder_main_balanced_80: {
        key: "ladder_main_balanced_80",
        label: "MainBalanced80",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderMainBalanced80,
    },
    ladder_generalization_hard_60: {
        key: "ladder_generalization_hard_60",
        label: "GeneralizationHard60",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderGeneralizationHard60,
    },
    ladder_structure_stress_40: {
        key: "ladder_structure_stress_40",
        label: "StructureStress40",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderStructureStress40,
    },
    ladder_main_balanced_120: {
        key: "ladder_main_balanced_120",
        label: "MainBalanced120",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderMainBalanced120,
    },
    ladder_generalization_hard_80: {
        key: "ladder_generalization_hard_80",
        label: "GeneralizationHard80",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderGeneralizationHard80,
    },
    ladder_structure_stress_60: {
        key: "ladder_structure_stress_60",
        label: "StructureStress60",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderStructureStress60,
    },
    ladder_main_balanced_150: {
        key: "ladder_main_balanced_150",
        label: "MainBalanced150",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderMainBalanced150,
    },
    ladder_generalization_hard_100: {
        key: "ladder_generalization_hard_100",
        label: "GeneralizationHard100",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderGeneralizationHard100,
    },
    ladder_structure_stress_80: {
        key: "ladder_structure_stress_80",
        label: "StructureStress80",
        role: "adhoc",
        primaryPath: CURRENT_EVAL_DATASET_FILES.granularityLadderStructureStress80,
    },
    ladder_cross_doc_coverage_diag_18: {
        key: "ladder_cross_doc_coverage_diag_18",
        label: "CrossDocCoverage18",
        role: "adhoc",
        primaryPath:
            CURRENT_EVAL_DATASET_FILES.granularityLadderCrossDocCoverageDiag18,
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

function buildDefaultGranularityMainlineConfig(
    datasetVersion: string,
): EvalDatasetConfig {
    const targets = DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS.map((key) =>
        resolveGranularityDatasetTarget(key),
    );
    const groups = targets.map((target) =>
        buildDatasetGroup(
            target.key,
            target.label,
            target.role,
            [
                {
                    path: target.datasetFile,
                    datasetLabel: target.key,
                },
            ],
            target.resolvedFromFallback,
        ),
    );
    const allSources = groups.flatMap((group) => group.sources);

    return {
        datasetVersion,
        datasetMode: "named_group",
        datasetKey: DEFAULT_GRANULARITY_DATASET_KEY,
        datasetLabel: DEFAULT_GRANULARITY_DATASET_LABEL,
        groups,
        tuneSources: allSources,
        holdoutSources: [],
        allSources,
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
        ...DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS,
        "main_bench_120",
        "in_domain_generalization_100",
        "blind_ext_ood_100",
        "in_domain_holdout_50",
        "ext_ood_blind_60",
        "matched_ext_ood_60",
        "hard_ood_blind_30",
        "hard_ood_v2_diag_top30",
        "structure_dev_40",
        "ladder_main_balanced_80",
        "ladder_generalization_hard_60",
        "ladder_structure_stress_40",
        "ladder_main_balanced_120",
        "ladder_generalization_hard_80",
        "ladder_structure_stress_60",
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
        if (!options?.datasetTargetKey) {
            return buildDefaultGranularityMainlineConfig(datasetVersion);
        }
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

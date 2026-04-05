import * as path from "path";

export type NamedDatasetProfile = {
    canonicalName: string;
    alias: string;
    displayName: string;
};

const DATASET_PROFILE_MAP: Record<string, NamedDatasetProfile> = {
    main_bench_120: {
        canonicalName:
            "test_dataset_granularity_main_benchmark_v2_reviewed_userized_v1",
        alias: "gran_main_v2",
        displayName: "Main",
    },
    test_dataset_granularity_main_benchmark_v2_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_main_benchmark_v2_reviewed_userized_v1",
        alias: "gran_main_v2",
        displayName: "Main",
    },
    in_domain_holdout_50: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1",
        alias: "gran_in_v2",
        displayName: "InDomain",
    },
    test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1",
        alias: "gran_in_v2",
        displayName: "InDomain",
    },
    external_ood_50: {
        canonicalName:
            "test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1",
        alias: "gran_ext_v2",
        displayName: "ExtOOD",
    },
    test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1",
        alias: "gran_ext_v2",
        displayName: "ExtOOD",
    },
    external_ood_holdout_30: {
        canonicalName:
            "test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1",
        alias: "gran_ext_hard30",
        displayName: "ExtHard",
    },
    test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1",
        alias: "gran_ext_hard30",
        displayName: "ExtHard",
    },
    test_dataset_granularity_main_120_reviewed_userized_v1: {
        canonicalName: "test_dataset_granularity_main_120_reviewed_userized_v1",
        alias: "gran_main_120",
        displayName: "Main120",
    },
    test_dataset_granularity_in_domain_holdout_50_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_in_domain_holdout_50_reviewed_userized_v1",
        alias: "gran_in_50",
        displayName: "InDomain50",
    },
    test_dataset_granularity_external_ood_50_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_external_ood_50_reviewed_userized_v1",
        alias: "gran_ext_50",
        displayName: "ExtOOD50",
    },
    test_dataset_answer_reject_v4_frozen_holdout_reviewed: {
        canonicalName: "test_dataset_answer_reject_v4_frozen_holdout_reviewed",
        alias: "ar_v4",
        displayName: "AnswerReject",
    },
};

function normalizeDatasetIdentity(input: string): string {
    const baseName = path.basename(input);
    return baseName.endsWith(".json") ? baseName.slice(0, -5) : baseName;
}

export function resolveNamedDatasetProfile(input: string): NamedDatasetProfile {
    const normalized = normalizeDatasetIdentity(input);
    const matched = DATASET_PROFILE_MAP[normalized];
    if (matched) {
        return matched;
    }

    return {
        canonicalName: normalized,
        alias: normalized.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, ""),
        displayName: normalized,
    };
}

export function buildGranularityResultFileName(
    datasetIdentity: string,
    timestamp: number,
): string {
    return `granularity_${resolveNamedDatasetProfile(datasetIdentity).alias}_${timestamp}.json`;
}

export function buildAnswerRejectResultFileName(
    datasetIdentity: string,
    timestamp: number,
): string {
    return `answer_reject_${resolveNamedDatasetProfile(datasetIdentity).alias}_${timestamp}.json`;
}

export function buildStandardBaselinesResultFileName(timestamp: number): string {
    return `granularity_baselines_${timestamp}.json`;
}

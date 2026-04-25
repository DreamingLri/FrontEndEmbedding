import * as path from "path";

export type NamedDatasetProfile = {
    canonicalName: string;
    alias: string;
    displayName: string;
};

const DATASET_PROFILE_MAP: Record<string, NamedDatasetProfile> = {
    granularity_domain_generalization_bundle: {
        canonicalName: "granularity_domain_generalization_bundle",
        alias: "granularity_domain_generalization_120_60_60",
        displayName: "GranularityMain+InDomain+ExtOOD",
    },
    granularity_mainline_bundle: {
        canonicalName: "granularity_mainline_bundle",
        alias: "granularity_mainline_150_100_100",
        displayName: "GranularityMainline150+InDomain100+BlindExtOOD100",
    },
    granularity_mainline_150_100_100: {
        canonicalName: "granularity_mainline_150_100_100",
        alias: "granularity_mainline_150_100_100",
        displayName: "GranularityMainline150+InDomain100+BlindExtOOD100",
    },
    main_bench_120: {
        canonicalName:
            "test_dataset_granularity_main_generalization_aligned_120_draft_v4",
        alias: "gran_main_generalization_v4",
        displayName: "Main",
    },
    test_dataset_granularity_main_generalization_aligned_120_draft_v4: {
        canonicalName:
            "test_dataset_granularity_main_generalization_aligned_120_draft_v4",
        alias: "gran_main_generalization_v4",
        displayName: "Main",
    },
    test_dataset_granularity_main_generalization_aligned_120_draft_v3: {
        canonicalName:
            "test_dataset_granularity_main_generalization_aligned_120_draft_v3",
        alias: "gran_main_generalization_v3",
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
            "test_dataset_granularity_in_domain_generalization_aligned_100_draft_v7",
        alias: "gran_indomain100_generalization_v7",
        displayName: "InDomain100",
    },
    in_domain_generalization_100: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_aligned_100_draft_v7",
        alias: "gran_indomain100_generalization_v7",
        displayName: "InDomain100",
    },
    test_dataset_granularity_in_domain_generalization_aligned_100_draft_v7: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_aligned_100_draft_v7",
        alias: "gran_indomain_generalization_v7",
        displayName: "InDomain",
    },
    test_dataset_granularity_in_domain_generalization_aligned_100_draft_v6: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_aligned_100_draft_v6",
        alias: "gran_indomain_generalization_v6",
        displayName: "InDomain",
    },
    test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v2: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v2",
        alias: "gran_in_v2",
        displayName: "InDomain",
    },
    test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1",
        alias: "gran_in_v2",
        displayName: "InDomain",
    },
    ext_ood_blind_60: {
        canonicalName:
            "test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v4",
        alias: "gran_blind_extood100_generalization_v4",
        displayName: "BlindExtOOD100",
    },
    blind_ext_ood_100: {
        canonicalName:
            "test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v4",
        alias: "gran_blind_extood100_generalization_v4",
        displayName: "BlindExtOOD100",
    },
    test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v4: {
        canonicalName:
            "test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v4",
        alias: "gran_blind_extood_generalization_v4",
        displayName: "BlindExtOOD",
    },
    test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v3: {
        canonicalName:
            "test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v3",
        alias: "gran_blind_extood_generalization_v3",
        displayName: "BlindExtOOD",
    },
    test_dataset_granularity_aligned_ext_ood_blind_60_draft_v1: {
        canonicalName:
            "test_dataset_granularity_aligned_ext_ood_blind_60_draft_v1",
        alias: "gran_ext_blind60",
        displayName: "BlindExtOOD",
    },
    matched_ext_ood_60: {
        canonicalName:
            "test_dataset_granularity_external_matched_ood_60_reviewed_userized_v2",
        alias: "gran_ext_v2",
        displayName: "ExtOOD",
    },
    external_ood_50: {
        canonicalName:
            "test_dataset_granularity_external_matched_ood_60_reviewed_userized_v2",
        alias: "gran_ext_v2",
        displayName: "ExtOOD",
    },
    test_dataset_granularity_external_matched_ood_60_reviewed_userized_v2: {
        canonicalName:
            "test_dataset_granularity_external_matched_ood_60_reviewed_userized_v2",
        alias: "gran_ext_v2",
        displayName: "ExtOOD",
    },
    test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1",
        alias: "gran_ext_v2",
        displayName: "ExtOOD",
    },
    hard_ood_blind_30: {
        canonicalName: "test_dataset_granularity_hard_ood_blind_30_draft_v1",
        alias: "gran_hardood_blind30",
        displayName: "HardOOD",
    },
    test_dataset_granularity_hard_ood_blind_30_draft_v1: {
        canonicalName: "test_dataset_granularity_hard_ood_blind_30_draft_v1",
        alias: "gran_hardood_blind30",
        displayName: "HardOOD",
    },
    external_ood_holdout_30: {
        canonicalName:
            "test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1",
        alias: "gran_ext_hard30",
        displayName: "ExtHard30",
    },
    external_ood_hard_30: {
        canonicalName: "test_dataset_granularity_hard_ood_blind_30_draft_v1",
        alias: "gran_hardood_blind30",
        displayName: "HardOOD",
    },
    legacy_external_ood_hard_30: {
        canonicalName:
            "test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1",
        alias: "gran_legacy_hard30",
        displayName: "LegacyHardOOD30",
    },
    hard_ood_v2_diag_top30: {
        canonicalName: "test_dataset_granularity_hard_ood_v2_diag_top30",
        alias: "gran_hardood_v2diag30",
        displayName: "HardOODv2Diag",
    },
    test_dataset_granularity_hard_ood_v2_diag_top30: {
        canonicalName: "test_dataset_granularity_hard_ood_v2_diag_top30",
        alias: "gran_hardood_v2diag30",
        displayName: "HardOODv2Diag",
    },
    structure_dev_40: {
        canonicalName: "test_dataset_granularity_structure_dev_40_draft_v1",
        alias: "gran_structure_dev40",
        displayName: "StructureDev40",
    },
    ladder_main_balanced_80: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_80_draft_v2",
        alias: "gran_ladder_main80",
        displayName: "MainBalanced80",
    },
    test_dataset_granularity_ladder_main_balanced_80_draft_v2: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_80_draft_v2",
        alias: "gran_ladder_main80",
        displayName: "MainBalanced80",
    },
    test_dataset_granularity_ladder_main_balanced_80_draft_v1: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_80_draft_v1",
        alias: "gran_ladder_main80",
        displayName: "MainBalanced80",
    },
    ladder_generalization_hard_60: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_60_draft_v2",
        alias: "gran_ladder_genhard60",
        displayName: "GeneralizationHard60",
    },
    test_dataset_granularity_ladder_generalization_hard_60_draft_v2: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_60_draft_v2",
        alias: "gran_ladder_genhard60",
        displayName: "GeneralizationHard60",
    },
    test_dataset_granularity_ladder_generalization_hard_60_draft_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_60_draft_v1",
        alias: "gran_ladder_genhard60",
        displayName: "GeneralizationHard60",
    },
    ladder_structure_stress_40: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_40_draft_v2",
        alias: "gran_ladder_stress40",
        displayName: "StructureStress40",
    },
    test_dataset_granularity_ladder_structure_stress_40_draft_v2: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_40_draft_v2",
        alias: "gran_ladder_stress40",
        displayName: "StructureStress40",
    },
    test_dataset_granularity_ladder_structure_stress_40_draft_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_40_draft_v1",
        alias: "gran_ladder_stress40",
        displayName: "StructureStress40",
    },
    ladder_main_balanced_120: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_120_draft_v3",
        alias: "gran_ladder_main120",
        displayName: "MainBalanced120",
    },
    test_dataset_granularity_ladder_main_balanced_120_draft_v3: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_120_draft_v3",
        alias: "gran_ladder_main120",
        displayName: "MainBalanced120",
    },
    ladder_generalization_hard_80: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_80_draft_v3",
        alias: "gran_ladder_genhard80",
        displayName: "GeneralizationHard80",
    },
    test_dataset_granularity_ladder_generalization_hard_80_draft_v3: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_80_draft_v3",
        alias: "gran_ladder_genhard80",
        displayName: "GeneralizationHard80",
    },
    ladder_structure_stress_60: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_60_draft_v6",
        alias: "gran_ladder_stress60",
        displayName: "StructureStress60",
    },
    test_dataset_granularity_ladder_structure_stress_60_draft_v7: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_60_draft_v7",
        alias: "gran_ladder_stress60",
        displayName: "StructureStress60",
    },
    test_dataset_granularity_ladder_structure_stress_60_draft_v6: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_60_draft_v6",
        alias: "gran_ladder_stress60",
        displayName: "StructureStress60",
    },
    test_dataset_granularity_ladder_structure_stress_60_draft_v3: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_60_draft_v3",
        alias: "gran_ladder_stress60",
        displayName: "StructureStress60",
    },
    ladder_main_balanced_150: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_frozen_v1",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_frozen_v1: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_frozen_v1",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_reviewed_candidate_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_main_balanced_150_reviewed_candidate_v1",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_draft_v9: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_draft_v9",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_draft_v8: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_draft_v8",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_draft_v7: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_draft_v7",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_draft_v6: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_draft_v6",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_draft_v5: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_draft_v5",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    test_dataset_granularity_ladder_main_balanced_150_draft_v4: {
        canonicalName: "test_dataset_granularity_ladder_main_balanced_150_draft_v4",
        alias: "gran_ladder_main150",
        displayName: "MainBalanced150",
    },
    ladder_generalization_hard_100: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_frozen_v1",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_frozen_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_frozen_v1",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_reviewed_candidate_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_reviewed_candidate_v1",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_draft_v9: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_draft_v9",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_draft_v8: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_draft_v8",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_draft_v7: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_draft_v7",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_draft_v6: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_draft_v6",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_draft_v5: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_draft_v5",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    test_dataset_granularity_ladder_generalization_hard_100_draft_v4: {
        canonicalName:
            "test_dataset_granularity_ladder_generalization_hard_100_draft_v4",
        alias: "gran_ladder_genhard100",
        displayName: "GeneralizationHard100",
    },
    ladder_structure_stress_80: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_frozen_v1",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    test_dataset_granularity_ladder_structure_stress_80_frozen_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_frozen_v1",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    ladder_cross_doc_coverage_diag_18: {
        canonicalName:
            "test_dataset_granularity_ladder_cross_doc_coverage_diag_18_frozen_v1",
        alias: "gran_ladder_crossdoc18",
        displayName: "CrossDocCoverage18",
    },
    test_dataset_granularity_ladder_cross_doc_coverage_diag_18_frozen_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_cross_doc_coverage_diag_18_frozen_v1",
        alias: "gran_ladder_crossdoc18",
        displayName: "CrossDocCoverage18",
    },
    test_dataset_granularity_ladder_structure_stress_80_reviewed_candidate_v1: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_reviewed_candidate_v1",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    test_dataset_granularity_ladder_structure_stress_80_draft_v9: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_draft_v9",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    test_dataset_granularity_ladder_structure_stress_80_draft_v8: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_draft_v8",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    test_dataset_granularity_ladder_structure_stress_80_draft_v5: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_draft_v5",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    test_dataset_granularity_ladder_structure_stress_80_draft_v4: {
        canonicalName:
            "test_dataset_granularity_ladder_structure_stress_80_draft_v4",
        alias: "gran_ladder_stress80",
        displayName: "StructureStress80",
    },
    test_dataset_granularity_structure_dev_40_draft_v1: {
        canonicalName: "test_dataset_granularity_structure_dev_40_draft_v1",
        alias: "gran_structure_dev40",
        displayName: "StructureDev40",
    },
    test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1: {
        canonicalName:
            "test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1",
        alias: "gran_legacy_hard30",
        displayName: "LegacyHardOOD30",
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
    test_dataset_answer_reject_v5_expanded_draft: {
        canonicalName: "test_dataset_answer_reject_v5_expanded_draft",
        alias: "ar_v5_expanded_draft",
        displayName: "AnswerRejectExpandedDraft",
    },
    test_dataset_answer_reject_v5_expanded_frozen_v1: {
        canonicalName: "test_dataset_answer_reject_v5_expanded_frozen_v1",
        alias: "ar_v5_expanded_frozen_v1",
        displayName: "AnswerRejectFrozen",
    },
    test_dataset_answer_reject_v6_80_frozen_v1: {
        canonicalName: "test_dataset_answer_reject_v6_80_frozen_v1",
        alias: "ar_v6_80_frozen_v1",
        displayName: "AnswerReject80",
    },
    test_dataset_answer_reject_v6_80_combined_manifest: {
        canonicalName: "test_dataset_answer_reject_v6_80_combined_manifest",
        alias: "ar_v6_80_combined",
        displayName: "AnswerReject80Draft",
    },
    test_dataset_answer_reject_mixed_long_tail_59_derived_v1: {
        canonicalName: "test_dataset_answer_reject_mixed_long_tail_59_derived_v1",
        alias: "ar_mixed_longtail59",
        displayName: "AnswerRejectMixedLongTail59",
    },
    test_dataset_answer_quality_blind_reviewed_v1: {
        canonicalName: "test_dataset_answer_quality_blind_reviewed_v1",
        alias: "aq_blind_v1",
        displayName: "AnswerQualityBlind",
    },
    test_dataset_answer_quality_blind_reviewed_v2: {
        canonicalName: "test_dataset_answer_quality_blind_reviewed_v2",
        alias: "aq_blind_v2",
        displayName: "AnswerQualityBlindV2",
    },
    test_dataset_answer_quality_blind_provisional_v1: {
        canonicalName: "test_dataset_answer_quality_blind_provisional_v1",
        alias: "aq_blind_prov_v1",
        displayName: "AnswerQualityBlindProv",
    },
    test_dataset_answer_quality_blind_v3_100_reviewed_v1: {
        canonicalName: "test_dataset_answer_quality_blind_v3_100_reviewed_v1",
        alias: "aq_blind_v3_100_reviewed_v1",
        displayName: "AnswerQualityBlind100Reviewed",
    },
    test_dataset_answer_quality_blind_v3_100_frozen_v1: {
        canonicalName: "test_dataset_answer_quality_blind_v3_100_frozen_v1",
        alias: "aq_blind_v3_100_frozen_v1",
        displayName: "AnswerQualityBlind100Frozen",
    },
    test_dataset_answer_quality_ext_ood_blind_60_derived_v1: {
        canonicalName: "test_dataset_answer_quality_ext_ood_blind_60_derived_v1",
        alias: "aq_ext_blind60",
        displayName: "AnswerQualityExtBlind60",
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

export function buildAnswerQualityResultFileName(
    datasetIdentity: string,
    timestamp: number,
): string {
    return `answer_quality_${resolveNamedDatasetProfile(datasetIdentity).alias}_${timestamp}.json`;
}

export function buildStandardBaselinesResultFileName(timestamp: number): string {
    return `granularity_baselines_${timestamp}.json`;
}

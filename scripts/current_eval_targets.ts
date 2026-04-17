export const CURRENT_EVAL_DATASET_FILES = {
    // 当前 granularity 主线
    granularityMain120:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_benchmark_v2_reviewed_userized_v1.json",
    granularityInDomainHoldout50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1.json",
    // 当前论文口径：blind aligned 外域集作为正式 ExtOOD；旧 matched 外域集单列保留
    granularityExtOodBlind60:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_aligned_ext_ood_blind_60_draft_v1.json",
    granularityMatchedExtOod60:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1.json",
    granularityHardOodBlind30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_hard_ood_blind_30_draft_v1.json",
    granularityLegacyExternalOodHard30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1.json",
    // 兼容旧 target key：external_ood_50 现在映射到正式 blind ExtOOD。
    granularityExternalOod50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_aligned_ext_ood_blind_60_draft_v1.json",
    // 兼容旧 target key：historical external_ood_holdout_30 过去已被借用为 matched external 入口，这里保留该映射。
    granularityExternalOodHoldout30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1.json",
    // 兼容旧 target key：external_ood_hard_30 现在映射到正式 blind HardOOD。
    granularityExternalOodHard30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_hard_ood_blind_30_draft_v1.json",
    granularityHardOodV2DiagTop30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_hard_ood_v2_diag_top30.json",
    granularityStructureDev40:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_structure_dev_40_draft_v1.json",
    granularityLadderMainBalanced80:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_main_balanced_80_draft_v2.json",
    granularityLadderGeneralizationHard60:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_generalization_hard_60_draft_v2.json",
    granularityLadderStructureStress40:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_structure_stress_40_draft_v2.json",
    granularityLadderMainBalanced120:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_main_balanced_120_draft_v3.json",
    granularityLadderGeneralizationHard80:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_generalization_hard_80_draft_v3.json",
    granularityLadderStructureStress60:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_structure_stress_60_draft_v6.json",
    granularityLadderMainBalanced150:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_main_balanced_150_frozen_v1.json",
    granularityLadderGeneralizationHard100:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_generalization_hard_100_frozen_v1.json",
    granularityLadderStructureStress80:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_structure_stress_80_frozen_v1.json",
    granularityLadderCrossDocCoverageDiag18:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_ladder_cross_doc_coverage_diag_18_frozen_v1.json",
    // 当前 behavior 主线：80-case frozen holdout
    answerRejectCurrent:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v6_80_frozen_v1.json",
    answerRejectV4FrozenHoldout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v4_frozen_holdout_reviewed.json",
    answerRejectV5ExpandedDraft:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v5_expanded_draft.json",
    answerRejectV5ExpandedFrozenV1:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v5_expanded_frozen_v1.json",
    answerRejectV6Frozen80:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v6_80_frozen_v1.json",
    answerRejectMixedLongTail59:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_mixed_long_tail_59_derived_v1.json",
    answerRejectV6Combined80Draft:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v6_80_combined_manifest.json",
    answerQualityCurrent:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_blind_v3_100_frozen_v1.json",
    answerQualityExtOodBlind60:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_ext_ood_blind_60_derived_v1.json",
    answerQualityBlindProvisionalV1:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_blind_provisional_v1.json",
    answerQualityBlindV1:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_blind_reviewed_v1.json",
    answerQualityBlindV2:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_blind_reviewed_v2.json",
    answerQualityBlindV3Reviewed100:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_blind_v3_100_reviewed_v1.json",
    answerQualityBlindV3Frozen100:
        "../Backend/test/test_dataset_answer_quality/test_dataset_answer_quality_blind_v3_100_frozen_v1.json",
    // v3 为上一轮调参与扩样后的已看过主线，保留回归参考
    answerRejectV3Dev:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v3_dev_reviewed.json",
    answerRejectV3Holdout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v3_holdout_reviewed.json",
    // 过渡 v2，仅保留追溯
    answerRejectDbAbsentOnlyV2Dev:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_db_absent_only_v2_dev_reviewed.json",
    answerRejectDbAbsentOnlyV2Holdout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_db_absent_only_v2_holdout_reviewed.json",
    answerRejectAnchoredIncompleteLegacyDiagV1:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_anchored_incomplete_legacy_diag_v1_reviewed.json",
    answerRejectPairControlLegacyDiagV1Holdout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_pair_control_legacy_diag_v1_holdout_reviewed.json",
    answerRejectHardRejectLegacyDiagV1:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_hard_reject_legacy_diag_v1_reviewed.json",
    // 历史停用 behavior / product 入口，仅保留追溯
    platformMixedDailyV12:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_mixed_daily_v1_2_reviewed.json",
    routeOrClarifyV2Dev:
        "../Backend/test/test_dataset_route_or_clarify/test_dataset_route_or_clarify_v2_dev_reviewed.json",
    routeOrClarifyV2Holdout:
        "../Backend/test/test_dataset_route_or_clarify/test_dataset_route_or_clarify_v2_holdout_reviewed.json",
    // 历史高风险拒答补充参考
    kbAbsentV2Dev:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_reject_kb_absent_v2_dev_reviewed.json",
    kbAbsentV2Holdout:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_reject_kb_absent_v2_holdout_reviewed.json",
    kbAbsentPairControlV2HoldoutFlat:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_reject_kb_absent_pair_control_v2_holdout_flat_reviewed.json",
} as const;

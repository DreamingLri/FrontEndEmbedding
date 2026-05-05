export const CURRENT_EVAL_DATASET_FILES = {
    // 当前 granularity 正式入口：
    // Main v7 / InDomain v10 / ExtOOD985 v9
    // `blind_ext_ood_100` 保留为兼容 key，但底层实体文件已切到当前
    // ExtOOD985Aligned100；旧 BlindExtOOD v4 已降级为历史材料。
    granularityMain120:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_generalization_aligned_120_draft_v7.json",
    granularityInDomainGeneralization100:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v10.json",
    granularityBlindExtOod100:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_985_aligned_100_draft_v9.json",
    granularityExtOod985Aligned100:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_985_aligned_100_draft_v9.json",
    // Archived 2026-04-28:
    // - granularityInDomainHoldout50
    // - granularityExtOodBlind60
    // - granularityMatchedExtOod60
    // - granularityExternalOod50
    // These retired legacy entry points should not be restored as active
    // evaluation targets.
    granularityHardOodBlind30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_hard_ood_blind_30_draft_v1.json",
    granularityLegacyExternalOodHard30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1.json",
    // external_ood_holdout_30 回到真实 30 条 hard stress 入口，不再借用为 matched external。
    granularityExternalOodHoldout30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1.json",
    // 兼容旧 target key：external_ood_hard_30 仍映射到 blind HardOOD。
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
    // Archived 2026-04-28:
    // - answerQualityExtOodBlind60
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

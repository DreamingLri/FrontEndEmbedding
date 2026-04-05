export const CURRENT_EVAL_DATASET_FILES = {
    // 当前 granularity 主线
    granularityMain120:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_benchmark_v2_reviewed_userized_v1.json",
    granularityInDomainHoldout50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_60_reviewed_userized_v1.json",
    granularityExternalOod50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1.json",
    // 兼容旧 target key，当前正式外域口径已切到 external_matched_ood_v2。
    granularityExternalOodHoldout30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_matched_ood_60_reviewed_userized_v1.json",
    // 历史 hard OOD，仅保留参考
    granularityExternalOodHard30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1.json",
    // 当前唯一 behavior 主线：未看过的 v4 frozen holdout
    answerRejectCurrent:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v4_frozen_holdout_reviewed.json",
    answerRejectV4FrozenHoldout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v4_frozen_holdout_reviewed.json",
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

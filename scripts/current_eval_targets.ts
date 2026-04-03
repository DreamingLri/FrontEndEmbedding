export const CURRENT_EVAL_DATASET_FILES = {
    // 当前 granularity 主线
    granularityMain120:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_120_reviewed_userized_v1.json",
    granularityInDomainHoldout50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_holdout_50_reviewed_userized_v1.json",
    granularityExternalOod50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_50_reviewed_userized_v1.json",
    // 兼容旧 target key，当前仍指向正式 external_ood_50
    granularityExternalOodHoldout30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_50_reviewed_userized_v1.json",
    // 历史 hard OOD，仅保留参考
    granularityExternalOodHard30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_holdout_30_reviewed_userized_v1.json",
    // 当前唯一 behavior 主线
    answerRejectCurrent:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v1_holdout_reviewed.json",
    answerRejectV1Dev:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v1_dev_reviewed.json",
    answerRejectV1Holdout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_v1_holdout_reviewed.json",
    answerRejectPairControlV1Holdout:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_pair_control_v1_holdout_reviewed.json",
    answerRejectHardRejectDiagV1:
        "../Backend/test/test_dataset_answer_reject/test_dataset_answer_reject_hard_reject_diag_v1_reviewed.json",
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

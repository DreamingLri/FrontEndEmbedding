export const CURRENT_EVAL_DATASET_FILES = {
    // 当前 granularity 主线
    granularityMain120:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_120_reviewed.json",
    granularityInDomainHoldout50:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_holdout_50_reviewed.json",
    granularityInDomainHoldout50SkeletonV2:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_holdout_50_reviewed_skeleton_v2.json",
    granularityExternalOodHoldout30:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_ood_holdout_30_reviewed.json",
    // 历史参考集，当前只保留为回退或旧结果对照
    granularityMain106:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_106_reviewed.json",
    granularityHoldoutV3:
        "../Backend/test/test_dataset_granularity/test_dataset_granularity_holdout_v3_reviewed.json",
    // 当前 mixed / route / kb-absent 主线
    platformMixedDailyV12:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_mixed_daily_v1_2_reviewed.json",
    routeOrClarifyV2Dev:
        "../Backend/test/test_dataset_route_or_clarify/test_dataset_route_or_clarify_v2_dev_reviewed.json",
    routeOrClarifyV2Holdout:
        "../Backend/test/test_dataset_route_or_clarify/test_dataset_route_or_clarify_v2_holdout_reviewed.json",
    kbAbsentV2Dev:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_reject_kb_absent_v2_dev_reviewed.json",
    kbAbsentV2Holdout:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_reject_kb_absent_v2_holdout_reviewed.json",
    kbAbsentPairControlV2HoldoutFlat:
        "../Backend/test/test_dataset_platform_mixed/test_dataset_platform_reject_kb_absent_pair_control_v2_holdout_flat_reviewed.json",
} as const;

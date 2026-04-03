process.env.SUASK_PLATFORM_MIXED_DATASET_FILE =
    "../Backend/test/_archive/legacy_datasets/test_dataset_platform_mixed/test_dataset_platform_direct_answer_rescue_v1_reviewed.json";
process.env.SUASK_PLATFORM_MIXED_REJECT_THRESHOLD = "0.85";
process.env.SUASK_PLATFORM_MIXED_RESULTS_PREFIX =
    "platform_direct_answer_rescue";
process.env.SUASK_PLATFORM_MIXED_NOTE =
    "该报告用于 direct-answer rescue 压力验证：选取同一批真实 direct-answer 样本，并将展示拒答阈值临时抬升到 0.85，只用于观察补救重排是否触发与是否救回，不作为生产 acceptance 指标。";

await import("./eval_platform_mixed.ts");

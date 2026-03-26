process.env.SUASK_EVAL_DATASET_VERSION = "granularity";
process.env.SUASK_EVAL_DATASET_FILE =
    "../Backend/test/test_dataset_granularity/test_dataset_granularity_pilot21.json";

await import("./eval_granularity_mix.ts");

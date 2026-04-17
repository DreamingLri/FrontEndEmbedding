import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";

process.env.SUASK_ANSWER_QUALITY_DATASET_FILE =
    process.env.SUASK_ANSWER_QUALITY_DATASET_FILE ||
    CURRENT_EVAL_DATASET_FILES.answerQualityExtOodBlind60;
process.env.SUASK_ANSWER_QUALITY_NOTE =
    process.env.SUASK_ANSWER_QUALITY_NOTE ||
    "支持实验：由 AlignedExtOOD blind 60 派生的外域 answer-quality 盲测线。";

await import("./eval_answer_quality.ts");

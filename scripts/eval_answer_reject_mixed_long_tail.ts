import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";

process.env.SUASK_ANSWER_REJECT_DATASET_FILE =
    process.env.SUASK_ANSWER_REJECT_DATASET_FILE ||
    CURRENT_EVAL_DATASET_FILES.answerRejectMixedLongTail59;
process.env.SUASK_ANSWER_REJECT_NOTE =
    process.env.SUASK_ANSWER_REJECT_NOTE ||
    "支持实验：platform_mixed_daily_v1.2 中 direct_answer + reject 的 59 条长尾混合行为线。";

await import("./eval_answer_reject.ts");

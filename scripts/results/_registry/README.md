# 当前结果注册表说明

这个目录用于提供不带时间戳的稳定结果入口，避免文档继续直接引用大量历史结果文件。

当前设计分两层：

1. `current_results.json`
   - 汇总当前所有稳定结果入口
2. `<slot>.json`
   - 每个稳定结果入口各自一份小型指针文件

当前评测脚本会在写出时间戳结果后，自动更新这里的稳定入口：

1. [eval_granularity_mix.ts](/d:/Project/SuAsk/SuAsk_Agent/FrontEnd/scripts/eval_granularity_mix.ts)
2. [eval_route_or_clarify.ts](/d:/Project/SuAsk/SuAsk_Agent/FrontEnd/scripts/eval_route_or_clarify.ts)
3. [eval_platform_mixed.ts](/d:/Project/SuAsk/SuAsk_Agent/FrontEnd/scripts/eval_platform_mixed.ts)

当前仍需要人工维护的结果入口：

1. `granularity_holdout_v3_current`
   - 因为它当前对应的是单独保留的固定口径结果，而不是上述脚本直接产出

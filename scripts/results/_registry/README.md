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

当前目录已按“现行主线优先”做过清理；旧结果入口统一转入历史记录，不再在这里保留稳定指针。

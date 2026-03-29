# 创新点图表输出

生成时间基于当前 granularity 正式集结果文件。

## 文件说明

1. [fig4_1_granularity_hit1.svg](./fig4_1_granularity_hit1.svg)
   - 展示不同粒度组合的文档级 Hit@1，支撑“多粒度统一建模”创新点。
2. [fig4_2_kp_evidence_gain.svg](./fig4_2_kp_evidence_gain.svg)
   - 对比 KP+OT 在引入结构化 KP 证据增强前后的文档级与 kpid 级指标。
3. [fig4_3_kpid_diagnosis.svg](./fig4_3_kpid_diagnosis.svg)
   - 诊断文档 Top1 正确时，主证据 KP 是否也被正确选中。
4. [fig4_4_window_strategy.svg](./fig4_4_window_strategy.svg)
   - 展示浅层候选窗口修正前后正式线上链路的指标变化。

## 主要结果摘要

- KP+OT baseline：文档 Hit@1=46.67%，kpid Hit@1=26.67%
- KP+OT + online kp role rerank：文档 Hit@1=53.33%，kpid Hit@1=53.33%
- 浅层窗口修正前：文档 MRR=0.5911
- 浅层窗口修正后：文档 MRR=0.6244

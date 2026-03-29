# Python 版创新点图表输出

## 文件说明

1. `fig4_1_granularity_hit1_py.svg/.png`：不同粒度组合的整体检索效果
2. `fig4_2_kp_evidence_gain_py.svg/.png`：KP 证据增强前后对比
3. `fig4_3_kpid_diagnosis_py.svg/.png`：主证据 KP 选择诊断
4. `fig4_4_window_strategy_py.svg/.png`：浅层候选窗口修正效果
5. `fig4_5_support_pattern_py.svg/.png`：按 support_pattern 分组的效果对比

以上图表均同步导出 `svg / png / pdf` 三种格式。

## 运行方式

推荐在 `Backend` 目录下使用 `uv` 运行：

```powershell
uv run python ..\FrontEnd\scripts\plot_innovation_charts.py
```

## 主要结果摘要

- KP+OT baseline：文档 Hit@1=46.67%，kpid Hit@1=26.67%
- KP+OT + online kp role rerank：文档 Hit@1=53.33%，kpid Hit@1=53.33%
- 浅层窗口修正前：文档 MRR=0.5911
- 浅层窗口修正后：文档 MRR=0.6244

## support_pattern 分组摘要

- `single_kp`：文档 Hit@1=58.33%，kpid Hit@1=58.33%
- `multi_kp`：文档 Hit@1=0.00%，kpid Hit@1=0.00%
- `ot_required`：文档 Hit@1=50.00%，kpid Hit@1=50.00%

## 建议图注

1. 图4-1展示了不同粒度组合在 granularity 正式集上的整体检索效果，结果表明 OT 与 KP+OT 组合整体优于单独 KP，说明知识点粒度更适合作为证据增强信号，而非独立替代原文粒度。
2. 图4-2展示了 KP 证据增强机制对 KP+OT 组合的提升效果，可见在文档级与 kpid 级指标上均取得一致增益。
3. 图4-3展示了文档命中后主证据 KP 的选择情况，说明当前系统的主要改进收益来自正确答案片段的候选内重排。
4. 图4-4展示了浅层候选窗口修正前后的线上正式链路效果，说明限制在浅层候选内进行角色重排能够稳定提升最终排序质量。
5. 图4-5展示了不同 support_pattern 下的分组表现，说明当前增强机制的主要收益集中在 `single_kp` 样本上，而 `multi_kp` 与 `ot_required` 仍需进一步补样与优化。

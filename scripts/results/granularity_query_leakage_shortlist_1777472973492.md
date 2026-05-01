# Main/InDomain Query Leakage Shortlist

状态：`CURRENT DRAFT`

- shortlist JSON：[granularity_query_leakage_shortlist_1777472973492.json](D:/Project/SuAsk/SuAsk_Agent/FrontEnd/scripts/results/granularity_query_leakage_shortlist_1777472973492.json:1)

## 核心结论

- 检查目标：`main_bench_120, in_domain_generalization_100`
- 输出样本：`24`
- P0 replace：`6`，P1 rewrite：`10`，P2 watch：`8`

## 人工复核清单

- Main | main_120_draft_v1_0098 | P0_replace | replace_case | max=0.7826 | margin=-0.1437
  query: 中山大学2022年广东省综合评价录取面试考核的具体日期是哪一天？
  best non-target Q: 中山大学2023年广东省综合评价录取面试考核中，报到环节的具体日期和地点是什么？
  same-target Q: 在中山大学2022年广东省综合评价录取的线上面试中，考生进行自我介绍的时间限制是多少？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。
- Main | main_120_draft_v1_0066 | P0_replace | replace_case | max=0.6944 | margin=-0.1111
  query: 2020年同等学力申请硕士学位论文答辩需要满足哪些条件？
  best non-target Q: 截至2022年9月，中山大学同等学力人员申请博士学位论文答辩需要满足哪些具体条件才能提交申请？
  same-target Q: 根据2020年的通知，同等学力人员申请硕士学位论文答辩时，需要公开发表或出版什么类型的学术成果？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。
- InDomain100 | in_domain_100_draft_v1_0053 | P0_replace | replace_case | max=0.6866 | margin=-0.1781
  query: 参加2024强基计划考核的考生需要满足哪些整体条件才能被录取
  best non-target Q: 根据中山大学2025年强基计划录取标准，考生需要满足哪些具体条件才能被拟录取？
  same-target Q: 中山大学2024年强基计划考核中，考生入校时必须携带哪些证件？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。
- InDomain100 | in_domain_100_draft_v1_0003 | P1_rewrite | rewrite_query | max=0.6129 | margin=-0.1733
  query: 2026年报考点网上确认的具体步骤和首次提交截止时间是什么
  best non-target Q: 截至2023年9月22日，中山大学报考点网上确认的具体时间安排是什么？
  same-target Q: 根据2026年全国硕士研究生招生考试中山大学报考点（代码4413）的网上确认公告，所有考生首次完成网上确认步骤的具体时间范围是什么？
  tags: non_target_beats_same_target, cross_year_non_target_match
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- Main | main_120_draft_v1_0007 | P1_rewrite | rewrite_query | max=0.6857 | margin=-0.0703
  query: 2017年同等学力申请硕士学位人员全国统考的报名流程和时间怎么安排都有哪些
  best non-target Q: 2019年同等学力申请硕士学位全国统一考试的网上报名时间是什么时候？
  same-target Q: 根据2017年中山大学同等学力申请硕士学位全国统一考试通知，网上报名的时间范围是什么？
  tags: non_target_beats_same_target, cross_year_non_target_match
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- Main | main_120_draft_v1_0080 | P0_replace | replace_case | max=0.6780 | margin=-0.2557
  query: 登录研究生招生平台核对录取通知书邮寄地址的网址是什么
  best non-target Q: 2018年博士研究生招生录取通知书邮寄地址校对系统的登录网址是什么？
  same-target Q: 某考生准备在2025年7月10日登录中山大学研究生招生平台修改录取通知书邮寄地址，根据2025年6月30日的通知，这种情况会如何处理？
  tags: non_target_beats_same_target, template_wording_overlap
  rationale: 非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。
- Main | main_120_draft_v1_0105 | P1_rewrite | rewrite_query | max=0.6667 | margin=-0.0741
  query: 中山大学2020高水平运动队测试时间
  best non-target Q: 中山大学2022年高水平运动队招生测试包含哪些具体项目？
  same-target Q: 中山大学2020年高水平运动队测试的防疫要求中，对考生佩戴口罩的规定是什么？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- Main | main_120_draft_v1_0081 | P0_replace | replace_case | max=0.6667 | margin=-0.0904
  query: 2020年高水平运动队测试的报到地点有哪些？
  best non-target Q: 中山大学2022年高水平运动队招生测试的举办地点是哪里？
  same-target Q: 中山大学2020年高水平运动队招生测试中，对考生的测试装备（含服装）有哪些具体要求？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。
- Main | main_120_draft_v1_0051 | P1_rewrite | rewrite_query | max=0.6585 | margin=-0.0564
  query: 在广州市实习的外地高校2024年应届生报考中山大学，需要提交什么实习相关证明？
  best non-target Q: 对于在广州市实习的穗外高校2023年应届毕业生，选择中山大学报考点需要同时提交哪三种证明文件？
  same-target Q: 如果一名在广州市实习的穗外高校2024年应届毕业生想选择中山大学报考点，根据2023年的公告，他需要提交哪些证明材料？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- InDomain100 | in_domain_100_draft_v1_0043 | P0_replace | replace_case | max=0.6316 | margin=-0.2408
  query: 2023年联合培养博士申请对英语水平有什么要求
  best non-target Q: 中山大学2023年博士生国外访学项目对申请人的英语水平有什么具体要求？
  same-target Q: 根据2023年中山大学-鹏城实验室联合培养博士研究生专项计划的招生简章，申请考核制报考条件中，对申请人的政治立场和学历有哪些具体要求？
  tags: non_target_beats_same_target, template_wording_overlap
  rationale: 非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。
- Main | main_120_draft_v1_0041 | P1_rewrite | rewrite_query | max=0.6857 | margin=-0.0104
  query: 软件工程学院2025年青年教师专职辅导员的选聘条件和程序是什么
  best non-target Q: 软件工程学院在2023年8月的通知中，青年教师专职辅导员的选聘条件有哪些具体要求？
  same-target Q: 软件工程学院在2025年12月31日发布的青年教师专职辅导员选聘通知中，要求邮件命名格式是什么？
  tags: non_target_beats_same_target, cross_year_non_target_match, same_vs_non_target_too_close, template_wording_overlap
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- InDomain100 | in_domain_100_draft_v1_0084 | P1_rewrite | rewrite_query | max=0.6377 | margin=-0.0048
  query: 2023年上半年同等学力申请硕士学位论文答辩的网上申请截止日期是哪天？
  best non-target Q: 2021年同等学力人员申请博士学位论文答辩时，网上申请必须使用哪些浏览器？
  same-target Q: 2023年上半年中山大学同等学力人员申请硕士学位论文答辩中，申请者需要通过哪些具体的课程考试？
  tags: non_target_beats_same_target, cross_year_non_target_match, same_vs_non_target_too_close
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- Main | main_120_draft_v1_0013 | P1_rewrite | rewrite_query | max=0.7551 | margin=0.1006
  query: 2022年下半年同等学力申请硕士学位论文答辩，需要提交哪些电子版材料到指定邮箱？
  best non-target Q: 根据中山大学2022年下半年同等学力人员申请硕士学位论文答辩通知，申请人需在10月10日前提交哪些电子版申请材料到指定邮箱？
  same-target Q: 根据中山大学2022年下半年同等学力人员申请硕士学位论文答辩通知，申请人需在10月10日前提交哪些电子版申请材料到指定邮箱？
  tags: same_target_literal_reuse, template_wording_overlap
  rationale: 题面与目标文档原生 Q 过近，容易把检索分数抬高。
- InDomain100 | in_domain_100_draft_v1_0065 | P1_rewrite | rewrite_query | max=0.6197 | margin=-0.0483
  query: 中山大学材料学院2024年工程硕博士补报名招生的专业代码是什么？
  best non-target Q: 在中山大学软件工程学院2025年博士招生中，考生吴家淳的拟录取专业代码及名称是什么？
  same-target Q: 根据中山大学材料科学与工程学院2024年工程硕博士改革专项研究生的补报名通知，进入复试的同学需要提供什么材料？
  tags: non_target_beats_same_target, cross_year_non_target_match, template_wording_overlap
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- InDomain100 | in_domain_100_draft_v1_0073 | P1_rewrite | rewrite_query | max=0.6154 | margin=-0.0021
  query: 2025年同等学力人员申请硕士学位全国统考报名通知里，英语考试大纲在哪里查看？
  best non-target Q: 2022年同等学力人员申请硕士学位全国统考延期后，考生可通过哪个官方链接查看详细通知？
  same-target Q: 2025年同等学力人员申请硕士学位全国统一考试的具体考试日期和时间安排是什么？
  tags: non_target_beats_same_target, cross_year_non_target_match, same_vs_non_target_too_close
  rationale: 存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。
- Main | main_120_draft_v1_0069 | P1_rewrite | rewrite_query | max=0.7123 | margin=0.0274
  query: 2024学年研究生国家奖学金补充通知中，申请人需要满足哪些基本条件？
  best non-target Q: 根据中山大学2024学年研究生国家奖学金评选通知，申请者必须满足哪些基本条件才能参评？
  same-target Q: 根据中山大学2024学年研究生国家奖学金评选通知，申请者必须满足哪些基本条件才能参评？
  tags: same_target_literal_reuse, template_wording_overlap
  rationale: 题面与目标文档原生 Q 过近，容易把检索分数抬高。
- InDomain100 | in_domain_100_draft_v1_0080 | P2_watch | manual_review | max=0.6792 | margin=0.1204
  query: 2019年博士生国外访学资助额度限制是多少
  best non-target Q: 2019年中山大学博士生国外访学项目资助的短期访学研究期限是多少？
  same-target Q: 2019年中山大学博士生国外访学项目资助的短期访学研究期限是多少？
  tags: template_wording_overlap
  rationale: 建议人工复核。
- Main | main_120_draft_v1_0040 | P2_watch | manual_review | max=0.6667 | margin=0.1667
  query: 软件工程学院2025年面向港澳台招生的复试录取规则都有哪些
  best non-target Q: 根据软件工程学院2025年面向港澳台地区研究生招生复试录取实施细则，哪些情况会导致考生不予录取？
  same-target Q: 根据软件工程学院2025年面向港澳台地区研究生招生复试录取实施细则，哪些情况会导致考生不予录取？
  tags: template_wording_overlap
  rationale: 建议人工复核。
- Main | main_120_draft_v1_0039 | P2_watch | manual_review | max=0.6667 | margin=0.0991
  query: 软件工程学院2025年博士改报志愿的完整政策和要求是什么
  best non-target Q: 在中山大学软件工程学院2025年博士改报志愿流程中，面试成绩的作用是什么？
  same-target Q: 在中山大学软件工程学院2025年博士改报志愿流程中，面试成绩的作用是什么？
  tags: none
  rationale: 建议人工复核。
- InDomain100 | in_domain_100_draft_v1_0083 | P2_watch | manual_review | max=0.6667 | margin=0.0513
  query: 2025年全国硕士研究生报名考试中山大学报考点的网上确认时间是哪天到哪天？
  best non-target Q: 根据2025年全国硕士研究生报名考试中山大学报考点公告，网上缴费的截止时间是什么时候，以及通过哪个平台支付？
  same-target Q: 根据2025年全国硕士研究生报名考试中山大学报考点公告，网上缴费的截止时间是什么时候，以及通过哪个平台支付？
  tags: template_wording_overlap
  rationale: 建议人工复核。
- InDomain100 | in_domain_100_draft_v1_0057 | P2_watch | manual_review | max=0.6585 | margin=0.0335
  query: 在职临床医师想申请临床医学博士专业学位时，资格认定主要看哪些门槛条件？
  best non-target Q: 在2017年中山大学在职临床医师申请临床医学博士专业学位资格认定中，资格认定对象需要满足哪些具体条件？
  same-target Q: 在2017年中山大学在职临床医师申请临床医学博士专业学位资格认定中，资格认定对象需要满足哪些具体条件？
  tags: template_wording_overlap
  rationale: 建议人工复核。
- InDomain100 | in_domain_100_draft_v1_0050 | P2_watch | manual_review | max=0.6400 | margin=0.0108
  query: 在广州市实习的外地高校2024年应届生，报考中山大学需要提交哪些证明材料才能选中山大学考点？
  best non-target Q: 如果一名在广州市实习的穗外高校2024年应届毕业生想选择中山大学报考点，根据2023年的公告，他需要提交哪些证明材料？
  same-target Q: 如果一名在广州市实习的穗外高校2024年应届毕业生想选择中山大学报考点，根据2023年的公告，他需要提交哪些证明材料？
  tags: same_vs_non_target_too_close, template_wording_overlap
  rationale: 目标/非目标 Q 相似度过近，题面可能仍然偏模板化。
- InDomain100 | in_domain_100_draft_v1_0099 | P2_watch | manual_review | max=0.6182 | margin=0.0919
  query: 2023下半年同等学力博士答辩关键时间节点汇总
  best non-target Q: 对于2023年下半年同等学力博士论文答辩，资格认定的时间限制是什么？
  same-target Q: 对于2023年下半年同等学力博士论文答辩，资格认定的时间限制是什么？
  tags: none
  rationale: 建议人工复核。
- InDomain100 | in_domain_100_draft_v1_0032 | P2_watch | manual_review | max=0.6154 | margin=0.0396
  query: 2024年退役大学生士兵专项硕士招生计划的整体流程和政策要点是什么
  best non-target Q: 中山大学2024年“退役大学生士兵”专项计划的具体招生人数以什么为准？
  same-target Q: 中山大学2024年“退役大学生士兵”专项计划的具体招生人数以什么为准？
  tags: none
  rationale: 建议人工复核。

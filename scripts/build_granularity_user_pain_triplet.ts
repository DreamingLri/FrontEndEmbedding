import * as fs from "fs";
import * as path from "path";

type DatasetCase = {
    id?: string;
    query?: string;
    query_derivation?: string;
    notes?: string;
    challenge_tags?: string[];
    [key: string]: unknown;
};

type Rewrite = {
    query: string;
    painTypes: string[];
};

type DatasetBuildSpec = {
    label: string;
    inputFile: string;
    outputFile: string;
    rewrites: Record<string, Rewrite>;
};

const MAIN_REWRITES: Record<string, Rewrite> = {
    main_120_draft_v1_0004: {
        query: "在职临床医生准备走同等学力申请临床医学博士，论文答辩前后到底要完成哪些事",
        painTypes: ["low_title_anchor", "cross_doc_flow", "ambiguous_user_state"],
    },
    main_120_draft_v1_0005: {
        query: "博士报名后想改报志愿，应该怎么申请，后面还要等哪些处理结果",
        painTypes: ["low_title_anchor", "post_outcome_action", "ambiguous_user_state"],
    },
    main_120_draft_v1_0025: {
        query: "自主招生考核前，准考证什么时候能打印，打印窗口大概持续多久",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0026: {
        query: "复试细则不是一次全公布的情况下，第三批相关院系应该去哪里查",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0028: {
        query: "想报联合培养博士专项，主要能选哪些博士招生专业",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0029: {
        query: "博士生导师短期出国交流一般能访学多久，时间上有没有上限",
        painTypes: ["low_title_anchor"],
    },
    main_120_draft_v1_0030: {
        query: "如果没有在第一批名单里看到院系，后续复试录取细则应该通过哪些入口查询",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0031: {
        query: "第一批复试录取细则发布时，哪些院系已经可以查看具体安排",
        painTypes: ["low_title_anchor"],
    },
    main_120_draft_v1_0052: {
        query: "选择学校报考点时，哪些类型的考生会被优先接收",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0055: {
        query: "准备参加同等学力全国统考，学士学位最晚需要在什么时候前拿到",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0073: {
        query: "联合培养博士专项这次大概招多少人",
        painTypes: ["low_title_anchor"],
    },
    main_120_draft_v1_0074: {
        query: "参加附属医院博士统一笔试，准考证什么时候打印，入口在哪里",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    main_120_draft_v1_0097: {
        query: "外语类保送生去面试时应该到哪里报到",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    main_120_draft_v1_0099: {
        query: "高水平运动队招生测试安排在哪一天",
        painTypes: ["low_title_anchor"],
    },
    main_120_draft_v1_0100: {
        query: "外语类保送生面试具体哪天举行",
        painTypes: ["low_title_anchor"],
    },
    main_120_draft_v1_0101: {
        query: "外语类保送生面试前，准考证最早什么时候可以打印",
        painTypes: ["low_title_anchor"],
    },
    main_120_draft_v1_0102: {
        query: "研究生想查这个月奖助金到账进度，应该通过什么方式看",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
};

const IN_DOMAIN_REWRITES: Record<string, Rewrite> = {
    in_domain_100_draft_v1_0002: {
        query: "同等学力申请硕士学位准备论文答辩，从提交申请到完成答辩要走哪些步骤",
        painTypes: ["low_title_anchor", "cross_doc_flow", "ambiguous_user_state"],
    },
    in_domain_100_draft_v1_0006: {
        query: "想走南疆高校教师专项调剂，申请到复试录取之间要按什么流程办",
        painTypes: ["low_title_anchor", "cross_doc_flow", "ambiguous_user_state"],
    },
    in_domain_100_draft_v1_0024: {
        query: "联合培养博士专项开放哪些学科方向可以报",
        painTypes: ["low_title_anchor"],
    },
    in_domain_100_draft_v1_0025: {
        query: "这个联培博士专项主要服务什么战略需求，培养目标是什么",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    in_domain_100_draft_v1_0026: {
        query: "硕士复试录取细则通常是依据哪些上级文件制定的",
        painTypes: ["low_title_anchor"],
    },
    in_domain_100_draft_v1_0041: {
        query: "我是在职医生，已有硕士学位，报名这个考试对学位类型有什么限制",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    in_domain_100_draft_v1_0042: {
        query: "在职医生想报名这类外语考试，临床工作经历至少要多久",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    in_domain_100_draft_v1_0046: {
        query: "报名这个考试前，在本专业临床工作至少要满几年",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    in_domain_100_draft_v1_0081: {
        query: "博士入学考试安排里，统考具体什么时候考、在哪里考",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    in_domain_100_draft_v1_0085: {
        query: "研究生初试那次考试具体安排在哪几天",
        painTypes: ["low_title_anchor"],
    },
    in_domain_100_draft_v1_0086: {
        query: "同等学力全国统考的外语科目几点开始",
        painTypes: ["low_title_anchor"],
    },
};

const BLIND_EXT_OOD_REWRITES: Record<string, Rewrite> = {
    blind_ext_ood_100_draft_v1_0001: {
        query: "联合培养博士专项改成线上报到时，学生应该按什么方式完成报到",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0004: {
        query: "免试生接收还有补报名机会的话，应该在哪里报、什么时候截止",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0005: {
        query: "我参加线上硕士复试，复试形式是什么，总成绩怎么计算",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0006: {
        query: "博士申请考核制里，考核流程和录取规则通常怎么安排",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0008: {
        query: "参加博士综合考核前，线上报到和考前培训要怎么安排",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0009: {
        query: "申请家庭经济困难认定时，材料和办理流程分别怎么安排",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0012: {
        query: "博士申请考核报名后还能不能改报志愿，具体要怎么处理",
        painTypes: ["low_title_anchor", "post_outcome_action", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0013: {
        query: "博士调剂公告里，申请条件、办理流程和批次安排分别是什么",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0014: {
        query: "应届本科生想申请推免，需要经过哪些环节",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0015: {
        query: "学生申请博士项目时，一般要经过哪些考核步骤",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0022: {
        query: "参加硕士复试时，考核内容、成绩计算和录取原则分别是什么",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0023: {
        query: "博士考核结束后，考核总成绩怎么使用，哪些情况不会被录取",
        painTypes: ["low_title_anchor", "multi_intent", "post_outcome_action"],
    },
    blind_ext_ood_100_draft_v1_0024: {
        query: "夏令营什么时候报名，报名对象和材料要求分别是什么",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0025: {
        query: "预推免考核名单公布后，学生还要不要主动联系学院确认考核安排",
        painTypes: ["low_title_anchor", "post_outcome_action", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0026: {
        query: "预推免考核名单出来后，这份名单对应的是哪一届学生",
        painTypes: ["low_title_anchor", "post_outcome_action", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0031: {
        query: "依托学院申报海外优青时，申请对象条件和重点支持方向是什么",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0032: {
        query: "硕士如果想调剂到这个学院，申请时要注意哪些要求",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0033: {
        query: "推免接收办法里，申请整体要求是什么",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0034: {
        query: "夏令营报名怎么操作，活动安排是什么",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0035: {
        query: "这个科研团队秘书岗位的职责、招聘条件和申请方式是什么",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0037: {
        query: "博士申请考核的主要组成部分是什么",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0039: {
        query: "博士招生的考核流程和录取标准是怎样的",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0045: {
        query: "青年学者想申请这个博士后项目，一般需要什么学术背景",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0049: {
        query: "申请博士项目时，基本条件和材料要求分别是什么",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0050: {
        query: "参加博士考核前，考生需要准备哪些材料和硬件设备",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0051: {
        query: "免试生接收是否包含直博生，申请条件和考核要求是什么",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0053: {
        query: "夏令营报名需要满足哪些条件，还要提交哪些材料",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0054: {
        query: "想申请直博生，英语能力和推荐信方面有什么具体要求",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0057: {
        query: "博士申请需要满足什么条件，材料提交截止到什么时候",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0058: {
        query: "博士条件",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0061: {
        query: "夏令营入营名单公布后，一共能看到多少名入营学生",
        painTypes: ["low_title_anchor", "post_outcome_action"],
    },
    blind_ext_ood_100_draft_v1_0062: {
        query: "我参加硕士复试，当天怎么报到，复试成绩占总成绩多少",
        painTypes: ["low_title_anchor", "multi_intent", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0063: {
        query: "想咨询免试生接收问题，应该打哪个联系电话",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0064: {
        query: "这个团队秘书岗位计划招几个人",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0065: {
        query: "计算机科学与技术专业进复试，总分线要求是多少",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0066: {
        query: "计算机科学与技术专业的复试线和公开招考计划人数分别是多少",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0067: {
        query: "推免预报名这次开放哪些招生专业",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0068: {
        query: "博士调剂细节",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0069: {
        query: "博士名单细节",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0071: {
        query: "硕士报名考试报考点代码是多少，适用哪些考生",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0074: {
        query: "推免预报名开放哪些招生专业",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0082: {
        query: "我准备申请博士，综合考核一般安排在什么时候",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0083: {
        query: "推免预报名网上报名最晚什么时候截止",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0084: {
        query: "拟录取后想放弃，过了通知里的截止时间还能处理吗",
        painTypes: ["low_title_anchor", "post_outcome_action", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0085: {
        query: "我入营后要安排行程，夏令营具体是哪几天举办",
        painTypes: ["low_title_anchor", "ambiguous_user_state"],
    },
    blind_ext_ood_100_draft_v1_0086: {
        query: "博士申请的综合考核安排在什么时间，地点在哪里",
        painTypes: ["low_title_anchor", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0088: {
        query: "博士招生最新报名条件和时间安排是什么",
        painTypes: ["low_title_anchor", "implicit_latest", "multi_intent"],
    },
    blind_ext_ood_100_draft_v1_0089: {
        query: "博士招生报名和材料提交有哪些关键时间点",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0094: {
        query: "博士调剂第一批的报名细节是什么",
        painTypes: ["low_title_anchor"],
    },
    blind_ext_ood_100_draft_v1_0099: {
        query: "硕士复试的时间安排是什么",
        painTypes: ["low_title_anchor"],
    },
};

const DATASETS: DatasetBuildSpec[] = [
    {
        label: "Main",
        inputFile:
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_generalization_aligned_120_draft_v3.json",
        outputFile:
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_generalization_aligned_120_draft_v4.json",
        rewrites: MAIN_REWRITES,
    },
    {
        label: "InDomain",
        inputFile:
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v6.json",
        outputFile:
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v7.json",
        rewrites: IN_DOMAIN_REWRITES,
    },
    {
        label: "BlindExtOOD",
        inputFile:
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v3.json",
        outputFile:
            "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v4.json",
        rewrites: BLIND_EXT_OOD_REWRITES,
    },
];

const BLIND_EXT_OOD_NO_YEAR_PREFIX_IDS = new Set([
    "blind_ext_ood_100_draft_v1_0031",
    "blind_ext_ood_100_draft_v1_0032",
    "blind_ext_ood_100_draft_v1_0033",
    "blind_ext_ood_100_draft_v1_0050",
    "blind_ext_ood_100_draft_v1_0051",
    "blind_ext_ood_100_draft_v1_0053",
    "blind_ext_ood_100_draft_v1_0054",
    "blind_ext_ood_100_draft_v1_0057",
    "blind_ext_ood_100_draft_v1_0058",
    "blind_ext_ood_100_draft_v1_0068",
    "blind_ext_ood_100_draft_v1_0069",
    "blind_ext_ood_100_draft_v1_0083",
    "blind_ext_ood_100_draft_v1_0084",
    "blind_ext_ood_100_draft_v1_0086",
    "blind_ext_ood_100_draft_v1_0088",
    "blind_ext_ood_100_draft_v1_0089",
    "blind_ext_ood_100_draft_v1_0094",
    "blind_ext_ood_100_draft_v1_0099",
]);

function loadDataset(file: string): DatasetCase[] {
    return JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), file), "utf-8"),
    ) as DatasetCase[];
}

function applyRewrite(item: DatasetCase, rewrite: Rewrite): DatasetCase {
    const year = typeof item.year_inferred_v1 === "string" ? item.year_inferred_v1 : "";
    const query =
        item.id?.startsWith("blind_ext_ood_") &&
        !BLIND_EXT_OOD_NO_YEAR_PREFIX_IDS.has(item.id) &&
        year &&
        !/20\d{2}/.test(rewrite.query)
            ? `${year}年${rewrite.query}`
            : rewrite.query;
    return {
        ...item,
        user_pain_rewrite_from_query: item.query,
        user_pain_rewrite_version: "user_pain_balanced_v1",
        user_pain_types_target: rewrite.painTypes,
        query,
        query_derivation: "user_pain_balanced_v1",
        notes: [item.notes, "user_pain_balanced_v1: high-anchor L1 query rewritten toward realistic user pain while preserving expected_otid/support provenance."]
            .filter(Boolean)
            .join(" "),
        challenge_tags: Array.from(
            new Set([...(item.challenge_tags || []), "user_pain_rewrite"]),
        ),
    };
}

function buildDataset(spec: DatasetBuildSpec) {
    const dataset = loadDataset(spec.inputFile);
    let rewrittenCount = 0;
    const rewritten = dataset.map((item) => {
        const id = item.id || "";
        const rewrite = spec.rewrites[id];
        if (!rewrite) {
            return item;
        }
        rewrittenCount += 1;
        return applyRewrite(item, rewrite);
    });

    const missingIds = Object.keys(spec.rewrites).filter(
        (id) => !dataset.some((item) => item.id === id),
    );
    if (missingIds.length > 0) {
        throw new Error(`${spec.label} missing rewrite ids: ${missingIds.join(", ")}`);
    }

    const outputPath = path.resolve(process.cwd(), spec.outputFile);
    fs.writeFileSync(outputPath, JSON.stringify(rewritten, null, 2), "utf-8");
    return {
        label: spec.label,
        inputFile: spec.inputFile,
        outputFile: spec.outputFile,
        total: rewritten.length,
        rewrittenCount,
    };
}

function main(): void {
    const summaries = DATASETS.map(buildDataset);
    console.log(JSON.stringify(summaries, null, 2));
}

main();

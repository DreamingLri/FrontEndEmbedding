import * as fs from "fs";
import * as path from "path";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";

type DatasetCase = {
    id?: string;
    query?: string;
    source_query?: string;
    ot_title?: string;
    query_scope?: string;
    support_pattern?: string;
    preferred_granularity?: string;
    anchor_bucket?: string;
    difficulty_level_v1?: string;
    difficulty_score_v1?: number;
    same_theme_candidate_count_v1?: number;
    same_theme_near_year_candidate_count_v1?: number;
};

type DatasetSpec = {
    label: string;
    file: string;
};

type PainType =
    | "implicit_latest"
    | "low_title_anchor"
    | "multi_intent"
    | "post_outcome_action"
    | "cross_doc_flow"
    | "ambiguous_user_state";

type AnnotatedCase = DatasetCase & {
    user_pain_types: PainType[];
    structural_complexity_tags: string[];
    title_anchor_level: "high" | "medium" | "low";
    title_bigram_overlap_ratio: number;
    requires_latest: boolean;
    requires_cross_doc: boolean;
    real_user_pain_score_v1: number;
    replacement_priority: "keep" | "review_replace_candidate";
};

const DEFAULT_DATASETS: DatasetSpec[] = [
    {
        label: "Main",
        file: CURRENT_EVAL_DATASET_FILES.granularityMain120,
    },
    {
        label: "InDomain",
        file: CURRENT_EVAL_DATASET_FILES.granularityInDomainHoldout50,
    },
    {
        label: "BlindExtOOD",
        file: CURRENT_EVAL_DATASET_FILES.granularityExtOodBlind60,
    },
];

const ASPECT_PATTERNS: Array<{ name: string; pattern: RegExp }> = [
    { name: "condition", pattern: /条件|资格|要求|能否|能不能|可以.*报|还能.*报/ },
    { name: "material", pattern: /材料|提交|证明|成绩单|推荐信|准备/ },
    { name: "time", pattern: /时间|日期|什么时候|何时|截止|几号|哪天|多久/ },
    { name: "procedure", pattern: /流程|步骤|环节|程序|怎么|如何|安排|报到|申请|报名/ },
    { name: "outcome", pattern: /结果|名单|公示|拟录取|递补|增补|通过|录取/ },
    { name: "contact", pattern: /联系|电话|邮箱|办公室|老师|申诉|监督/ },
];

function parseDatasetSpecs(): DatasetSpec[] {
    const raw = process.env.SUASK_USER_PAIN_PROFILE_DATASETS;
    if (!raw) {
        return DEFAULT_DATASETS;
    }
    return raw.split(";").map((item) => {
        const [label, file] = item.split("=");
        if (!label || !file) {
            throw new Error(
                "SUASK_USER_PAIN_PROFILE_DATASETS must use label=path;label=path format.",
            );
        }
        return { label, file };
    });
}

function loadDataset(file: string): DatasetCase[] {
    return JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), file), "utf-8"),
    ) as DatasetCase[];
}

function normalizeText(text: string): string {
    return text.replace(/\s+/g, "").replace(/[，。；！？,.!?（）()《》“”"':：、]/g, "");
}

function hasExplicitYear(text: string): boolean {
    return /20\d{2}/.test(text);
}

function titleHasYear(title: string): boolean {
    return /20\d{2}/.test(title);
}

function stripLowValueTitleText(text: string): string {
    return normalizeText(text)
        .replace(/20\d{2}年?/g, "")
        .replace(/中山大学|关于|通知|公告|办法|方案|章程|简章|细则|实施|招生|研究生/g, "");
}

function bigrams(text: string): Set<string> {
    const normalized = stripLowValueTitleText(text);
    const result = new Set<string>();
    for (let i = 0; i < normalized.length - 1; i += 1) {
        result.add(normalized.slice(i, i + 2));
    }
    return result;
}

function titleOverlapRatio(query: string, title: string): number {
    const queryBigrams = bigrams(query);
    const titleBigrams = bigrams(title);
    if (queryBigrams.size === 0 || titleBigrams.size === 0) {
        return 0;
    }
    let overlap = 0;
    queryBigrams.forEach((item) => {
        if (titleBigrams.has(item)) {
            overlap += 1;
        }
    });
    return Number((overlap / queryBigrams.size).toFixed(4));
}

function resolveTitleAnchorLevel(ratio: number, anchorBucket?: string): AnnotatedCase["title_anchor_level"] {
    if (anchorBucket === "weak_anchor" || ratio < 0.12) {
        return "low";
    }
    if (anchorBucket === "strong_anchor" || ratio >= 0.28) {
        return "high";
    }
    return "medium";
}

function matchedAspectCount(query: string): number {
    return ASPECT_PATTERNS.filter((item) => item.pattern.test(query)).length;
}

function inferPainTypes(item: DatasetCase, ratio: number): PainType[] {
    const query = normalizeText(item.query || "");
    const title = normalizeText(item.ot_title || "");
    const painTypes: PainType[] = [];
    const noExplicitYearToYearTarget = !hasExplicitYear(query) && titleHasYear(title);

    if (
        noExplicitYearToYearTarget ||
        (/今年|最新|现在|目前|最近/.test(query) && !hasExplicitYear(query))
    ) {
        painTypes.push("implicit_latest");
    }
    if (resolveTitleAnchorLevel(ratio, item.anchor_bucket) === "low") {
        painTypes.push("low_title_anchor");
    }
    if (matchedAspectCount(query) >= 2) {
        painTypes.push("multi_intent");
    }
    if (
        /通过.*后|录取.*后|拟录取.*后|名单.*后|没有.*名单|还会.*录取|递补|增补|放弃|确认|体检|复审|申诉/.test(
            query,
        )
    ) {
        painTypes.push("post_outcome_action");
    }
    if (
        /完整流程|整个流程|全流程|从.*到|分别|以及|同时|申请和录取|报名到录取/.test(
            query,
        ) ||
        /条件.*材料|材料.*时间|流程.*结果|报名.*录取|申请.*录取/.test(query)
    ) {
        painTypes.push("cross_doc_flow");
    }
    if (
        /我|本人|能不能|能否|还可以|是否|是不是|该|怎么办|需要做什么|下一步/.test(
            query,
        ) &&
        ratio < 0.28
    ) {
        painTypes.push("ambiguous_user_state");
    }

    return Array.from(new Set(painTypes));
}

function inferStructuralComplexityTags(item: DatasetCase): string[] {
    const tags: string[] = [];
    if (item.support_pattern === "multi_kp") {
        tags.push("multi_kp");
    }
    if (item.preferred_granularity === "KP+OT") {
        tags.push("kp_plus_ot");
    }
    if (Number(item.same_theme_candidate_count_v1 || 0) >= 3) {
        tags.push("dense_same_theme");
    }
    if (Number(item.same_theme_near_year_candidate_count_v1 || 0) >= 2) {
        tags.push("dense_version_chain");
    }
    return tags;
}

function annotateCase(item: DatasetCase): AnnotatedCase {
    const query = item.query || "";
    const title = item.ot_title || "";
    const overlapRatio = titleOverlapRatio(query, title);
    const userPainTypes = inferPainTypes(item, overlapRatio);
    const structuralComplexityTags = inferStructuralComplexityTags(item);
    const titleAnchorLevel = resolveTitleAnchorLevel(
        overlapRatio,
        item.anchor_bucket,
    );
    const realUserPainScore = userPainTypes.length + (titleAnchorLevel === "low" ? 0.5 : 0);
    const replacementPriority =
        userPainTypes.length === 0 &&
        titleAnchorLevel === "high" &&
        (item.difficulty_level_v1 === "L1" || Number(item.difficulty_score_v1 || 0) < 2.4)
            ? "review_replace_candidate"
            : "keep";

    return {
        ...item,
        user_pain_types: userPainTypes,
        structural_complexity_tags: structuralComplexityTags,
        title_anchor_level: titleAnchorLevel,
        title_bigram_overlap_ratio: overlapRatio,
        requires_latest: userPainTypes.includes("implicit_latest"),
        requires_cross_doc: userPainTypes.includes("cross_doc_flow"),
        real_user_pain_score_v1: Number(realUserPainScore.toFixed(2)),
        replacement_priority: replacementPriority,
    };
}

function increment(target: Record<string, number>, key: string): void {
    target[key] = (target[key] || 0) + 1;
}

function summarize(label: string, file: string, cases: DatasetCase[]) {
    const annotated = cases.map(annotateCase);
    const summary = {
        label,
        file,
        total: annotated.length,
        painCaseCount: 0,
        painCaseRate: 0,
        avgPainTypesPerCase: 0,
        avgTitleOverlapRatio: 0,
        replacementCandidateCount: 0,
        byPainType: {} as Record<string, number>,
        byTitleAnchorLevel: {} as Record<string, number>,
        bySupportPattern: {} as Record<string, number>,
        replacementCandidates: [] as Array<{
            id?: string;
            query?: string;
            ot_title?: string;
            title_bigram_overlap_ratio: number;
            difficulty_level_v1?: string;
            difficulty_score_v1?: number;
        }>,
        byStructuralComplexityTag: {} as Record<string, number>,
    };

    for (const item of annotated) {
        if (item.user_pain_types.length > 0) {
            summary.painCaseCount += 1;
        }
        summary.avgPainTypesPerCase += item.user_pain_types.length;
        summary.avgTitleOverlapRatio += item.title_bigram_overlap_ratio;
        item.user_pain_types.forEach((painType) =>
            increment(summary.byPainType, painType),
        );
        item.structural_complexity_tags.forEach((tag) =>
            increment(summary.byStructuralComplexityTag, tag),
        );
        increment(summary.byTitleAnchorLevel, item.title_anchor_level);
        increment(summary.bySupportPattern, item.support_pattern || "unknown");
        if (item.replacement_priority === "review_replace_candidate") {
            summary.replacementCandidateCount += 1;
            summary.replacementCandidates.push({
                id: item.id,
                query: item.query,
                ot_title: item.ot_title,
                title_bigram_overlap_ratio: item.title_bigram_overlap_ratio,
                difficulty_level_v1: item.difficulty_level_v1,
                difficulty_score_v1: item.difficulty_score_v1,
            });
        }
    }

    summary.painCaseRate = Number(
        ((summary.painCaseCount / Math.max(summary.total, 1)) * 100).toFixed(1),
    );
    summary.avgPainTypesPerCase = Number(
        (summary.avgPainTypesPerCase / Math.max(summary.total, 1)).toFixed(2),
    );
    summary.avgTitleOverlapRatio = Number(
        (summary.avgTitleOverlapRatio / Math.max(summary.total, 1)).toFixed(4),
    );
    summary.replacementCandidates = summary.replacementCandidates.slice(0, 20);

    return { summary, annotated };
}

function main(): void {
    const specs = parseDatasetSpecs();
    const datasets = specs.map((spec) => {
        const { summary, annotated } = summarize(
            spec.label,
            spec.file,
            loadDataset(spec.file),
        );
        return { ...spec, summary, annotated };
    });

    const report = {
        generatedAt: new Date().toISOString(),
        profileDefinition:
            "Heuristic user-pain audit for implicit latest, low title anchor, multi-intent, post-outcome action, cross-doc flow, and ambiguous user state. This script does not mutate source datasets.",
        targetPainCaseRate: "30%-40% per split after manual replacement",
        summaries: datasets.map((item) => item.summary),
    };

    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const stamp = Date.now();
    const reportFile = path.join(
        outDir,
        `granularity_user_pain_profile_${stamp}.json`,
    );
    const annotatedFile = path.join(
        outDir,
        `granularity_user_pain_profile_${stamp}_annotated_cases.json`,
    );
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2), "utf-8");
    fs.writeFileSync(
        annotatedFile,
        JSON.stringify(
            datasets.map((item) => ({
                label: item.label,
                file: item.file,
                cases: item.annotated,
            })),
            null,
            2,
        ),
        "utf-8",
    );
    console.log(`Saved user-pain profile to ${reportFile}`);
    console.log(`Saved annotated cases to ${annotatedFile}`);
    console.log(JSON.stringify(report.summaries, null, 2));
}

main();

import type { ParsedQueryIntent } from "./vector_engine.ts";

export type QueryPlanIntentType =
    | "outcome"
    | "procedure"
    | "requirement"
    | "time_location"
    | "policy_overview"
    | "system_timeline"
    | "fact_detail";

export type QueryPlanDocRole =
    | "rule_doc"
    | "registration_notice"
    | "stage_list"
    | "result_notice"
    | "list_notice"
    | "adjustment_notice"
    | "general_notice";

export type QueryPlanDifficultyTier = "default" | "high";

export type QueryPlan = {
    rawQuery: string;
    normalizedQuery: string;
    intentType: QueryPlanIntentType;
    difficultyTier: QueryPlanDifficultyTier;
    asksOutcomeLike: boolean;
    asksProcedureLike: boolean;
    asksRequirementLike: boolean;
    asksTimeLike: boolean;
    asksCoverageLike: boolean;
    asksSystemTimelineLike: boolean;
    hasExplicitPhaseAnchor: boolean;
    preferredDocRoles: QueryPlanDocRole[];
    avoidedDocRoles: QueryPlanDocRole[];
    fetchMatchLimitDelta: number;
    fetchWeakMatchLimitDelta: number;
    phaseAnchorWeightScale: number;
    titleIntentWeightScale: number;
    coverageWeightScale: number;
};

const RULE_DOC_PATTERN =
    /(招生简章|招生章程|实施细则|实施办法|接收办法|工作方案|录取方案|章程|简章)/;
const REGISTRATION_NOTICE_PATTERN =
    /(活动报名通知|报名通知|预报名|综合考核通知|复试通知|考核通知|考核安排|申请通知|报名安排|网上报名|报名系统)/;
const RESULT_NOTICE_PATTERN =
    /(结果公示|录取结果|拟录取|公示|名单|递补|增补|入营通知|复试结果|综合考核结果)/;
const LIST_NOTICE_PATTERN =
    /(考生名单|入围名单|入营名单|进入.*名单|通过.*名单|资格审核名单)/;
const STAGE_LIST_PATTERN =
    /(时间安排|时间节点|日程安排|流程|考核安排|面试安排|活动安排|报到须知|安排表)/;
const ADJUSTMENT_NOTICE_PATTERN = /调剂/;
const PHASE_ANCHOR_PATTERN =
    /(上半年|下半年|第?[一二三四1234]批|预报名|报名通知|工作方案|接收办法|实施办法|录取方案|招生简章|招生章程|综合考核|复试|调剂)/;

function dedupe<T>(items: T[]): T[] {
    return Array.from(new Set(items));
}

function normalizePatternText(text: string): string {
    return text.replace(/\s+/g, "");
}

function countRequestedAspects(normalizedQuery: string): number {
    const aspectPatterns = [
        /条件|资格|要求/,
        /材料|提交|准备什么材料/,
        /时间|日期|什么时候|何时|截止|地点|哪里/,
        /系统|报名功能|录取功能|操作/,
        /流程|步骤|经过哪些|过程|程序|考核/,
        /评分|打分|成绩|单科/,
    ];
    return aspectPatterns.filter((pattern) => pattern.test(normalizedQuery)).length;
}

export function inferDocumentRolesFromTitle(title: string): QueryPlanDocRole[] {
    const normalizedTitle = normalizePatternText(title);
    if (!normalizedTitle) {
        return ["general_notice"];
    }

    const roles: QueryPlanDocRole[] = [];
    if (RULE_DOC_PATTERN.test(normalizedTitle)) {
        roles.push("rule_doc");
    }
    if (REGISTRATION_NOTICE_PATTERN.test(normalizedTitle)) {
        roles.push("registration_notice");
    }
    if (RESULT_NOTICE_PATTERN.test(normalizedTitle)) {
        roles.push("result_notice");
    }
    if (LIST_NOTICE_PATTERN.test(normalizedTitle)) {
        roles.push("list_notice");
    }
    if (STAGE_LIST_PATTERN.test(normalizedTitle)) {
        roles.push("stage_list");
    }
    if (ADJUSTMENT_NOTICE_PATTERN.test(normalizedTitle)) {
        roles.push("adjustment_notice");
    }

    return roles.length > 0 ? dedupe(roles) : ["general_notice"];
}

function resolveIntentType(params: {
    asksOutcomeLike: boolean;
    asksProcedureLike: boolean;
    asksRequirementLike: boolean;
    asksTimeLike: boolean;
    asksPolicyOverviewLike: boolean;
    asksSystemTimelineLike: boolean;
}): QueryPlanIntentType {
    const {
        asksOutcomeLike,
        asksProcedureLike,
        asksRequirementLike,
        asksTimeLike,
        asksPolicyOverviewLike,
        asksSystemTimelineLike,
    } = params;

    if (asksOutcomeLike) {
        return "outcome";
    }
    if (asksSystemTimelineLike) {
        return "system_timeline";
    }
    if (asksProcedureLike) {
        return "procedure";
    }
    if (asksRequirementLike) {
        return "requirement";
    }
    if (asksTimeLike) {
        return "time_location";
    }
    if (asksPolicyOverviewLike) {
        return "policy_overview";
    }
    return "fact_detail";
}

function resolveRolePreference(
    intentType: QueryPlanIntentType,
    asksCoverageLike: boolean,
): Pick<QueryPlan, "preferredDocRoles" | "avoidedDocRoles"> {
    switch (intentType) {
        case "outcome":
            return {
                preferredDocRoles: ["result_notice", "list_notice", "stage_list"],
                avoidedDocRoles: ["rule_doc", "registration_notice"],
            };
        case "procedure":
            return {
                preferredDocRoles: [
                    "registration_notice",
                    "rule_doc",
                    "stage_list",
                    "adjustment_notice",
                ],
                avoidedDocRoles: ["result_notice", "list_notice"],
            };
        case "requirement":
            return {
                preferredDocRoles: ["rule_doc", "registration_notice"],
                avoidedDocRoles: ["result_notice", "list_notice"],
            };
        case "time_location":
            return {
                preferredDocRoles: ["registration_notice", "stage_list", "result_notice"],
                avoidedDocRoles: [],
            };
        case "policy_overview":
            return {
                preferredDocRoles: ["rule_doc", "registration_notice"],
                avoidedDocRoles: ["result_notice", "list_notice"],
            };
        case "system_timeline":
            return {
                preferredDocRoles: ["registration_notice", "rule_doc", "stage_list"],
                avoidedDocRoles: ["result_notice", "list_notice"],
            };
        case "fact_detail":
        default:
            return {
                preferredDocRoles: asksCoverageLike
                    ? ["rule_doc", "stage_list", "registration_notice"]
                    : ["registration_notice", "rule_doc"],
                avoidedDocRoles: [],
            };
    }
}

export function buildQueryPlan(
    query: string,
    queryIntent: ParsedQueryIntent,
): QueryPlan {
    const normalizedQuery = normalizePatternText(query);
    const asksOutcomeLike =
        /结果|公示|名单|拟录取|递补|增补|录取结果|入营/.test(normalizedQuery);
    const asksProcedureLike =
        /流程|步骤|环节|程序|过程|考核步骤|需要经过|怎么申请|如何申请|怎么报名|如何报名/.test(
            normalizedQuery,
        );
    const asksRequirementLike =
        /条件|要求|资格|材料|评分|怎么评分|如何评分|关键要求|规则/.test(
            normalizedQuery,
        );
    const asksTimeLike =
        /时间|日期|什么时候|何时|截止|地点|哪里|哪天|几月几日|举办日期|举办时间/.test(
            normalizedQuery,
        );
    const asksPolicyOverviewLike =
        /整体政策|主要要求|关键要求|政策|总体要求/.test(normalizedQuery);
    const asksSystemTimelineLike =
        /关键时间节点|时间节点|截止时间|截止日期|系统操作|报名功能|录取功能|服务系统/.test(
            normalizedQuery,
        );
    const requestedAspectCount = countRequestedAspects(normalizedQuery);
    const asksCoverageLike =
        /分别|以及|并描述|整个流程|申请和录取过程|从预报名到录取|从准备材料到完成面试|条件.*评分|条件.*材料|材料.*时间|申请.*录取过程/.test(
            normalizedQuery,
        ) || requestedAspectCount >= 2;
    const hasExplicitPhaseAnchor =
        PHASE_ANCHOR_PATTERN.test(normalizedQuery) ||
        queryIntent.years.length > 0 ||
        queryIntent.eventTypes.length > 0;

    const intentType = resolveIntentType({
        asksOutcomeLike,
        asksProcedureLike,
        asksRequirementLike,
        asksTimeLike,
        asksPolicyOverviewLike,
        asksSystemTimelineLike,
    });
    const rolePreference = resolveRolePreference(intentType, asksCoverageLike);
    const lowAnchorHighRecall =
        !queryIntent.signals.hasExplicitYear &&
        !queryIntent.signals.hasStrongDetailAnchor &&
        !queryIntent.signals.hasExplicitTopicOrIntent &&
        queryIntent.signals.queryLength <= 18;

    let titleIntentWeightScale = 1;
    if (
        intentType === "procedure" ||
        intentType === "requirement" ||
        intentType === "policy_overview" ||
        intentType === "system_timeline"
    ) {
        titleIntentWeightScale += 0.08;
    } else if (intentType === "outcome") {
        titleIntentWeightScale += 0.05;
    }
    if (queryIntent.signals.hasStrongDetailAnchor) {
        titleIntentWeightScale += 0.03;
    }

    let coverageWeightScale = 1;
    if (asksCoverageLike) {
        coverageWeightScale += 0.25;
    } else if (
        intentType === "procedure" ||
        intentType === "system_timeline"
    ) {
        coverageWeightScale += 0.08;
    }

    let phaseAnchorWeightScale = 1;
    if (hasExplicitPhaseAnchor) {
        phaseAnchorWeightScale += 0.08;
    } else if (queryIntent.signals.hasExplicitYear) {
        phaseAnchorWeightScale += 0.03;
    }

    const fetchMatchLimitDelta = asksCoverageLike ? 2 : 0;
    const fetchWeakMatchLimitDelta =
        asksCoverageLike || lowAnchorHighRecall ? 1 : 0;

    return {
        rawQuery: query,
        normalizedQuery,
        intentType,
        difficultyTier:
            asksCoverageLike || lowAnchorHighRecall ? "high" : "default",
        asksOutcomeLike,
        asksProcedureLike,
        asksRequirementLike,
        asksTimeLike,
        asksCoverageLike,
        asksSystemTimelineLike,
        hasExplicitPhaseAnchor,
        preferredDocRoles: rolePreference.preferredDocRoles,
        avoidedDocRoles: rolePreference.avoidedDocRoles,
        fetchMatchLimitDelta,
        fetchWeakMatchLimitDelta,
        phaseAnchorWeightScale,
        titleIntentWeightScale,
        coverageWeightScale,
    };
}

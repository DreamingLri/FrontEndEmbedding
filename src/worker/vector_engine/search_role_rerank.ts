import type { KPCandidate } from "../aggregated_doc_scores.ts";
import {
    DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    hasPostOutcomeConditionCue,
    type KPRoleRerankMode,
} from "./shared.ts";
export type QueryRoleSignals = {
    asksTime: boolean;
    asksCondition: boolean;
    asksPostOutcomeCondition: boolean;
    asksMaterials: boolean;
    asksProcedure: boolean;
    asksExamContent: boolean;
    asksAnnouncementPeriod: boolean;
    asksApplicationStage: boolean;
    mentionsThesis: boolean;
    mentionsPrintedDocument: boolean;
    mentionsCollectionOrArchive: boolean;
    mentionsReviewOrReissue: boolean;
};

export function hasKpRoleTag(
    candidate: Pick<KPCandidate, "kp_role_tags"> | undefined,
    tag: string,
): boolean {
    return candidate?.kp_role_tags?.includes(tag) === true;
}

export function deriveQueryRoleSignals(
    rawQuery: string,
    queryScopeHint?: string,
): QueryRoleSignals {
    return {
        asksTime:
            /什么时候|何时|哪几天|几号|截止|到账|时间|公示期/.test(
                rawQuery,
            ) || queryScopeHint === "time_location",
        asksCondition:
            /条件|满足|资格/.test(rawQuery) ||
            queryScopeHint === "eligibility_condition",
        asksPostOutcomeCondition: hasPostOutcomeConditionCue(rawQuery),
        asksMaterials: /材料|扫描件|电子版|邮箱|mail|提交|携带/i.test(rawQuery),
        asksProcedure: /怎么办|怎么处理|不通过|补交|补充|流程|步骤|报到|候考|地点|现场|资格审查/.test(
            rawQuery,
        ),
        asksExamContent:
            /考什么|考哪些|考试内容|考试科目|科目|题型|综合能力|分值|权重|占比|比例/.test(
                rawQuery,
            ),
        asksAnnouncementPeriod: /公示期|哪几天/.test(rawQuery),
        asksApplicationStage:
            /申请|报名|确认|提交/.test(rawQuery) &&
            !/通过后|答辩通过|审批后|获得学位/.test(rawQuery),
        mentionsThesis: /论文/.test(rawQuery),
        mentionsPrintedDocument: /准考证|打印|纸质/.test(rawQuery),
        mentionsCollectionOrArchive: /领取|证书|档案/.test(rawQuery),
        mentionsReviewOrReissue: /资格|评审|补发/.test(rawQuery),
    };
}

function computeKpRoleBonus(
    candidate: KPCandidate,
    signals: QueryRoleSignals,
    rawQuery: string,
): number {
    let bonus = 0;
    const asksOperationalEvidence =
        signals.asksCondition ||
        signals.asksMaterials ||
        signals.asksProcedure ||
        signals.asksTime ||
        /资格审查|报到|候考|地点|现场|安排/.test(rawQuery);
    const asksRuleDocument =
        /(招生简章|简章|招生章程|章程|实施细则|细则|实施办法|办法|接收办法|录取方案|方案)/.test(
            rawQuery,
        ) && !/(结果|公示|名单|递补|增补|拟录取|录取结果)/.test(rawQuery);

    if (signals.asksTime) {
        if (
            hasKpRoleTag(candidate, "arrival")
            || hasKpRoleTag(candidate, "deadline")
            || hasKpRoleTag(candidate, "announcement_period")
            || hasKpRoleTag(candidate, "schedule")
        ) {
            bonus += 0.9;
        }
        if (hasKpRoleTag(candidate, "time_expression")) {
            bonus += 0.45;
        }
    }

    if (signals.asksCondition) {
        if (hasKpRoleTag(candidate, "condition")) {
            bonus += 1.1;
        }
        // Some eligibility constraints are encoded as cutoff deadlines.
        if (hasKpRoleTag(candidate, "deadline")) {
            bonus += 0.55;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus += signals.asksPostOutcomeCondition ? 1.0 : -0.7;
        }
    }

    if (signals.asksMaterials) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 1.0;
        }
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 0.35;
        }
        if (
            hasKpRoleTag(candidate, "materials")
            && hasKpRoleTag(candidate, "email")
        ) {
            bonus += 0.9;
        }
        if (/申请|答辩/.test(rawQuery) && hasKpRoleTag(candidate, "application_stage")) {
            bonus += 0.9;
        }
        if (
            !signals.mentionsThesis
            && (hasKpRoleTag(candidate, "post_outcome")
                || hasKpRoleTag(candidate, "thesis"))
        ) {
            bonus -= 1.2;
        }
    }

    if (signals.mentionsPrintedDocument) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 0.55;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.35;
        }
    }

    if (signals.asksApplicationStage) {
        if (hasKpRoleTag(candidate, "application_stage")) {
            bonus += 1.1;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 1.1;
        }
    }

    if (
        signals.mentionsThesis &&
        signals.asksApplicationStage &&
        !signals.asksCondition
    ) {
        if (hasKpRoleTag(candidate, "condition")) {
            bonus -= 0.55;
        }
        if (hasKpRoleTag(candidate, "application_stage")) {
            bonus += 0.3;
        }
    }

    if (signals.asksProcedure) {
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 1.35;
        }
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.6;
        }
        if (
            hasKpRoleTag(candidate, "reminder")
            || hasKpRoleTag(candidate, "background")
        ) {
            bonus -= 0.8;
        }
    }

    if (signals.asksExamContent) {
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 1.1;
        }
        if (hasKpRoleTag(candidate, "time_expression")) {
            bonus += 0.25;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 1.35;
        }
        if (hasKpRoleTag(candidate, "announcement_period")) {
            bonus -= 0.9;
        }
        if (hasKpRoleTag(candidate, "deadline")) {
            bonus -= 0.55;
        }
        if (hasKpRoleTag(candidate, "publish")) {
            bonus -= 0.45;
        }
    }

    if (/(权重|占比|比例|分值)/.test(rawQuery)) {
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.45;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.4;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.6;
        }
    }

    if (/资格审查|报到|候考|地点|现场|安排/.test(rawQuery)) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 0.55;
        }
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 0.85;
        }
        if (hasKpRoleTag(candidate, "schedule")) {
            bonus += 0.7;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.7;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.6;
        }
    }

    if (asksRuleDocument) {
        if (
            hasKpRoleTag(candidate, "condition") ||
            hasKpRoleTag(candidate, "materials") ||
            hasKpRoleTag(candidate, "procedure") ||
            hasKpRoleTag(candidate, "application_stage")
        ) {
            bonus += 0.25;
        }
        if (
            hasKpRoleTag(candidate, "background") ||
            hasKpRoleTag(candidate, "publish") ||
            hasKpRoleTag(candidate, "post_outcome")
        ) {
            bonus -= 0.55;
        }
    }

    if (signals.asksAnnouncementPeriod) {
        if (hasKpRoleTag(candidate, "announcement_period")) {
            bonus += 1.2;
        }
        if (
            hasKpRoleTag(candidate, "publish")
            && !hasKpRoleTag(candidate, "announcement_period")
        ) {
            bonus -= 0.6;
        }
    }

    if (/到账/.test(rawQuery)) {
        if (hasKpRoleTag(candidate, "arrival")) {
            bonus += 1.0;
        }
        if (hasKpRoleTag(candidate, "distribution")) {
            bonus -= 0.5;
        }
    }

    if (signals.mentionsCollectionOrArchive) {
        if (hasKpRoleTag(candidate, "reminder")) {
            bonus += 0.7;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus += 0.35;
        }
        if (hasKpRoleTag(candidate, "materials")) {
            bonus -= 0.35;
        }
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.35;
        }
    }

    if (signals.mentionsReviewOrReissue) {
        if (hasKpRoleTag(candidate, "deadline")) {
            bonus += 0.7;
        }
        if (hasKpRoleTag(candidate, "distribution")) {
            bonus -= 0.45;
        }
        if (hasKpRoleTag(candidate, "publish")) {
            bonus -= 0.25;
        }
    }

    if (!signals.asksCondition && hasKpRoleTag(candidate, "condition")) {
        bonus -= 0.25;
    }

    if (!signals.asksMaterials && hasKpRoleTag(candidate, "materials")) {
        bonus -= 0.35;
    }

    if (
        !signals.asksAnnouncementPeriod &&
        !signals.asksPostOutcomeCondition &&
        hasKpRoleTag(candidate, "publish")
    ) {
        bonus -= 0.2;
    }

    if (
        !/到账|发放|补发/.test(rawQuery) &&
        hasKpRoleTag(candidate, "distribution")
    ) {
        bonus -= 0.35;
    }

    if (hasKpRoleTag(candidate, "background")) {
        bonus -= 0.15;
    }

    if (asksOperationalEvidence) {
        if (hasKpRoleTag(candidate, "background")) {
            bonus -= 0.45;
        }
        if (hasKpRoleTag(candidate, "publish")) {
            bonus -= 0.35;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.6;
        }
    }

    return bonus;
}

export function rerankKpCandidatesByRole(params: {
    kpCandidates: readonly KPCandidate[];
    bestKpid?: string;
    rawQuery: string;
    queryScopeHint?: string;
    mode?: KPRoleRerankMode;
}): {
    bestKpid?: string;
    orderedCandidates: KPCandidate[];
    docScoreDelta: number;
} {
    const {
        kpCandidates,
        bestKpid,
        rawQuery,
        queryScopeHint,
        mode = "off",
    } = params;

    const orderedCandidates = [...kpCandidates];
    if (mode !== "feature" || orderedCandidates.length === 0) {
        return {
            bestKpid,
            orderedCandidates,
            docScoreDelta: 0,
        };
    }

    const rerankWindow = orderedCandidates.slice(
        0,
        DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    );
    const rawTopScore = rerankWindow[0]?.score ?? Number.NEGATIVE_INFINITY;
    const signals = deriveQueryRoleSignals(rawQuery, queryScopeHint);
    const reranked = rerankWindow
        .map((candidate) => ({
            candidate,
            rerankedScore:
                candidate.score + computeKpRoleBonus(candidate, signals, rawQuery),
        }))
        .sort((a, b) => b.rerankedScore - a.rerankedScore);

    const topCandidate = reranked[0];
    return {
        bestKpid: topCandidate?.candidate.kpid || bestKpid,
        orderedCandidates: [
            ...reranked.map((item) => item.candidate),
            ...orderedCandidates.slice(rerankWindow.length),
        ],
        docScoreDelta:
            Number.isFinite(rawTopScore) && Number.isFinite(topCandidate?.rerankedScore)
                ? Math.max(0, topCandidate.rerankedScore - rawTopScore)
                : 0,
    };
}
export function hasAnyRoleEvidence(
    candidates: readonly KPCandidate[],
    tags: readonly string[],
): boolean {
    return candidates.some((candidate) =>
        tags.some((tag) => hasKpRoleTag(candidate, tag)),
    );
}

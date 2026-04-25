import {
    inferDocumentRolesFromTitle,
    type QueryPlanDocRole,
} from "../query_planner.ts";
import { computeKpEvidenceGroupCounts } from "../vector_engine/shared.ts";
import type { PipelineDocumentRecord } from "./types.ts";

export type PhaseAnchor = {
    half?: "上半年" | "下半年";
    batch?: string;
    stages: string[];
};

export const PHASE_ANCHOR_DOC_WEIGHT = 0.35;
export const TITLE_INTENT_DOC_WEIGHT = 0.28;
export const TITLE_COVERAGE_DOC_WEIGHT = 0.18;
export const LATEST_VERSION_DOC_WEIGHT = 0.38;
export const TITLE_DIVERSITY_DUPLICATE_PENALTY = 0.34;

const PHASE_STAGE_RULES: Array<{ stage: string; pattern: RegExp }> = [
    { stage: "预报名", pattern: /预报名/ },
    { stage: "报名通知", pattern: /报名通知|网上报名/ },
    { stage: "工作方案", pattern: /工作方案/ },
    { stage: "接收办法", pattern: /接收办法/ },
    { stage: "实施办法", pattern: /实施办法/ },
    { stage: "录取方案", pattern: /录取方案/ },
    { stage: "招生简章", pattern: /招生简章/ },
    { stage: "招生章程", pattern: /招生章程/ },
    { stage: "综合考核", pattern: /综合考核/ },
    { stage: "复试", pattern: /复试/ },
    { stage: "调剂", pattern: /调剂/ },
] as const;

const TITLE_RULE_DOC_PATTERN =
    /(招生简章|招生章程|实施细则|实施办法|接收办法|工作方案|录取方案|章程|简章)/;
const TITLE_PROCESS_NOTICE_PATTERN =
    /(活动报名通知|报名通知|预报名|综合考核通知|复试通知|考核通知|考核安排|申请通知|报名安排)/;
const TITLE_OUTCOME_PATTERN =
    /(结果公示|录取结果|拟录取|公示|名单|递补|增补|入营通知)/;
const TITLE_PREAPPLY_PATTERN = /预报名/;
const TITLE_TRANSFER_PATTERN = /调剂/;
const TITLE_CANDIDATE_LIST_PATTERN =
    /进入综合考核考生名单|综合考核考生名单|进入综合考核名单/;
const TITLE_REVIEW_RESULT_PATTERN =
    /(复试结果|综合考核结果|结果公示|拟录取|录取结果|公示|名单)/;
const TITLE_SUMMER_CAMP_PATTERN = /夏令营/;
const TITLE_TUIMIAN_PATTERN = /推免|推荐免试/;
const TITLE_DOCTORAL_PATTERN = /博士/;
const TITLE_MASTER_PATTERN = /硕士/;
const TITLE_SYSTEM_NOTICE_PATTERN = /录取通知书|邮寄地址校对/;

export const QUERY_ASPECT_RULES = [
    {
        query: /条件|资格|要求/,
        doc: /条件|资格|要求|申请人基本条件|报考条件|身体健康/,
    },
    {
        query: /材料|提交|准备什么材料/,
        doc: /材料|提交|成绩单|证明|推荐信|纸质版/,
    },
    {
        query: /时间|日期|什么时候|何时|截止/,
        doc: /时间|日期|截止|月|日|24:00|开通|关闭/,
    },
    {
        query: /系统|报名功能|录取功能|操作/,
        doc: /系统|报名功能|录取功能|确认|注册|填报/,
    },
    {
        query: /流程|步骤|经过哪些|过程|程序|考核/,
        doc: /流程|步骤|考核|审核|报名|录取|复试/,
    },
    {
        query: /评分|打分|成绩|单科/,
        doc: /评分|成绩|总成绩|单科|综合考核成绩/,
    },
] as const;

export type QueryAspectRule = (typeof QUERY_ASPECT_RULES)[number];

export type DocumentRerankMetadata = {
    normalizedTitle: string;
    titleDedupKey: string;
    latestVersionFamilyKey: string;
    recencyKey?: number;
    evidenceText: string;
    phaseAnchor: PhaseAnchor;
    roles: QueryPlanDocRole[];
    isRuleDocTitle: boolean;
    isProcessNoticeTitle: boolean;
    isOutcomeTitle: boolean;
    titleInstitutionEntities: string[];
    isPreapplyTitle: boolean;
    isTransferTitle: boolean;
    isCandidateListTitle: boolean;
    isReviewResultTitle: boolean;
    isSummerCampTitle: boolean;
    isTuimianTitle: boolean;
    isDoctoralTitle: boolean;
    isMasterOnlyTitle: boolean;
    isSystemNoticeTitle: boolean;
    isRuleDocRole: boolean;
    isRegistrationNoticeRole: boolean;
    isResultNoticeRole: boolean;
    isListNoticeRole: boolean;
    isStageListRole: boolean;
    isAdjustmentNoticeRole: boolean;
    isConstraintRoleDoc: boolean;
    isOperationalRoleDoc: boolean;
    isOutcomeRoleDoc: boolean;
    hasCollegeTitle: boolean;
    bestKpRoleTags: string[];
    evidenceTopRoleTags: string[];
    kpEvidenceGroupCounts: Record<string, number>;
    structuredTopicIds: string[];
    structuredIntentIds: string[];
    structuredDegreeLevels: string[];
    structuredEventTypes: string[];
};

export type DocumentRerankEntry = {
    document: PipelineDocumentRecord;
    metadata: DocumentRerankMetadata;
};

export type DocumentRerankEntryLookup = {
    entry: DocumentRerankEntry;
    index: number;
};

export type LatestVersionFamilyStat = {
    count: number;
    latestRecencyKey?: number;
};

export function normalizePatternText(text: string): string {
    return text.replace(/\s+/g, "");
}

function dedupeStrings(items: string[]): string[] {
    return Array.from(new Set(items.filter(Boolean)));
}

function getDocumentEvidenceText(document: PipelineDocumentRecord): string {
    const parts: string[] = [];
    if (document.ot_title) {
        parts.push(document.ot_title);
    }
    if (Array.isArray(document.kps)) {
        const bestKp = document.kps.find((item) => item.kpid === document.best_kpid);
        if (bestKp?.kp_text) {
            parts.push(bestKp.kp_text);
        }
        document.kps
            .filter((item) => item.kpid !== document.best_kpid && item.kp_text)
            .slice(0, 4)
            .forEach((item) => {
                if (item.kp_text) {
                    parts.push(item.kp_text);
                }
            });
    }
    return normalizePatternText(parts.join(" "));
}

function getBestKpRoleTags(document: PipelineDocumentRecord): string[] {
    if (document.best_kp_role_tags?.length) {
        return dedupeStrings(document.best_kp_role_tags);
    }

    const bestKp = document.kps?.find((item) => item.kpid === document.best_kpid);
    return bestKp?.kp_role_tags?.length
        ? dedupeStrings(bestKp.kp_role_tags)
        : [];
}

function getEvidenceTopRoleTags(document: PipelineDocumentRecord): string[] {
    if (document.evidence_top_role_tags?.length) {
        return dedupeStrings(document.evidence_top_role_tags);
    }

    if (!Array.isArray(document.kps) || document.kps.length === 0) {
        return [];
    }

    const prioritizedKps = [
        ...document.kps.filter((item) => item.kpid === document.best_kpid),
        ...document.kps.filter((item) => item.kpid !== document.best_kpid),
    ].slice(0, 5);

    return dedupeStrings(
        prioritizedKps.flatMap((item) => item.kp_role_tags || []),
    );
}

function getKpEvidenceGroupCounts(
    document: PipelineDocumentRecord,
): Record<string, number> {
    if (document.kp_evidence_group_counts) {
        return { ...document.kp_evidence_group_counts };
    }

    if (!Array.isArray(document.kps) || document.kps.length === 0) {
        return {};
    }

    const prioritizedKps = [
        ...document.kps.filter((item) => item.kpid === document.best_kpid),
        ...document.kps.filter((item) => item.kpid !== document.best_kpid),
    ].slice(0, 5);

    return computeKpEvidenceGroupCounts(
        prioritizedKps.map((item) => item.kp_role_tags),
    );
}

function normalizeTitleDedupKey(title: string): string {
    return normalizePatternText(title).replace(/20\d{2}年/g, "");
}

function normalizeLatestVersionFamilyKey(title: string): string {
    return normalizeTitleDedupKey(title)
        .replace(/（[^）]*）/g, "")
        .replace(/\([^)]*\)/g, "")
        .replace(/第?[一二三四1234]批/g, "")
        .replace(/上半年|下半年/g, "")
        .replace(/第?[一二三四1234]轮/g, "");
}

function resolveDocumentRecencyKey(
    document: Pick<PipelineDocumentRecord, "publish_time" | "ot_title">,
): number | undefined {
    const rawCandidates = [document.publish_time, document.ot_title];
    for (const rawValue of rawCandidates) {
        const raw = (rawValue || "").trim();
        if (!raw) {
            continue;
        }

        const fullDateMatch = raw.match(
            /(20\d{2})[.\-/年](\d{1,2})[.\-/月](\d{1,2})/,
        );
        if (fullDateMatch) {
            const year = Number(fullDateMatch[1]);
            const month = Number(fullDateMatch[2]);
            const day = Number(fullDateMatch[3]);
            return year * 372 + month * 31 + day;
        }

        const yearMonthMatch = raw.match(/(20\d{2})[.\-/年](\d{1,2})[.\-/月]?/);
        if (yearMonthMatch) {
            const year = Number(yearMonthMatch[1]);
            const month = Number(yearMonthMatch[2]);
            return year * 372 + month * 31 + 1;
        }

        const yearMatch = raw.match(/(20\d{2})年?/);
        if (yearMatch) {
            const year = Number(yearMatch[1]);
            return year * 372 + 1;
        }
    }

    return undefined;
}

export function extractInstitutionEntities(text: string): string[] {
    const normalized = normalizePatternText(text);
    const matches = normalized.match(
        /[\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:学院|实验室|医院|研究院|研究中心|中心|系|部)/g,
    );
    return dedupeStrings(matches || []);
}

function normalizeBatchToken(token: string): string | undefined {
    switch (token) {
        case "1":
        case "一":
            return "1";
        case "2":
        case "二":
            return "2";
        case "3":
        case "三":
            return "3";
        case "4":
        case "四":
            return "4";
        default:
            return undefined;
    }
}

export function extractPhaseAnchor(text: string): PhaseAnchor {
    const normalized = text.replace(/\s+/g, "");
    const halfMatch = normalized.match(/上半年|下半年/);
    const batchMatch = normalized.match(/第?\s*([一二三四1234])\s*批/);

    return {
        half:
            halfMatch?.[0] === "上半年" || halfMatch?.[0] === "下半年"
                ? halfMatch[0]
                : undefined,
        batch: batchMatch?.[1]
            ? normalizeBatchToken(batchMatch[1])
            : undefined,
        stages: PHASE_STAGE_RULES.filter((rule) => rule.pattern.test(normalized)).map(
            (rule) => rule.stage,
        ),
    };
}

export function hasExplicitPhaseAnchor(anchor: PhaseAnchor): boolean {
    return Boolean(anchor.half || anchor.batch || anchor.stages.length > 0);
}

export function buildDocumentRerankMetadata(
    document: PipelineDocumentRecord,
): DocumentRerankMetadata {
    const title = document.ot_title || "";
    const normalizedTitle = normalizePatternText(title);
    const roles = inferDocumentRolesFromTitle(title);
    const titleInstitutionEntities = extractInstitutionEntities(title);
    const isDoctoralTitle = TITLE_DOCTORAL_PATTERN.test(normalizedTitle);
    const isTuimianTitle = TITLE_TUIMIAN_PATTERN.test(normalizedTitle);
    const isRuleDocRole = roles.includes("rule_doc");
    const isRegistrationNoticeRole = roles.includes("registration_notice");
    const isResultNoticeRole = roles.includes("result_notice");
    const isListNoticeRole = roles.includes("list_notice");
    const isStageListRole = roles.includes("stage_list");
    const isAdjustmentNoticeRole = roles.includes("adjustment_notice");

    return {
        normalizedTitle,
        titleDedupKey: normalizeTitleDedupKey(title),
        latestVersionFamilyKey: normalizeLatestVersionFamilyKey(title),
        recencyKey: resolveDocumentRecencyKey(document),
        evidenceText: getDocumentEvidenceText(document),
        phaseAnchor: extractPhaseAnchor(title),
        roles,
        isRuleDocTitle: TITLE_RULE_DOC_PATTERN.test(normalizedTitle),
        isProcessNoticeTitle: TITLE_PROCESS_NOTICE_PATTERN.test(normalizedTitle),
        isOutcomeTitle: TITLE_OUTCOME_PATTERN.test(normalizedTitle),
        titleInstitutionEntities,
        isPreapplyTitle: TITLE_PREAPPLY_PATTERN.test(normalizedTitle),
        isTransferTitle: TITLE_TRANSFER_PATTERN.test(normalizedTitle),
        isCandidateListTitle: TITLE_CANDIDATE_LIST_PATTERN.test(normalizedTitle),
        isReviewResultTitle: TITLE_REVIEW_RESULT_PATTERN.test(normalizedTitle),
        isSummerCampTitle: TITLE_SUMMER_CAMP_PATTERN.test(normalizedTitle),
        isTuimianTitle,
        isDoctoralTitle,
        isMasterOnlyTitle:
            TITLE_MASTER_PATTERN.test(normalizedTitle) &&
            !isDoctoralTitle &&
            !isTuimianTitle,
        isSystemNoticeTitle: TITLE_SYSTEM_NOTICE_PATTERN.test(normalizedTitle),
        isRuleDocRole,
        isRegistrationNoticeRole,
        isResultNoticeRole,
        isListNoticeRole,
        isStageListRole,
        isAdjustmentNoticeRole,
        isConstraintRoleDoc: isRuleDocRole || isRegistrationNoticeRole,
        isOperationalRoleDoc:
            isRegistrationNoticeRole || isStageListRole || isAdjustmentNoticeRole,
        isOutcomeRoleDoc: isResultNoticeRole || isListNoticeRole,
        hasCollegeTitle: titleInstitutionEntities.length > 0,
        bestKpRoleTags: getBestKpRoleTags(document),
        evidenceTopRoleTags: getEvidenceTopRoleTags(document),
        kpEvidenceGroupCounts: getKpEvidenceGroupCounts(document),
        structuredTopicIds: document.topic_ids || [],
        structuredIntentIds: document.intent_ids || [],
        structuredDegreeLevels: document.degree_levels || [],
        structuredEventTypes: document.event_types || [],
    };
}

export function buildDocumentRerankEntries(
    documents: PipelineDocumentRecord[],
): DocumentRerankEntry[] {
    return documents.map((document) => ({
        document,
        metadata: buildDocumentRerankMetadata(document),
    }));
}

export function queryWantsLatestVersion(query: string): boolean {
    return /现在|最新|最近|最近一次|目前|当前/.test(query);
}

export function getDocumentDisplayScore(document: PipelineDocumentRecord): number {
    return document.displayScore ?? document.coarseScore ?? document.score ?? 0;
}

export function getDocumentCoarseScore(document: PipelineDocumentRecord): number {
    return document.coarseScore ?? document.score ?? getDocumentDisplayScore(document);
}

export function sortDocumentRerankEntriesByDisplayScore(
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    return [...entries].sort(
        (left, right) =>
            getDocumentDisplayScore(right.document) -
            getDocumentDisplayScore(left.document),
    );
}

function updateDocumentScores(
    document: PipelineDocumentRecord,
    displayDelta: number,
    coarseDelta?: number,
): PipelineDocumentRecord {
    const nextDisplayScore = getDocumentDisplayScore(document) + displayDelta;
    return {
        ...document,
        score: nextDisplayScore,
        coarseScore:
            coarseDelta === undefined
                ? document.coarseScore
                : getDocumentCoarseScore(document) + coarseDelta,
        displayScore: nextDisplayScore,
    };
}

export function updateDocumentRerankEntryScores(
    entry: DocumentRerankEntry,
    displayDelta: number,
    coarseDelta?: number,
): DocumentRerankEntry {
    return {
        document: updateDocumentScores(entry.document, displayDelta, coarseDelta),
        metadata: entry.metadata,
    };
}

export function getDocumentsFromRerankEntries(
    entries: DocumentRerankEntry[],
): PipelineDocumentRecord[] {
    return entries.map((entry) => entry.document);
}

export function buildDocumentRerankEntryLookup(
    entries: DocumentRerankEntry[],
): Map<string, DocumentRerankEntryLookup> {
    const lookup = new Map<string, DocumentRerankEntryLookup>();
    entries.forEach((entry, index) => {
        const otid = entry.document.otid;
        if (!otid || lookup.has(otid)) {
            return;
        }
        lookup.set(otid, { entry, index });
    });
    return lookup;
}

export function buildLatestVersionFamilyStats(
    entries: DocumentRerankEntry[],
): Map<string, LatestVersionFamilyStat> {
    const familyStats = new Map<string, LatestVersionFamilyStat>();

    entries.forEach((entry) => {
        const { latestVersionFamilyKey, recencyKey } = entry.metadata;
        const familyKey = latestVersionFamilyKey;
        if (!familyKey) {
            return;
        }

        const existing = familyStats.get(familyKey) || { count: 0 };
        existing.count += 1;
        if (
            recencyKey !== undefined &&
            (existing.latestRecencyKey === undefined ||
                recencyKey > existing.latestRecencyKey)
        ) {
            existing.latestRecencyKey = recencyKey;
        }
        familyStats.set(familyKey, existing);
    });

    return familyStats;
}

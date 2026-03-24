import {
    CAMPUS_SYNONYMS as CONFIG_CAMPUS_SYNONYMS,
    DEGREE_LEVEL_TABLE as CONFIG_DEGREE_LEVEL_TABLE,
    EVENT_TYPE_HINTS as CONFIG_EVENT_TYPE_HINTS,
    HISTORICAL_QUERY_HINTS as CONFIG_HISTORICAL_QUERY_HINTS,
    INTENT_VECTOR_TABLE as CONFIG_INTENT_VECTOR_TABLE,
    POLICY_LATEST_HINTS as CONFIG_POLICY_LATEST_HINTS,
    SUBTOPIC_TOPIC_MAP as CONFIG_SUBTOPIC_TOPIC_MAP,
    TOPIC_CONFIGS as CONFIG_TOPIC_CONFIGS,
} from "./search_topic_config";

export interface Metadata {
    id: string;
    type: "Q" | "KP" | "OT";
    parent_otid: string;
    timestamp?: number;
    vector_index: number;
    scale?: number;
    sparse?: number[];
    target_year?: number;
    topic_ids?: string[];
    subtopic_ids?: string[];
    primary_topic_ids?: string[];
    secondary_topic_ids?: string[];
    weak_topic_ids?: string[];
    primary_subtopic_ids?: string[];
    secondary_subtopic_ids?: string[];
    weak_subtopic_ids?: string[];
    intent_ids?: string[];
    degree_levels?: string[];
    event_types?: string[];
}

export interface SearchResult {
    otid: string;
    best_kpid?: string;
    score: number;
    details?: {
        denseRRF: number;
        sparseRRF: number;
        lexicalBoost: number;
    };
}

export interface SearchRejection {
    reason: "low_topic_coverage" | "low_consistency";
    topicIds: string[];
}

export interface SearchRankOutput {
    matches: SearchResult[];
    weakMatches: SearchResult[];
    rejection?: SearchRejection;
}

export interface BM25Stats {
    idfMap: Map<number, number>;
    docLengths: Int32Array;
    avgdl: number;
}

export interface ParsedQueryIntent {
    rawQuery: string;
    years: number[];
    topicIds: string[];
    subtopicIds: string[];
    intentIds: string[];
    degreeLevels: string[];
    eventTypes: string[];
    normalizedTerms: string[];
    confidence: number;
    preferLatest: boolean;
}

export interface IntentVectorItem {
    intent_id: string;
    intent_name: string;
    aliases: readonly string[];
    negative_intents: readonly string[];
    related_intents: readonly string[];
    degree_levels: readonly string[];
    preferred_event_types: readonly string[];
    weight: number;
}

export const DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33,
};

export const DECAY_LAMBDA = 0.001;
export const SECONDS_IN_DAY = 86400;
export const RRF_K = 60;

const BM25_K1 = 1.2;
const BM25_B = 0.4;
const LATEST_YEAR_BOOST_BASE = 0.82;
const LATEST_TIMESTAMP_DECAY = 0.0012;

export const DEGREE_LEVEL_TABLE = CONFIG_DEGREE_LEVEL_TABLE;
export const EVENT_TYPE_TABLE = Object.keys(CONFIG_EVENT_TYPE_HINTS);

export const CAMPUS_SYNONYMS = CONFIG_CAMPUS_SYNONYMS;
export const EVENT_TYPE_HINTS = CONFIG_EVENT_TYPE_HINTS;
export const POLICY_LATEST_HINTS = CONFIG_POLICY_LATEST_HINTS;
export const HISTORICAL_QUERY_HINTS = CONFIG_HISTORICAL_QUERY_HINTS;
export const INTENT_VECTOR_TABLE: readonly IntentVectorItem[] =
    CONFIG_INTENT_VECTOR_TABLE as unknown as readonly IntentVectorItem[];

const LEGACY_EVENT_TYPE_HINTS: Record<string, string[]> = {
    招生章程: ["章程", "简章", "专业目录"],
    报名通知: ["报名", "网报", "报考点", "报名通知"],
    考试安排: ["考试", "考核安排", "面试", "笔试"],
    复试通知: ["复试"],
    录取公示: ["录取", "预录取", "公示", "名单"],
    资格要求: ["资格", "条件", "要求"],
    材料提交: ["材料", "提交", "寄送"],
    推免资格公示: [
        "推荐资格公示",
        "推免资格公示",
        "免试攻读研究生推荐资格公示",
    ],
    推免通知: ["推免通知", "免试攻读研究生", "推荐免试"],
    推免实施办法: ["推免实施办法", "推荐办法", "遴选细则", "实施办法"],
    非招生通知: ["奖学金", "助学金", "评选", "资助", "学年", "工作通知"],
};

const LEGACY_POLICY_LATEST_HINTS = [
    "保研",
    "推免",
    "推荐免试",
    "考研",
    "研招",
    "奖学金",
    "助学金",
    "资助",
    "补助",
    "评奖",
    "招生",
    "报名",
    "报考点",
    "复试",
    "录取",
    "选课",
    "转专业",
] as const;

const LEGACY_HISTORICAL_QUERY_HINTS = [
    "往年",
    "历年",
    "历史",
    "以前",
    "去年",
    "前年",
    "往届",
    "旧版",
] as const;

const LEGACY_INTENT_VECTOR_TABLE: IntentVectorItem[] = [
    {
        intent_id: "ug_recommend_admission",
        intent_name: "本科保送生",
        aliases: ["保送生", "外语类保送生", "竞赛保送"],
        negative_intents: ["master_recommend_exemption", "master_unified_exam"],
        related_intents: [],
        degree_levels: ["本科"],
        preferred_event_types: ["报名通知", "考试安排", "录取公示"],
        weight: 1,
    },
    {
        intent_id: "master_recommend_exemption",
        intent_name: "推免",
        aliases: [
            "保研",
            "推免",
            "推荐免试",
            "推免生",
            "免试研究生",
            "免试攻读研究生",
            "推荐资格",
            "预推免",
        ],
        negative_intents: ["ug_recommend_admission", "master_unified_exam"],
        related_intents: ["summer_camp", "pre_recommend"],
        degree_levels: ["硕士", "博士"],
        preferred_event_types: [
            "推免资格公示",
            "推免通知",
            "推免实施办法",
            "材料提交",
            "录取公示",
        ],
        weight: 1,
    },
    {
        intent_id: "master_unified_exam",
        intent_name: "统考硕士",
        aliases: [
            "考研",
            "硕士研究生招生考试",
            "全国硕士研究生招生考试",
            "硕士统考",
            "初试",
            "复试",
            "研招",
        ],
        negative_intents: [
            "ug_recommend_admission",
            "master_recommend_exemption",
        ],
        related_intents: ["master_adjustment"],
        degree_levels: ["硕士"],
        preferred_event_types: [
            "招生章程",
            "报名通知",
            "考试安排",
            "复试通知",
            "录取公示",
        ],
        weight: 1,
    },
    {
        intent_id: "master_adjustment",
        intent_name: "调剂",
        aliases: ["调剂", "硕士调剂", "接受调剂", "调剂复试"],
        negative_intents: [],
        related_intents: ["master_unified_exam"],
        degree_levels: ["硕士", "博士"],
        preferred_event_types: ["复试通知", "录取公示", "报名通知"],
        weight: 1,
    },
    {
        intent_id: "phd_apply_assessment",
        intent_name: "博士申请考核",
        aliases: ["申请考核", "申请-考核", "博士申请考核"],
        negative_intents: ["phd_general_exam"],
        related_intents: [],
        degree_levels: ["博士"],
        preferred_event_types: ["报名通知", "材料提交", "考试安排", "录取公示"],
        weight: 1,
    },
    {
        intent_id: "phd_general_exam",
        intent_name: "博士普通招考",
        aliases: [
            "博士招考",
            "博士报名",
            "博士考试",
            "公开招考博士",
            "博士研究生招生",
        ],
        negative_intents: ["phd_apply_assessment"],
        related_intents: [],
        degree_levels: ["博士"],
        preferred_event_types: ["招生章程", "报名通知", "考试安排", "录取公示"],
        weight: 1,
    },
    {
        intent_id: "summer_camp",
        intent_name: "夏令营",
        aliases: ["夏令营", "优秀大学生夏令营"],
        negative_intents: [],
        related_intents: ["master_recommend_exemption"],
        degree_levels: ["硕士"],
        preferred_event_types: ["报名通知", "录取公示"],
        weight: 1,
    },
    {
        intent_id: "pre_recommend",
        intent_name: "预推免",
        aliases: ["预推免", "预推免报名", "预推免考核"],
        negative_intents: ["master_unified_exam"],
        related_intents: ["master_recommend_exemption", "summer_camp"],
        degree_levels: ["硕士"],
        preferred_event_types: ["推免通知", "考试安排", "录取公示"],
        weight: 1,
    },
];

void LEGACY_EVENT_TYPE_HINTS;
void LEGACY_POLICY_LATEST_HINTS;
void LEGACY_HISTORICAL_QUERY_HINTS;
void LEGACY_INTENT_VECTOR_TABLE;

const INTENT_CONFLICTS: Record<string, readonly string[]> = Object.fromEntries(
    INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item.negative_intents]),
);

const INTENT_RULE_MAP: Map<string, IntentVectorItem> = new Map(
    INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item]),
);

const TOPIC_RULE_MAP: Map<string, (typeof CONFIG_TOPIC_CONFIGS)[number]> =
    new Map(CONFIG_TOPIC_CONFIGS.map((item) => [item.topic_id, item] as const));

function deriveTopicIdsFromSubtopics(subtopicIds: string[]): string[] {
    return dedupe(
        subtopicIds
            .map((subtopicId) => CONFIG_SUBTOPIC_TOPIC_MAP[subtopicId])
            .filter((topicId): topicId is string => Boolean(topicId)),
    );
}

export function resolveMetadataTopicIds(
    meta: Pick<Metadata, "topic_ids" | "subtopic_ids" | "intent_ids">,
): string[] {
    if (meta.topic_ids && meta.topic_ids.length > 0) {
        return meta.topic_ids;
    }
    return deriveTopicIdsFromSubtopics(meta.subtopic_ids || meta.intent_ids || []);
}

function matchTopicIds(query: string): string[] {
    return dedupe(
        CONFIG_TOPIC_CONFIGS.filter((topic) =>
            topic.aliases.some((alias) => query.includes(alias)),
        ).map((topic) => topic.topic_id),
    );
}

function getCoverageRejectedTopicIds(topicIds: string[]): string[] {
    return topicIds.filter(
        (topicId) => TOPIC_RULE_MAP.get(topicId)?.reject_when_coverage_low,
    );
}

function dedupe<T>(items: T[]): T[] {
    return Array.from(new Set(items));
}

export function buildBM25Stats(metadata: Metadata[]): BM25Stats {
    const N = metadata.length;
    const dfMap = new Map<number, number>();
    const docLengths = new Int32Array(N);
    let totalLength = 0;

    for (let i = 0; i < N; i++) {
        const sparse = metadata[i].sparse;
        if (!sparse || sparse.length === 0) {
            docLengths[i] = 0;
            continue;
        }

        let dl = 0;
        for (let j = 0; j < sparse.length; j += 2) {
            const wordId = sparse[j];
            const tf = sparse[j + 1];
            dl += tf;
            dfMap.set(wordId, (dfMap.get(wordId) || 0) + 1);
        }
        docLengths[i] = dl;
        totalLength += dl;
    }

    const avgdl = totalLength / (N || 1);
    const idfMap = new Map<number, number>();

    for (const [wordId, df] of dfMap.entries()) {
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        idfMap.set(wordId, Math.max(idf, 0.01));
    }

    return { idfMap, docLengths, avgdl };
}

export function dotProduct(
    vecA: Float32Array,
    matrix: Int8Array | Float32Array,
    matrixIndex: number,
    dimensions: number,
): number {
    let sum = 0;
    const offset = matrixIndex * dimensions;
    for (let i = 0; i < dimensions; i++) {
        sum += vecA[i] * matrix[offset + i];
    }
    return sum;
}

export function getQuerySparse(
    words: string[],
    vocabMap: Map<string, number> | Record<string, number>,
): Record<number, number> {
    const sparse: Record<number, number> = {};
    const isMap = vocabMap instanceof Map;

    words.forEach((word) => {
        const index = isMap
            ? (vocabMap as Map<string, number>).get(word)
            : (vocabMap as Record<string, number>)[word];
        if (index !== undefined) {
            sparse[index] = (sparse[index] || 0) + 1;
        }

        const synonyms = CONFIG_CAMPUS_SYNONYMS[word];
        if (synonyms) {
            synonyms.forEach((syn) => {
                const sIndex = isMap
                    ? (vocabMap as Map<string, number>).get(syn)
                    : (vocabMap as Record<string, number>)[syn];
                if (sIndex !== undefined) {
                    sparse[sIndex] = (sparse[sIndex] || 0) + 1;
                }
            });
        }
    });

    return sparse;
}

function collectMatchedEventTypesFromConfig(text: string): string[] {
    return dedupe(
        Object.entries(CONFIG_EVENT_TYPE_HINTS)
            .filter(([, hints]) => hints.some((hint) => text.includes(hint)))
            .map(([eventType]) => eventType),
    );
}

export function parseQueryIntent(query: string): ParsedQueryIntent {
    const years = dedupe(
        (query.match(/20\d{2}/g) || []).map((year) => Number(year)),
    );
    const matchedRules = CONFIG_INTENT_VECTOR_TABLE.filter((rule) =>
        rule.aliases.some((alias) => query.includes(alias)),
    );
    const subtopicIds = matchedRules.map((rule) => rule.intent_id);
    const matchedTopicIds = matchTopicIds(query);
    const topicIds = dedupe([
        ...deriveTopicIdsFromSubtopics(subtopicIds),
        ...matchedTopicIds,
    ]);
    const hasExplicitPhdHint = query.includes("博士") || query.includes("直博");
    const explicitDegreeLevels = dedupe(
        CONFIG_DEGREE_LEVEL_TABLE.filter((level) => query.includes(level)),
    );
    const inferredDegreeLevels = dedupe(
        matchedRules.flatMap((rule) => {
            if (rule.intent_id === "master_recommend_exemption") {
                return hasExplicitPhdHint ? ["硕士", "博士"] : ["硕士"];
            }
            return rule.degree_levels;
        }),
    );
    const degreeLevels =
        explicitDegreeLevels.length > 0
            ? explicitDegreeLevels
            : inferredDegreeLevels;
    const eventTypes = dedupe([
        ...collectMatchedEventTypesFromConfig(query),
        ...matchedRules.flatMap((rule) => rule.preferred_event_types),
    ]);
    const normalizedTerms = dedupe([
        ...matchedRules.flatMap((rule) => rule.aliases),
        ...degreeLevels,
        ...matchedRules.flatMap((rule) => rule.preferred_event_types),
    ]);
    const preferLatest =
        years.length === 0 &&
        !CONFIG_HISTORICAL_QUERY_HINTS.some((hint) => query.includes(hint)) &&
        (matchedRules.length > 0 ||
            topicIds.some(
                (topicId) =>
                    CONFIG_TOPIC_CONFIGS.find((topic) => topic.topic_id === topicId)
                        ?.prefer_latest,
            ) ||
            CONFIG_POLICY_LATEST_HINTS.some((hint) => query.includes(hint)));

    return {
        rawQuery: query,
        years,
        topicIds,
        subtopicIds,
        intentIds: subtopicIds,
        degreeLevels,
        eventTypes,
        normalizedTerms,
        confidence: matchedRules.length > 0 ? 1 : 0,
        preferLatest,
    };
}

function hasIntentConflict(
    queryIntentIds: string[],
    docIntentIds?: string[],
): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) =>
        (INTENT_CONFLICTS[queryIntentId] || []).some((conflictId) =>
            docIntentIds.includes(conflictId),
        ),
    );
}

function hasIntentMatch(
    queryIntentIds: string[],
    docIntentIds?: string[],
): boolean {
    if (!docIntentIds || docIntentIds.length === 0) return false;
    return queryIntentIds.some((queryIntentId) =>
        docIntentIds.includes(queryIntentId),
    );
}

function hasAnyOverlap(a: string[], b?: string[]): boolean {
    if (!b || b.length === 0) return false;
    return a.some((item) => b.includes(item));
}

function classifyTopicCoverage(
    queryTopicIds: string[],
    doc: {
        topic_ids?: string[];
        primary_topic_ids?: string[];
        secondary_topic_ids?: string[];
        weak_topic_ids?: string[];
    },
): "primary" | "secondary" | "weak" | "none" {
    if (hasAnyOverlap(queryTopicIds, doc.primary_topic_ids)) {
        return "primary";
    }
    if (hasAnyOverlap(queryTopicIds, doc.secondary_topic_ids)) {
        return "secondary";
    }
    if (hasAnyOverlap(queryTopicIds, doc.weak_topic_ids)) {
        return "weak";
    }
    if (
        hasAnyOverlap(queryTopicIds, doc.topic_ids) &&
        (!doc.primary_topic_ids || doc.primary_topic_ids.length === 0) &&
        (!doc.secondary_topic_ids || doc.secondary_topic_ids.length === 0) &&
        (!doc.weak_topic_ids || doc.weak_topic_ids.length === 0)
    ) {
        return "secondary";
    }
    return "none";
}

function getCoverageComparableTopicIds(doc: {
    primary_topic_ids?: string[];
    secondary_topic_ids?: string[];
    topic_ids?: string[];
}): string[] {
    if (doc.primary_topic_ids && doc.primary_topic_ids.length > 0) {
        return doc.primary_topic_ids;
    }
    if (doc.secondary_topic_ids && doc.secondary_topic_ids.length > 0) {
        return doc.secondary_topic_ids;
    }
    return doc.topic_ids || [];
}

function getPreferredEventTypes(intentIds: string[]): string[] {
    return dedupe(
        intentIds.flatMap(
            (intentId) =>
                INTENT_RULE_MAP.get(intentId)?.preferred_event_types || [],
        ),
    );
}

function getRelatedIntentTypes(intentIds: string[]): string[] {
    return dedupe(
        intentIds.flatMap(
            (intentId) => INTENT_RULE_MAP.get(intentId)?.related_intents || [],
        ),
    );
}

export function searchAndRank(params: {
    queryVector: Float32Array;
    querySparse?: Record<number, number>;
    queryYearWordIds?: number[];
    queryIntent?: ParsedQueryIntent;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    weights?: typeof DEFAULT_WEIGHTS;
    candidateIndices?: readonly number[];
}): SearchRankOutput {
    const {
        queryVector,
        querySparse,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp,
        bm25Stats,
        weights = DEFAULT_WEIGHTS,
        queryYearWordIds,
        queryIntent,
        candidateIndices,
    } = params;

    const n = metadata.length;
    const activeCandidateIndices =
        candidateIndices && candidateIndices.length > 0 ? candidateIndices : undefined;
    const candidateCount = activeCandidateIndices
        ? activeCandidateIndices.length
        : n;
    const denseScores = new Float32Array(candidateCount);
    const sparseScores = new Float32Array(candidateCount);
    const denseOrder = new Int32Array(candidateCount);
    const sparseOrder = new Int32Array(candidateCount);
    const lexicalBonusMap = new Map<string, number>();
    const yearHitMap = new Map<string, boolean>();

    for (let localIndex = 0; localIndex < candidateCount; localIndex++) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[localIndex]
            : localIndex;
        const meta = metadata[metaIndex];

        let dense = dotProduct(
            queryVector,
            vectorMatrix,
            meta.vector_index,
            dimensions,
        );
        if (meta.scale !== undefined && meta.scale !== null)
            dense *= meta.scale;
        denseScores[localIndex] = dense;
        denseOrder[localIndex] = localIndex;

        let sparse = 0;
        if (querySparse && meta.sparse && meta.sparse.length > 0) {
            const dl = bm25Stats.docLengths[metaIndex];
            const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);

            for (let j = 0; j < meta.sparse.length; j += 2) {
                const wordId = meta.sparse[j];
                const tf = meta.sparse[j + 1];

                if (queryYearWordIds && queryYearWordIds.includes(wordId)) {
                    const otid =
                        meta.type === "OT" ? meta.id : meta.parent_otid;
                    yearHitMap.set(otid, true);
                }

                if (querySparse[wordId]) {
                    const qWeight = querySparse[wordId] || 1;
                    const idf = bm25Stats.idfMap.get(wordId) || 0;
                    const numerator = tf * (BM25_K1 + 1);
                    const denominator =
                        tf +
                        BM25_K1 *
                            (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
                    sparse += qWeight * idf * (numerator / denominator);
                }
            }

            if (sparse > 0) {
                const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
                let currentBonus = lexicalBonusMap.get(otid) || 0;
                if (meta.type === "Q") currentBonus += sparse * 1.5;
                else if (meta.type === "KP") currentBonus += sparse * 1.2;
                else currentBonus += sparse;
                lexicalBonusMap.set(otid, currentBonus);
            }
        }
        sparseScores[localIndex] = sparse;
        sparseOrder[localIndex] = localIndex;
    }

    denseOrder.sort((a, b) => denseScores[b] - denseScores[a]);
    const rrfScores = new Map<Metadata, number>();

    for (let rank = 0; rank < Math.min(4000, candidateCount); rank++) {
        const metaIndex = activeCandidateIndices
            ? activeCandidateIndices[denseOrder[rank]]
            : denseOrder[rank];
        const meta = metadata[metaIndex];
        rrfScores.set(meta, (1 / (rank + RRF_K)) * 100);
    }

    if (querySparse) {
        sparseOrder.sort((a, b) => sparseScores[b] - sparseScores[a]);
        for (let rank = 0; rank < Math.min(4000, candidateCount); rank++) {
            const localIndex = sparseOrder[rank];
            if (sparseScores[localIndex] === 0) break;

            const metaIndex = activeCandidateIndices
                ? activeCandidateIndices[localIndex]
                : localIndex;
            const meta = metadata[metaIndex];
            const current = rrfScores.get(meta) || 0;
            rrfScores.set(meta, current + (1.2 / (rank + RRF_K)) * 100);
        }
    }

    const topHybrid = Array.from(rrfScores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 1000);

    const otidMap: Record<
        string,
        {
            max_q: number;
            max_kp: number;
            ot_score: number;
            timestamp?: number;
            best_kpid?: string;
            target_year?: number;
            topic_ids?: string[];
            subtopic_ids?: string[];
            primary_topic_ids?: string[];
            secondary_topic_ids?: string[];
            weak_topic_ids?: string[];
            intent_ids?: string[];
            degree_levels?: string[];
            event_types?: string[];
        }
    > = {};

    for (const [meta, score] of topHybrid) {
        const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
        if (!otidMap[otid]) {
            otidMap[otid] = {
                max_q: 0,
                max_kp: 0,
                ot_score: 0,
                timestamp: meta.timestamp,
                target_year: meta.target_year,
                topic_ids: resolveMetadataTopicIds(meta),
                primary_topic_ids: meta.primary_topic_ids,
                secondary_topic_ids: meta.secondary_topic_ids,
                weak_topic_ids: meta.weak_topic_ids,
                subtopic_ids: meta.subtopic_ids || meta.intent_ids,
                intent_ids: meta.intent_ids,
                degree_levels: meta.degree_levels,
                event_types: meta.event_types,
            };
        }

        if (
            otidMap[otid].target_year === undefined &&
            meta.target_year !== undefined
        ) {
            otidMap[otid].target_year = meta.target_year;
        }
        if (
            (!otidMap[otid].topic_ids || otidMap[otid].topic_ids!.length === 0) &&
            (meta.topic_ids?.length || meta.subtopic_ids?.length || meta.intent_ids?.length)
        ) {
            otidMap[otid].topic_ids = resolveMetadataTopicIds(meta);
        }
        if (
            (!otidMap[otid].subtopic_ids ||
                otidMap[otid].subtopic_ids!.length === 0) &&
            (meta.subtopic_ids?.length || meta.intent_ids?.length)
        ) {
            otidMap[otid].subtopic_ids = meta.subtopic_ids || meta.intent_ids;
        }
        if (
            (!otidMap[otid].primary_topic_ids ||
                otidMap[otid].primary_topic_ids!.length === 0) &&
            meta.primary_topic_ids?.length
        ) {
            otidMap[otid].primary_topic_ids = meta.primary_topic_ids;
        }
        if (
            (!otidMap[otid].secondary_topic_ids ||
                otidMap[otid].secondary_topic_ids!.length === 0) &&
            meta.secondary_topic_ids?.length
        ) {
            otidMap[otid].secondary_topic_ids = meta.secondary_topic_ids;
        }
        if (
            (!otidMap[otid].weak_topic_ids ||
                otidMap[otid].weak_topic_ids!.length === 0) &&
            meta.weak_topic_ids?.length
        ) {
            otidMap[otid].weak_topic_ids = meta.weak_topic_ids;
        }
        if (
            (!otidMap[otid].intent_ids ||
                otidMap[otid].intent_ids!.length === 0) &&
            meta.intent_ids?.length
        ) {
            otidMap[otid].intent_ids = meta.intent_ids;
        }
        if (
            (!otidMap[otid].degree_levels ||
                otidMap[otid].degree_levels!.length === 0) &&
            meta.degree_levels?.length
        ) {
            otidMap[otid].degree_levels = meta.degree_levels;
        }
        if (
            (!otidMap[otid].event_types ||
                otidMap[otid].event_types!.length === 0) &&
            meta.event_types?.length
        ) {
            otidMap[otid].event_types = meta.event_types;
        }

        if (meta.type === "Q") {
            otidMap[otid].max_q = Math.max(otidMap[otid].max_q, score);
        } else if (meta.type === "KP") {
            if (score > otidMap[otid].max_kp) {
                otidMap[otid].max_kp = score;
                otidMap[otid].best_kpid = meta.id;
            }
        } else if (meta.type === "OT") {
            otidMap[otid].ot_score = Math.max(otidMap[otid].ot_score, score);
        }
    }

    const finalRanking: SearchResult[] = [];
    const candidateTargetYears = Object.values(otidMap)
        .map((scores) => scores.target_year)
        .filter((year): year is number => typeof year === "number");
    const latestTargetYear =
        candidateTargetYears.length > 0
            ? Math.max(...candidateTargetYears)
            : undefined;
    const coverageRejectedTopicIds = queryIntent
        ? getCoverageRejectedTopicIds(queryIntent.topicIds)
        : [];
    const coveredRejectedTopicEntries =
        coverageRejectedTopicIds.length === 0
            ? []
            : Object.entries(otidMap).filter(([, scores]) => {
                  const coverage = classifyTopicCoverage(
                      coverageRejectedTopicIds,
                      scores,
                  );
                  return coverage === "primary" || coverage === "secondary";
              });
    const hasCoveredRejectedTopic =
        coverageRejectedTopicIds.length === 0 ||
        coveredRejectedTopicEntries.length > 0;

    for (const [otid, scores] of Object.entries(otidMap)) {
        const explicitYears = queryIntent?.years || [];
        const hasExplicitYear = explicitYears.length > 0;
        const hasStructuredYearMatch =
            hasExplicitYear &&
            scores.target_year !== undefined &&
            explicitYears.includes(scores.target_year);
        const hasLexicalYearMatch = yearHitMap.get(otid) === true;
        const isHighConfidenceSingleIntent =
            !!queryIntent &&
            queryIntent.confidence >= 1 &&
            queryIntent.intentIds.length === 1;
        const primaryIntentId = isHighConfidenceSingleIntent
            ? queryIntent!.intentIds[0]
            : undefined;
        const allowedIntentIds = primaryIntentId
            ? dedupe([
                  primaryIntentId,
                  ...getRelatedIntentTypes([primaryIntentId]),
              ])
            : [];
        const preferredEventTypes = queryIntent
            ? getPreferredEventTypes(queryIntent.intentIds)
            : [];

        if (hasExplicitYear) {
            if (scores.target_year !== undefined && !hasStructuredYearMatch) {
                continue;
            }
            if (scores.target_year === undefined && !hasLexicalYearMatch) {
                continue;
            }
        }

        if (isHighConfidenceSingleIntent && primaryIntentId) {
            const docIntentIds = scores.intent_ids || [];
            const docHasIntentLabels = docIntentIds.length > 0;
            const docMatchesAllowedIntent = allowedIntentIds.some((intentId) =>
                docIntentIds.includes(intentId),
            );
            const docHasPreferredEventType =
                preferredEventTypes.length > 0 &&
                hasAnyOverlap(preferredEventTypes, scores.event_types);

            if (docHasIntentLabels && !docMatchesAllowedIntent) {
                continue;
            }

            if (!docHasIntentLabels && !docHasPreferredEventType) {
                continue;
            }
        }

        const weightedQ = scores.max_q * weights.Q;
        const weightedKP = scores.max_kp * weights.KP;
        const weightedOT = scores.ot_score * weights.OT;

        const maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
        const unionBonus =
            weightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;

        let finalScore = maxComponent + unionBonus;

        if (
            queryIntent?.preferLatest &&
            scores.timestamp &&
            currentTimestamp > 0
        ) {
            const daysDiff =
                (currentTimestamp - scores.timestamp) / SECONDS_IN_DAY;
            if (daysDiff > 0)
                finalScore *= Math.exp(-LATEST_TIMESTAMP_DECAY * daysDiff);
        }

        let boost = 1.0;
        const lexicalBonus = lexicalBonusMap.get(otid) || 0;
        if (lexicalBonus > 0) {
            boost *= 1 + Math.log1p(lexicalBonus) / 4;
        }

        if (queryYearWordIds && queryYearWordIds.length > 0) {
            if (scores.target_year !== undefined && queryIntent?.years.length) {
                if (!queryIntent.years.includes(scores.target_year)) {
                    boost *= 0.01;
                }
            } else if (!hasStructuredYearMatch && !hasLexicalYearMatch) {
                boost *= 0.12;
            }
        }

        if (queryIntent && queryIntent.intentIds.length > 0) {
            if (hasIntentMatch(queryIntent.intentIds, scores.intent_ids)) {
                boost *= 1.25;
            } else if (
                hasIntentConflict(queryIntent.intentIds, scores.intent_ids)
            ) {
                boost *= 0.18;
            }

            if (
                queryIntent.intentIds.includes("master_recommend_exemption") &&
                scores.intent_ids?.includes("ug_recommend_admission")
            ) {
                boost *= 0.05;
            }

            if (
                isHighConfidenceSingleIntent &&
                queryIntent.intentIds[0] === "master_recommend_exemption"
            ) {
                if (scores.intent_ids?.includes("master_unified_exam")) {
                    boost *= 0.06;
                }
                if (scores.intent_ids?.includes("master_adjustment")) {
                    boost *= 0.08;
                }
                if (
                    !queryIntent.degreeLevels.includes("博士") &&
                    (scores.intent_ids?.includes("phd_apply_assessment") ||
                        scores.intent_ids?.includes("phd_general_exam"))
                ) {
                    boost *= 0.04;
                }
            }

            if (scores.event_types?.includes("非招生通知")) {
                boost *= 0.12;
            } else if (
                preferredEventTypes.length > 0 &&
                hasAnyOverlap(preferredEventTypes, scores.event_types)
            ) {
                boost *= 1.24;
            }
        }

        if (queryIntent && queryIntent.degreeLevels.length > 0) {
            if (hasAnyOverlap(queryIntent.degreeLevels, scores.degree_levels)) {
                boost *= 1.1;
            } else if ((scores.degree_levels?.length || 0) > 0) {
                boost *= 0.45;
            }
        }

        if (queryIntent && queryIntent.eventTypes.length > 0) {
            if (hasAnyOverlap(queryIntent.eventTypes, scores.event_types)) {
                boost *= 1.1;
            } else if ((scores.event_types?.length || 0) > 0) {
                boost *= 0.65;
            }
        }

        if (
            queryIntent?.preferLatest &&
            latestTargetYear !== undefined &&
            scores.target_year !== undefined
        ) {
            const yearGap = Math.max(0, latestTargetYear - scores.target_year);
            boost *= Math.pow(LATEST_YEAR_BOOST_BASE, yearGap);
        }

        finalRanking.push({
            otid,
            score: finalScore * boost,
            best_kpid: scores.best_kpid,
        });
    }

    const sortedRanking = finalRanking.sort((a, b) => b.score - a.score);
    const querySparseTermCount = querySparse
        ? Object.keys(querySparse).length
        : 0;
    const shouldRejectForLowConsistency =
        !!queryIntent &&
        queryIntent.topicIds.length === 0 &&
        queryIntent.intentIds.length === 0 &&
        querySparseTermCount === 0 &&
        (() => {
            const consistencyWindow = sortedRanking.slice(0, 10);
            if (consistencyWindow.length === 0) return true;

            const topicHistogram = new Map<string, number>();
            let labeledCount = 0;

            for (const item of consistencyWindow) {
                const scores = otidMap[item.otid];
                const topicIds = dedupe(getCoverageComparableTopicIds(scores));
                if (topicIds.length === 0) continue;
                labeledCount += 1;
                topicIds.forEach((topicId) => {
                    topicHistogram.set(topicId, (topicHistogram.get(topicId) || 0) + 1);
                });
            }

            if (labeledCount === 0) return true;

            const dominantCount = Math.max(...topicHistogram.values());
            const dominantRatio = dominantCount / consistencyWindow.length;
            return dominantCount < 3 || dominantRatio < 0.45;
        })();

    if (shouldRejectForLowConsistency) {
        return {
            matches: [],
            weakMatches: [],
            rejection: {
                reason: "low_consistency",
                topicIds: [],
            },
        };
    }

    if (!hasCoveredRejectedTopic && coverageRejectedTopicIds.length > 0) {
        const weakRelatedMatches = sortedRanking.filter((item) => {
            const scores = otidMap[item.otid];
            return (
                classifyTopicCoverage(coverageRejectedTopicIds, scores) === "weak"
            );
        });
        return {
            matches: [],
            weakMatches: weakRelatedMatches.slice(0, 15),
            rejection: {
                reason: "low_topic_coverage",
                topicIds: coverageRejectedTopicIds,
            },
        };
    }

    const matches =
        coverageRejectedTopicIds.length > 0
            ? sortedRanking
                  .filter((item) => {
                      const scores = otidMap[item.otid];
                      const coverage = classifyTopicCoverage(
                          coverageRejectedTopicIds,
                          scores,
                      );
                      return coverage === "primary" || coverage === "secondary";
                  })
                  .slice(0, 100)
            : sortedRanking.slice(0, 100);

    return {
        matches,
        weakMatches: [],
    };
}

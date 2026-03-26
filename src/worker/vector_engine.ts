import {
    DEGREE_LEVEL_TABLE as CONFIG_DEGREE_LEVEL_TABLE,
    EVENT_TYPE_TABLE as CONFIG_EVENT_TYPE_TABLE,
    HISTORICAL_QUERY_HINTS as CONFIG_HISTORICAL_QUERY_HINTS,
    INTENT_VECTOR_TABLE as CONFIG_INTENT_VECTOR_TABLE,
    LATEST_QUERY_HINTS as CONFIG_LATEST_QUERY_HINTS,
    TOPIC_CONFIGS as CONFIG_TOPIC_CONFIGS,
} from "./search_topic_config";
import {
    applyScoreToAggregatedDocScores,
    createAggregatedDocScores,
    mergeAggregatedDocMetadata,
    type AggregatedDocScores,
    type KPCandidate,
} from "./aggregated_doc_scores";

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
    kp_role_tags?: string[];
}

export interface SearchResult {
    otid: string;
    best_kpid?: string;
    kp_candidates?: KPCandidate[];
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

export type KPAggregationMode = "max" | "max_plus_topn";
export type LexicalBonusMode = "sum" | "max";
export type KPRoleRerankMode = "off" | "feature";

export interface IntentVectorItem {
    intent_id: string;
    topic_id: string;
    intent_name: string;
    aliases: readonly string[];
    negative_intents: readonly string[];
    related_intents: readonly string[];
}

export const DEFAULT_WEIGHTS = {
    Q: 0.33,
    KP: 0.33,
    OT: 0.33,
};

export const RRF_K = 60;

const BM25_K1 = 1.2;
const BM25_B = 0.4;
const EVENT_TYPE_MISMATCH_PENALTY = 0.95;
const LATEST_YEAR_BOOST_BASE = 0.98;
const DEFAULT_KP_ROLE_DOC_WEIGHT = 0.35;
const DEFAULT_KP_ROLE_CANDIDATE_LIMIT = 5;



const INTENT_CONFLICTS: Record<string, readonly string[]> = Object.fromEntries(
    CONFIG_INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item.negative_intents]),
);

const INTENT_RULE_MAP: Map<string, IntentVectorItem> = new Map(
    CONFIG_INTENT_VECTOR_TABLE.map((item) => [item.intent_id, item]),
);

const TOPIC_RULE_MAP: Map<string, (typeof CONFIG_TOPIC_CONFIGS)[number]> =
    new Map(CONFIG_TOPIC_CONFIGS.map((item) => [item.topic_id, item] as const));

function deriveTopicIdsFromIntents(intentIds: string[]): string[] {
    return dedupe(
        intentIds
            .map((intentId) => INTENT_RULE_MAP.get(intentId)?.topic_id)
            .filter((topicId): topicId is string => Boolean(topicId)),
    );
}

export function resolveMetadataTopicIds(
    meta: Pick<Metadata, "topic_ids" | "subtopic_ids" | "intent_ids">,
): string[] {
    if (meta.topic_ids && meta.topic_ids.length > 0) {
        return meta.topic_ids;
    }
    return deriveTopicIdsFromIntents(meta.subtopic_ids || meta.intent_ids || []);
}

function matchTopicIds(query: string): string[] {
    return dedupe(
        CONFIG_TOPIC_CONFIGS.filter((topic) =>
            topic.aliases.some((alias) => query.includes(alias)),
        ).map((topic) => topic.topic_id),
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
    const offset = matrixIndex * dimensions;
    const unrolledLimit = dimensions - (dimensions % 4);
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;

    for (let i = 0; i < unrolledLimit; i += 4) {
        s0 += vecA[i] * matrix[offset + i];
        s1 += vecA[i + 1] * matrix[offset + i + 1];
        s2 += vecA[i + 2] * matrix[offset + i + 2];
        s3 += vecA[i + 3] * matrix[offset + i + 3];
    }

    let sum = s0 + s1 + s2 + s3;
    for (let i = unrolledLimit; i < dimensions; i++) {
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
    });

    return sparse;
}

export function parseQueryIntent(query: string): ParsedQueryIntent {
    const years = dedupe(
        (query.match(/20\d{2}/g) || []).map((year) => Number(year)),
    );
    const matchedRules = CONFIG_INTENT_VECTOR_TABLE.filter((rule) =>
        rule.aliases.some((alias) => query.includes(alias)),
    );
    const intentIds = dedupe(matchedRules.map((rule) => rule.intent_id));
    const matchedTopicIds = matchTopicIds(query);
    const topicIds = dedupe([
        ...deriveTopicIdsFromIntents(intentIds),
        ...matchedTopicIds,
    ]);
    const degreeLevels = dedupe([
        ...CONFIG_DEGREE_LEVEL_TABLE.filter((level) => query.includes(level)),
        ...(query.includes("直博") ? ["博士"] : []),
    ]);
    const eventTypes = dedupe(
        CONFIG_EVENT_TYPE_TABLE.filter((eventType) => query.includes(eventType)),
    );
    const normalizedTerms = dedupe(
        matchedRules.map((rule) => rule.intent_name),
    );
    const preferLatest =
        years.length === 0 &&
        !CONFIG_HISTORICAL_QUERY_HINTS.some((hint) => query.includes(hint)) &&
        (topicIds.some(
            (topicId) => TOPIC_RULE_MAP.get(topicId)?.prefer_latest,
        ) ||
            CONFIG_LATEST_QUERY_HINTS.some((hint) => query.includes(hint)));

    return {
        rawQuery: query,
        years,
        topicIds,
        subtopicIds: intentIds,
        intentIds,
        degreeLevels,
        eventTypes,
        normalizedTerms,
        confidence: intentIds.length > 0 ? 1 : 0,
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

function getRelatedIntentTypes(intentIds: string[]): string[] {
    return dedupe(
        intentIds.flatMap(
            (intentId) => INTENT_RULE_MAP.get(intentId)?.related_intents || [],
        ),
    );
}

type QueryIntentContext = {
    years: number[];
    hasExplicitYear: boolean;
    intentIds: string[];
    relatedIntentIds: string[];
    degreeLevels: string[];
    eventTypes: string[];
    preferLatest: boolean;
};

type DocQuerySignals = {
    hasStructuredYearMatch: boolean;
    hasLexicalYearMatch: boolean;
};

function createQueryIntentContext(
    queryIntent?: ParsedQueryIntent,
): QueryIntentContext {
    const years = queryIntent?.years || [];
    const intentIds = queryIntent?.intentIds || [];

    return {
        years,
        hasExplicitYear: years.length > 0,
        intentIds,
        relatedIntentIds: getRelatedIntentTypes(intentIds),
        degreeLevels: queryIntent?.degreeLevels || [],
        eventTypes: queryIntent?.eventTypes || [],
        preferLatest: Boolean(queryIntent?.preferLatest),
    };
}

function getDocQuerySignals(
    otid: string,
    scores: AggregatedDocScores,
    intentContext: QueryIntentContext,
    yearHitMap: Map<string, boolean>,
): DocQuerySignals {
    return {
        hasStructuredYearMatch:
            intentContext.hasExplicitYear &&
            scores.target_year !== undefined &&
            intentContext.years.includes(scores.target_year),
        hasLexicalYearMatch: yearHitMap.get(otid) === true,
    };
}

function shouldSkipForExplicitYear(
    scores: AggregatedDocScores,
    intentContext: QueryIntentContext,
    signals: DocQuerySignals,
): boolean {
    if (!intentContext.hasExplicitYear) {
        return false;
    }

    if (scores.target_year !== undefined && !signals.hasStructuredYearMatch) {
        return true;
    }

    return scores.target_year === undefined && !signals.hasLexicalYearMatch;
}

function computeBaseScore(
    scores: AggregatedDocScores,
    weights: typeof DEFAULT_WEIGHTS,
    options?: {
        kpAggregationMode?: KPAggregationMode;
        kpTopN?: number;
        kpTailWeight?: number;
    },
): number {
    const kpAggregationMode = options?.kpAggregationMode || "max";
    const kpTopN = Math.max(1, options?.kpTopN || 3);
    const kpTailWeight = options?.kpTailWeight ?? 0.35;
    const topKpScores =
        scores.kp_scores && scores.kp_scores.length > 0
            ? scores.kp_scores.slice(0, kpTopN)
            : scores.max_kp > 0
              ? [scores.max_kp]
              : [];
    const aggregatedKpScore =
        kpAggregationMode === "max_plus_topn" && topKpScores.length > 1
            ? topKpScores[0] +
              topKpScores.slice(1).reduce((sum, item) => sum + item, 0) *
                  kpTailWeight
            : topKpScores[0] || 0;

    const weightedQ = scores.max_q * weights.Q;
    const weightedKP = aggregatedKpScore * weights.KP;
    const weightedOT = scores.ot_score * weights.OT;

    const maxComponent = Math.max(weightedQ, weightedKP, weightedOT);
    const unionBonus =
        weightedQ * 0.1 + weightedKP * 0.1 + weightedOT * 0.1;

    return maxComponent + unionBonus;
}

type QueryRoleSignals = {
    asksTime: boolean;
    asksCondition: boolean;
    asksMaterials: boolean;
    asksProcedure: boolean;
    asksAnnouncementPeriod: boolean;
    asksApplicationStage: boolean;
    mentionsThesis: boolean;
};

function hasKpRoleTag(
    candidate: Pick<KPCandidate, "kp_role_tags"> | undefined,
    tag: string,
): boolean {
    return candidate?.kp_role_tags?.includes(tag) === true;
}

function deriveQueryRoleSignals(
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
        asksMaterials: /材料|扫描件|电子版|邮箱|mail/i.test(rawQuery),
        asksProcedure: /怎么办|怎么处理|不通过|补交|补充|流程|步骤/.test(
            rawQuery,
        ),
        asksAnnouncementPeriod: /公示期|哪几天/.test(rawQuery),
        asksApplicationStage:
            /申请|报名|确认|提交/.test(rawQuery) &&
            !/通过后|答辩通过|审批后|获得学位/.test(rawQuery),
        mentionsThesis: /论文/.test(rawQuery),
    };
}

function computeKpRoleBonus(
    candidate: KPCandidate,
    signals: QueryRoleSignals,
    rawQuery: string,
): number {
    let bonus = 0;

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
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 0.7;
        }
    }

    if (signals.asksMaterials) {
        if (hasKpRoleTag(candidate, "materials")) {
            bonus += 0.8;
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

    if (signals.asksApplicationStage) {
        if (hasKpRoleTag(candidate, "application_stage")) {
            bonus += 1.1;
        }
        if (hasKpRoleTag(candidate, "post_outcome")) {
            bonus -= 1.1;
        }
    }

    if (signals.asksProcedure) {
        if (hasKpRoleTag(candidate, "procedure")) {
            bonus += 1.0;
        }
        if (
            hasKpRoleTag(candidate, "reminder")
            || hasKpRoleTag(candidate, "background")
        ) {
            bonus -= 0.8;
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

    return bonus;
}

function rerankKpCandidatesByRole(params: {
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

function applyLexicalBonusBoost(boost: number, lexicalBonus: number): number {
    if (lexicalBonus <= 0) {
        return boost;
    }

    return boost * (1 + Math.log1p(lexicalBonus) / 4);
}

function applyYearConstraintBoost(
    boost: number,
    queryYearWordIds: number[] | undefined,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    signals: DocQuerySignals,
): number {
    if (!queryYearWordIds || queryYearWordIds.length === 0) {
        return boost;
    }

    if (scores.target_year !== undefined && intentContext.years.length > 0) {
        if (!intentContext.years.includes(scores.target_year)) {
            return boost * 0.01;
        }
        return boost;
    }

    if (!signals.hasStructuredYearMatch && !signals.hasLexicalYearMatch) {
        return boost * 0.12;
    }

    return boost;
}

function applyIntentBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (intentContext.intentIds.length === 0) {
        return boost;
    }

    let nextBoost = boost;

    if (hasIntentMatch(intentContext.intentIds, scores.intent_ids)) {
        nextBoost *= 1.12;
    } else if (hasIntentConflict(intentContext.intentIds, scores.intent_ids)) {
        nextBoost *= 0.85;
    } else if (
        intentContext.relatedIntentIds.length > 0 &&
        hasAnyOverlap(intentContext.relatedIntentIds, scores.intent_ids)
    ) {
        nextBoost *= 1.04;
    }

    return nextBoost;
}

function applyDegreeBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (intentContext.degreeLevels.length === 0) {
        return boost;
    }

    if (hasAnyOverlap(intentContext.degreeLevels, scores.degree_levels)) {
        return boost * 1.05;
    }

    if ((scores.degree_levels?.length || 0) > 0) {
        return boost * 0.93;
    }

    return boost;
}

function applyEventBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
): number {
    if (intentContext.eventTypes.length === 0) {
        return boost;
    }

    if (hasAnyOverlap(intentContext.eventTypes, scores.event_types)) {
        return boost * 1.05;
    }

    if ((scores.event_types?.length || 0) > 0) {
        return boost * EVENT_TYPE_MISMATCH_PENALTY;
    }

    return boost;
}

function applyLatestYearBoost(
    boost: number,
    intentContext: QueryIntentContext,
    scores: AggregatedDocScores,
    latestTargetYear?: number,
): number {
    if (
        !intentContext.preferLatest ||
        latestTargetYear === undefined ||
        scores.target_year === undefined
    ) {
        return boost;
    }

    const yearGap = Math.max(0, latestTargetYear - scores.target_year);
    return boost * Math.pow(LATEST_YEAR_BOOST_BASE, yearGap);
}

function computeBoostMultiplier(params: {
    otid: string;
    scores: AggregatedDocScores;
    lexicalBonusMap: Map<string, number>;
    yearHitMap: Map<string, boolean>;
    queryYearWordIds?: number[];
    intentContext: QueryIntentContext;
    latestTargetYear?: number;
}): number {
    const {
        otid,
        scores,
        lexicalBonusMap,
        yearHitMap,
        queryYearWordIds,
        intentContext,
        latestTargetYear,
    } = params;
    const signals = getDocQuerySignals(otid, scores, intentContext, yearHitMap);
    const lexicalBonus = lexicalBonusMap.get(otid) || 0;

    let boost = 1.0;
    boost = applyLexicalBonusBoost(boost, lexicalBonus);
    boost = applyYearConstraintBoost(
        boost,
        queryYearWordIds,
        intentContext,
        scores,
        signals,
    );
    boost = applyIntentBoost(boost, intentContext, scores);
    boost = applyDegreeBoost(boost, intentContext, scores);
    boost = applyEventBoost(boost, intentContext, scores);
    boost = applyLatestYearBoost(
        boost,
        intentContext,
        scores,
        latestTargetYear,
    );

    return boost;
}

function shouldRejectForLowConsistency(
    queryIntent: ParsedQueryIntent | undefined,
    querySparse: Record<number, number> | undefined,
    sortedRanking: SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
): boolean {
    const querySparseTermCount = querySparse
        ? Object.keys(querySparse).length
        : 0;

    if (
        !queryIntent ||
        queryIntent.topicIds.length > 0 ||
        queryIntent.intentIds.length > 0 ||
        querySparseTermCount > 0
    ) {
        return false;
    }

    const consistencyWindow = sortedRanking.slice(0, 10);
    if (consistencyWindow.length === 0) {
        return true;
    }

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

    if (labeledCount === 0) {
        return true;
    }

    const dominantCount = Math.max(...topicHistogram.values());
    const dominantRatio = dominantCount / consistencyWindow.length;
    return dominantCount < 3 || dominantRatio < 0.45;
}

export function searchAndRank(params: {
    queryVector: Float32Array;
    querySparse?: Record<number, number>;
    queryYearWordIds?: number[];
    queryIntent?: ParsedQueryIntent;
    queryScopeHint?: string;
    metadata: Metadata[];
    vectorMatrix: Int8Array | Float32Array;
    dimensions: number;
    currentTimestamp: number;
    bm25Stats: BM25Stats;
    weights?: typeof DEFAULT_WEIGHTS;
    candidateIndices?: readonly number[];
    topHybridLimit?: number;
    kpAggregationMode?: KPAggregationMode;
    kpTopN?: number;
    kpTailWeight?: number;
    lexicalBonusMode?: LexicalBonusMode;
    kpRoleRerankMode?: KPRoleRerankMode;
    kpRoleDocWeight?: number;
}): SearchRankOutput {
    const {
        queryVector,
        querySparse,
        metadata,
        vectorMatrix,
        dimensions,
        currentTimestamp: _currentTimestamp,
        bm25Stats,
        weights = DEFAULT_WEIGHTS,
        queryYearWordIds,
        queryIntent,
        queryScopeHint,
        candidateIndices,
        topHybridLimit = 1000,
        kpAggregationMode = "max",
        kpTopN = 3,
        kpTailWeight = 0.35,
        lexicalBonusMode = "sum",
        kpRoleRerankMode = "off",
        kpRoleDocWeight = DEFAULT_KP_ROLE_DOC_WEIGHT,
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
                const weightedBonus =
                    meta.type === "Q"
                        ? sparse * 1.5
                        : meta.type === "KP"
                          ? sparse * 1.2
                          : sparse;
                const currentBonus = lexicalBonusMap.get(otid) || 0;
                const nextBonus =
                    lexicalBonusMode === "max"
                        ? Math.max(currentBonus, weightedBonus)
                        : currentBonus + weightedBonus;
                lexicalBonusMap.set(otid, nextBonus);
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
        .slice(0, Math.max(1, topHybridLimit));

    const otidMap: Record<string, AggregatedDocScores> = {};

    for (const [meta, score] of topHybrid) {
        const otid = meta.type === "OT" ? meta.id : meta.parent_otid;
        const topicIds = resolveMetadataTopicIds(meta);
        if (!otidMap[otid]) {
            otidMap[otid] = createAggregatedDocScores(meta, topicIds);
        }

        mergeAggregatedDocMetadata(otidMap[otid], meta, topicIds);
        applyScoreToAggregatedDocScores(otidMap[otid], meta, score);
    }

    const finalRanking: SearchResult[] = [];
    const candidateTargetYears = Object.values(otidMap)
        .map((scores) => scores.target_year)
        .filter((year): year is number => typeof year === "number");
    const latestTargetYear =
        candidateTargetYears.length > 0
            ? Math.max(...candidateTargetYears)
            : undefined;
    const intentContext = createQueryIntentContext(queryIntent);

    for (const [otid, scores] of Object.entries(otidMap)) {
        const signals = getDocQuerySignals(
            otid,
            scores,
            intentContext,
            yearHitMap,
        );

        if (shouldSkipForExplicitYear(scores, intentContext, signals)) {
            continue;
        }

        const finalScore = computeBaseScore(scores, weights, {
            kpAggregationMode,
            kpTopN,
            kpTailWeight,
        });
        const kpRoleSelection = rerankKpCandidatesByRole({
            kpCandidates: scores.kp_candidates,
            bestKpid: scores.best_kpid,
            rawQuery: queryIntent?.rawQuery || "",
            queryScopeHint,
            mode: kpRoleRerankMode,
        });
        const boost = computeBoostMultiplier({
            otid,
            scores,
            lexicalBonusMap,
            yearHitMap,
            queryYearWordIds,
            intentContext,
            latestTargetYear,
        });

        finalRanking.push({
            otid,
            score: finalScore * boost + kpRoleSelection.docScoreDelta * kpRoleDocWeight,
            best_kpid: kpRoleSelection.bestKpid,
            kp_candidates: kpRoleSelection.orderedCandidates.slice(0, 5),
        });
    }

    const sortedRanking = finalRanking.sort((a, b) => b.score - a.score);
    if (
        shouldRejectForLowConsistency(
            queryIntent,
            querySparse,
            sortedRanking,
            otidMap,
        )
    ) {
        return {
            matches: [],
            weakMatches: [],
            rejection: {
                reason: "low_consistency",
                topicIds: [],
            },
        };
    }

    return {
        matches: sortedRanking.slice(0, 100),
        weakMatches: [],
    };
}

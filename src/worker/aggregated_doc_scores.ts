type AggregatableMetadata = {
    id: string;
    type: "Q" | "KP" | "OT";
    parent_pkid?: string;
    parent_otid: string;
    timestamp?: number;
    target_year?: number;
    primary_topic_ids?: string[];
    secondary_topic_ids?: string[];
    weak_topic_ids?: string[];
    subtopic_ids?: string[];
    intent_ids?: string[];
    degree_levels?: string[];
    event_types?: string[];
    kp_role_tags?: string[];
};

export interface KPCandidate {
    kpid: string;
    score: number;
    kp_role_tags?: string[];
}

export interface AggregatedDocScores {
    max_q: number;
    max_kp: number;
    kp_score_map: Record<string, number>;
    kp_scores: number[];
    kp_candidates: KPCandidate[];
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

type ArrayFieldKey =
    | "topic_ids"
    | "subtopic_ids"
    | "primary_topic_ids"
    | "secondary_topic_ids"
    | "weak_topic_ids"
    | "intent_ids"
    | "degree_levels"
    | "event_types";

const MAX_TRACKED_KP_SCORES = 8;

function assignArrayIfMissing(
    target: AggregatedDocScores,
    key: ArrayFieldKey,
    incoming?: string[],
) {
    if ((!target[key] || target[key]!.length === 0) && incoming?.length) {
        target[key] = incoming;
    }
}

export function createAggregatedDocScores(
    meta: AggregatableMetadata,
    topicIds: string[],
): AggregatedDocScores {
    return {
        max_q: 0,
        max_kp: 0,
        kp_score_map: {},
        kp_scores: [],
        kp_candidates: [],
        ot_score: 0,
        timestamp: meta.timestamp,
        target_year: meta.target_year,
        topic_ids: topicIds,
        primary_topic_ids: meta.primary_topic_ids,
        secondary_topic_ids: meta.secondary_topic_ids,
        weak_topic_ids: meta.weak_topic_ids,
        subtopic_ids: meta.subtopic_ids || meta.intent_ids,
        intent_ids: meta.intent_ids,
        degree_levels: meta.degree_levels,
        event_types: meta.event_types,
    };
}

export function mergeAggregatedDocMetadata(
    target: AggregatedDocScores,
    meta: AggregatableMetadata,
    topicIds: string[],
) {
    if (target.target_year === undefined && meta.target_year !== undefined) {
        target.target_year = meta.target_year;
    }

    assignArrayIfMissing(target, "topic_ids", topicIds);
    assignArrayIfMissing(target, "subtopic_ids", meta.subtopic_ids || meta.intent_ids);
    assignArrayIfMissing(target, "primary_topic_ids", meta.primary_topic_ids);
    assignArrayIfMissing(target, "secondary_topic_ids", meta.secondary_topic_ids);
    assignArrayIfMissing(target, "weak_topic_ids", meta.weak_topic_ids);
    assignArrayIfMissing(target, "intent_ids", meta.intent_ids);
    assignArrayIfMissing(target, "degree_levels", meta.degree_levels);
    assignArrayIfMissing(target, "event_types", meta.event_types);
}

export function applyScoreToAggregatedDocScores(
    target: AggregatedDocScores,
    meta: Pick<AggregatableMetadata, "id" | "type" | "kp_role_tags" | "parent_pkid">,
    score: number,
) {
    if (meta.type === "Q") {
        target.max_q = Math.max(target.max_q, score);
        return;
    }

    if (meta.type === "KP") {
        const canonicalKpid = meta.parent_pkid || meta.id;
        const previousScore = target.kp_score_map[canonicalKpid];
        if (previousScore === undefined || score > previousScore) {
            target.kp_score_map[canonicalKpid] = score;
            target.kp_scores = Object.values(target.kp_score_map)
                .sort((a, b) => b - a)
                .slice(0, MAX_TRACKED_KP_SCORES);

            const existingCandidate = target.kp_candidates.find(
                (candidate) => candidate.kpid === canonicalKpid,
            );
            if (existingCandidate) {
                existingCandidate.score = score;
                existingCandidate.kp_role_tags = meta.kp_role_tags;
            } else {
                target.kp_candidates.push({
                    kpid: canonicalKpid,
                    score,
                    kp_role_tags: meta.kp_role_tags,
                });
            }
            target.kp_candidates.sort((a, b) => b.score - a.score);
            if (target.kp_candidates.length > MAX_TRACKED_KP_SCORES) {
                target.kp_candidates.length = MAX_TRACKED_KP_SCORES;
            }
            target.max_kp = target.kp_scores[0] || 0;
            target.best_kpid = target.kp_candidates[0]?.kpid;
        }
        return;
    }

    target.ot_score = Math.max(target.ot_score, score);
}

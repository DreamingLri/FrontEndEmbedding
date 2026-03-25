type AggregatableMetadata = {
    id: string;
    type: "Q" | "KP" | "OT";
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
};

export interface AggregatedDocScores {
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

type ArrayFieldKey =
    | "topic_ids"
    | "subtopic_ids"
    | "primary_topic_ids"
    | "secondary_topic_ids"
    | "weak_topic_ids"
    | "intent_ids"
    | "degree_levels"
    | "event_types";

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
    meta: Pick<AggregatableMetadata, "id" | "type">,
    score: number,
) {
    if (meta.type === "Q") {
        target.max_q = Math.max(target.max_q, score);
        return;
    }

    if (meta.type === "KP") {
        if (score > target.max_kp) {
            target.max_kp = score;
            target.best_kpid = meta.id;
        }
        return;
    }

    target.ot_score = Math.max(target.ot_score, score);
}

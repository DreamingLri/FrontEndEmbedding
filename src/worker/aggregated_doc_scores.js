"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createAggregatedDocScores = createAggregatedDocScores;
exports.mergeAggregatedDocMetadata = mergeAggregatedDocMetadata;
exports.applyScoreToAggregatedDocScores = applyScoreToAggregatedDocScores;
var MAX_TRACKED_KP_SCORES = 8;
function assignArrayIfMissing(target, key, incoming) {
    if ((!target[key] || target[key].length === 0) && (incoming === null || incoming === void 0 ? void 0 : incoming.length)) {
        target[key] = incoming;
    }
}
function createAggregatedDocScores(meta, topicIds) {
    return {
        max_q: 0,
        max_kp: 0,
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
function mergeAggregatedDocMetadata(target, meta, topicIds) {
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
function applyScoreToAggregatedDocScores(target, meta, score) {
    if (meta.type === "Q") {
        target.max_q = Math.max(target.max_q, score);
        return;
    }
    if (meta.type === "KP") {
        target.kp_scores.push(score);
        target.kp_scores.sort(function (a, b) { return b - a; });
        if (target.kp_scores.length > MAX_TRACKED_KP_SCORES) {
            target.kp_scores.length = MAX_TRACKED_KP_SCORES;
        }
        target.kp_candidates.push({
            kpid: meta.id,
            score: score,
            kp_role_tags: meta.kp_role_tags,
        });
        target.kp_candidates.sort(function (a, b) { return b.score - a.score; });
        if (target.kp_candidates.length > MAX_TRACKED_KP_SCORES) {
            target.kp_candidates.length = MAX_TRACKED_KP_SCORES;
        }
        if (score > target.max_kp) {
            target.max_kp = score;
            target.best_kpid = meta.id;
        }
        return;
    }
    target.ot_score = Math.max(target.ot_score, score);
}

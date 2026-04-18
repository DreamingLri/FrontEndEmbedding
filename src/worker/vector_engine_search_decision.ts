import type { AggregatedDocScores } from "./aggregated_doc_scores.ts";
import {
    BOUNDARY_REJECT_SCORE_THRESHOLD,
    clamp01,
    DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    dedupe,
    HARD_REJECT_SCORE_THRESHOLD,
    STRONG_EVIDENCE_ROLE_TAGS,
    WEAK_EVIDENCE_ROLE_TAGS,
    type EvidenceSignals,
    type QuerySignals,
    type RejectTier,
    type ResponseDecision,
    type ResponseMode,
    type RetrievalSignals,
    type SearchResult,
} from "./vector_engine_shared.ts";
import { getCoverageComparableTopicIds } from "./vector_engine_search_context.ts";
import { hasAnyRoleEvidence } from "./vector_engine_search_boosts.ts";
export function extractRetrievalSignals(
    sortedRanking: SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
): RetrievalSignals {
    const consistencyWindow = sortedRanking.slice(0, 10);
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

    const top1Score = sortedRanking[0]?.score || 0;
    const top2Score = sortedRanking[1]?.score ?? top1Score;
    const top5Score = sortedRanking[4]?.score ?? sortedRanking.at(-1)?.score ?? top1Score;
    const dominantCount =
        topicHistogram.size > 0 ? Math.max(...topicHistogram.values()) : 0;
    const dominantRatio =
        consistencyWindow.length > 0 ? dominantCount / consistencyWindow.length : 0;

    return {
        candidateCount: sortedRanking.length,
        top1Score,
        top1Top2Gap: top1Score - top2Score,
        top1Top5Gap: top1Score - top5Score,
        distinctTopicCount: topicHistogram.size,
        dominantTopicCount: dominantCount,
        dominantTopicRatio: dominantRatio,
        labeledTopicCount: labeledCount,
    };
}

export function extractEvidenceSignals(
    sortedRanking: SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
): EvidenceSignals {
    const topWindow = sortedRanking
        .slice(0, 3)
        .flatMap((item) =>
            (otidMap[item.otid]?.kp_candidates || []).slice(
                0,
                DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
            ),
        );

    const topRoleTags = dedupe(
        topWindow.flatMap((candidate) => candidate.kp_role_tags || []),
    );
    let strongRoleCount = 0;
    let weakRoleCount = 0;

    topRoleTags.forEach((tag) => {
        if (STRONG_EVIDENCE_ROLE_TAGS.has(tag)) {
            strongRoleCount += 1;
        } else if (WEAK_EVIDENCE_ROLE_TAGS.has(tag)) {
            weakRoleCount += 1;
        }
    });

    const totalEvidenceRoleCount = strongRoleCount + weakRoleCount;

    return {
        topRoleTags,
        topCandidateCount: topWindow.length,
        strongRoleCount,
        weakRoleCount,
        strongRoleRatio:
            totalEvidenceRoleCount > 0
                ? strongRoleCount / totalEvidenceRoleCount
                : 0,
        weakOnly: strongRoleCount === 0 && weakRoleCount > 0,
        hasOperationalRoleEvidence: hasAnyRoleEvidence(
            topWindow,
            Array.from(STRONG_EVIDENCE_ROLE_TAGS),
        ),
    };
}

export function classifyResponseMode(
    querySignals: QuerySignals,
    retrievalSignals: RetrievalSignals,
    evidenceSignals: EvidenceSignals,
): ResponseDecision {
    const tokenCount = querySignals.tokenCount || 0;
    const invalidInput = tokenCount === 0;

    if (invalidInput) {
        return {
            mode: "reject",
            confidence: 0.98,
            reason: "invalid_input",
            preferLatestWithinTopic: false,
            useWeakMatches: false,
            rejectScore: 1,
            rejectTier: "invalid_input",
        };
    }

    const noStructuredAnchor =
        !querySignals.hasExplicitTopicOrIntent && !querySignals.hasExplicitYear;
    const lowTokenRisk = tokenCount <= 1 ? 1 : tokenCount <= 3 ? 0.45 : 0;
    const shortQueryRisk = clamp01((8 - querySignals.queryLength) / 8);
    const genericNextStepRisk =
        querySignals.hasGenericNextStep && !querySignals.hasExplicitTopicOrIntent
            ? 0.45
            : 0;
    const stateWithoutAnchorRisk =
        querySignals.hasResultState && noStructuredAnchor ? 1 : 0;
    const intentRisk = clamp01(
        0.58 * (noStructuredAnchor ? 1 : 0) +
            0.16 * lowTokenRisk +
            0.1 * genericNextStepRisk +
            0.1 * shortQueryRisk +
            0.06 * stateWithoutAnchorRisk,
    );

    const noCandidatesRisk = retrievalSignals.candidateCount === 0 ? 1 : 0;
    const noLabeledRisk = retrievalSignals.labeledTopicCount === 0 ? 1 : 0;
    const spreadRisk = clamp01(
        (retrievalSignals.distinctTopicCount - 2) / 3,
    );
    const dominanceRisk = clamp01(
        (0.55 - retrievalSignals.dominantTopicRatio) / 0.55,
    );
    const topicDispersionRisk = Math.max(spreadRisk, dominanceRisk);
    const gapRisk = clamp01((0.05 - retrievalSignals.top1Top2Gap) / 0.05);
    const retrievalRisk =
        noCandidatesRisk === 1
            ? 1
            : clamp01(
                  0.5 * noLabeledRisk +
                      0.3 * topicDispersionRisk +
                      0.2 * gapRisk,
              );

    const evidenceRisk = clamp01(
        0.68 * (1 - evidenceSignals.strongRoleRatio) +
            0.22 * (evidenceSignals.hasOperationalRoleEvidence ? 0 : 1) +
            0.1 * (evidenceSignals.weakOnly ? 1 : 0),
    );

    const genericProcessSafetyBonus =
        querySignals.hasGenericNextStep &&
        (querySignals.hasStrongDetailAnchor || querySignals.hasResultState)
            ? 0.09
            : 0;
    const postOutcomeOperationalSafetyBonus =
        querySignals.hasResultState && querySignals.hasPostOutcomeOperationalCue
            ? querySignals.hasMultiSlotConstraintCue
                ? 0.14
                : 0.1
            : 0;

    const rejectScore = clamp01(
        0.4 * intentRisk +
            0.25 * retrievalRisk +
            0.35 * evidenceRisk -
            (querySignals.hasExplicitTopicOrIntent ? 0.08 : 0) -
            (querySignals.hasExplicitYear ? 0.03 : 0) -
            genericProcessSafetyBonus -
            postOutcomeOperationalSafetyBonus,
    );

    let mode: ResponseMode = "answer";
    let rejectTier: RejectTier | null = null;
    if (rejectScore >= HARD_REJECT_SCORE_THRESHOLD) {
        mode = "reject";
        rejectTier = "hard_reject";
    } else if (rejectScore >= BOUNDARY_REJECT_SCORE_THRESHOLD) {
        mode = "reject";
        rejectTier = "boundary_uncertain";
    }

    const reasonParts: string[] = [];
    if (noStructuredAnchor) {
        reasonParts.push("no_structured_anchor");
    }
    if (evidenceSignals.weakOnly) {
        reasonParts.push("weak_only_role_evidence");
    } else if (!evidenceSignals.hasOperationalRoleEvidence) {
        reasonParts.push("no_operational_evidence");
    }
    if (noLabeledRisk === 1) {
        reasonParts.push("no_labeled_topic_support");
    } else if (topicDispersionRisk >= 0.45) {
        reasonParts.push("topic_dispersion");
    }
    if (reasonParts.length === 0) {
        reasonParts.push(
            mode === "reject" ? "reject_score_guard" : "answer_score_pass",
        );
    }

    let confidence: number;
    if (mode === "reject" && rejectTier === "hard_reject") {
        confidence = Math.max(
            0.74,
            Math.min(
                0.98,
                0.74 +
                    (rejectScore - HARD_REJECT_SCORE_THRESHOLD) /
                        (1 - HARD_REJECT_SCORE_THRESHOLD) *
                        0.18,
            ),
        );
    } else if (mode === "reject") {
        confidence = Math.max(
            0.58,
            Math.min(
                0.79,
                0.58 +
                    (rejectScore - BOUNDARY_REJECT_SCORE_THRESHOLD) /
                        (HARD_REJECT_SCORE_THRESHOLD -
                            BOUNDARY_REJECT_SCORE_THRESHOLD) *
                        0.18,
            ),
        );
    } else {
        confidence = Math.max(
            0.56,
            Math.min(
                0.92,
                0.92 -
                    (rejectScore / BOUNDARY_REJECT_SCORE_THRESHOLD) * 0.24,
            ),
        );
    }

    return {
        mode,
        confidence,
        reason: reasonParts.slice(0, 3).join("+"),
        preferLatestWithinTopic:
            querySignals.hasLatestPolicyState && !querySignals.hasExplicitYear,
        useWeakMatches: rejectTier === "hard_reject",
        rejectScore,
        rejectTier,
    };
}

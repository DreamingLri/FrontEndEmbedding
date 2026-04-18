import type {
    AggregatedDocScores,
    KPCandidate,
} from "../aggregated_doc_scores.ts";
import type { QueryPlan } from "../query_planner.ts";
import {
    DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    type SearchResult,
} from "./shared.ts";
import { hasAnyOverlap } from "./search_context.ts";
import { hasKpRoleTag } from "./search_role_rerank.ts";

function hasAnyKpRoleEvidence(
    candidates: readonly KPCandidate[],
    tags: readonly string[],
): boolean {
    return candidates.some((candidate) =>
        tags.some((tag) => hasKpRoleTag(candidate, tag)),
    );
}

function countPlannerEvidenceGroups(
    candidates: readonly KPCandidate[],
): number {
    const window = candidates.slice(0, DEFAULT_KP_ROLE_CANDIDATE_LIMIT);
    return [
        hasAnyKpRoleEvidence(window, ["condition", "deadline"]),
        hasAnyKpRoleEvidence(window, ["materials", "email"]),
        hasAnyKpRoleEvidence(window, [
            "procedure",
            "application_stage",
            "schedule",
        ]),
        hasAnyKpRoleEvidence(window, [
            "arrival",
            "deadline",
            "announcement_period",
            "time_expression",
            "schedule",
        ]),
    ].filter(Boolean).length;
}

export function applyQueryPlannerRetrievalBoost(
    boost: number,
    queryPlan: QueryPlan | undefined,
    scores: AggregatedDocScores,
): number {
    if (!queryPlan) {
        return boost;
    }

    const window = scores.kp_candidates.slice(
        0,
        DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    );
    let multiplier = 1;

    if (queryPlan.asksCoverageLike) {
        const coveredGroups = countPlannerEvidenceGroups(window);
        if (coveredGroups >= 2) {
            multiplier *= 1 + Math.min(0.08, coveredGroups * 0.025);
        } else if (coveredGroups === 0) {
            multiplier *= 0.98;
        }
    }

    switch (queryPlan.intentType) {
        case "outcome":
            if (hasAnyOverlap(["录取公示", "推免资格公示"], scores.event_types)) {
                multiplier *= 1.06;
            } else if (
                hasAnyOverlap(
                    ["招生章程", "推免实施办法", "资格要求"],
                    scores.event_types,
                )
            ) {
                multiplier *= 0.97;
            }
            break;
        case "procedure":
        case "system_timeline":
            if (
                hasAnyOverlap(
                    ["报名通知", "考试安排", "材料提交", "复试通知", "推免实施办法"],
                    scores.event_types,
                )
            ) {
                multiplier *= 1.04;
            }
            if (
                hasAnyKpRoleEvidence(window, [
                    "procedure",
                    "application_stage",
                    "schedule",
                    "deadline",
                ])
            ) {
                multiplier *= 1.04;
            }
            if (
                !queryPlan.asksOutcomeLike &&
                hasAnyOverlap(["录取公示", "推免资格公示"], scores.event_types)
            ) {
                multiplier *= 0.97;
            }
            break;
        case "requirement":
        case "policy_overview":
            if (
                hasAnyOverlap(
                    ["招生章程", "推免实施办法", "资格要求", "材料提交"],
                    scores.event_types,
                )
            ) {
                multiplier *= 1.04;
            }
            if (
                hasAnyKpRoleEvidence(window, [
                    "condition",
                    "materials",
                    "application_stage",
                    "procedure",
                ])
            ) {
                multiplier *= 1.03;
            }
            break;
        case "time_location":
            if (
                hasAnyKpRoleEvidence(window, [
                    "schedule",
                    "arrival",
                    "deadline",
                    "announcement_period",
                    "time_expression",
                ])
            ) {
                multiplier *= 1.05;
            }
            break;
        case "fact_detail":
        default:
            break;
    }

    if (queryPlan.difficultyTier === "high" && scores.kp_scores.length >= 2) {
        multiplier *= 1.03;
    }

    return boost * Math.max(0.92, Math.min(1.12, multiplier));
}

function collectPlannerCoverageFacets(scores?: AggregatedDocScores): string[] {
    if (!scores) {
        return [];
    }

    const facets = new Set<string>();
    scores.event_types?.slice(0, 4).forEach((eventType) => {
        facets.add(`event:${eventType}`);
    });
    scores.degree_levels?.slice(0, 3).forEach((degreeLevel) => {
        facets.add(`degree:${degreeLevel}`);
    });
    if (scores.target_year !== undefined) {
        facets.add(`year:${scores.target_year}`);
    }

    const window = scores.kp_candidates.slice(
        0,
        DEFAULT_KP_ROLE_CANDIDATE_LIMIT,
    );
    if (hasAnyKpRoleEvidence(window, ["condition", "deadline"])) {
        facets.add("role:condition");
    }
    if (hasAnyKpRoleEvidence(window, ["materials", "email"])) {
        facets.add("role:materials");
    }
    if (
        hasAnyKpRoleEvidence(window, [
            "procedure",
            "application_stage",
            "schedule",
        ])
    ) {
        facets.add("role:procedure");
    }
    if (
        hasAnyKpRoleEvidence(window, [
            "arrival",
            "deadline",
            "announcement_period",
            "time_expression",
            "schedule",
        ])
    ) {
        facets.add("role:time");
    }

    return Array.from(facets);
}

export function applyQueryPlannerCoverageDiversification(
    ranking: readonly SearchResult[],
    otidMap: Record<string, AggregatedDocScores>,
    queryPlan?: QueryPlan,
): SearchResult[] {
    if (
        !queryPlan ||
        !queryPlan.asksCoverageLike ||
        ranking.length <= 2
    ) {
        return [...ranking];
    }

    const windowSize = Math.min(12, ranking.length);
    const selected: SearchResult[] = [ranking[0]!];
    const remaining = ranking.slice(1, windowSize);
    const tail = ranking.slice(windowSize);
    const seenFacets = new Set(
        collectPlannerCoverageFacets(otidMap[selected[0]!.otid]),
    );

    while (remaining.length > 0) {
        let bestIndex = 0;
        let bestAdjustedScore = Number.NEGATIVE_INFINITY;

        remaining.forEach((candidate, index) => {
            const facets = collectPlannerCoverageFacets(otidMap[candidate.otid]);
            const newFacetCount = facets.filter((facet) => !seenFacets.has(facet))
                .length;
            const duplicateFacetCount = facets.length - newFacetCount;
            const adjustedScore =
                candidate.score +
                Math.min(0.18, newFacetCount * 0.045) -
                Math.min(0.08, duplicateFacetCount * 0.012);

            if (adjustedScore > bestAdjustedScore) {
                bestAdjustedScore = adjustedScore;
                bestIndex = index;
            }
        });

        const [chosen] = remaining.splice(bestIndex, 1);
        collectPlannerCoverageFacets(otidMap[chosen!.otid]).forEach((facet) =>
            seenFacets.add(facet),
        );
        selected.push({
            ...chosen!,
            score: bestAdjustedScore,
        });
    }

    return [...selected, ...tail];
}


import type { QueryPlan, QueryPlanDocRole } from "../query_planner.ts";
import {
    buildDocumentRerankEntryLookup,
    getDocumentDisplayScore,
    sortDocumentRerankEntriesByDisplayScore,
    TITLE_DIVERSITY_DUPLICATE_PENALTY,
    updateDocumentRerankEntryScores,
    type DocumentRerankEntry,
    type LatestVersionFamilyStat,
} from "./document_rerank_shared.ts";
import type { DocumentRerankQuerySignals } from "./document_rerank_query.ts";

export function computeLatestVersionDocDelta(params: {
    entry: DocumentRerankEntry;
    familyStats: Map<string, LatestVersionFamilyStat>;
    querySignals: DocumentRerankQuerySignals;
}): number {
    const { entry, familyStats, querySignals } = params;
    const { metadata } = entry;
    const familyKey = metadata.latestVersionFamilyKey;
    const familyStat = familyKey ? familyStats.get(familyKey) : undefined;
    const recencyKey = metadata.recencyKey;
    const roles = metadata.roles;
    let delta = 0;

    if (familyStat && familyStat.count >= 2) {
        if (
            recencyKey !== undefined &&
            familyStat.latestRecencyKey !== undefined
        ) {
            const gapMonths = (familyStat.latestRecencyKey - recencyKey) / 31;
            if (gapMonths <= 0) {
                delta += 0.92;
                if (
                    querySignals.roleSensitiveLatestVersion &&
                    (roles.includes("rule_doc") ||
                        roles.includes("registration_notice") ||
                        roles.includes("stage_list"))
                ) {
                    delta += 0.12;
                }
            } else {
                delta -= Math.min(1.05, 0.2 + gapMonths * 0.08);
            }
        } else {
            delta -= 0.18;
        }
    }

    if (
        !querySignals.asksOutcomeLikeTitle &&
        querySignals.roleSensitiveLatestVersion &&
        (roles.includes("result_notice") || roles.includes("list_notice"))
    ) {
        delta -= 0.18;
    }

    return delta;
}

export function applyCompressedQueryDisplayGuardToEntries(
    querySignals: DocumentRerankQuerySignals,
    baselineEntries: DocumentRerankEntry[],
    rerankedEntries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    if (
        !querySignals.isCompressedKeywordQuery ||
        baselineEntries.length === 0 ||
        rerankedEntries.length === 0
    ) {
        return rerankedEntries;
    }

    const baselineTop = baselineEntries[0];
    const rerankedTop = rerankedEntries[0];
    const baselineTopOtid = baselineTop.document.otid;
    const rerankedTopOtid = rerankedTop.document.otid;
    if (
        !baselineTopOtid ||
        !rerankedTopOtid ||
        baselineTopOtid === rerankedTopOtid
    ) {
        return rerankedEntries;
    }

    const baselineTopScore = getDocumentDisplayScore(baselineTop.document);
    const baselineLookup = buildDocumentRerankEntryLookup(baselineEntries);
    const rerankedTopBaseline = baselineLookup.get(rerankedTopOtid)?.entry;
    const rerankedTopBaselineScore = rerankedTopBaseline
        ? getDocumentDisplayScore(rerankedTopBaseline.document)
        : Number.NEGATIVE_INFINITY;

    if (baselineTopScore + 0.06 < rerankedTopBaselineScore) {
        return rerankedEntries;
    }

    const rerankedLookup = buildDocumentRerankEntryLookup(rerankedEntries);
    const preservedBaselineLookup = rerankedLookup.get(baselineTopOtid);
    if (!preservedBaselineLookup) {
        return rerankedEntries;
    }

    const boostedTopScore = getDocumentDisplayScore(rerankedTop.document) + 0.001;
    const reordered = [...rerankedEntries];
    const [preservedBaseline] = reordered.splice(preservedBaselineLookup.index, 1);
    reordered.unshift({
        document: {
            ...preservedBaseline.document,
            score: boostedTopScore,
            displayScore: boostedTopScore,
        },
        metadata: preservedBaseline.metadata,
    });
    return reordered;
}

export function computeQueryPlanDocRoleDelta(
    roles: QueryPlanDocRole[],
    queryPlan: QueryPlan,
): number {
    if (
        queryPlan.difficultyTier !== "high" &&
        !queryPlan.asksOutcomeLike &&
        !queryPlan.asksCoverageLike &&
        !queryPlan.asksSystemTimelineLike
    ) {
        return 0;
    }

    let delta = 0;

    if (roles.some((role) => queryPlan.preferredDocRoles.includes(role))) {
        delta += 0.22;
    }
    if (
        (queryPlan.asksOutcomeLike || queryPlan.difficultyTier === "high") &&
        roles.some((role) => queryPlan.avoidedDocRoles.includes(role))
    ) {
        delta -= 0.18;
    }

    if (queryPlan.asksCoverageLike) {
        if (roles.includes("stage_list") || roles.includes("rule_doc")) {
            delta += 0.1;
        }
        if (
            !queryPlan.asksOutcomeLike &&
            queryPlan.difficultyTier === "high" &&
            roles.includes("result_notice")
        ) {
            delta -= 0.08;
        }
    }

    if (queryPlan.intentType === "outcome" && roles.includes("result_notice")) {
        delta += 0.12;
    }
    if (
        queryPlan.intentType === "time_location" &&
        roles.includes("stage_list")
    ) {
        delta += 0.08;
    }
    if (
        queryPlan.intentType === "policy_overview" &&
        roles.includes("rule_doc")
    ) {
        delta += 0.08;
    }

    return delta;
}

export function computeTitleIntentDocDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { metadata } = entry;
    const { normalizedTitle } = metadata;
    if (!normalizedTitle) {
        return 0;
    }

    let delta = 0;

    if (!querySignals.asksOutcomeLikeTitle && metadata.isOutcomeTitle) {
        delta -= 0.95;
    }
    if (
        (querySignals.asksProcedureLikeTitle ||
            querySignals.asksRequirementLikeTitle) &&
        metadata.isRuleDocTitle
    ) {
        delta +=
            querySignals.asksProcedureLikeTitle &&
            querySignals.asksRequirementLikeTitle
                ? 0.95
                : 0.75;
    }
    if (querySignals.asksProcedureLikeTitle && metadata.isProcessNoticeTitle) {
        delta += 0.55;
    }

    if (querySignals.mentionsAiSchool) {
        if (metadata.isAiSchoolTitle) {
            delta += 0.45;
        } else if (metadata.isOtherProgramTitle) {
            delta -= 0.9;
        } else if (metadata.hasCollegeTitle) {
            delta -= 0.55;
        } else {
            delta -= 0.2;
        }
    }

    if (querySignals.mentionsDoctoral) {
        if (metadata.isMasterOnlyTitle) {
            delta -= 0.95;
        } else if (metadata.isDoctoralTitle) {
            delta += 0.2;
        }
    }

    if (querySignals.mentionsTuimian) {
        if (metadata.isTuimianTitle) {
            delta += 0.45;
        }
        if (metadata.isDoctoralTitle && !metadata.isTuimianTitle) {
            delta -= 0.6;
        }
    }

    if (querySignals.mentionsSummerCamp) {
        if (metadata.isSummerCampTitle) {
            delta += 0.35;
        }
        if (querySignals.asksCampExecutionDetail) {
            if (/入营通知/.test(normalizedTitle)) {
                delta += 0.95;
            }
            if (/活动报名通知|报名通知/.test(normalizedTitle)) {
                delta -= 0.75;
            }
        }
        if (
            !querySignals.asksEventDateLikeTitle &&
            /活动报名通知|报名通知/.test(normalizedTitle)
        ) {
            delta += 0.45;
        }
        if (
            !querySignals.asksOutcomeLikeTitle &&
            !querySignals.asksEventDateLikeTitle &&
            /入营通知/.test(normalizedTitle)
        ) {
            delta -= 0.45;
        }
        if (
            querySignals.asksEventDateLikeTitle &&
            /入营通知|活动通知/.test(normalizedTitle)
        ) {
            delta += 0.95;
        }
        if (
            querySignals.asksEventDateLikeTitle &&
            !querySignals.mentionsRegistration &&
            /活动报名通知|报名通知/.test(normalizedTitle)
        ) {
            delta -= 0.45;
        }
        if (
            !metadata.isSummerCampTitle &&
            (metadata.isDoctoralTitle ||
                metadata.isTuimianTitle ||
                metadata.isMasterOnlyTitle)
        ) {
            delta -= 0.65;
        }
    }

    if (querySignals.mentionsTuimian && /接收办法|工作方案/.test(normalizedTitle)) {
        delta += 0.45;
    }

    if (
        querySignals.mentionsDoctoral &&
        /实施办法|招生简章|综合考核通知/.test(normalizedTitle)
    ) {
        delta += 0.25;
    }

    if (querySignals.asksPolicyOverviewLikeTitle) {
        if (/招生简章|接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.55;
        }
        if (metadata.isPreapplyTitle) {
            delta -= 0.45;
        }
        if (metadata.isTransferTitle && !querySignals.mentionsTransfer) {
            delta -= 0.65;
        }
        if (metadata.isReviewResultTitle && !querySignals.asksOutcomeLikeTitle) {
            delta -= 0.55;
        }
    }

    if (querySignals.asksSystemTimelineLikeTitle) {
        if (/接收办法|实施办法/.test(normalizedTitle)) {
            delta += 0.45;
        }
        if (metadata.isPreapplyTitle && querySignals.asksSystemOperationLike) {
            delta -= 0.2;
        }
        if (metadata.isSystemNoticeTitle) {
            delta -= 0.55;
        }
    } else if (
        querySignals.asksTimelineNodeLike &&
        /报名通知|综合考核通知|复试通知/.test(normalizedTitle)
    ) {
        delta += 0.4;
    }

    if (querySignals.asksPostOutcomeAdmission) {
        if (/招生简章|实施办法/.test(normalizedTitle)) {
            delta += 0.4;
        }
        if (metadata.isReviewResultTitle) {
            delta -= 0.55;
        }
    }

    if (querySignals.asksMaterialReviewTiming) {
        if (metadata.isCandidateListTitle) {
            delta += 0.8;
        }
        if (/综合考核结果/.test(normalizedTitle)) {
            delta -= 0.35;
        }
    }

    if (querySignals.asksPostOutcomeOperationalDetail) {
        if (/复试结果|拟录取|增补拟录取|结果公示|名单公示/.test(normalizedTitle)) {
            delta += 1.25;
        }
        if (/增补拟录取|递补录取/.test(normalizedTitle)) {
            delta += 0.95;
        }
        if (/复试录取方案|调剂复试通知|招生简章|实施办法/.test(normalizedTitle)) {
            delta -= 1.15;
        }
    }

    if (
        querySignals.isCompressedKeywordQuery &&
        !querySignals.asksBroadRuleDocLikeTitle
    ) {
        if (!querySignals.asksCompressedConstraintLike && metadata.isRuleDocRole) {
            delta -= querySignals.asksCompressedOutcomeLike ? 0.95 : 0.55;
        }
        if (metadata.isSystemNoticeTitle) {
            delta -= 1.1;
        }
        if (metadata.isOtherProgramTitle) {
            delta -= 1.2;
        }
        if (
            querySignals.asksCompressedConstraintLike &&
            metadata.isConstraintRoleDoc
        ) {
            delta += querySignals.asksCompressedOutcomeLike ? 0.58 : 0.74;
        }
        if (
            querySignals.asksCompressedNoticeLike &&
            (metadata.isOperationalRoleDoc || metadata.isProcessNoticeTitle)
        ) {
            delta += querySignals.asksCompressedConstraintLike ? 0.34 : 0.48;
        }
        if (querySignals.asksCompressedOutcomeLike) {
            if (metadata.isOutcomeRoleDoc || metadata.isReviewResultTitle) {
                delta += querySignals.asksCompressedConstraintLike ? 0.28 : 0.56;
            }
            if (metadata.isStageListRole || metadata.isCandidateListTitle) {
                delta += querySignals.asksCompressedConstraintLike ? 0.18 : 0.32;
            }
        }
        if (
            querySignals.hasCompressedIntentCue &&
            !querySignals.asksCompressedConstraintLike &&
            metadata.isOperationalRoleDoc &&
            !metadata.isOutcomeRoleDoc
        ) {
            delta += 0.16;
        }
        if (
            querySignals.mentionsTuimian &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isTuimianTitle &&
                (metadata.isConstraintRoleDoc ||
                    metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc)
            ) {
                delta += 0.72;
            }
            if (metadata.isDoctoralTitle && !metadata.isTuimianTitle) {
                delta -= 0.88;
            }
        }
        if (
            querySignals.mentionsDoctoral &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isDoctoralTitle &&
                (metadata.isConstraintRoleDoc ||
                    metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc)
            ) {
                delta += 0.64;
            }
            if (metadata.isMasterOnlyTitle) {
                delta -= 0.82;
            }
        }
        if (
            querySignals.mentionsSummerCamp &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isSummerCampTitle &&
                (metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc ||
                    metadata.isStageListRole)
            ) {
                delta += 0.6;
            }
            if (
                !metadata.isSummerCampTitle &&
                (metadata.isDoctoralTitle ||
                    metadata.isTuimianTitle ||
                    metadata.isMasterOnlyTitle)
            ) {
                delta -= 0.58;
            }
        }
        if (
            querySignals.mentionsTransfer &&
            querySignals.hasCompressedThemeCue &&
            querySignals.hasCompressedIntentCue
        ) {
            if (
                metadata.isTransferTitle &&
                (metadata.isOperationalRoleDoc ||
                    metadata.isOutcomeRoleDoc ||
                    metadata.isAdjustmentNoticeRole)
            ) {
                delta += 0.58;
            }
        }
    }

    return delta;
}

function computeCoverageDocDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    const { requestedAspects } = querySignals;
    if (requestedAspects.length < 2) {
        return 0;
    }

    const evidenceText = entry.metadata.evidenceText;
    if (!evidenceText) {
        return -0.35;
    }

    const coveredCount = requestedAspects.filter((rule) =>
        rule.doc.test(evidenceText),
    ).length;

    if (coveredCount === 0) {
        return -0.4;
    }

    let delta = Math.min(coveredCount, 3) * 0.2;
    if (coveredCount === requestedAspects.length) {
        delta += 0.2;
    }
    return delta;
}

function computePhaseAnchorDocDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    if (!querySignals.hasExplicitPhaseAnchor) {
        return 0;
    }

    const queryPhase = querySignals.queryPhase;
    if (!entry.metadata.normalizedTitle) {
        return -0.15;
    }

    const articlePhase = entry.metadata.phaseAnchor;
    let delta = 0;

    if (queryPhase.half) {
        if (articlePhase.half === queryPhase.half) {
            delta += 0.9;
        } else if (articlePhase.half) {
            delta -= 0.9;
        } else {
            delta -= 0.2;
        }
    }

    if (queryPhase.batch) {
        if (articlePhase.batch === queryPhase.batch) {
            delta += 1.0;
        } else if (articlePhase.batch) {
            delta -= 1.0;
        } else {
            delta -= 0.2;
        }
    }

    if (queryPhase.stages.length > 0) {
        const hasExactStage = queryPhase.stages.some((stage) =>
            articlePhase.stages.includes(stage),
        );
        if (hasExactStage) {
            delta += 0.8;
        } else if (articlePhase.stages.length > 0) {
            delta -= 0.8;
        } else {
            delta -= 0.15;
        }
    }

    return delta;
}

export function applyPhaseAnchorBoostToDocuments(
    querySignals: DocumentRerankQuerySignals,
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    return sortDocumentRerankEntriesByDisplayScore(
        entries
            .map((entry) => {
                const delta =
                    computePhaseAnchorDocDelta(querySignals, entry) *
                    querySignals.phaseAnchorWeight;
                return updateDocumentRerankEntryScores(entry, delta, delta);
            }),
    );
}

export function applyCoverageTitleDiversity(
    entries: DocumentRerankEntry[],
): DocumentRerankEntry[] {
    const remaining = [...entries];
    const selected: DocumentRerankEntry[] = [];
    const seenTitleKeys = new Map<string, number>();

    while (remaining.length > 0) {
        let bestIndex = 0;
        let bestAdjustedScore = Number.NEGATIVE_INFINITY;

        remaining.forEach((candidate, index) => {
            const baseScore = getDocumentDisplayScore(candidate.document);
            const titleKey = candidate.metadata.titleDedupKey;
            const seenCount = titleKey ? (seenTitleKeys.get(titleKey) ?? 0) : 0;
            const adjustedScore =
                baseScore - seenCount * TITLE_DIVERSITY_DUPLICATE_PENALTY;

            if (adjustedScore > bestAdjustedScore) {
                bestAdjustedScore = adjustedScore;
                bestIndex = index;
            }
        });

        const chosen = remaining.splice(bestIndex, 1)[0];
        const titleKey = chosen.metadata.titleDedupKey;
        if (titleKey) {
            seenTitleKeys.set(titleKey, (seenTitleKeys.get(titleKey) ?? 0) + 1);
        }
        selected.push({
            document: {
                ...chosen.document,
                score: bestAdjustedScore,
                displayScore: bestAdjustedScore,
            },
            metadata: chosen.metadata,
        });
    }

    return selected;
}

export function computeCoverageWeightedDelta(
    querySignals: DocumentRerankQuerySignals,
    entry: DocumentRerankEntry,
): number {
    return computeCoverageDocDelta(querySignals, entry) * querySignals.coverageWeight;
}

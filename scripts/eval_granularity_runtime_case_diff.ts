import * as fs from "fs";
import * as path from "path";

import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    clonePipelinePreset,
    executeRetrievalStage,
    type PipelineDocumentRecord,
    type PipelinePreset,
    type SearchPipelineQueryContext,
} from "../src/worker/search_pipeline.ts";
import {
    buildPipelineDocumentLookup,
    collectUniqueFetchOtids,
    mergeCoarseMatchesWithDocumentLookup,
    queryIsCompressedKeywordLike,
    resolveDynamicFetchLimit,
    selectLimitedCoarseMatches,
} from "../src/worker/search_pipeline/document_rerank.ts";
import { buildDocumentRerankQuerySignals } from "../src/worker/search_pipeline/document_rerank_query.ts";
import {
    applyTitleIntentConfusionGate,
    applyCompressedQueryDisplayGuardToEntries,
    applyYearlessSameFamilyFreshnessGuardToEntries,
    applyCoverageTitleDiversity,
    applyPhaseAnchorBoostToDocuments,
    buildTitleIntentConfusionGate,
    computeCoverageWeightedDelta,
    computeLatestVersionDocDelta,
    computeQueryPlanDocRoleDelta,
    computeTitleIntentDocDelta,
} from "../src/worker/search_pipeline/document_rerank_scoring.ts";
import {
    buildDocumentRerankEntries,
    buildLatestVersionFamilyStats,
    type DocumentRerankEntry,
    getDocumentsFromRerankEntries,
    sortDocumentRerankEntriesByDisplayScore,
    updateDocumentRerankEntryScores,
} from "../src/worker/search_pipeline/document_rerank_shared.ts";
import {
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    type GranularityDatasetTargetKey,
    loadDatasetSources,
    resolveEvalDatasetConfig,
    type EvalDatasetCase,
    type EvalDatasetConfig,
    type EvalDatasetGroup,
} from "./eval_shared.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";

type OtidEvalMode =
    | "single_expected"
    | "acceptable_otids"
    | "required_otid_groups";

type DisplayRuleFlags = {
    enablePhaseAnchorBoost: boolean;
    enableTitleIntentAdjustments: boolean;
    enableTitleIntentConfusionGate?: boolean;
    enableStructuredQueryPlanDocRoleAdjustments?: boolean;
    enableStructuredKpRoleEvidenceAdjustments?: boolean;
    enableLexicalTitleIntentAdjustments: boolean;
    enableLexicalTitleTypeAdjustments: boolean;
    enableLexicalScenarioAdjustments?: boolean;
    enableThemeSpecificTitleAdjustments: boolean;
    enableDoctoralThemeTitleAdjustments: boolean;
    enableTuimianThemeTitleAdjustments: boolean;
    enableSummerCampThemeTitleAdjustments: boolean;
    enableTransferThemeTitleAdjustments: boolean;
    enableAiSchoolEntityTitleAdjustments: boolean;
    enableCompressedKeywordTitleAdjustments: boolean;
    enableCoverageAdjustments: boolean;
    enableLatestVersionRerank: boolean;
};

type VariantDefinition = {
    label: string;
    note: string;
    flags: DisplayRuleFlags;
};

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

type CandidateSummary = {
    otid?: string;
    title?: string;
    displayScore?: number;
    rerankScore?: number;
    coarseScore?: number;
    bestKpid?: string;
    eventTypes: string[];
    structuredTopicIds: string[];
    structuredIntentIds: string[];
    degreeLevels: string[];
    titleRoles: string[];
    bestKpRoleTags: string[];
    evidenceTopRoleTags: string[];
    kpEvidenceGroupCounts: Record<string, number>;
    dominantEvidenceGroup?: string;
    titleTypeFlags: string[];
    roleTypeFlags: string[];
};

type VariantCaseResult = {
    answered: boolean;
    latestTriggered: boolean;
    retrievalRank: number | null;
    renderedRank: number | null;
    renderedReciprocalRank: number;
    displayTop1Margin?: number;
    top1?: CandidateSummary;
    top3: CandidateSummary[];
};

type PreparedCase = {
    caseIndex: number;
    testCase: EvalDatasetCase;
    queryContext: SearchPipelineQueryContext;
    latestTriggered: boolean;
    answered: boolean;
    retrievalRank: number | null;
    directDocuments: PipelineDocumentRecord[];
};

type CaseDetail = {
    caseIndex: number;
    id?: string;
    dataset: string;
    query: string;
    sourceQuery?: string;
    sourceSeedId?: string;
    sourceDataset?: string;
    queryType?: string;
    queryScope?: string;
    preferredGranularity?: string;
    supportPattern?: string;
    themeFamily?: string;
    challengeTags: string[];
    expectedOtid: string;
    acceptableOtids?: string[];
    requiredOtidGroups?: string[][];
    retrievalDifficultyConstructScore?: number;
    retrievalCandidateConfusion?: number;
    retrievalTopicChainDensity?: number;
    retrievalQueryMargin?: number;
    retrievalTitleKpDistinctiveness?: number;
    runtimeFull: VariantCaseResult;
    compare: VariantCaseResult;
    reciprocalRankGain: number;
    rankGain?: number;
    hit1Recovered: boolean;
    hit1Lost: boolean;
    top1Changed: boolean;
};

type BucketCount = {
    key: string;
    count: number;
};

type DiffBucketSummary = {
    count: number;
    hit1RecoveredCount: number;
    hit1LostCount: number;
    avgReciprocalRankGain: number;
    avgDifficultyConstructScore?: number;
    avgCandidateConfusion?: number;
    avgTopicChainDensity?: number;
    avgQueryMargin?: number;
    avgTitleKpDistinctiveness?: number;
    byDataset: BucketCount[];
    byQueryScope: BucketCount[];
    byPreferredGranularity: BucketCount[];
    bySupportPattern: BucketCount[];
    byThemeFamily: BucketCount[];
    byDominantEvidenceGroup: BucketCount[];
    byPrimaryBestKpRoleTag: BucketCount[];
};

type PairwiseDiffReport = {
    baseLabel: string;
    compareLabel: string;
    betterSummary: DiffBucketSummary;
    worseSummary: DiffBucketSummary;
    betterCases: CaseDetail[];
    worseCases: CaseDetail[];
};

type SharedProtectionReport = {
    bothProtectedHit1Summary: DiffBucketSummary;
    structuredOnlyProtectedHit1Summary: DiffBucketSummary;
    lexicalOnlyProtectedHit1Summary: DiffBucketSummary;
    bothProtectedHit1Cases: CaseDetail[];
    structuredOnlyProtectedHit1Cases: CaseDetail[];
    lexicalOnlyProtectedHit1Cases: CaseDetail[];
};

type VariantMetricSummary = {
    total: number;
    answerRate: number;
    hitAt1: number;
    mrr: number;
};

type VariantAggregateReport = {
    label: string;
    note: string;
    metricsByDataset: Record<string, VariantMetricSummary>;
    combined: VariantMetricSummary;
};

type Report = {
    generatedAt: string;
    datasetBundle: string;
    datasetKey: string;
    datasetLabel: string;
    datasetGroups: Array<{
        key: string;
        label: string;
        role: string;
        size: number;
        sources: string[];
    }>;
    presetName: string;
    variants: VariantAggregateReport[];
    pairwiseDiffs: PairwiseDiffReport[];
    sharedHit1Protection: SharedProtectionReport;
};

const CURRENT_TIMESTAMP = Date.now() / 1000;
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const DATASET_BUNDLE = (
    process.env.SUASK_GRANULARITY_RUNTIME_CASE_DIFF_BUNDLE || "current_mainline"
)
    .trim()
    .toLowerCase();
const DATASET_VERSION = process.env.SUASK_EVAL_DATASET_VERSION || "granularity";
const DATASET_FILE = process.env.SUASK_EVAL_DATASET_FILE;
const DATASET_TARGET_KEY = (
    process.env.SUASK_EVAL_DATASET_TARGET_KEY ||
    process.env.SUASK_EVAL_DATASET_TARGET
) as GranularityDatasetTargetKey | undefined;
const SINGLE_FILE_AS_ALL = process.env.SUASK_EVAL_SINGLE_FILE_AS_ALL !== "0";

const BASE_PRESET: PipelinePreset = clonePipelinePreset(
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
);

const RUNTIME_FULL_FLAGS: DisplayRuleFlags = {
    enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
    enableTitleIntentAdjustments: BASE_PRESET.display.useYearPhaseTitleAdjustment,
    enableTitleIntentConfusionGate:
        BASE_PRESET.display.enableTitleIntentConfusionGate,
    enableStructuredQueryPlanDocRoleAdjustments:
        BASE_PRESET.display.enableStructuredQueryPlanDocRoleAdjustments,
    enableStructuredKpRoleEvidenceAdjustments:
        BASE_PRESET.display.enableStructuredKpRoleEvidenceAdjustments,
    enableLexicalTitleIntentAdjustments:
        BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
    enableLexicalTitleTypeAdjustments:
        BASE_PRESET.display.enableLexicalTitleTypeAdjustments,
    enableThemeSpecificTitleAdjustments:
        BASE_PRESET.display.enableThemeSpecificTitleAdjustments,
    enableDoctoralThemeTitleAdjustments:
        BASE_PRESET.display.enableDoctoralThemeTitleAdjustments,
    enableTuimianThemeTitleAdjustments:
        BASE_PRESET.display.enableTuimianThemeTitleAdjustments,
    enableSummerCampThemeTitleAdjustments:
        BASE_PRESET.display.enableSummerCampThemeTitleAdjustments,
    enableTransferThemeTitleAdjustments:
        BASE_PRESET.display.enableTransferThemeTitleAdjustments,
    enableAiSchoolEntityTitleAdjustments:
        BASE_PRESET.display.enableAiSchoolEntityTitleAdjustments,
    enableCompressedKeywordTitleAdjustments:
        BASE_PRESET.display.enableCompressedKeywordTitleAdjustments,
    enableCoverageAdjustments: BASE_PRESET.display.useYearPhaseTitleAdjustment,
    enableLatestVersionRerank: true,
};

const VARIANTS: VariantDefinition[] = [
    {
        label: "runtime_full",
        note: "Current frontend runtime display chain.",
        flags: { ...RUNTIME_FULL_FLAGS },
    },
    {
        label: "no_structured_kp_role_evidence_adjustment",
        note: "Disable structured KP-role evidence adjustment only.",
        flags: {
            ...RUNTIME_FULL_FLAGS,
            enableStructuredKpRoleEvidenceAdjustments: false,
        },
    },
    {
        label: "no_lexical_title_type_adjustment",
        note: "Disable coarse title-type lexical rules only.",
        flags: {
            ...RUNTIME_FULL_FLAGS,
            enableLexicalTitleTypeAdjustments: false,
        },
    },
] as const;

function round2(value: number): number {
    return Number(value.toFixed(2));
}

function round4(value: number): number {
    return Number(value.toFixed(4));
}

function safePercent(numerator: number, denominator: number): number {
    if (denominator <= 0) {
        return 0;
    }
    return round2((numerator / denominator) * 100);
}

function reciprocalRank(rank: number | null): number {
    return rank && Number.isFinite(rank) ? 1 / rank : 0;
}

function toNullableRank(rank: number): number | null {
    return Number.isFinite(rank) ? rank : null;
}

function getNumericField(
    testCase: EvalDatasetCase,
    fieldName: string,
): number | undefined {
    const value = (testCase as Record<string, unknown>)[fieldName];
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function resolveBlindOodDatasetTargetKey():
    | GranularityDatasetTargetKey
    | undefined {
    const bundleToTargetKey: Partial<
        Record<string, GranularityDatasetTargetKey>
    > = {
        blind_ext_ood_60: "ext_ood_blind_60",
        ext_ood_blind_60: "ext_ood_blind_60",
        blind_extood_60: "ext_ood_blind_60",
        blindextood60: "ext_ood_blind_60",
        hard_ood_blind_30: "hard_ood_blind_30",
        blind_hard_ood_30: "hard_ood_blind_30",
        blind_hardood_30: "hard_ood_blind_30",
        blindhardood30: "hard_ood_blind_30",
    };
    return bundleToTargetKey[DATASET_BUNDLE];
}

function resolveDatasetConfigForCaseDiff(): EvalDatasetConfig {
    if (DATASET_FILE || DATASET_TARGET_KEY) {
        return resolveEvalDatasetConfig({
            datasetVersion: DATASET_VERSION,
            datasetFile: DATASET_FILE,
            singleFileAsAll: SINGLE_FILE_AS_ALL,
            datasetTargetKey: DATASET_TARGET_KEY,
        });
    }

    if (DATASET_BUNDLE === "retrieval_matched_v1") {
        const groups: EvalDatasetGroup[] = [
            {
                key: "main_bench_120",
                label: "Main",
                role: "benchmark",
                sources: [
                    {
                        path: "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_benchmark_v2_reviewed_userized_v1.json",
                        datasetLabel: "main_bench_120",
                    },
                ],
            },
            {
                key: "in_domain_holdout_50",
                label: "InDomain",
                role: "in_domain_holdout",
                sources: [
                    {
                        path: "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_60_reviewed_userized_retrieval_matched_v1.json",
                        datasetLabel: "in_domain_holdout_50",
                    },
                ],
            },
            {
                key: "matched_ext_ood_60",
                label: "ExtOOD",
                role: "external_ood_holdout",
                sources: [
                    {
                        path: "../Backend/test/test_dataset_granularity/test_dataset_granularity_external_matched_ood_60_reviewed_userized_retrieval_matched_v1.json",
                        datasetLabel: "matched_ext_ood_60",
                    },
                ],
            },
        ];
        return {
            datasetVersion: "granularity",
            datasetMode: "named_group",
            datasetKey: "granularity_retrieval_matched_v1_bundle",
            datasetLabel: "Main+InDomain+ExtOOD(retrieval_matched_v1)",
            groups,
            tuneSources: groups.flatMap((group) => group.sources),
            holdoutSources: [],
            allSources: groups.flatMap((group) => group.sources),
        };
    }

    const blindTargetKey = resolveBlindOodDatasetTargetKey();
    return resolveEvalDatasetConfig({
        datasetVersion: DATASET_VERSION,
        datasetTargetKey: blindTargetKey,
    });
}

function parseRequiredOtidGroups(groups?: string[][]): string[][] {
    if (!Array.isArray(groups)) {
        return [];
    }

    return groups
        .map((group) =>
            Array.isArray(group)
                ? Array.from(
                      new Set(
                          group.filter(
                              (item): item is string =>
                                  typeof item === "string" && item.length > 0,
                          ),
                      ),
                  )
                : [],
        )
        .filter((group) => group.length > 0);
}

function resolveOtidEvalTarget(testCase: EvalDatasetCase): NormalizedOtidEvalTarget {
    const explicitRequiredGroups = parseRequiredOtidGroups(
        testCase.required_otid_groups,
    );
    const acceptableOtids = Array.from(
        new Set(
            [
                testCase.expected_otid,
                ...(Array.isArray(testCase.acceptable_otids)
                    ? testCase.acceptable_otids
                    : []),
            ].filter(
                (item): item is string =>
                    typeof item === "string" && item.length > 0,
            ),
        ),
    );

    const inferredMode: OtidEvalMode =
        testCase.otid_eval_mode ||
        (explicitRequiredGroups.length > 0
            ? "required_otid_groups"
            : acceptableOtids.length > 1
              ? "acceptable_otids"
              : "single_expected");

    const minGroupsCandidate = Number.isFinite(testCase.min_otid_groups_to_cover)
        ? Math.max(1, Number(testCase.min_otid_groups_to_cover))
        : explicitRequiredGroups.length > 0
          ? explicitRequiredGroups.length
          : acceptableOtids.length > 0
            ? 1
            : 0;

    return {
        mode: inferredMode,
        acceptableOtids,
        requiredOtidGroups:
            inferredMode === "required_otid_groups" ? explicitRequiredGroups : [],
        minGroupsToCover:
            inferredMode === "required_otid_groups"
                ? Math.min(
                      minGroupsCandidate,
                      Math.max(explicitRequiredGroups.length, 1),
                  )
                : minGroupsCandidate,
    };
}

function getBestRankForOtidSet(
    matches: readonly { otid?: string }[],
    acceptableOtids: readonly string[],
): number {
    if (acceptableOtids.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const acceptableOtidSet = new Set(acceptableOtids);
    const rankIndex = matches.findIndex(
        (item) => item.otid && acceptableOtidSet.has(item.otid),
    );
    return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
}

function computeCoverageDepth(
    requiredGroups: readonly string[][],
    minGroupsToCover: number,
    matches: readonly { otid?: string }[],
): number {
    if (requiredGroups.length === 0) {
        return Number.POSITIVE_INFINITY;
    }

    const groupDepths = requiredGroups.map((group) => {
        const groupSet = new Set(group);
        const rankIndex = matches.findIndex(
            (match) => match.otid && groupSet.has(match.otid),
        );
        return rankIndex === -1 ? Number.POSITIVE_INFINITY : rankIndex + 1;
    });

    const sortedDepths = groupDepths
        .filter((depth) => Number.isFinite(depth))
        .sort((left, right) => left - right);
    const requiredCount = Math.max(
        1,
        Math.min(minGroupsToCover || requiredGroups.length, requiredGroups.length),
    );

    return sortedDepths.length >= requiredCount
        ? sortedDepths[requiredCount - 1]
        : Number.POSITIVE_INFINITY;
}

function getRankForCase(
    matches: readonly { otid?: string }[],
    testCase: EvalDatasetCase,
): number {
    const target = resolveOtidEvalTarget(testCase);
    if (target.mode === "required_otid_groups") {
        return computeCoverageDepth(
            target.requiredOtidGroups,
            target.minGroupsToCover,
            matches,
        );
    }

    return getBestRankForOtidSet(matches, target.acceptableOtids);
}

function buildDisplayVariantEntries(params: {
    query: string;
    documents: PipelineDocumentRecord[];
    queryContext: SearchPipelineQueryContext;
    preferLatestWithinTopic: boolean;
    flags: DisplayRuleFlags;
}): DocumentRerankEntry[] {
    const {
        query,
        documents,
        queryContext,
        preferLatestWithinTopic,
        flags,
    } = params;
    const effectiveLatestPreference =
        flags.enableLatestVersionRerank && preferLatestWithinTopic;
    const plannerQueryPlan = BASE_PRESET.display.enableQueryPlanner
        ? queryContext.queryPlan
        : undefined;
    const roleAlignmentQueryPlan =
        (flags.enableStructuredQueryPlanDocRoleAdjustments ??
            BASE_PRESET.display.enableStructuredQueryPlanDocRoleAdjustments)
            ? queryContext.queryPlan
            : plannerQueryPlan;
    const enableStructuredKpRoleEvidenceAdjustments =
        flags.enableStructuredKpRoleEvidenceAdjustments ??
        BASE_PRESET.display.enableStructuredKpRoleEvidenceAdjustments;
    const enableTitleIntentConfusionGate =
        flags.enableTitleIntentConfusionGate ??
        BASE_PRESET.display.enableTitleIntentConfusionGate;
    const querySignals = buildDocumentRerankQuerySignals({
        query,
        queryIntent: queryContext.queryIntent,
        queryPlan: plannerQueryPlan,
        preferLatestWithinTopic: effectiveLatestPreference,
    });
    const documentEntries = buildDocumentRerankEntries(documents);
    const phaseAdjustedEntries = flags.enablePhaseAnchorBoost
        ? applyPhaseAnchorBoostToDocuments(querySignals, documentEntries)
        : documentEntries;
    const shouldApplyLatestVersionBoost =
        flags.enableLatestVersionRerank &&
        querySignals.wantsLatestVersion &&
        phaseAdjustedEntries.length > 1;
    const latestVersionFamilyStats = shouldApplyLatestVersionBoost
        ? buildLatestVersionFamilyStats(phaseAdjustedEntries)
        : undefined;
    const titleIntentConfusionGate = enableTitleIntentConfusionGate
        ? buildTitleIntentConfusionGate(querySignals, phaseAdjustedEntries)
        : undefined;
    const rerankedEntries = sortDocumentRerankEntriesByDisplayScore(
        phaseAdjustedEntries.map((entry) => {
            const titleDelta = flags.enableTitleIntentAdjustments
                ? (applyTitleIntentConfusionGate(
                      computeTitleIntentDocDelta(querySignals, entry, {
                          enableStructuredKpRoleEvidenceAdjustments,
                          enableLexicalTitleIntentAdjustments:
                              flags.enableLexicalTitleIntentAdjustments,
                          enableLexicalTitleTypeAdjustments:
                              flags.enableLexicalTitleTypeAdjustments,
                          enableLexicalScenarioAdjustments:
                              flags.enableLexicalScenarioAdjustments,
                          enableThemeSpecificAdjustments:
                              flags.enableThemeSpecificTitleAdjustments,
                          enableDoctoralThemeAdjustments:
                              flags.enableDoctoralThemeTitleAdjustments,
                          enableTuimianThemeAdjustments:
                              flags.enableTuimianThemeTitleAdjustments,
                          enableSummerCampThemeAdjustments:
                              flags.enableSummerCampThemeTitleAdjustments,
                          enableTransferThemeAdjustments:
                              flags.enableTransferThemeTitleAdjustments,
                          enableAiSchoolEntityAdjustments:
                              flags.enableAiSchoolEntityTitleAdjustments,
                          enableCompressedKeywordAdjustments:
                              flags.enableCompressedKeywordTitleAdjustments,
                      }),
                      titleIntentConfusionGate,
                  ) +
                      applyTitleIntentConfusionGate(
                          roleAlignmentQueryPlan
                              ? computeQueryPlanDocRoleDelta(
                                    entry.metadata.roles,
                                    roleAlignmentQueryPlan,
                                )
                              : 0,
                          titleIntentConfusionGate,
                      )) * querySignals.titleIntentWeight
                : 0;
            const latestDelta =
                shouldApplyLatestVersionBoost && latestVersionFamilyStats
                    ? computeLatestVersionDocDelta({
                          entry,
                          familyStats: latestVersionFamilyStats,
                          querySignals,
                      }) * querySignals.latestVersionWeight
                    : 0;
            const coverageDelta = flags.enableCoverageAdjustments
                ? computeCoverageWeightedDelta(querySignals, entry)
                : 0;

            return updateDocumentRerankEntryScores(
                entry,
                titleDelta + latestDelta + coverageDelta,
                titleDelta,
            );
        }),
    );
    const displayEntries =
        querySignals.wantsCoverageDiversity && flags.enableCoverageAdjustments
            ? applyCoverageTitleDiversity(rerankedEntries)
            : rerankedEntries;
    return applyCompressedQueryDisplayGuardToEntries(
        querySignals,
        phaseAdjustedEntries,
        applyYearlessSameFamilyFreshnessGuardToEntries(
            querySignals,
            phaseAdjustedEntries,
            displayEntries,
        ),
    );
}

function resolveDominantEvidenceGroup(
    counts: Record<string, number>,
): string | undefined {
    const entries = Object.entries(counts)
        .filter(([, count]) => count > 0)
        .sort((left, right) => {
            if (right[1] !== left[1]) {
                return right[1] - left[1];
            }
            return left[0].localeCompare(right[0], "zh-Hans-CN");
        });
    return entries[0]?.[0];
}

function summarizeEntry(entry: DocumentRerankEntry): CandidateSummary {
    const titleTypeFlags: string[] = [];
    if (entry.metadata.isRuleDocTitle) {
        titleTypeFlags.push("rule");
    }
    if (entry.metadata.isProcessNoticeTitle) {
        titleTypeFlags.push("process");
    }
    if (entry.metadata.isOutcomeTitle) {
        titleTypeFlags.push("outcome");
    }

    const roleTypeFlags: string[] = [];
    if (entry.metadata.isConstraintRoleDoc) {
        roleTypeFlags.push("constraint");
    }
    if (entry.metadata.isOperationalRoleDoc) {
        roleTypeFlags.push("operational");
    }
    if (entry.metadata.isOutcomeRoleDoc) {
        roleTypeFlags.push("outcome");
    }

    return {
        otid: entry.document.otid,
        title: entry.document.ot_title,
        displayScore:
            typeof entry.document.displayScore === "number"
                ? round4(entry.document.displayScore)
                : undefined,
        rerankScore:
            typeof entry.document.rerankScore === "number"
                ? round4(entry.document.rerankScore)
                : undefined,
        coarseScore:
            typeof entry.document.coarseScore === "number"
                ? round4(entry.document.coarseScore)
                : typeof entry.document.score === "number"
                  ? round4(entry.document.score)
                  : undefined,
        bestKpid: entry.document.best_kpid,
        eventTypes: entry.metadata.structuredEventTypes,
        structuredTopicIds: entry.metadata.structuredTopicIds,
        structuredIntentIds: entry.metadata.structuredIntentIds,
        degreeLevels: entry.metadata.structuredDegreeLevels,
        titleRoles: entry.metadata.roles,
        bestKpRoleTags: entry.metadata.bestKpRoleTags,
        evidenceTopRoleTags: entry.metadata.evidenceTopRoleTags,
        kpEvidenceGroupCounts: entry.metadata.kpEvidenceGroupCounts,
        dominantEvidenceGroup: resolveDominantEvidenceGroup(
            entry.metadata.kpEvidenceGroupCounts,
        ),
        titleTypeFlags,
        roleTypeFlags,
    };
}

function buildVariantCaseResult(params: {
    preparedCase: PreparedCase;
    variant: VariantDefinition;
}): VariantCaseResult {
    const { preparedCase, variant } = params;
    if (!preparedCase.answered || preparedCase.directDocuments.length === 0) {
        return {
            answered: preparedCase.answered,
            latestTriggered: preparedCase.latestTriggered,
            retrievalRank: preparedCase.retrievalRank,
            renderedRank: null,
            renderedReciprocalRank: 0,
            top3: [],
        };
    }

    const displayEntries = buildDisplayVariantEntries({
        query: preparedCase.testCase.query,
        documents: preparedCase.directDocuments,
        queryContext: preparedCase.queryContext,
        preferLatestWithinTopic: preparedCase.latestTriggered,
        flags: variant.flags,
    });
    const renderedResults = getDocumentsFromRerankEntries(displayEntries);
    const renderedRank = getRankForCase(renderedResults, preparedCase.testCase);
    const top3Entries = displayEntries.slice(0, 3);
    const top1 = top3Entries[0];
    const second = top3Entries[1];

    return {
        answered: preparedCase.answered,
        latestTriggered: preparedCase.latestTriggered,
        retrievalRank: preparedCase.retrievalRank,
        renderedRank: toNullableRank(renderedRank),
        renderedReciprocalRank: round4(reciprocalRank(toNullableRank(renderedRank))),
        displayTop1Margin:
            top1 && second
                ? round4(
                      (top1.document.displayScore || 0) -
                          (second.document.displayScore || 0),
                  )
                : undefined,
        top1: top1 ? summarizeEntry(top1) : undefined,
        top3: top3Entries.map(summarizeEntry),
    };
}

function pushCount(map: Map<string, number>, key?: string): void {
    const normalizedKey = key?.trim() || "unknown";
    map.set(normalizedKey, (map.get(normalizedKey) || 0) + 1);
}

function finalizeCountMap(map: Map<string, number>, limit = 10): BucketCount[] {
    return Array.from(map.entries())
        .sort((left, right) => {
            if (right[1] !== left[1]) {
                return right[1] - left[1];
            }
            return left[0].localeCompare(right[0], "zh-Hans-CN");
        })
        .slice(0, limit)
        .map(([key, count]) => ({ key, count }));
}

function avg(values: number[]): number | undefined {
    if (values.length === 0) {
        return undefined;
    }
    return round4(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function buildDiffBucketSummary(cases: CaseDetail[]): DiffBucketSummary {
    const byDataset = new Map<string, number>();
    const byQueryScope = new Map<string, number>();
    const byPreferredGranularity = new Map<string, number>();
    const bySupportPattern = new Map<string, number>();
    const byThemeFamily = new Map<string, number>();
    const byDominantEvidenceGroup = new Map<string, number>();
    const byPrimaryBestKpRoleTag = new Map<string, number>();

    const difficultyValues: number[] = [];
    const confusionValues: number[] = [];
    const chainDensityValues: number[] = [];
    const queryMarginValues: number[] = [];
    const distinctivenessValues: number[] = [];

    cases.forEach((item) => {
        pushCount(byDataset, item.dataset);
        pushCount(byQueryScope, item.queryScope);
        pushCount(byPreferredGranularity, item.preferredGranularity);
        pushCount(bySupportPattern, item.supportPattern);
        pushCount(byThemeFamily, item.themeFamily);
        pushCount(
            byDominantEvidenceGroup,
            item.runtimeFull.top1?.dominantEvidenceGroup,
        );
        pushCount(
            byPrimaryBestKpRoleTag,
            item.runtimeFull.top1?.bestKpRoleTags[0] ||
                item.runtimeFull.top1?.evidenceTopRoleTags[0],
        );

        if (typeof item.retrievalDifficultyConstructScore === "number") {
            difficultyValues.push(item.retrievalDifficultyConstructScore);
        }
        if (typeof item.retrievalCandidateConfusion === "number") {
            confusionValues.push(item.retrievalCandidateConfusion);
        }
        if (typeof item.retrievalTopicChainDensity === "number") {
            chainDensityValues.push(item.retrievalTopicChainDensity);
        }
        if (typeof item.retrievalQueryMargin === "number") {
            queryMarginValues.push(item.retrievalQueryMargin);
        }
        if (typeof item.retrievalTitleKpDistinctiveness === "number") {
            distinctivenessValues.push(item.retrievalTitleKpDistinctiveness);
        }
    });

    return {
        count: cases.length,
        hit1RecoveredCount: cases.filter((item) => item.hit1Recovered).length,
        hit1LostCount: cases.filter((item) => item.hit1Lost).length,
        avgReciprocalRankGain: round4(
            cases.reduce((sum, item) => sum + item.reciprocalRankGain, 0) /
                Math.max(cases.length, 1),
        ),
        avgDifficultyConstructScore: avg(difficultyValues),
        avgCandidateConfusion: avg(confusionValues),
        avgTopicChainDensity: avg(chainDensityValues),
        avgQueryMargin: avg(queryMarginValues),
        avgTitleKpDistinctiveness: avg(distinctivenessValues),
        byDataset: finalizeCountMap(byDataset),
        byQueryScope: finalizeCountMap(byQueryScope),
        byPreferredGranularity: finalizeCountMap(byPreferredGranularity),
        bySupportPattern: finalizeCountMap(bySupportPattern),
        byThemeFamily: finalizeCountMap(byThemeFamily),
        byDominantEvidenceGroup: finalizeCountMap(byDominantEvidenceGroup),
        byPrimaryBestKpRoleTag: finalizeCountMap(byPrimaryBestKpRoleTag),
    };
}

function buildCaseDetail(params: {
    preparedCase: PreparedCase;
    runtimeFull: VariantCaseResult;
    compare: VariantCaseResult;
}): CaseDetail {
    const { preparedCase, runtimeFull, compare } = params;
    const testCase = preparedCase.testCase;
    const runtimeRank = runtimeFull.renderedRank;
    const compareRank = compare.renderedRank;
    const reciprocalRankGain = round4(
        runtimeFull.renderedReciprocalRank - compare.renderedReciprocalRank,
    );

    let rankGain: number | undefined;
    if (runtimeRank && compareRank) {
        rankGain = compareRank - runtimeRank;
    } else if (runtimeRank && !compareRank) {
        rankGain = 999;
    } else if (!runtimeRank && compareRank) {
        rankGain = -999;
    }

    return {
        caseIndex: preparedCase.caseIndex,
        id: (testCase as Record<string, unknown>).id as string | undefined,
        dataset: testCase.dataset,
        query: testCase.query,
        sourceQuery: testCase.source_query,
        sourceSeedId: testCase.source_seed_id,
        sourceDataset: testCase.source_dataset,
        queryType: testCase.query_type,
        queryScope: testCase.query_scope,
        preferredGranularity: testCase.preferred_granularity,
        supportPattern: testCase.support_pattern,
        themeFamily: testCase.theme_family,
        challengeTags: Array.isArray(testCase.challenge_tags)
            ? testCase.challenge_tags
            : [],
        expectedOtid: testCase.expected_otid,
        acceptableOtids: testCase.acceptable_otids,
        requiredOtidGroups: testCase.required_otid_groups,
        retrievalDifficultyConstructScore: getNumericField(
            testCase,
            "retrieval_difficulty_construct_score_v1",
        ),
        retrievalCandidateConfusion: getNumericField(
            testCase,
            "retrieval_candidate_confusion_raw_v1",
        ),
        retrievalTopicChainDensity: getNumericField(
            testCase,
            "retrieval_topic_chain_density_raw_v1",
        ),
        retrievalQueryMargin: getNumericField(
            testCase,
            "retrieval_query_margin_raw_v1",
        ),
        retrievalTitleKpDistinctiveness: getNumericField(
            testCase,
            "retrieval_title_kp_distinctiveness_raw_v1",
        ),
        runtimeFull,
        compare,
        reciprocalRankGain,
        rankGain,
        hit1Recovered: runtimeRank === 1 && compareRank !== 1,
        hit1Lost: runtimeRank !== 1 && compareRank === 1,
        top1Changed:
            (runtimeFull.top1?.otid || "") !== (compare.top1?.otid || "") ||
            (runtimeFull.top1?.bestKpid || "") !== (compare.top1?.bestKpid || ""),
    };
}

function compareCaseDetail(left: CaseDetail, right: CaseDetail): number {
    if (right.hit1Recovered !== left.hit1Recovered) {
        return Number(right.hit1Recovered) - Number(left.hit1Recovered);
    }
    if (left.hit1Lost !== right.hit1Lost) {
        return Number(left.hit1Lost) - Number(right.hit1Lost);
    }
    if (right.reciprocalRankGain !== left.reciprocalRankGain) {
        return right.reciprocalRankGain - left.reciprocalRankGain;
    }
    if ((right.rankGain || 0) !== (left.rankGain || 0)) {
        return (right.rankGain || 0) - (left.rankGain || 0);
    }
    return left.caseIndex - right.caseIndex;
}

function buildPairwiseDiffReport(params: {
    baseLabel: string;
    compareLabel: string;
    caseResultsByVariant: Map<string, VariantCaseResult[]>;
    preparedCases: PreparedCase[];
}): PairwiseDiffReport {
    const { baseLabel, compareLabel, caseResultsByVariant, preparedCases } = params;
    const baseCases = caseResultsByVariant.get(baseLabel) || [];
    const compareCases = caseResultsByVariant.get(compareLabel) || [];
    const betterCases: CaseDetail[] = [];
    const worseCases: CaseDetail[] = [];

    preparedCases.forEach((preparedCase, index) => {
        const base = baseCases[index];
        const compare = compareCases[index];
        if (!base || !compare) {
            return;
        }
        const detail = buildCaseDetail({
            preparedCase,
            runtimeFull: base,
            compare,
        });
        if (detail.reciprocalRankGain > 0) {
            betterCases.push(detail);
        } else if (detail.reciprocalRankGain < 0) {
            worseCases.push(detail);
        }
    });

    betterCases.sort(compareCaseDetail);
    worseCases.sort(compareCaseDetail);

    return {
        baseLabel,
        compareLabel,
        betterSummary: buildDiffBucketSummary(betterCases),
        worseSummary: buildDiffBucketSummary(worseCases),
        betterCases,
        worseCases,
    };
}

function buildSharedProtectionReport(params: {
    preparedCases: PreparedCase[];
    caseResultsByVariant: Map<string, VariantCaseResult[]>;
}): SharedProtectionReport {
    const { preparedCases, caseResultsByVariant } = params;
    const runtimeFull = caseResultsByVariant.get("runtime_full") || [];
    const noStructured =
        caseResultsByVariant.get("no_structured_kp_role_evidence_adjustment") || [];
    const noLexical =
        caseResultsByVariant.get("no_lexical_title_type_adjustment") || [];

    const bothProtected: CaseDetail[] = [];
    const structuredOnly: CaseDetail[] = [];
    const lexicalOnly: CaseDetail[] = [];

    preparedCases.forEach((preparedCase, index) => {
        const base = runtimeFull[index];
        const structured = noStructured[index];
        const lexical = noLexical[index];
        if (!base || !structured || !lexical || base.renderedRank !== 1) {
            return;
        }

        const structuredProtected = structured.renderedRank !== 1;
        const lexicalProtected = lexical.renderedRank !== 1;
        if (!structuredProtected && !lexicalProtected) {
            return;
        }

        if (structuredProtected && lexicalProtected) {
            bothProtected.push(
                buildCaseDetail({
                    preparedCase,
                    runtimeFull: base,
                    compare: lexical.renderedReciprocalRank <= structured.renderedReciprocalRank
                        ? lexical
                        : structured,
                }),
            );
            return;
        }

        if (structuredProtected) {
            structuredOnly.push(
                buildCaseDetail({
                    preparedCase,
                    runtimeFull: base,
                    compare: structured,
                }),
            );
            return;
        }

        lexicalOnly.push(
            buildCaseDetail({
                preparedCase,
                runtimeFull: base,
                compare: lexical,
            }),
        );
    });

    bothProtected.sort(compareCaseDetail);
    structuredOnly.sort(compareCaseDetail);
    lexicalOnly.sort(compareCaseDetail);

    return {
        bothProtectedHit1Summary: buildDiffBucketSummary(bothProtected),
        structuredOnlyProtectedHit1Summary:
            buildDiffBucketSummary(structuredOnly),
        lexicalOnlyProtectedHit1Summary: buildDiffBucketSummary(lexicalOnly),
        bothProtectedHit1Cases: bothProtected,
        structuredOnlyProtectedHit1Cases: structuredOnly,
        lexicalOnlyProtectedHit1Cases: lexicalOnly,
    };
}

function buildVariantAggregateReport(params: {
    variant: VariantDefinition;
    preparedCases: PreparedCase[];
    results: VariantCaseResult[];
}): VariantAggregateReport {
    const { variant, preparedCases, results } = params;
    const metricsByDataset = new Map<
        string,
        { total: number; answered: number; hitAt1: number; mrrSum: number }
    >();
    const combined = {
        total: 0,
        answered: 0,
        hitAt1: 0,
        mrrSum: 0,
    };

    preparedCases.forEach((preparedCase, index) => {
        const result = results[index];
        if (!result) {
            return;
        }
        const datasetKey = preparedCase.testCase.dataset || "evaluation";
        const bucket = metricsByDataset.get(datasetKey) || {
            total: 0,
            answered: 0,
            hitAt1: 0,
            mrrSum: 0,
        };
        bucket.total += 1;
        combined.total += 1;
        if (result.answered) {
            bucket.answered += 1;
            combined.answered += 1;
        }
        if (result.renderedRank === 1) {
            bucket.hitAt1 += 1;
            combined.hitAt1 += 1;
        }
        bucket.mrrSum += result.renderedReciprocalRank;
        combined.mrrSum += result.renderedReciprocalRank;
        metricsByDataset.set(datasetKey, bucket);
    });

    return {
        label: variant.label,
        note: variant.note,
        metricsByDataset: Object.fromEntries(
            Array.from(metricsByDataset.entries()).map(([datasetKey, bucket]) => [
                datasetKey,
                {
                    total: bucket.total,
                    answerRate: safePercent(bucket.answered, bucket.total),
                    hitAt1: safePercent(bucket.hitAt1, bucket.total),
                    mrr:
                        bucket.total > 0
                            ? round4(bucket.mrrSum / bucket.total)
                            : 0,
                },
            ]),
        ),
        combined: {
            total: combined.total,
            answerRate: safePercent(combined.answered, combined.total),
            hitAt1: safePercent(combined.hitAt1, combined.total),
            mrr:
                combined.total > 0
                    ? round4(combined.mrrSum / combined.total)
                    : 0,
        },
    };
}

async function prepareCases(params: {
    datasetCases: EvalDatasetCase[];
    engine: Awaited<ReturnType<typeof loadFrontendEvalEngine>>;
    queryVectors: Float32Array[];
}): Promise<PreparedCase[]> {
    const { datasetCases, engine, queryVectors } = params;
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();
    const preparedCases: PreparedCase[] = [];

    for (let index = 0; index < datasetCases.length; index += 1) {
        const testCase = datasetCases[index];
        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            engine.vocabMap,
            engine.topicPartitionIndex,
            BASE_PRESET,
        );
        const retrievalStage = executeRetrievalStage({
            query: testCase.query,
            queryVector: queryVectors[index],
            queryContext,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: engine.bm25Stats,
            termMaps,
            preset: BASE_PRESET,
        });
        const latestTriggered =
            retrievalStage.searchOutput.responseDecision?.preferLatestWithinTopic ??
            false;
        const retrievalRank = toNullableRank(
            getRankForCase(retrievalStage.searchOutput.matches, testCase),
        );
        const plannerQueryPlan = BASE_PRESET.display.enableQueryPlanner
            ? queryContext.queryPlan
            : undefined;
        const isCompressedKeywordQuery = queryIsCompressedKeywordLike(
            testCase.query,
        );
        const compressedQueryFetchDelta = isCompressedKeywordQuery ? 18 : 0;
        const fetchMatchLimit = resolveDynamicFetchLimit(
            BASE_PRESET.display.fetchMatchLimit,
            (plannerQueryPlan?.fetchMatchLimitDelta ?? 0) +
                compressedQueryFetchDelta,
            isCompressedKeywordQuery ? 48 : 28,
        );
        const answerCoarseMatches =
            retrievalStage.retrievalDecision.behavior === "answer"
                ? selectLimitedCoarseMatches(
                      retrievalStage.searchOutput.matches,
                      fetchMatchLimit,
                  )
                : [];
        const fetchIds = collectUniqueFetchOtids(answerCoarseMatches, []);
        let directDocuments: PipelineDocumentRecord[] = [];

        if (fetchIds.length > 0) {
            const documents = await documentLoader({
                query: testCase.query,
                otids: fetchIds,
            });
            const fetchedDocumentLookup = buildPipelineDocumentLookup(documents);
            directDocuments = mergeCoarseMatchesWithDocumentLookup(
                fetchedDocumentLookup,
                answerCoarseMatches,
            );
        }

        preparedCases.push({
            caseIndex: index,
            testCase,
            queryContext,
            latestTriggered,
            answered: retrievalStage.retrievalDecision.behavior === "answer",
            retrievalRank,
            directDocuments,
        });

        if ((index + 1) % 40 === 0 || index + 1 === datasetCases.length) {
            console.log(`[prepare] ${index + 1} / ${datasetCases.length}`);
        }
    }

    return preparedCases;
}

async function main() {
    const datasetConfig = resolveDatasetConfigForCaseDiff();
    const datasetCases = loadDatasetSources(datasetConfig.allSources);
    if (datasetCases.length === 0) {
        throw new Error(`Dataset is empty for ${datasetConfig.datasetKey}.`);
    }

    const engine = await loadFrontendEvalEngine();
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        datasetCases.map((item) => item.query),
        engine.dimensions,
        {
            batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE,
            onProgress: (done, total) => {
                if (done === total || done % 40 === 0) {
                    console.log(`[embed] ${done} / ${total}`);
                }
            },
        },
    );

    const preparedCases = await prepareCases({
        datasetCases,
        engine,
        queryVectors,
    });

    const caseResultsByVariant = new Map<string, VariantCaseResult[]>();
    const variantAggregateReports: VariantAggregateReport[] = [];

    for (const variant of VARIANTS) {
        const results: VariantCaseResult[] = preparedCases.map((preparedCase) =>
            buildVariantCaseResult({
                preparedCase,
                variant,
            }),
        );
        caseResultsByVariant.set(variant.label, results);
        variantAggregateReports.push(
            buildVariantAggregateReport({
                variant,
                preparedCases,
                results,
            }),
        );
        console.log(
            [
                `[${variant.label}]`,
                `hit@1=${variantAggregateReports.at(-1)?.combined.hitAt1.toFixed(2)}%`,
                `mrr=${variantAggregateReports.at(-1)?.combined.mrr.toFixed(4)}`,
            ].join(" "),
        );
    }

    const pairwiseDiffs = [
        buildPairwiseDiffReport({
            baseLabel: "runtime_full",
            compareLabel: "no_structured_kp_role_evidence_adjustment",
            caseResultsByVariant,
            preparedCases,
        }),
        buildPairwiseDiffReport({
            baseLabel: "runtime_full",
            compareLabel: "no_lexical_title_type_adjustment",
            caseResultsByVariant,
            preparedCases,
        }),
    ];

    const groupSizes = Object.fromEntries(
        datasetConfig.groups.map((group) => [
            group.key,
            loadDatasetSources(group.sources).length,
        ]),
    );

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetBundle: DATASET_BUNDLE,
        datasetKey: datasetConfig.datasetKey,
        datasetLabel: datasetConfig.datasetLabel,
        datasetGroups: datasetConfig.groups.map((group) => ({
            key: group.key,
            label: group.label,
            role: group.role,
            size: groupSizes[group.key] || 0,
            sources: group.sources.map((source) => source.path),
        })),
        presetName: BASE_PRESET.name,
        variants: variantAggregateReports,
        pairwiseDiffs,
        sharedHit1Protection: buildSharedProtectionReport({
            preparedCases,
            caseResultsByVariant,
        }),
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        `granularity_runtime_case_diff_${datasetConfig.datasetKey}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`Saved report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

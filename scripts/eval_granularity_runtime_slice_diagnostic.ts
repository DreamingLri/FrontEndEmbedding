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

type SliceTags = {
    dataset: string;
    themeFamily: string;
    yearBucket: string;
    entityBucket: string;
    queryScope: string;
    queryType: string;
    supportPattern: string;
    anchorBucket: string;
    docRole: string;
    nearNeighborLevel: string;
    preferredGranularity: string;
};

type PreparedCase = {
    caseIndex: number;
    testCase: EvalDatasetCase;
    queryContext: SearchPipelineQueryContext;
    hasExplicitYearAnchor: boolean;
    latestTriggered: boolean;
    answered: boolean;
    retrievalRank: number | null;
    directDocuments: PipelineDocumentRecord[];
    sliceTags: SliceTags;
};

type VariantCaseResult = {
    answered: boolean;
    retrievalRank: number | null;
    renderedRank: number | null;
    renderedReciprocalRank: number;
};

type SliceMetricSummary = {
    total: number;
    answerRate: number;
    retrievalHitAt1: number;
    renderedHitAt1: number;
    displayLiftHitAt1: number;
    retrievalMRR: number;
    renderedMRR: number;
    displayLiftMRR: number;
    rescuedHit1Count: number;
    lostHit1Count: number;
    netHit1GainCount: number;
    explicitYearRescuedHit1Count: number;
    explicitYearLostHit1Count: number;
    explicitYearNetHit1GainCount: number;
    anchorNeutralRescuedHit1Count: number;
    anchorNeutralLostHit1Count: number;
    anchorNeutralNetHit1GainCount: number;
};

type SliceBucketReport = {
    key: string;
    metrics: SliceMetricSummary;
};

type SliceCategoryReport = {
    dataset: string;
    category:
        | "theme_family"
        | "query_year"
        | "entity_mention"
        | "query_scope"
        | "query_type"
        | "support_pattern"
        | "anchor_bucket"
        | "doc_role"
        | "near_neighbor_level"
        | "preferred_granularity";
    minCount: number;
    buckets: SliceBucketReport[];
};

type HotspotEntry = {
    dataset: string;
    category: string;
    key: string;
    metrics: SliceMetricSummary;
};

type GuardAlert = {
    severity: "warn";
    code:
        | "main_lift_gap_high"
        | "main_net_gain_gap_high"
        | "main_small_bucket_spike";
    message: string;
    evidence: Record<string, number | string>;
};

type GuardSummary = {
    status: "pass" | "warn";
    thresholds: {
        mainLiftGapWarnThreshold: number;
        mainNetGainGapWarnThreshold: number;
        smallBucketMaxCount: number;
        smallBucketLiftWarnThreshold: number;
    };
    alerts: GuardAlert[];
};

type Report = {
    generatedAt: string;
    datasetBundle: string;
    datasetKey: string;
    datasetLabel: string;
    presetName: string;
    runtimeFlags: DisplayRuleFlags;
    overallByDataset: Record<string, SliceMetricSummary>;
    sliceReports: SliceCategoryReport[];
    hotspots: {
        strongestPositiveLift: HotspotEntry[];
        strongestNegativeLift: HotspotEntry[];
    };
    guard: GuardSummary;
};

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

type BucketAccumulator = {
    total: number;
    answered: number;
    retrievalHitAt1: number;
    renderedHitAt1: number;
    retrievalMrrSum: number;
    renderedMrrSum: number;
    rescuedHit1Count: number;
    lostHit1Count: number;
    explicitYearRescuedHit1Count: number;
    explicitYearLostHit1Count: number;
    anchorNeutralRescuedHit1Count: number;
    anchorNeutralLostHit1Count: number;
};

const CURRENT_TIMESTAMP = Date.now() / 1000;
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const DATASET_BUNDLE = (
    process.env.SUASK_GRANULARITY_RUNTIME_SLICE_BUNDLE || "current_mainline"
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
const MAIN_LIFT_GAP_WARN_THRESHOLD = Number(
    process.env.SUASK_RUNTIME_GUARD_MAIN_LIFT_GAP_WARN_THRESHOLD || 1.5,
);
const MAIN_NET_GAIN_GAP_WARN_THRESHOLD = Number(
    process.env.SUASK_RUNTIME_GUARD_MAIN_NET_GAIN_GAP_WARN_THRESHOLD || 3,
);
const SMALL_BUCKET_MAX_COUNT = Number(
    process.env.SUASK_RUNTIME_GUARD_SMALL_BUCKET_MAX_COUNT || 6,
);
const SMALL_BUCKET_LIFT_WARN_THRESHOLD = Number(
    process.env.SUASK_RUNTIME_GUARD_SMALL_BUCKET_LIFT_WARN_THRESHOLD || 15,
);
const FAIL_ON_GUARD_WARN =
    process.env.SUASK_RUNTIME_GUARD_FAIL_ON_WARN === "1";

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

function round4(value: number): number {
    return Number(value.toFixed(4));
}

function safePercent(numerator: number, denominator: number): number {
    if (denominator <= 0) {
        return 0;
    }
    return round4((numerator / denominator) * 100);
}

function reciprocalRank(rank: number | null): number {
    return rank && Number.isFinite(rank) ? 1 / rank : 0;
}

function toNullableRank(rank: number): number | null {
    return Number.isFinite(rank) ? rank : null;
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

function resolveDatasetConfig(): EvalDatasetConfig {
    if (DATASET_FILE || DATASET_TARGET_KEY) {
        return resolveEvalDatasetConfig({
            datasetVersion: DATASET_VERSION,
            datasetFile: DATASET_FILE,
            singleFileAsAll: SINGLE_FILE_AS_ALL,
            datasetTargetKey: DATASET_TARGET_KEY,
        });
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

function buildDisplayVariantDocuments(params: {
    query: string;
    documents: PipelineDocumentRecord[];
    queryContext: SearchPipelineQueryContext;
    preferLatestWithinTopic: boolean;
    flags: DisplayRuleFlags;
}): PipelineDocumentRecord[] {
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
    const freshnessGuardedEntries = applyYearlessSameFamilyFreshnessGuardToEntries(
        querySignals,
        phaseAdjustedEntries,
        displayEntries,
    );
    return getDocumentsFromRerankEntries(
        applyCompressedQueryDisplayGuardToEntries(
            querySignals,
            phaseAdjustedEntries,
            freshnessGuardedEntries,
        ),
    );
}

function yearBucketFromYears(years: number[]): string {
    if (years.length === 0) {
        return "none";
    }
    const uniqueYears = Array.from(new Set(years)).sort((left, right) => left - right);
    if (uniqueYears.length === 1) {
        return String(uniqueYears[0]);
    }
    return `multi:${uniqueYears.join("_")}`;
}

function readCaseStringField(
    testCase: EvalDatasetCase,
    field:
        | "support_pattern"
        | "anchor_bucket"
        | "doc_role"
        | "near_neighbor_level"
        | "preferred_granularity",
): string {
    const value = (testCase as EvalDatasetCase & Record<string, unknown>)[field];
    return typeof value === "string" && value.length > 0 ? value : "unknown";
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
        const querySignals = buildDocumentRerankQuerySignals({
            query: testCase.query,
            queryIntent: queryContext.queryIntent,
            queryPlan: plannerQueryPlan,
            preferLatestWithinTopic: latestTriggered,
        });
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
            hasExplicitYearAnchor: querySignals.queryYears.length > 0,
            latestTriggered,
            answered: retrievalStage.retrievalDecision.behavior === "answer",
            retrievalRank,
            directDocuments,
            sliceTags: {
                dataset: testCase.dataset || "evaluation",
                themeFamily: testCase.theme_family || "unknown",
                yearBucket: yearBucketFromYears(querySignals.queryYears),
                entityBucket: querySignals.mentionsCollegeEntity
                    ? "entity_mentioned"
                    : "no_entity_mentioned",
                queryScope: testCase.query_scope || "unknown",
                queryType: testCase.query_type || "unknown",
                supportPattern: readCaseStringField(testCase, "support_pattern"),
                anchorBucket: readCaseStringField(testCase, "anchor_bucket"),
                docRole: readCaseStringField(testCase, "doc_role"),
                nearNeighborLevel: readCaseStringField(
                    testCase,
                    "near_neighbor_level",
                ),
                preferredGranularity: readCaseStringField(
                    testCase,
                    "preferred_granularity",
                ),
            },
        });

        if ((index + 1) % 40 === 0 || index + 1 === datasetCases.length) {
            console.log(`[prepare] ${index + 1} / ${datasetCases.length}`);
        }
    }

    return preparedCases;
}

function buildRuntimeResults(preparedCases: PreparedCase[]): VariantCaseResult[] {
    return preparedCases.map((preparedCase) => {
        if (!preparedCase.answered || preparedCase.directDocuments.length === 0) {
            return {
                answered: preparedCase.answered,
                retrievalRank: preparedCase.retrievalRank,
                renderedRank: null,
                renderedReciprocalRank: 0,
            };
        }

        const renderedResults = buildDisplayVariantDocuments({
            query: preparedCase.testCase.query,
            documents: preparedCase.directDocuments,
            queryContext: preparedCase.queryContext,
            preferLatestWithinTopic: preparedCase.latestTriggered,
            flags: RUNTIME_FULL_FLAGS,
        });
        const renderedRank = toNullableRank(
            getRankForCase(renderedResults, preparedCase.testCase),
        );
        return {
            answered: preparedCase.answered,
            retrievalRank: preparedCase.retrievalRank,
            renderedRank,
            renderedReciprocalRank: round4(reciprocalRank(renderedRank)),
        };
    });
}

function createBucketAccumulator(): BucketAccumulator {
    return {
        total: 0,
        answered: 0,
        retrievalHitAt1: 0,
        renderedHitAt1: 0,
        retrievalMrrSum: 0,
        renderedMrrSum: 0,
        rescuedHit1Count: 0,
        lostHit1Count: 0,
        explicitYearRescuedHit1Count: 0,
        explicitYearLostHit1Count: 0,
        anchorNeutralRescuedHit1Count: 0,
        anchorNeutralLostHit1Count: 0,
    };
}

function finalizeBucket(bucket: BucketAccumulator): SliceMetricSummary {
    const retrievalMRR =
        bucket.total > 0 ? round4(bucket.retrievalMrrSum / bucket.total) : 0;
    const renderedMRR =
        bucket.total > 0 ? round4(bucket.renderedMrrSum / bucket.total) : 0;
    return {
        total: bucket.total,
        answerRate: safePercent(bucket.answered, bucket.total),
        retrievalHitAt1: safePercent(bucket.retrievalHitAt1, bucket.total),
        renderedHitAt1: safePercent(bucket.renderedHitAt1, bucket.total),
        displayLiftHitAt1: round4(
            safePercent(bucket.renderedHitAt1, bucket.total) -
                safePercent(bucket.retrievalHitAt1, bucket.total),
        ),
        retrievalMRR,
        renderedMRR,
        displayLiftMRR: round4(renderedMRR - retrievalMRR),
        rescuedHit1Count: bucket.rescuedHit1Count,
        lostHit1Count: bucket.lostHit1Count,
        netHit1GainCount: bucket.rescuedHit1Count - bucket.lostHit1Count,
        explicitYearRescuedHit1Count: bucket.explicitYearRescuedHit1Count,
        explicitYearLostHit1Count: bucket.explicitYearLostHit1Count,
        explicitYearNetHit1GainCount:
            bucket.explicitYearRescuedHit1Count -
            bucket.explicitYearLostHit1Count,
        anchorNeutralRescuedHit1Count: bucket.anchorNeutralRescuedHit1Count,
        anchorNeutralLostHit1Count: bucket.anchorNeutralLostHit1Count,
        anchorNeutralNetHit1GainCount:
            bucket.anchorNeutralRescuedHit1Count -
            bucket.anchorNeutralLostHit1Count,
    };
}

function updateAccumulator(
    accumulator: BucketAccumulator,
    preparedCase: PreparedCase,
    result: VariantCaseResult,
): void {
    accumulator.total += 1;
    if (result.answered) {
        accumulator.answered += 1;
    }
    if (preparedCase.retrievalRank === 1) {
        accumulator.retrievalHitAt1 += 1;
    }
    if (result.renderedRank === 1) {
        accumulator.renderedHitAt1 += 1;
    }
    accumulator.retrievalMrrSum += reciprocalRank(preparedCase.retrievalRank);
    accumulator.renderedMrrSum += result.renderedReciprocalRank;
    if (preparedCase.retrievalRank !== 1 && result.renderedRank === 1) {
        accumulator.rescuedHit1Count += 1;
        if (preparedCase.hasExplicitYearAnchor) {
            accumulator.explicitYearRescuedHit1Count += 1;
        } else {
            accumulator.anchorNeutralRescuedHit1Count += 1;
        }
    }
    if (preparedCase.retrievalRank === 1 && result.renderedRank !== 1) {
        accumulator.lostHit1Count += 1;
        if (preparedCase.hasExplicitYearAnchor) {
            accumulator.explicitYearLostHit1Count += 1;
        } else {
            accumulator.anchorNeutralLostHit1Count += 1;
        }
    }
}

function buildSliceCategoryReport(params: {
    preparedCases: PreparedCase[];
    results: VariantCaseResult[];
    dataset: string;
    category: SliceCategoryReport["category"];
    minCount: number;
    getBucketKey: (preparedCase: PreparedCase) => string;
}): SliceCategoryReport {
    const { preparedCases, results, dataset, category, minCount, getBucketKey } = params;
    const buckets = new Map<string, BucketAccumulator>();

    preparedCases.forEach((preparedCase, index) => {
        if (preparedCase.sliceTags.dataset !== dataset) {
            return;
        }
        const result = results[index];
        if (!result) {
            return;
        }
        const key = getBucketKey(preparedCase);
        const accumulator = buckets.get(key) || createBucketAccumulator();
        updateAccumulator(accumulator, preparedCase, result);
        buckets.set(key, accumulator);
    });

    return {
        dataset,
        category,
        minCount,
        buckets: Array.from(buckets.entries())
            .map(([key, bucket]) => ({
                key,
                metrics: finalizeBucket(bucket),
            }))
            .filter((item) => item.metrics.total >= minCount)
            .sort((left, right) => {
                if (
                    right.metrics.displayLiftHitAt1 !== left.metrics.displayLiftHitAt1
                ) {
                    return (
                        right.metrics.displayLiftHitAt1 -
                        left.metrics.displayLiftHitAt1
                    );
                }
                if (right.metrics.total !== left.metrics.total) {
                    return right.metrics.total - left.metrics.total;
                }
                return left.key.localeCompare(right.key, "zh-Hans-CN");
            }),
    };
}

function buildOverallByDataset(
    preparedCases: PreparedCase[],
    results: VariantCaseResult[],
): Record<string, SliceMetricSummary> {
    const buckets = new Map<string, BucketAccumulator>();
    preparedCases.forEach((preparedCase, index) => {
        const result = results[index];
        if (!result) {
            return;
        }
        const key = preparedCase.sliceTags.dataset;
        const accumulator = buckets.get(key) || createBucketAccumulator();
        updateAccumulator(accumulator, preparedCase, result);
        buckets.set(key, accumulator);
    });
    return Object.fromEntries(
        Array.from(buckets.entries()).map(([key, bucket]) => [
            key,
            finalizeBucket(bucket),
        ]),
    );
}

function collectHotspots(sliceReports: SliceCategoryReport[]): {
    strongestPositiveLift: HotspotEntry[];
    strongestNegativeLift: HotspotEntry[];
} {
    const entries: HotspotEntry[] = sliceReports.flatMap((report) =>
        report.buckets.map((bucket) => ({
            dataset: report.dataset,
            category: report.category,
            key: bucket.key,
            metrics: bucket.metrics,
        })),
    );

    const sortable = entries.filter((item) => item.metrics.total >= 4);

    return {
        strongestPositiveLift: [...sortable]
            .sort((left, right) => {
                if (
                    right.metrics.displayLiftHitAt1 !== left.metrics.displayLiftHitAt1
                ) {
                    return (
                        right.metrics.displayLiftHitAt1 -
                        left.metrics.displayLiftHitAt1
                    );
                }
                return right.metrics.total - left.metrics.total;
            })
            .slice(0, 12),
        strongestNegativeLift: [...sortable]
            .sort((left, right) => {
                if (
                    left.metrics.displayLiftHitAt1 !== right.metrics.displayLiftHitAt1
                ) {
                    return (
                        left.metrics.displayLiftHitAt1 -
                        right.metrics.displayLiftHitAt1
                    );
                }
                return right.metrics.total - left.metrics.total;
            })
            .slice(0, 12),
    };
}

function buildGuardSummary(params: {
    overallByDataset: Record<string, SliceMetricSummary>;
    sliceReports: SliceCategoryReport[];
}): GuardSummary {
    const { overallByDataset, sliceReports } = params;
    const alerts: GuardAlert[] = [];
    const main = overallByDataset.main_bench_120;
    const inDomain = overallByDataset.in_domain_holdout_50;
    const extOOD = overallByDataset.matched_ext_ood_60;

    if (main && inDomain && extOOD) {
        const maxOtherLift = Math.max(
            inDomain.displayLiftHitAt1,
            extOOD.displayLiftHitAt1,
        );
        const mainLiftGap = round4(main.displayLiftHitAt1 - maxOtherLift);
        if (mainLiftGap >= MAIN_LIFT_GAP_WARN_THRESHOLD) {
            alerts.push({
                severity: "warn",
                code: "main_lift_gap_high",
                message:
                    "Main display lift is materially higher than the strongest generalization split.",
                evidence: {
                    mainDisplayLiftHitAt1: main.displayLiftHitAt1,
                    inDomainDisplayLiftHitAt1: inDomain.displayLiftHitAt1,
                    extOODDisplayLiftHitAt1: extOOD.displayLiftHitAt1,
                    mainLiftGapVsBestGeneralization: mainLiftGap,
                },
            });
        }

        const maxOtherNetGain = Math.max(
            inDomain.anchorNeutralNetHit1GainCount,
            extOOD.anchorNeutralNetHit1GainCount,
        );
        const mainNetGainGap =
            main.anchorNeutralNetHit1GainCount - maxOtherNetGain;
        if (mainNetGainGap >= MAIN_NET_GAIN_GAP_WARN_THRESHOLD) {
            alerts.push({
                severity: "warn",
                code: "main_net_gain_gap_high",
                message:
                    "Main anchor-neutral net Hit@1 rescue count is substantially higher than the generalization splits.",
                evidence: {
                    mainAnchorNeutralNetHit1GainCount:
                        main.anchorNeutralNetHit1GainCount,
                    inDomainAnchorNeutralNetHit1GainCount:
                        inDomain.anchorNeutralNetHit1GainCount,
                    extOODAnchorNeutralNetHit1GainCount:
                        extOOD.anchorNeutralNetHit1GainCount,
                    mainAnchorNeutralNetGainGapVsBestGeneralization:
                        mainNetGainGap,
                },
            });
        }
    }

    sliceReports
        .filter((report) => report.dataset === "main_bench_120")
        .forEach((report) => {
            report.buckets
                .filter(
                    (bucket) =>
                        bucket.metrics.total <= SMALL_BUCKET_MAX_COUNT &&
                        Math.abs(bucket.metrics.displayLiftHitAt1) >=
                            SMALL_BUCKET_LIFT_WARN_THRESHOLD &&
                        (bucket.metrics.anchorNeutralRescuedHit1Count > 0 ||
                            bucket.metrics.anchorNeutralLostHit1Count > 0),
                )
                .forEach((bucket) => {
                    alerts.push({
                        severity: "warn",
                        code: "main_small_bucket_spike",
                        message:
                            "A small Main slice shows an unusually large display lift spike.",
                        evidence: {
                            dataset: report.dataset,
                            category: report.category,
                            bucket: bucket.key,
                            total: bucket.metrics.total,
                            displayLiftHitAt1: bucket.metrics.displayLiftHitAt1,
                            netHit1GainCount: bucket.metrics.netHit1GainCount,
                            anchorNeutralNetHit1GainCount:
                                bucket.metrics.anchorNeutralNetHit1GainCount,
                            explicitYearNetHit1GainCount:
                                bucket.metrics.explicitYearNetHit1GainCount,
                        },
                    });
                });
        });

    return {
        status: alerts.length > 0 ? "warn" : "pass",
        thresholds: {
            mainLiftGapWarnThreshold: MAIN_LIFT_GAP_WARN_THRESHOLD,
            mainNetGainGapWarnThreshold: MAIN_NET_GAIN_GAP_WARN_THRESHOLD,
            smallBucketMaxCount: SMALL_BUCKET_MAX_COUNT,
            smallBucketLiftWarnThreshold: SMALL_BUCKET_LIFT_WARN_THRESHOLD,
        },
        alerts,
    };
}

async function main() {
    const datasetConfig = resolveDatasetConfig();
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
    const runtimeResults = buildRuntimeResults(preparedCases);
    const datasetKeys = Array.from(
        new Set(preparedCases.map((item) => item.sliceTags.dataset)),
    );

    const sliceReports: SliceCategoryReport[] = [];
    datasetKeys.forEach((dataset) => {
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "theme_family",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.themeFamily,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "query_year",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.yearBucket,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "entity_mention",
                minCount: 1,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.entityBucket,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "query_scope",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.queryScope,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "query_type",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.queryType,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "support_pattern",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.supportPattern,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "anchor_bucket",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.anchorBucket,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "doc_role",
                minCount: 3,
                getBucketKey: (preparedCase) => preparedCase.sliceTags.docRole,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "near_neighbor_level",
                minCount: 3,
                getBucketKey: (preparedCase) =>
                    preparedCase.sliceTags.nearNeighborLevel,
            }),
        );
        sliceReports.push(
            buildSliceCategoryReport({
                preparedCases,
                results: runtimeResults,
                dataset,
                category: "preferred_granularity",
                minCount: 3,
                getBucketKey: (preparedCase) =>
                    preparedCase.sliceTags.preferredGranularity,
            }),
        );
    });

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetBundle: DATASET_BUNDLE,
        datasetKey: datasetConfig.datasetKey,
        datasetLabel: datasetConfig.datasetLabel,
        presetName: BASE_PRESET.name,
        runtimeFlags: RUNTIME_FULL_FLAGS,
        overallByDataset: buildOverallByDataset(preparedCases, runtimeResults),
        sliceReports,
        hotspots: collectHotspots(sliceReports),
        guard: buildGuardSummary({
            overallByDataset: buildOverallByDataset(preparedCases, runtimeResults),
            sliceReports,
        }),
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        `${datasetConfig.datasetVersion}_runtime_slice_diagnostic_${datasetConfig.datasetKey}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");
    console.log(`[guard] status=${report.guard.status}`);
    report.guard.alerts.forEach((alert, index) => {
        console.log(
            `[guard][${index + 1}] ${alert.code}: ${alert.message} ${JSON.stringify(alert.evidence)}`,
        );
    });
    console.log(`Saved report to ${outputPath}`);
    if (FAIL_ON_GUARD_WARN && report.guard.status === "warn") {
        process.exitCode = 2;
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

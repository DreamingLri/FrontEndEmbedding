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

type MetricAccumulator = {
    total: number;
    answered: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrrSum: number;
    latestTriggered: number;
};

type MetricSummary = {
    total: number;
    answerRate: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
    latestTriggeredRate: number;
};

type VariantDatasetSummary = {
    retrieval: MetricSummary;
    rendered: MetricSummary;
    displayLiftHitAt1: number;
    displayLiftMRR: number;
};

type VariantReport = {
    label: string;
    note: string;
    flags: DisplayRuleFlags;
    metricsByDataset: Record<string, VariantDatasetSummary>;
    combined: VariantDatasetSummary;
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
    variants: VariantReport[];
};

type NormalizedOtidEvalTarget = {
    mode: OtidEvalMode;
    acceptableOtids: string[];
    requiredOtidGroups: string[][];
    minGroupsToCover: number;
};

const CURRENT_TIMESTAMP = Date.now() / 1000;
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const DATASET_BUNDLE = (
    process.env.SUASK_GRANULARITY_RUNTIME_ABLATION_BUNDLE || "retrieval_matched_v1"
)
    .trim()
    .toLowerCase();

const BASE_PRESET: PipelinePreset = clonePipelinePreset(
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
);

const VARIANTS: VariantDefinition[] = [
    {
        label: "runtime_full",
        note: "Current frontend runtime display chain.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_title_intent_confusion_gate",
        note: "Disable ambiguity-aware display downscaling and keep full title-intent rescue.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableTitleIntentConfusionGate: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_phase_anchor_boost",
        note: "Disable phase-anchor document boost only.",
        flags: {
            enablePhaseAnchorBoost: false,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_latest_version_rerank",
        note: "Keep other display rules, but force latest-version preference off.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: false,
        },
    },
    {
        label: "no_title_intent_adjustment",
        note: "Disable title-intent and query-plan role adjustment, keep coverage and latest-version.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments: false,
            enableLexicalTitleIntentAdjustments: false,
            enableLexicalTitleTypeAdjustments: false,
            enableThemeSpecificTitleAdjustments: false,
            enableDoctoralThemeTitleAdjustments: false,
            enableTuimianThemeTitleAdjustments: false,
            enableSummerCampThemeTitleAdjustments: false,
            enableTransferThemeTitleAdjustments: false,
            enableAiSchoolEntityTitleAdjustments: false,
            enableCompressedKeywordTitleAdjustments: false,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_structured_kp_role_evidence_adjustment",
        note: "Disable structured KP-role evidence adjustment only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableStructuredQueryPlanDocRoleAdjustments:
                BASE_PRESET.display.enableStructuredQueryPlanDocRoleAdjustments,
            enableStructuredKpRoleEvidenceAdjustments: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_lexical_title_intent_adjustment",
        note: "Disable lexical title-intent rules while keeping structured title signals.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableStructuredQueryPlanDocRoleAdjustments:
                BASE_PRESET.display.enableStructuredQueryPlanDocRoleAdjustments,
            enableLexicalTitleIntentAdjustments: false,
            enableLexicalTitleTypeAdjustments: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_structured_query_plan_doc_role_adjustment",
        note: "Disable structured query-plan doc-role alignment only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableStructuredQueryPlanDocRoleAdjustments: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_lexical_title_type_adjustment",
        note: "Disable coarse title-type lexical rules only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLexicalTitleIntentAdjustments:
                BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
            enableLexicalTitleTypeAdjustments: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_lexical_scenario_title_adjustment",
        note: "Disable scenario-specific lexical title rules only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLexicalTitleIntentAdjustments:
                BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
            enableLexicalTitleTypeAdjustments:
                BASE_PRESET.display.enableLexicalTitleTypeAdjustments,
            enableLexicalScenarioAdjustments: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_theme_specific_title_adjustment",
        note: "Disable theme-specific title adjustments only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLexicalTitleIntentAdjustments:
                BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
            enableLexicalTitleTypeAdjustments:
                BASE_PRESET.display.enableLexicalTitleTypeAdjustments,
            enableThemeSpecificTitleAdjustments: false,
            enableDoctoralThemeTitleAdjustments: false,
            enableTuimianThemeTitleAdjustments: false,
            enableSummerCampThemeTitleAdjustments: false,
            enableTransferThemeTitleAdjustments: false,
            enableAiSchoolEntityTitleAdjustments:
                BASE_PRESET.display.enableAiSchoolEntityTitleAdjustments,
            enableCompressedKeywordTitleAdjustments:
                BASE_PRESET.display.enableCompressedKeywordTitleAdjustments,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_ai_school_entity_title_adjustment",
        note: "Disable AI-school entity title adjustment only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableAiSchoolEntityTitleAdjustments: false,
            enableCompressedKeywordTitleAdjustments:
                BASE_PRESET.display.enableCompressedKeywordTitleAdjustments,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_doctoral_theme_title_adjustment",
        note: "Disable doctoral residual title fallback only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLexicalTitleIntentAdjustments:
                BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
            enableLexicalTitleTypeAdjustments:
                BASE_PRESET.display.enableLexicalTitleTypeAdjustments,
            enableThemeSpecificTitleAdjustments:
                BASE_PRESET.display.enableThemeSpecificTitleAdjustments,
            enableDoctoralThemeTitleAdjustments: false,
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
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_tuimian_theme_title_adjustment",
        note: "Disable tuimian residual title fallback only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLexicalTitleIntentAdjustments:
                BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
            enableLexicalTitleTypeAdjustments:
                BASE_PRESET.display.enableLexicalTitleTypeAdjustments,
            enableThemeSpecificTitleAdjustments:
                BASE_PRESET.display.enableThemeSpecificTitleAdjustments,
            enableDoctoralThemeTitleAdjustments:
                BASE_PRESET.display.enableDoctoralThemeTitleAdjustments,
            enableTuimianThemeTitleAdjustments: false,
            enableSummerCampThemeTitleAdjustments:
                BASE_PRESET.display.enableSummerCampThemeTitleAdjustments,
            enableTransferThemeTitleAdjustments:
                BASE_PRESET.display.enableTransferThemeTitleAdjustments,
            enableAiSchoolEntityTitleAdjustments:
                BASE_PRESET.display.enableAiSchoolEntityTitleAdjustments,
            enableCompressedKeywordTitleAdjustments:
                BASE_PRESET.display.enableCompressedKeywordTitleAdjustments,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_summer_camp_theme_title_adjustment",
        note: "Disable summer-camp residual title fallback only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableSummerCampThemeTitleAdjustments: false,
            enableTransferThemeTitleAdjustments:
                BASE_PRESET.display.enableTransferThemeTitleAdjustments,
            enableAiSchoolEntityTitleAdjustments:
                BASE_PRESET.display.enableAiSchoolEntityTitleAdjustments,
            enableCompressedKeywordTitleAdjustments:
                BASE_PRESET.display.enableCompressedKeywordTitleAdjustments,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_transfer_theme_title_adjustment",
        note: "Disable transfer residual title fallback only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableTransferThemeTitleAdjustments: false,
            enableAiSchoolEntityTitleAdjustments:
                BASE_PRESET.display.enableAiSchoolEntityTitleAdjustments,
            enableCompressedKeywordTitleAdjustments:
                BASE_PRESET.display.enableCompressedKeywordTitleAdjustments,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_compressed_keyword_title_adjustment",
        note: "Disable compressed-keyword title specialization only.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
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
            enableCompressedKeywordTitleAdjustments: false,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "no_specialized_title_adjustment",
        note: "Disable theme-specific and compressed-keyword title specialization together.",
        flags: {
            enablePhaseAnchorBoost: BASE_PRESET.retrieval.enablePhaseAnchorBoost,
            enableTitleIntentAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLexicalTitleIntentAdjustments:
                BASE_PRESET.display.enableLexicalTitleIntentAdjustments,
            enableLexicalTitleTypeAdjustments:
                BASE_PRESET.display.enableLexicalTitleTypeAdjustments,
            enableThemeSpecificTitleAdjustments: false,
            enableDoctoralThemeTitleAdjustments: false,
            enableTuimianThemeTitleAdjustments: false,
            enableSummerCampThemeTitleAdjustments: false,
            enableTransferThemeTitleAdjustments: false,
            enableAiSchoolEntityTitleAdjustments: false,
            enableCompressedKeywordTitleAdjustments: false,
            enableCoverageAdjustments:
                BASE_PRESET.display.useYearPhaseTitleAdjustment,
            enableLatestVersionRerank: true,
        },
    },
    {
        label: "all_display_rules_off",
        note: "Disable phase, title-intent, coverage, and latest-version display rerank.",
        flags: {
            enablePhaseAnchorBoost: false,
            enableTitleIntentAdjustments: false,
            enableLexicalTitleIntentAdjustments: false,
            enableLexicalTitleTypeAdjustments: false,
            enableThemeSpecificTitleAdjustments: false,
            enableDoctoralThemeTitleAdjustments: false,
            enableTuimianThemeTitleAdjustments: false,
            enableSummerCampThemeTitleAdjustments: false,
            enableTransferThemeTitleAdjustments: false,
            enableAiSchoolEntityTitleAdjustments: false,
            enableCompressedKeywordTitleAdjustments: false,
            enableCoverageAdjustments: false,
            enableLatestVersionRerank: false,
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

function resolveDatasetConfigForAblation(): EvalDatasetConfig {
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

    return resolveEvalDatasetConfig({
        datasetVersion: "granularity",
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

function createAccumulator(): MetricAccumulator {
    return {
        total: 0,
        answered: 0,
        hitAt1: 0,
        hitAt3: 0,
        hitAt5: 0,
        mrrSum: 0,
        latestTriggered: 0,
    };
}

function updateAccumulator(
    accumulator: MetricAccumulator,
    rank: number,
    answered: boolean,
    latestTriggered: boolean,
): void {
    accumulator.total += 1;
    if (answered) {
        accumulator.answered += 1;
    }
    if (latestTriggered) {
        accumulator.latestTriggered += 1;
    }
    if (Number.isFinite(rank)) {
        if (rank === 1) {
            accumulator.hitAt1 += 1;
        }
        if (rank <= 3) {
            accumulator.hitAt3 += 1;
        }
        if (rank <= 5) {
            accumulator.hitAt5 += 1;
        }
        accumulator.mrrSum += 1 / rank;
    }
}

function finalizeAccumulator(accumulator: MetricAccumulator): MetricSummary {
    return {
        total: accumulator.total,
        answerRate: safePercent(accumulator.answered, accumulator.total),
        hitAt1: safePercent(accumulator.hitAt1, accumulator.total),
        hitAt3: safePercent(accumulator.hitAt3, accumulator.total),
        hitAt5: safePercent(accumulator.hitAt5, accumulator.total),
        mrr:
            accumulator.total > 0
                ? round4(accumulator.mrrSum / accumulator.total)
                : 0,
        latestTriggeredRate: safePercent(
            accumulator.latestTriggered,
            accumulator.total,
        ),
    };
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
    const guardedEntries = applyCompressedQueryDisplayGuardToEntries(
        querySignals,
        phaseAdjustedEntries,
        freshnessGuardedEntries,
    );
    return getDocumentsFromRerankEntries(guardedEntries);
}

async function evaluateVariant(params: {
    variant: VariantDefinition;
    datasetCases: EvalDatasetCase[];
    engine: Awaited<ReturnType<typeof loadFrontendEvalEngine>>;
    queryVectors: Float32Array[];
}): Promise<VariantReport> {
    const { variant, datasetCases, engine, queryVectors } = params;
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();

    const retrievalAccumulators = new Map<string, MetricAccumulator>();
    const renderedAccumulators = new Map<string, MetricAccumulator>();
    const combinedRetrieval = createAccumulator();
    const combinedRendered = createAccumulator();

    for (let index = 0; index < datasetCases.length; index += 1) {
        const testCase = datasetCases[index];
        const datasetKey = testCase.dataset || "evaluation";
        const retrievalAccumulator =
            retrievalAccumulators.get(datasetKey) || createAccumulator();
        const renderedAccumulator =
            renderedAccumulators.get(datasetKey) || createAccumulator();
        retrievalAccumulators.set(datasetKey, retrievalAccumulator);
        renderedAccumulators.set(datasetKey, renderedAccumulator);

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
        const retrievalRank = getRankForCase(
            retrievalStage.searchOutput.matches,
            testCase,
        );
        updateAccumulator(
            retrievalAccumulator,
            retrievalRank,
            retrievalStage.retrievalDecision.behavior === "answer",
            latestTriggered,
        );
        updateAccumulator(
            combinedRetrieval,
            retrievalRank,
            retrievalStage.retrievalDecision.behavior === "answer",
            latestTriggered,
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
        let renderedResults: PipelineDocumentRecord[] = [];

        if (fetchIds.length > 0) {
            const documents = await documentLoader({
                query: testCase.query,
                otids: fetchIds,
            });
            const fetchedDocumentLookup = buildPipelineDocumentLookup(documents);
            const directDocuments = mergeCoarseMatchesWithDocumentLookup(
                fetchedDocumentLookup,
                answerCoarseMatches,
            );
            renderedResults = buildDisplayVariantDocuments({
                query: testCase.query,
                documents: directDocuments,
                queryContext,
                preferLatestWithinTopic: latestTriggered,
                flags: variant.flags,
            });
        }

        const renderedRank = getRankForCase(renderedResults, testCase);
        updateAccumulator(
            renderedAccumulator,
            renderedRank,
            retrievalStage.retrievalDecision.behavior === "answer",
            latestTriggered,
        );
        updateAccumulator(
            combinedRendered,
            renderedRank,
            retrievalStage.retrievalDecision.behavior === "answer",
            latestTriggered,
        );

        if ((index + 1) % 40 === 0 || index + 1 === datasetCases.length) {
            console.log(
                `[${variant.label}] processed ${index + 1} / ${datasetCases.length}`,
            );
        }
    }

    const metricsByDataset: Record<string, VariantDatasetSummary> = {};
    retrievalAccumulators.forEach((retrieval, datasetKey) => {
        const rendered = renderedAccumulators.get(datasetKey) || createAccumulator();
        const retrievalSummary = finalizeAccumulator(retrieval);
        const renderedSummary = finalizeAccumulator(rendered);
        metricsByDataset[datasetKey] = {
            retrieval: retrievalSummary,
            rendered: renderedSummary,
            displayLiftHitAt1: round2(
                renderedSummary.hitAt1 - retrievalSummary.hitAt1,
            ),
            displayLiftMRR: round4(renderedSummary.mrr - retrievalSummary.mrr),
        };
    });

    const combinedRetrievalSummary = finalizeAccumulator(combinedRetrieval);
    const combinedRenderedSummary = finalizeAccumulator(combinedRendered);

    return {
        label: variant.label,
        note: variant.note,
        flags: variant.flags,
        metricsByDataset,
        combined: {
            retrieval: combinedRetrievalSummary,
            rendered: combinedRenderedSummary,
            displayLiftHitAt1: round2(
                combinedRenderedSummary.hitAt1 - combinedRetrievalSummary.hitAt1,
            ),
            displayLiftMRR: round4(
                combinedRenderedSummary.mrr - combinedRetrievalSummary.mrr,
            ),
        },
    };
}

async function main() {
    const datasetConfig = resolveDatasetConfigForAblation();
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

    const variants: VariantReport[] = [];
    for (const variant of VARIANTS) {
        console.log(`Evaluating ${variant.label} ...`);
        variants.push(
            await evaluateVariant({
                variant,
                datasetCases,
                engine,
                queryVectors,
            }),
        );
    }

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
        variants,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        `granularity_runtime_ablation_${datasetConfig.datasetKey}_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    for (const variant of variants) {
        const combined = variant.combined;
        console.log(
            [
                variant.label,
                `retrieval@1=${combined.retrieval.hitAt1.toFixed(2)}%`,
                `rendered@1=${combined.rendered.hitAt1.toFixed(2)}%`,
                `lift@1=${combined.displayLiftHitAt1.toFixed(2)}pt`,
                `rendered_mrr=${combined.rendered.mrr.toFixed(4)}`,
            ].join(" | "),
        );
    }
    console.log(`Saved report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

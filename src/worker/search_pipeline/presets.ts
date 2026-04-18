import type { PipelinePreset } from "./types.ts";

const DEFAULT_DISPLAY_CONFIG: PipelinePreset["display"] = {
    rejectThreshold: 0.4,
    rerankBlendAlpha: 0.15,
    bestSentenceThreshold: 0.4,
    fetchMatchLimit: 15,
    fetchWeakMatchLimit: 10,
    useYearPhaseTitleAdjustment: false,
    enableQueryPlanner: false,
};

export const PAPER_FROZEN_MAIN_PIPELINE_PRESET: PipelinePreset = {
    name: "paper_frozen_main_v1",
    retrieval: {
        weights: {
            Q: 0,
            KP: 0.28571428571428575,
            OT: 0.7142857142857143,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
        qConfusionMode: "off",
        qConfusionWeight: 0.2,
        useQueryExpansion: true,
        useTopicPartition: true,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: false,
        minimalMode: false,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

// Historical preset retained only for explicit compatibility / replay.
// Do not use as the default runtime preset.
export const PRODUCT_CANONICAL_FULL_PIPELINE_PRESET: PipelinePreset = {
    name: "product_canonical_full_v1",
    retrieval: {
        weights: {
            Q: 0.3333333333333333,
            KP: 0.13333333333333333,
            OT: 0.5333333333333333,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "feature",
        kpRoleDocWeight: 0.35,
        qConfusionMode: "off",
        qConfusionWeight: 0.2,
        useQueryExpansion: true,
        useTopicPartition: true,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: false,
        minimalMode: false,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PAPER_TAIL_TOP3_W020_PIPELINE_PRESET: PipelinePreset = {
    name: "paper_tail_top3_w020_v1",
    retrieval: {
        ...PAPER_FROZEN_MAIN_PIPELINE_PRESET.retrieval,
        kpAggregationMode: "max_plus_topn",
        kpTopN: 3,
        kpTailWeight: 0.2,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET: PipelinePreset = {
    name: "product_tail_top3_w020_v1",
    retrieval: {
        ...PRODUCT_CANONICAL_FULL_PIPELINE_PRESET.retrieval,
        kpAggregationMode: "max_plus_topn",
        kpTopN: 3,
        kpTailWeight: 0.2,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const MINIMAL_BASELINE_PIPELINE_PRESET: PipelinePreset = {
    name: "minimal_q_kp_ot_v1",
    retrieval: {
        weights: {
            Q: 0.3333333333333333,
            KP: 0.3333333333333333,
            OT: 0.3333333333333333,
        },
        topHybridLimit: 1000,
        kpAggregationMode: "max",
        kpTopN: 3,
        kpTailWeight: 0.35,
        lexicalBonusMode: "sum",
        kpRoleRerankMode: "off",
        kpRoleDocWeight: 0,
        qConfusionMode: "off",
        qConfusionWeight: 0.2,
        useQueryExpansion: false,
        useTopicPartition: false,
        enableExplicitYearFilter: false,
        enablePhaseAnchorBoost: false,
        minimalMode: true,
    },
    display: { ...DEFAULT_DISPLAY_CONFIG },
};

export const FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET: PipelinePreset = {
    name: "frontend_research_sync_v1",
    retrieval: {
        ...MINIMAL_BASELINE_PIPELINE_PRESET.retrieval,
        qConfusionMode: "combined",
        qConfusionWeight: 0.2,
        enableExplicitYearFilter: true,
        enablePhaseAnchorBoost: true,
    },
    display: {
        ...DEFAULT_DISPLAY_CONFIG,
        fetchMatchLimit: 20,
        fetchWeakMatchLimit: 12,
        useYearPhaseTitleAdjustment: true,
    },
};

export const FRONTEND_RESEARCH_SYNC_QUERY_PLANNER_PIPELINE_PRESET: PipelinePreset =
    {
        name: "frontend_research_sync_query_planner_v1",
        retrieval: { ...FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.retrieval },
        display: {
            ...FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.display,
            enableQueryPlanner: true,
        },
    };

export const PIPELINE_PRESET_REGISTRY = {
    paper_frozen_main_v1: PAPER_FROZEN_MAIN_PIPELINE_PRESET,
    product_canonical_full_v1: PRODUCT_CANONICAL_FULL_PIPELINE_PRESET,
    paper_tail_top3_w020_v1: PAPER_TAIL_TOP3_W020_PIPELINE_PRESET,
    product_tail_top3_w020_v1: PRODUCT_TAIL_TOP3_W020_PIPELINE_PRESET,
    minimal_q_kp_ot_v1: MINIMAL_BASELINE_PIPELINE_PRESET,
    frontend_research_sync_v1: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    frontend_research_sync_query_planner_v1:
        FRONTEND_RESEARCH_SYNC_QUERY_PLANNER_PIPELINE_PRESET,
} as const;

export type PipelinePresetName = keyof typeof PIPELINE_PRESET_REGISTRY;

export const CANONICAL_PIPELINE_PRESET =
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET;

export function clonePipelinePreset(preset: PipelinePreset): PipelinePreset {
    return {
        ...preset,
        retrieval: {
            ...preset.retrieval,
            weights: { ...preset.retrieval.weights },
        },
        display: { ...preset.display },
    };
}

export function resolvePipelinePresetByName(
    presetName?: string,
): PipelinePreset {
    if (!presetName) {
        return clonePipelinePreset(CANONICAL_PIPELINE_PRESET);
    }

    const preset = PIPELINE_PRESET_REGISTRY[
        presetName as PipelinePresetName
    ];
    return clonePipelinePreset(preset || CANONICAL_PIPELINE_PRESET);
}


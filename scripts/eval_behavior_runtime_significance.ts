import * as fs from "fs";
import * as path from "path";

import { loadAnswerQualityDataset } from "./answer_quality_dataset.ts";
import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    clonePipelinePreset,
    executeSearchPipeline,
    type PipelinePreset,
} from "../src/worker/search_pipeline.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";

type VariantDefinition = {
    label: string;
    note: string;
    preset: PipelinePreset;
};

type CaseOutcome = {
    id: string;
    query: string;
    expectedOtid: string;
    predictedBehavior: "answer" | "reject";
    renderedDocRank: number | null;
    retrievalDocRank: number | null;
    correctAt1: boolean;
    retrievalHitAt1: boolean;
};

type VariantOutcomeReport = {
    label: string;
    note: string;
    correctAt1Count: number;
    correctAt1Rate: number;
    retrievalHitAt1Count: number;
    retrievalHitAt1Rate: number;
    caseOutcomes: CaseOutcome[];
};

type McNemarComparison = {
    compareLabel: string;
    compareNote: string;
    metric: "correctAt1" | "retrievalHitAt1";
    baseOnlyCorrect: number;
    compareOnlyCorrect: number;
    bothCorrect: number;
    bothWrong: number;
    exactTwoSidedPValue: number;
    baseRate: number;
    compareRate: number;
    rateDiff: number;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    baseLabel: string;
    variants: Array<{
        label: string;
        note: string;
        correctAt1Count: number;
        correctAt1Rate: number;
        retrievalHitAt1Count: number;
        retrievalHitAt1Rate: number;
    }>;
    comparisons: McNemarComparison[];
};

const ANSWER_QUALITY_DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_ANSWER_QUALITY_DATASET_FILE ||
        CURRENT_EVAL_DATASET_FILES.answerQualityCurrent,
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;

function createVariant(
    label: string,
    note: string,
    mutate: (preset: PipelinePreset) => void,
): VariantDefinition {
    const preset = clonePipelinePreset(FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET);
    preset.name = label;
    mutate(preset);
    return {
        label,
        note,
        preset,
    };
}

const VARIANTS: VariantDefinition[] = [
    createVariant("runtime_full", "Current runtime preset.", () => {}),
    createVariant(
        "no_q_confusion",
        "Disable qConfusion while keeping other runtime modules unchanged.",
        (preset) => {
            preset.retrieval.qConfusionMode = "off";
        },
    ),
    createVariant(
        "no_year_filter",
        "Disable explicit year filtering while keeping other runtime modules unchanged.",
        (preset) => {
            preset.retrieval.enableExplicitYearFilter = false;
        },
    ),
    createVariant(
        "no_phase_anchor_boost",
        "Disable document-level phase anchor boost before display ordering.",
        (preset) => {
            preset.retrieval.enablePhaseAnchorBoost = false;
        },
    ),
    createVariant(
        "no_display_adjustment",
        "Disable title-intent and coverage adjustments in the display stage.",
        (preset) => {
            preset.display.useYearPhaseTitleAdjustment = false;
        },
    ),
];

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function getRankByOtid(otids: string[], expectedOtid: string): number | null {
    const index = otids.findIndex((otid) => otid === expectedOtid);
    return index >= 0 ? index + 1 : null;
}

function combination(n: number, k: number): number {
    if (k < 0 || k > n) {
        return 0;
    }
    const effectiveK = Math.min(k, n - k);
    let result = 1;
    for (let index = 1; index <= effectiveK; index += 1) {
        result = (result * (n - effectiveK + index)) / index;
    }
    return result;
}

function exactMcNemarTwoSidedPValue(b: number, c: number): number {
    const discordant = b + c;
    if (discordant === 0) {
        return 1;
    }
    const tail = Math.min(b, c);
    let cumulative = 0;
    for (let index = 0; index <= tail; index += 1) {
        cumulative += combination(discordant, index);
    }
    return Math.min(1, (2 * cumulative) / 2 ** discordant);
}

async function evaluateVariant(params: {
    preset: PipelinePreset;
    engine: Awaited<ReturnType<typeof loadFrontendEvalEngine>>;
    queryVectors: Float32Array[];
}): Promise<CaseOutcome[]> {
    const { cases } = loadAnswerQualityDataset(ANSWER_QUALITY_DATASET_FILE);
    const termMaps = buildPipelineTermMaps(params.engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();

    const outcomes: CaseOutcome[] = [];

    for (let index = 0; index < cases.length; index += 1) {
        const testCase = cases[index];
        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            params.engine.vocabMap,
            params.engine.topicPartitionIndex,
            params.preset,
        );
        const pipelineResult = await executeSearchPipeline({
            query: testCase.query,
            queryVector: params.queryVectors[index],
            queryContext,
            metadata: params.engine.metadataList,
            vectorMatrix: params.engine.vectorMatrix,
            dimensions: params.engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: params.engine.bm25Stats,
            documentLoader,
            termMaps,
            preset: params.preset,
        });

        const predictedBehavior = pipelineResult.finalDecision.behavior;
        const retrievalOtids = pipelineResult.searchOutput.matches.map(
            (item) => item.otid,
        );
        const renderedOtids = pipelineResult.results.map((item) => item.otid);
        const retrievalDocRank = getRankByOtid(
            retrievalOtids,
            testCase.expected_otid,
        );
        const renderedDocRank =
            predictedBehavior === "answer"
                ? getRankByOtid(renderedOtids, testCase.expected_otid)
                : null;

        outcomes.push({
            id: testCase.id || `aq_case_${index + 1}`,
            query: testCase.query,
            expectedOtid: testCase.expected_otid,
            predictedBehavior,
            renderedDocRank,
            retrievalDocRank,
            correctAt1: renderedDocRank === 1,
            retrievalHitAt1: retrievalDocRank === 1,
        });
    }

    return outcomes;
}

function buildVariantOutcomeReport(
    variant: VariantDefinition,
    caseOutcomes: CaseOutcome[],
): VariantOutcomeReport {
    const correctAt1Count = caseOutcomes.filter((item) => item.correctAt1).length;
    const retrievalHitAt1Count = caseOutcomes.filter(
        (item) => item.retrievalHitAt1,
    ).length;
    return {
        label: variant.label,
        note: variant.note,
        correctAt1Count,
        correctAt1Rate: safeRate(correctAt1Count, caseOutcomes.length),
        retrievalHitAt1Count,
        retrievalHitAt1Rate: safeRate(retrievalHitAt1Count, caseOutcomes.length),
        caseOutcomes,
    };
}

function compareVariants(
    base: VariantOutcomeReport,
    compare: VariantOutcomeReport,
    metric: "correctAt1" | "retrievalHitAt1",
): McNemarComparison {
    let baseOnlyCorrect = 0;
    let compareOnlyCorrect = 0;
    let bothCorrect = 0;
    let bothWrong = 0;

    for (let index = 0; index < base.caseOutcomes.length; index += 1) {
        const baseValue = base.caseOutcomes[index][metric];
        const compareValue = compare.caseOutcomes[index][metric];
        if (baseValue && compareValue) {
            bothCorrect += 1;
        } else if (!baseValue && !compareValue) {
            bothWrong += 1;
        } else if (baseValue) {
            baseOnlyCorrect += 1;
        } else {
            compareOnlyCorrect += 1;
        }
    }

    const baseRate =
        metric === "correctAt1"
            ? base.correctAt1Rate
            : base.retrievalHitAt1Rate;
    const compareRate =
        metric === "correctAt1"
            ? compare.correctAt1Rate
            : compare.retrievalHitAt1Rate;

    return {
        compareLabel: compare.label,
        compareNote: compare.note,
        metric,
        baseOnlyCorrect,
        compareOnlyCorrect,
        bothCorrect,
        bothWrong,
        exactTwoSidedPValue: exactMcNemarTwoSidedPValue(
            baseOnlyCorrect,
            compareOnlyCorrect,
        ),
        baseRate,
        compareRate,
        rateDiff: baseRate - compareRate,
    };
}

async function main() {
    const { cases } = loadAnswerQualityDataset(ANSWER_QUALITY_DATASET_FILE);
    const engine = await loadFrontendEvalEngine();
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        cases.map((item) => item.query),
        engine.dimensions,
    );

    const variantReports: VariantOutcomeReport[] = [];
    for (const variant of VARIANTS) {
        console.log(`Evaluating significance variant: ${variant.label}`);
        const caseOutcomes = await evaluateVariant({
            preset: variant.preset,
            engine,
            queryVectors,
        });
        variantReports.push(buildVariantOutcomeReport(variant, caseOutcomes));
    }

    const baseReport = variantReports[0];
    if (!baseReport) {
        throw new Error("Missing runtime_full baseline report.");
    }

    const comparisons = variantReports
        .slice(1)
        .flatMap((variantReport) => [
            compareVariants(baseReport, variantReport, "correctAt1"),
            compareVariants(baseReport, variantReport, "retrievalHitAt1"),
        ]);

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetFile: ANSWER_QUALITY_DATASET_FILE,
        baseLabel: baseReport.label,
        variants: variantReports.map((item) => ({
            label: item.label,
            note: item.note,
            correctAt1Count: item.correctAt1Count,
            correctAt1Rate: item.correctAt1Rate,
            retrievalHitAt1Count: item.retrievalHitAt1Count,
            retrievalHitAt1Rate: item.retrievalHitAt1Rate,
        })),
        comparisons,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputFile = path.resolve(
        RESULTS_DIR,
        `behavior_runtime_significance_${Date.now()}.json`,
    );
    fs.writeFileSync(outputFile, JSON.stringify(report, null, 2), "utf8");

    console.log(`Saved report to ${outputFile}`);
    comparisons.forEach((item) => {
        console.log(
            [
                `${item.metric}`,
                `${baseReport.label} vs ${item.compareLabel}`,
                `b=${item.baseOnlyCorrect}`,
                `c=${item.compareOnlyCorrect}`,
                `p=${item.exactTwoSidedPValue.toFixed(6)}`,
                `diff=${(item.rateDiff * 100).toFixed(2)}pp`,
            ].join(" | "),
        );
    });
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});

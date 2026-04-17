import * as fs from "fs";
import * as path from "path";

import { loadAnswerRejectDataset } from "./answer_reject_dataset.ts";
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
    preset: PipelinePreset;
    note: string;
};

type AnswerRejectSummary = {
    total: number;
    binaryBehaviorAccuracy: number;
    answerHitRate: number;
    rejectRecall: number;
    falseRejectRate: number;
    unsafeAnswerRate: number;
    answerDocHitRate: number;
};

type AnswerQualitySummary = {
    total: number;
    answerRate: number;
    correctAt1Rate: number;
    correctAt3Rate: number;
    correctAt5Rate: number;
    falseRejectRate: number;
    potentiallyMisleadingRate: number;
    retrievalDocHitAt1Rate: number;
};

type VariantReport = {
    label: string;
    note: string;
    preset: {
        name: string;
        qConfusionMode: string;
        qConfusionWeight: number;
        enableExplicitYearFilter: boolean;
        enablePhaseAnchorBoost: boolean;
        useYearPhaseTitleAdjustment: boolean;
    };
    answerReject: AnswerRejectSummary;
    answerQuality: AnswerQualitySummary;
};

type Report = {
    generatedAt: string;
    answerRejectDatasetFile: string;
    answerQualityDatasetFile: string;
    variants: VariantReport[];
};

const ANSWER_REJECT_DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_ANSWER_REJECT_DATASET_FILE ||
        CURRENT_EVAL_DATASET_FILES.answerRejectCurrent,
);
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
        preset,
        note,
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

async function evaluateAnswerRejectVariant(params: {
    datasetFile: string;
    preset: PipelinePreset;
    engine: Awaited<ReturnType<typeof loadFrontendEvalEngine>>;
    queryVectors: Float32Array[];
}): Promise<AnswerRejectSummary> {
    const { cases } = loadAnswerRejectDataset(params.datasetFile);
    const termMaps = buildPipelineTermMaps(params.engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();

    let binaryCorrect = 0;
    let answerTotal = 0;
    let answerHit = 0;
    let rejectTotal = 0;
    let rejectHit = 0;
    let falseReject = 0;
    let unsafeAnswer = 0;
    let answerDocEligible = 0;
    let answerDocHit = 0;

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
            extractor: params.engine.extractor,
            documentLoader,
            termMaps,
            preset: params.preset,
        });

        const predictedBehavior = pipelineResult.finalDecision.behavior;
        if (predictedBehavior === testCase.expected_behavior) {
            binaryCorrect += 1;
        }

        if (testCase.expected_behavior === "answer") {
            answerTotal += 1;
            if (predictedBehavior === "answer") {
                answerHit += 1;
            } else {
                falseReject += 1;
            }

            if (testCase.expected_otid) {
                answerDocEligible += 1;
                const topResultOtid =
                    predictedBehavior === "answer"
                        ? pipelineResult.results[0]?.otid ||
                          pipelineResult.searchOutput.matches[0]?.otid ||
                          null
                        : null;
                if (topResultOtid === testCase.expected_otid) {
                    answerDocHit += 1;
                }
            }
        } else {
            rejectTotal += 1;
            if (predictedBehavior === "reject") {
                rejectHit += 1;
            } else {
                unsafeAnswer += 1;
            }
        }
    }

    return {
        total: cases.length,
        binaryBehaviorAccuracy: safeRate(binaryCorrect, cases.length),
        answerHitRate: safeRate(answerHit, answerTotal),
        rejectRecall: safeRate(rejectHit, rejectTotal),
        falseRejectRate: safeRate(falseReject, answerTotal),
        unsafeAnswerRate: safeRate(unsafeAnswer, rejectTotal),
        answerDocHitRate: safeRate(answerDocHit, answerDocEligible),
    };
}

async function evaluateAnswerQualityVariant(params: {
    datasetFile: string;
    preset: PipelinePreset;
    engine: Awaited<ReturnType<typeof loadFrontendEvalEngine>>;
    queryVectors: Float32Array[];
}): Promise<AnswerQualitySummary> {
    const { cases } = loadAnswerQualityDataset(params.datasetFile);
    const termMaps = buildPipelineTermMaps(params.engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();

    let answered = 0;
    let correctAt1 = 0;
    let correctAt3 = 0;
    let correctAt5 = 0;
    let falseReject = 0;
    let potentiallyMisleading = 0;
    let retrievalDocHitAt1 = 0;

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
            extractor: params.engine.extractor,
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

        if (predictedBehavior === "answer") {
            answered += 1;
        } else {
            falseReject += 1;
        }

        if (renderedDocRank === 1) {
            correctAt1 += 1;
        }
        if (renderedDocRank !== null && renderedDocRank <= 3) {
            correctAt3 += 1;
        }
        if (renderedDocRank !== null && renderedDocRank <= 5) {
            correctAt5 += 1;
        }
        if (predictedBehavior === "answer" && renderedDocRank !== 1) {
            potentiallyMisleading += 1;
        }
        if (retrievalDocRank === 1) {
            retrievalDocHitAt1 += 1;
        }
    }

    return {
        total: cases.length,
        answerRate: safeRate(answered, cases.length),
        correctAt1Rate: safeRate(correctAt1, cases.length),
        correctAt3Rate: safeRate(correctAt3, cases.length),
        correctAt5Rate: safeRate(correctAt5, cases.length),
        falseRejectRate: safeRate(falseReject, cases.length),
        potentiallyMisleadingRate: safeRate(
            potentiallyMisleading,
            cases.length,
        ),
        retrievalDocHitAt1Rate: safeRate(retrievalDocHitAt1, cases.length),
    };
}

async function main() {
    const answerRejectCases = loadAnswerRejectDataset(
        ANSWER_REJECT_DATASET_FILE,
    ).cases;
    const answerQualityCases = loadAnswerQualityDataset(
        ANSWER_QUALITY_DATASET_FILE,
    ).cases;

    const engine = await loadFrontendEvalEngine();
    const answerRejectQueryVectors = await embedFrontendQueries(
        engine.extractor,
        answerRejectCases.map((item) => item.query),
        engine.dimensions,
    );
    const answerQualityQueryVectors = await embedFrontendQueries(
        engine.extractor,
        answerQualityCases.map((item) => item.query),
        engine.dimensions,
    );

    const reports: VariantReport[] = [];
    for (const variant of VARIANTS) {
        console.log(`Evaluating variant: ${variant.label}`);
        const answerReject = await evaluateAnswerRejectVariant({
            datasetFile: ANSWER_REJECT_DATASET_FILE,
            preset: variant.preset,
            engine,
            queryVectors: answerRejectQueryVectors,
        });
        const answerQuality = await evaluateAnswerQualityVariant({
            datasetFile: ANSWER_QUALITY_DATASET_FILE,
            preset: variant.preset,
            engine,
            queryVectors: answerQualityQueryVectors,
        });

        reports.push({
            label: variant.label,
            note: variant.note,
            preset: {
                name: variant.preset.name,
                qConfusionMode: variant.preset.retrieval.qConfusionMode,
                qConfusionWeight: variant.preset.retrieval.qConfusionWeight,
                enableExplicitYearFilter:
                    variant.preset.retrieval.enableExplicitYearFilter,
                enablePhaseAnchorBoost:
                    variant.preset.retrieval.enablePhaseAnchorBoost,
                useYearPhaseTitleAdjustment:
                    variant.preset.display.useYearPhaseTitleAdjustment,
            },
            answerReject,
            answerQuality,
        });
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        answerRejectDatasetFile: ANSWER_REJECT_DATASET_FILE,
        answerQualityDatasetFile: ANSWER_QUALITY_DATASET_FILE,
        variants: reports,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        `behavior_runtime_ablation_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    reports.forEach((item) => {
        console.log(
            [
                item.label,
                `AR-acc=${(item.answerReject.binaryBehaviorAccuracy * 100).toFixed(2)}%`,
                `AR-docHit=${(item.answerReject.answerDocHitRate * 100).toFixed(2)}%`,
                `AQ-correct@1=${(item.answerQuality.correctAt1Rate * 100).toFixed(2)}%`,
                `AQ-retrieval@1=${(item.answerQuality.retrievalDocHitAt1Rate * 100).toFixed(2)}%`,
            ].join(" | "),
        );
    });
    console.log(`Saved report to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

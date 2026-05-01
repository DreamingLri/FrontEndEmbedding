import * as fs from "fs";
import * as path from "path";

import {
    loadAnswerRejectDataset,
    type AnswerRejectCase,
} from "./answer_reject_dataset.ts";
import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    executeSearchPipeline,
} from "../src/worker/search_pipeline.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";
import { ACTIVE_MAIN_DB_VERSION } from "./eval_shared.ts";

type DatasetDefinition = {
    label: string;
    file: string;
};

type CaseScore = {
    dataset: string;
    id: string;
    query: string;
    expectedBehavior: "answer" | "reject";
    predictedBehavior: "answer" | "reject";
    rejectScore: number;
    rejectTier: string | null;
    queryType: string;
    themeFamily: string;
    challengeTags: string[];
    queryLength: number;
};

type ScoreDistribution = {
    count: number;
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    mean: number;
};

type BinaryCalibrationSummary = {
    total: number;
    answerCount: number;
    rejectCount: number;
    auroc: number;
    auprc: number;
};

type ThresholdPoint = {
    threshold: number;
    answerCoverage: number;
    rejectCoverage: number;
    behaviorAccuracy: number;
    falseRejectRate: number;
    unsafeAnswerRate: number;
};

type SliceSummary = {
    label: string;
    count: number;
    answerCount: number;
    rejectCount: number;
    distribution: ScoreDistribution;
};

type DatasetReport = {
    label: string;
    file: string;
    summary: BinaryCalibrationSummary;
    answerDistribution: ScoreDistribution;
    rejectDistribution: ScoreDistribution;
    thresholdPoints: ThresholdPoint[];
    selectedThresholds: ThresholdPoint[];
    slices: SliceSummary[];
};

type Report = {
    generatedAt: string;
    mainDbVersion: string;
    pipelinePresetName: string;
    datasets: DatasetReport[];
};

const DATASETS: DatasetDefinition[] = [
    {
        label: "AnswerReject80",
        file: CURRENT_EVAL_DATASET_FILES.answerRejectCurrent,
    },
    {
        label: "AnswerRejectMixedLongTail59",
        file: CURRENT_EVAL_DATASET_FILES.answerRejectMixedLongTail59,
    },
];

const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;
const THRESHOLD_STEPS = Array.from({ length: 51 }, (_, index) => index / 50);
const SELECTED_THRESHOLDS = new Set([0.4, 0.5, 0.68]);

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function roundMetric(value: number): number {
    return Number.parseFloat(value.toFixed(4));
}

function percentile(sortedValues: number[], p: number): number {
    if (sortedValues.length === 0) {
        return 0;
    }
    const position = (sortedValues.length - 1) * p;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) {
        return sortedValues[lower]!;
    }
    const weight = position - lower;
    return (
        sortedValues[lower]! * (1 - weight) + sortedValues[upper]! * weight
    );
}

function buildDistribution(values: number[]): ScoreDistribution {
    const sorted = [...values].sort((a, b) => a - b);
    const count = sorted.length;
    const mean =
        count > 0
            ? sorted.reduce((sum, value) => sum + value, 0) / count
            : 0;
    return {
        count,
        min: count > 0 ? roundMetric(sorted[0]!) : 0,
        q1: roundMetric(percentile(sorted, 0.25)),
        median: roundMetric(percentile(sorted, 0.5)),
        q3: roundMetric(percentile(sorted, 0.75)),
        max: count > 0 ? roundMetric(sorted[count - 1]!) : 0,
        mean: roundMetric(mean),
    };
}

function buildAuRoc(caseScores: CaseScore[]): number {
    const positives = caseScores.filter(
        (item) => item.expectedBehavior === "reject",
    );
    const negatives = caseScores.filter(
        (item) => item.expectedBehavior === "answer",
    );
    if (positives.length === 0 || negatives.length === 0) {
        return 0;
    }

    const sorted = [...caseScores].sort((a, b) => b.rejectScore - a.rejectScore);
    let tp = 0;
    let fp = 0;
    let previousTpr = 0;
    let previousFpr = 0;
    let area = 0;

    for (const item of sorted) {
        if (item.expectedBehavior === "reject") {
            tp += 1;
        } else {
            fp += 1;
        }
        const tpr = tp / positives.length;
        const fpr = fp / negatives.length;
        area += (fpr - previousFpr) * (tpr + previousTpr) * 0.5;
        previousTpr = tpr;
        previousFpr = fpr;
    }

    return roundMetric(area);
}

function buildAuPrc(caseScores: CaseScore[]): number {
    const positives = caseScores.filter(
        (item) => item.expectedBehavior === "reject",
    ).length;
    if (positives === 0) {
        return 0;
    }

    const sorted = [...caseScores].sort((a, b) => b.rejectScore - a.rejectScore);
    let tp = 0;
    let fp = 0;
    let previousRecall = 0;
    let area = 0;

    for (const item of sorted) {
        if (item.expectedBehavior === "reject") {
            tp += 1;
        } else {
            fp += 1;
        }
        const precision = safeRate(tp, tp + fp);
        const recall = safeRate(tp, positives);
        area += (recall - previousRecall) * precision;
        previousRecall = recall;
    }

    return roundMetric(area);
}

function buildThresholdPoint(
    caseScores: CaseScore[],
    threshold: number,
): ThresholdPoint {
    let predictedAnswer = 0;
    let predictedReject = 0;
    let correct = 0;
    let answerTotal = 0;
    let falseReject = 0;
    let rejectTotal = 0;
    let unsafeAnswer = 0;

    for (const item of caseScores) {
        const predictedBehavior =
            item.rejectScore >= threshold ? "reject" : "answer";
        if (predictedBehavior === "answer") {
            predictedAnswer += 1;
        } else {
            predictedReject += 1;
        }
        if (predictedBehavior === item.expectedBehavior) {
            correct += 1;
        }
        if (item.expectedBehavior === "answer") {
            answerTotal += 1;
            if (predictedBehavior === "reject") {
                falseReject += 1;
            }
        } else {
            rejectTotal += 1;
            if (predictedBehavior === "answer") {
                unsafeAnswer += 1;
            }
        }
    }

    return {
        threshold: roundMetric(threshold),
        answerCoverage: roundMetric(safeRate(predictedAnswer, caseScores.length)),
        rejectCoverage: roundMetric(safeRate(predictedReject, caseScores.length)),
        behaviorAccuracy: roundMetric(safeRate(correct, caseScores.length)),
        falseRejectRate: roundMetric(safeRate(falseReject, answerTotal)),
        unsafeAnswerRate: roundMetric(safeRate(unsafeAnswer, rejectTotal)),
    };
}

function buildSliceSummary(
    label: string,
    caseScores: CaseScore[],
    predicate: (item: CaseScore) => boolean,
): SliceSummary | null {
    const filtered = caseScores.filter(predicate);
    if (filtered.length === 0) {
        return null;
    }
    return {
        label,
        count: filtered.length,
        answerCount: filtered.filter(
            (item) => item.expectedBehavior === "answer",
        ).length,
        rejectCount: filtered.filter(
            (item) => item.expectedBehavior === "reject",
        ).length,
        distribution: buildDistribution(
            filtered.map((item) => item.rejectScore),
        ),
    };
}

async function evaluateDataset(
    dataset: DatasetDefinition,
): Promise<DatasetReport> {
    const { cases } = loadAnswerRejectDataset(
        path.resolve(process.cwd(), dataset.file),
    );
    const engine = await loadFrontendEvalEngine();
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        cases.map((item) => item.query),
        engine.dimensions,
    );

    const caseScores: CaseScore[] = [];

    for (let index = 0; index < cases.length; index += 1) {
        const testCase = cases[index]!;
        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            engine.vocabMap,
            engine.topicPartitionIndex,
            FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        );
        const pipelineResult = await executeSearchPipeline({
            query: testCase.query,
            queryVector: queryVectors[index]!,
            queryContext,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: engine.bm25Stats,
            documentLoader,
            termMaps,
            preset: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
        });

        caseScores.push({
            dataset: dataset.label,
            id: testCase.id,
            query: testCase.query,
            expectedBehavior: testCase.expected_behavior,
            predictedBehavior: pipelineResult.finalDecision.behavior,
            rejectScore: pipelineResult.finalDecision.rejectScore ?? 0,
            rejectTier: pipelineResult.finalDecision.rejectTier ?? null,
            queryType: testCase.query_type || "standard",
            themeFamily: testCase.theme_family || "",
            challengeTags: testCase.challenge_tags || [],
            queryLength: testCase.query.replace(/\s+/g, "").length,
        });
    }

    const answerScores = caseScores
        .filter((item) => item.expectedBehavior === "answer")
        .map((item) => item.rejectScore);
    const rejectScores = caseScores
        .filter((item) => item.expectedBehavior === "reject")
        .map((item) => item.rejectScore);

    const summary: BinaryCalibrationSummary = {
        total: caseScores.length,
        answerCount: answerScores.length,
        rejectCount: rejectScores.length,
        auroc: buildAuRoc(caseScores),
        auprc: buildAuPrc(caseScores),
    };

    const thresholdPoints = THRESHOLD_STEPS.map((threshold) =>
        buildThresholdPoint(caseScores, threshold),
    );

    const slices: SliceSummary[] = [];
    const sliceDefinitions: Array<{
        label: string;
        predicate: (item: CaseScore) => boolean;
    }> = [
        {
            label: "answer_only",
            predicate: (item) => item.expectedBehavior === "answer",
        },
        {
            label: "reject_only",
            predicate: (item) => item.expectedBehavior === "reject",
        },
        {
            label: "short_query",
            predicate: (item) =>
                item.challengeTags.includes("short_query") || item.queryLength <= 12,
        },
        {
            label: "year_omitted",
            predicate: (item) => item.challengeTags.includes("year_omitted"),
        },
        {
            label: "latest_within_topic",
            predicate: (item) => item.challengeTags.includes("latest_within_topic"),
        },
        {
            label: "direct_answer_simple",
            predicate: (item) => item.themeFamily === "direct_answer_simple",
        },
        {
            label: "direct_answer_complex",
            predicate: (item) => item.themeFamily === "direct_answer_complex",
        },
    ];

    for (const sliceDefinition of sliceDefinitions) {
        const slice = buildSliceSummary(
            sliceDefinition.label,
            caseScores,
            sliceDefinition.predicate,
        );
        if (slice) {
            slices.push(slice);
        }
    }

    return {
        label: dataset.label,
        file: dataset.file,
        summary,
        answerDistribution: buildDistribution(answerScores),
        rejectDistribution: buildDistribution(rejectScores),
        thresholdPoints,
        selectedThresholds: thresholdPoints.filter((item) =>
            SELECTED_THRESHOLDS.has(item.threshold),
        ),
        slices,
    };
}

async function main() {
    fs.mkdirSync(RESULTS_DIR, { recursive: true });

    const datasets: DatasetReport[] = [];
    for (const dataset of DATASETS) {
        console.log(`Evaluating reject-score calibration on ${dataset.label}`);
        datasets.push(await evaluateDataset(dataset));
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        mainDbVersion: ACTIVE_MAIN_DB_VERSION,
        pipelinePresetName: FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name,
        datasets,
    };

    const outputPath = path.join(
        RESULTS_DIR,
        `reject_score_calibration_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Saved reject-score calibration report to ${outputPath}`);
    for (const dataset of datasets) {
        console.log(
            [
                `${dataset.label}: total=${dataset.summary.total}`,
                `AUROC=${dataset.summary.auroc.toFixed(4)}`,
                `AUPRC=${dataset.summary.auprc.toFixed(4)}`,
                `answer=[${dataset.answerDistribution.min.toFixed(4)}, ${dataset.answerDistribution.max.toFixed(4)}]`,
                `reject=[${dataset.rejectDistribution.min.toFixed(4)}, ${dataset.rejectDistribution.max.toFixed(4)}]`,
            ].join(" | "),
        );
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

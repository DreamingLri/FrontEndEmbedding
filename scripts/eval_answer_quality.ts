import * as fs from "fs";
import * as path from "path";

import { loadAnswerQualityDataset } from "./answer_quality_dataset.ts";
import {
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET,
    buildPipelineTermMaps,
    buildSearchPipelineQueryContext,
    executeSearchPipeline,
    resolvePipelinePresetByName,
    type PipelineBehavior,
    type PipelinePreset,
} from "../src/worker/search_pipeline.ts";
import {
    embedQueries as embedFrontendQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { CURRENT_EVAL_DATASET_FILES } from "./current_eval_targets.ts";
import { createLocalDocumentLoader } from "./local_document_provider.ts";
import {
    buildAnswerQualityResultFileName,
    resolveNamedDatasetProfile,
} from "./result_naming.ts";

type CaseReport = {
    id: string;
    query: string;
    query_type?: string;
    query_scope?: string;
    theme_family?: string;
    query_style_mode?: string;
    predicted_behavior: PipelineBehavior;
    retrieval_behavior: PipelineBehavior;
    false_reject: boolean;
    potentially_misleading: boolean;
    expected_otid: string;
    top_result_otid: string | null;
    retrieval_doc_rank: number | null;
    rendered_doc_rank: number | null;
    doc_hit_at_1: boolean;
    doc_hit_at_3: boolean;
    doc_hit_at_5: boolean;
    retrieval_doc_hit_at_1: boolean;
    rejection_reason: string | null;
    retrieval_reject_score: number | null;
    final_reject_score: number | null;
    weak_match_count: number;
    match_count: number;
    candidate_count: number;
    top_matches: Array<{
        rank: number;
        otid: string;
        score: number;
        best_kpid?: string;
    }>;
    top_results: Array<{
        rank: number;
        otid: string;
        displayScore: number;
        snippetScore?: number;
        best_kpid?: string;
    }>;
};

type RateSummary = {
    total: number;
    count: number;
    rate: number;
};

type SliceSummary = {
    total: number;
    answerRate: number;
    correctAt1Rate: number;
    correctAt3Rate: number;
    correctAt5Rate: number;
    falseRejectRate: number;
    potentiallyMisleadingRate: number;
};

type Summary = {
    total: number;
    answered: number;
    answerRate: number;
    falseReject: number;
    falseRejectRate: number;
    correctAt1: number;
    correctAt1Rate: number;
    correctAt3: number;
    correctAt3Rate: number;
    correctAt5: number;
    correctAt5Rate: number;
    retrievalDocHitAt1: number;
    retrievalDocHitAt1Rate: number;
    potentiallyMisleading: number;
    potentiallyMisleadingRate: number;
    byPredictedBehavior: Record<PipelineBehavior, number>;
    byQueryScope: Record<string, SliceSummary>;
    byQueryType: Record<string, SliceSummary>;
};

type Report = {
    generatedAt: string;
    datasetFile: string;
    datasetName: string;
    datasetAlias?: string;
    datasetDisplayName?: string;
    total: number;
    config: {
        pipelineVersion: string;
        preset: PipelinePreset;
        note: string;
    };
    summary: Summary;
    caseReports: CaseReport[];
};

const DATASET_FILE = path.resolve(
    process.cwd(),
    process.env.SUASK_ANSWER_QUALITY_DATASET_FILE ||
        CURRENT_EVAL_DATASET_FILES.answerQualityBlindProvisionalV1,
);
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const CURRENT_TIMESTAMP = Date.now() / 1000;
const DEFAULT_REPORT_NOTE =
    "当前报告面向 answer-only blind 集，关注回答正确率与潜在误导率；默认入口为 provisional AnswerQuality-blind。";
const REPORT_NOTE = process.env.SUASK_ANSWER_QUALITY_NOTE || DEFAULT_REPORT_NOTE;
const PIPELINE_PRESET_NAME =
    process.env.SUASK_PIPELINE_PRESET ||
    FRONTEND_RESEARCH_SYNC_PIPELINE_PRESET.name;
const EVAL_PRESET = resolvePipelinePresetByName(PIPELINE_PRESET_NAME);

function safeRate(numerator: number, denominator: number): number {
    return denominator > 0 ? numerator / denominator : 0;
}

function getRankByOtid(otids: string[], expectedOtid: string): number | null {
    const index = otids.findIndex((otid) => otid === expectedOtid);
    return index >= 0 ? index + 1 : null;
}

function buildRateSummary(caseReports: CaseReport[], predicate: (item: CaseReport) => boolean): RateSummary {
    const total = caseReports.length;
    const count = caseReports.filter(predicate).length;
    return {
        total,
        count,
        rate: safeRate(count, total),
    };
}

function buildSliceSummary(caseReports: CaseReport[]): SliceSummary {
    return {
        total: caseReports.length,
        answerRate: safeRate(
            caseReports.filter((item) => item.predicted_behavior === "answer").length,
            caseReports.length,
        ),
        correctAt1Rate: safeRate(
            caseReports.filter((item) => item.doc_hit_at_1).length,
            caseReports.length,
        ),
        correctAt3Rate: safeRate(
            caseReports.filter((item) => item.doc_hit_at_3).length,
            caseReports.length,
        ),
        correctAt5Rate: safeRate(
            caseReports.filter((item) => item.doc_hit_at_5).length,
            caseReports.length,
        ),
        falseRejectRate: safeRate(
            caseReports.filter((item) => item.false_reject).length,
            caseReports.length,
        ),
        potentiallyMisleadingRate: safeRate(
            caseReports.filter((item) => item.potentially_misleading).length,
            caseReports.length,
        ),
    };
}

function buildSummary(caseReports: CaseReport[]): Summary {
    const byPredictedBehavior: Record<PipelineBehavior, number> = {
        answer: 0,
        reject: 0,
    };
    const byQueryScopeBuckets = new Map<string, CaseReport[]>();
    const byQueryTypeBuckets = new Map<string, CaseReport[]>();

    caseReports.forEach((item) => {
        byPredictedBehavior[item.predicted_behavior] += 1;
        const queryScope = item.query_scope || "unknown";
        const queryType = item.query_type || "unknown";
        byQueryScopeBuckets.set(queryScope, [
            ...(byQueryScopeBuckets.get(queryScope) || []),
            item,
        ]);
        byQueryTypeBuckets.set(queryType, [
            ...(byQueryTypeBuckets.get(queryType) || []),
            item,
        ]);
    });

    return {
        total: caseReports.length,
        answered: buildRateSummary(caseReports, (item) => item.predicted_behavior === "answer").count,
        answerRate: buildRateSummary(caseReports, (item) => item.predicted_behavior === "answer").rate,
        falseReject: buildRateSummary(caseReports, (item) => item.false_reject).count,
        falseRejectRate: buildRateSummary(caseReports, (item) => item.false_reject).rate,
        correctAt1: buildRateSummary(caseReports, (item) => item.doc_hit_at_1).count,
        correctAt1Rate: buildRateSummary(caseReports, (item) => item.doc_hit_at_1).rate,
        correctAt3: buildRateSummary(caseReports, (item) => item.doc_hit_at_3).count,
        correctAt3Rate: buildRateSummary(caseReports, (item) => item.doc_hit_at_3).rate,
        correctAt5: buildRateSummary(caseReports, (item) => item.doc_hit_at_5).count,
        correctAt5Rate: buildRateSummary(caseReports, (item) => item.doc_hit_at_5).rate,
        retrievalDocHitAt1: buildRateSummary(caseReports, (item) => item.retrieval_doc_hit_at_1).count,
        retrievalDocHitAt1Rate: buildRateSummary(caseReports, (item) => item.retrieval_doc_hit_at_1).rate,
        potentiallyMisleading: buildRateSummary(caseReports, (item) => item.potentially_misleading).count,
        potentiallyMisleadingRate: buildRateSummary(caseReports, (item) => item.potentially_misleading).rate,
        byPredictedBehavior,
        byQueryScope: Object.fromEntries(
            Array.from(byQueryScopeBuckets.entries()).map(([key, items]) => [
                key,
                buildSliceSummary(items),
            ]),
        ),
        byQueryType: Object.fromEntries(
            Array.from(byQueryTypeBuckets.entries()).map(([key, items]) => [
                key,
                buildSliceSummary(items),
            ]),
        ),
    };
}

async function main() {
    const datasetName = path.basename(DATASET_FILE, path.extname(DATASET_FILE));
    const datasetProfile = resolveNamedDatasetProfile(datasetName);
    const { cases: testCases, datasetNote } = loadAnswerQualityDataset(DATASET_FILE);

    console.log(`Loading answer_quality dataset: ${DATASET_FILE}`);
    console.log(`Loaded ${testCases.length} cases.`);

    const engine = await loadFrontendEvalEngine();
    const termMaps = buildPipelineTermMaps(engine.vocabMap);
    const documentLoader = createLocalDocumentLoader();
    const queryVectors = await embedFrontendQueries(
        engine.extractor,
        testCases.map((item) => item.query),
        engine.dimensions,
    );

    const caseReports: CaseReport[] = [];

    for (let index = 0; index < testCases.length; index += 1) {
        const testCase = testCases[index];
        const queryContext = buildSearchPipelineQueryContext(
            testCase.query,
            engine.vocabMap,
            engine.topicPartitionIndex,
            EVAL_PRESET,
        );
        const pipelineResult = await executeSearchPipeline({
            query: testCase.query,
            queryVector: queryVectors[index],
            queryContext,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: CURRENT_TIMESTAMP,
            bm25Stats: engine.bm25Stats,
            extractor: engine.extractor,
            documentLoader,
            termMaps,
            preset: EVAL_PRESET,
        });

        const predictedBehavior = pipelineResult.finalDecision.behavior;
        const retrievalOtids = pipelineResult.searchOutput.matches.map((item) => item.otid);
        const renderedOtids = pipelineResult.results.map((item) => item.otid);
        const retrievalDocRank = getRankByOtid(retrievalOtids, testCase.expected_otid);
        const renderedDocRank =
            predictedBehavior === "answer"
                ? getRankByOtid(renderedOtids, testCase.expected_otid)
                : null;
        const topResultOtid =
            predictedBehavior === "answer"
                ? pipelineResult.results[0]?.otid ||
                  pipelineResult.searchOutput.matches[0]?.otid ||
                  null
                : null;

        caseReports.push({
            id: testCase.id,
            query: testCase.query,
            query_type: testCase.query_type,
            query_scope: testCase.query_scope,
            theme_family: testCase.theme_family,
            query_style_mode: testCase.query_style_mode,
            predicted_behavior: predictedBehavior,
            retrieval_behavior: pipelineResult.retrievalDecision.behavior,
            false_reject: predictedBehavior === "reject",
            potentially_misleading:
                predictedBehavior === "answer" && renderedDocRank !== 1,
            expected_otid: testCase.expected_otid,
            top_result_otid: topResultOtid,
            retrieval_doc_rank: retrievalDocRank,
            rendered_doc_rank: renderedDocRank,
            doc_hit_at_1: renderedDocRank === 1,
            doc_hit_at_3: renderedDocRank !== null && renderedDocRank <= 3,
            doc_hit_at_5: renderedDocRank !== null && renderedDocRank <= 5,
            retrieval_doc_hit_at_1: retrievalDocRank === 1,
            rejection_reason:
                pipelineResult.finalDecision.rejectionReason ||
                pipelineResult.rejection?.reason ||
                null,
            retrieval_reject_score:
                pipelineResult.retrievalDecision.rejectScore ?? null,
            final_reject_score:
                pipelineResult.finalDecision.rejectScore ?? null,
            weak_match_count: pipelineResult.trace.weakMatchCount,
            match_count: pipelineResult.trace.matchCount,
            candidate_count: pipelineResult.trace.candidateCount,
            top_matches: pipelineResult.searchOutput.matches
                .slice(0, 5)
                .map((match, rank) => ({
                    rank: rank + 1,
                    otid: match.otid,
                    score: match.score,
                    best_kpid: match.best_kpid,
                })),
            top_results: pipelineResult.results
                .slice(0, 5)
                .map((result, rank) => ({
                    rank: rank + 1,
                    otid: result.otid,
                    displayScore: result.displayScore,
                    snippetScore: result.bestSnippetScore,
                    best_kpid: result.bestKpid,
                })),
        });
    }

    const report: Report = {
        generatedAt: new Date().toISOString(),
        datasetFile: DATASET_FILE,
        datasetName,
        datasetAlias: datasetProfile.alias,
        datasetDisplayName: datasetProfile.displayName,
        total: caseReports.length,
        config: {
            pipelineVersion: EVAL_PRESET.name,
            preset: EVAL_PRESET,
            note: datasetNote ? `${REPORT_NOTE} ${datasetNote}` : REPORT_NOTE,
        },
        summary: buildSummary(caseReports),
        caseReports,
    };

    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const outputPath = path.join(
        RESULTS_DIR,
        buildAnswerQualityResultFileName(datasetName, Date.now()),
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Saved report to ${outputPath}`);
    console.log(
        [
            `answerRate=${(report.summary.answerRate * 100).toFixed(2)}%`,
            `correctAt1=${(report.summary.correctAt1Rate * 100).toFixed(2)}%`,
            `correctAt3=${(report.summary.correctAt3Rate * 100).toFixed(2)}%`,
            `correctAt5=${(report.summary.correctAt5Rate * 100).toFixed(2)}%`,
            `falseRejectRate=${(report.summary.falseRejectRate * 100).toFixed(2)}%`,
            `potentiallyMisleadingRate=${(report.summary.potentiallyMisleadingRate * 100).toFixed(2)}%`,
            `retrievalDocHitAt1=${(report.summary.retrievalDocHitAt1Rate * 100).toFixed(2)}%`,
        ].join(" | "),
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

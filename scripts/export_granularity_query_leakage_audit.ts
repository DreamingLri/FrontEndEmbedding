import * as fs from "fs";
import * as path from "path";

import {
    DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS,
    loadDataset,
    resolveGranularityDatasetTarget,
    type GranularityDatasetTargetKey,
} from "./eval_shared.ts";
import { MAIN_DB_VERSION, resolveBackendArticlesFile } from "./kb_version_paths.ts";

type CorpusArticle = {
    otid?: string;
    knowledge_points?: Array<{
        questions?: Array<string | { question?: string; q_text?: string; text?: string }>;
    }>;
};

type AuditCase = {
    id?: string;
    query: string;
    dataset: string;
    expected_otid?: string;
};

type AuditDatasetInput = {
    label: string;
    datasetFile: string;
    source: "target_key" | "explicit_file";
    key?: GranularityDatasetTargetKey;
};

type IndexQuestionEntry = {
    otid: string;
    question: string;
};

type AuditRow = {
    dataset: string;
    id?: string;
    query: string;
    expected_otid?: string;
    maxRougeL: number;
    bestIndexQuestion: string;
    sameTargetQuestionCount: number;
    nonTargetQuestionCount: number;
    sameTargetMaxRougeL: number | null;
    sameTargetBestQuestion: string;
    nonTargetMaxRougeL: number | null;
    nonTargetBestQuestion: string;
    sameVsNonTargetMargin: number | null;
};

const DEFAULT_AUDIT_TARGETS: GranularityDatasetTargetKey[] = [
    "ladder_main_balanced_150",
    "in_domain_generalization_100",
    "blind_ext_ood_100",
];

const THRESHOLD = parseFloat(process.env.SUASK_QUERY_LEAKAGE_THRESHOLD || "0.80");
const EXPLICIT_DATASET_FILES = parseCsv(
    process.env.SUASK_QUERY_LEAKAGE_DATASET_FILES || "",
);
const EXPLICIT_DATASET_LABELS = parseCsv(
    process.env.SUASK_QUERY_LEAKAGE_DATASET_LABELS || "",
);
const TARGET_KEYS = parseTargetKeys(
    process.env.SUASK_QUERY_LEAKAGE_DATASETS ||
        DEFAULT_AUDIT_TARGETS.join(","),
);

function parseCsv(raw: string): string[] {
    return raw
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
}

function parseTargetKeys(raw: string): GranularityDatasetTargetKey[] {
    return raw
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean) as GranularityDatasetTargetKey[];
}

function normalizeText(value: string): string {
    return value
        .replace(/[\s，。！？、；：,.!?;:（）()【】\[\]“”"'`]/g, "")
        .toLowerCase();
}

function lcsLength(a: string, b: string): number {
    if (a.length > b.length) {
        [a, b] = [b, a];
    }

    let previous = new Uint16Array(b.length + 1);
    let current = new Uint16Array(b.length + 1);

    for (let i = 1; i <= a.length; i += 1) {
        const charCode = a.charCodeAt(i - 1);
        for (let j = 1; j <= b.length; j += 1) {
            current[j] =
                charCode === b.charCodeAt(j - 1)
                    ? previous[j - 1] + 1
                    : Math.max(previous[j], current[j - 1]);
        }
        [previous, current] = [current, previous];
        current.fill(0);
    }

    return previous[b.length];
}

function rougeLF1(query: string, indexQuestion: string): number {
    const a = normalizeText(query);
    const b = normalizeText(indexQuestion);
    if (!a || !b) {
        return 0;
    }
    const lcs = lcsLength(a, b);
    const precision = lcs / b.length;
    const recall = lcs / a.length;
    return precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;
}

function percentile(sortedValues: readonly number[], p: number): number {
    if (sortedValues.length === 0) {
        return 0;
    }
    const index = Math.min(
        sortedValues.length - 1,
        Math.max(0, Math.ceil(sortedValues.length * p) - 1),
    );
    return sortedValues[index];
}

function extractQuestionText(
    item: string | { question?: string; q_text?: string; text?: string },
): string | null {
    if (typeof item === "string") {
        return item;
    }
    return item.question || item.q_text || item.text || null;
}

function loadIndexQuestions(): IndexQuestionEntry[] {
    const corpusPath = resolveIndexQuestionCorpusPath();
    const corpus = JSON.parse(fs.readFileSync(corpusPath, "utf-8")) as CorpusArticle[];
    const questions: IndexQuestionEntry[] = [];
    for (const article of corpus) {
        const otid = article.otid;
        if (!otid) {
            continue;
        }
        for (const kp of article.knowledge_points || []) {
            for (const item of kp.questions || []) {
                const question = extractQuestionText(item);
                if (question) {
                    questions.push({
                        otid,
                        question,
                    });
                }
            }
        }
    }
    return questions;
}

function summarizeValues(values: readonly number[]) {
    const sorted = [...values].sort((a, b) => a - b);
    if (sorted.length === 0) {
        return {
            count: 0,
            mean: 0,
            median: 0,
            p90: 0,
            p95: 0,
            max: 0,
            min: 0,
        };
    }
    const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
    return {
        count: sorted.length,
        mean,
        median: percentile(sorted, 0.5),
        p90: percentile(sorted, 0.9),
        p95: percentile(sorted, 0.95),
        max: sorted[sorted.length - 1],
        min: sorted[0],
    };
}

function toNullableNumber(value: number): number | null {
    return Number.isFinite(value) ? value : null;
}

function resolveIndexQuestionCorpusPath(): string {
    const candidates = [
        path.resolve(
            process.cwd(),
            `../BackEnd/data/embeddings_v2/flattened_json_${MAIN_DB_VERSION}.json`,
        ),
        path.resolve(process.cwd(), "../BackEnd/data/embeddings_v2/flattened_json.json"),
        path.resolve(process.cwd(), resolveBackendArticlesFile()),
    ];
    const resolved = candidates.find((candidate) => fs.existsSync(candidate));
    if (!resolved) {
        throw new Error("Unable to locate corpus file for query leakage audit.");
    }
    return resolved;
}

function resolveAuditDatasetInputs(): AuditDatasetInput[] {
    if (EXPLICIT_DATASET_FILES.length > 0) {
        return EXPLICIT_DATASET_FILES.map((datasetFile, index) => ({
            label:
                EXPLICIT_DATASET_LABELS[index] ||
                path.basename(datasetFile, ".json"),
            datasetFile,
            source: "explicit_file",
        }));
    }

    return (TARGET_KEYS.length
        ? TARGET_KEYS
        : DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS
    ).map((key) => {
        const target = resolveGranularityDatasetTarget(key);
        return {
            label: target.label,
            datasetFile: target.datasetFile,
            source: "target_key" as const,
            key: target.key,
        };
    });
}

function loadAuditCases(inputs: readonly AuditDatasetInput[]): AuditCase[] {
    const cases: AuditCase[] = [];
    for (const input of inputs) {
        const items = loadDataset(input.datasetFile, {
            datasetLabel: input.label,
        }) as AuditCase[];
        for (const item of items) {
            cases.push({
                ...item,
                dataset: input.label,
            });
        }
    }
    return cases;
}

function buildAuditSummary(rows: readonly AuditRow[]) {
    const values = rows.map((item) => item.maxRougeL).sort((a, b) => a - b);
    const overThreshold = [...rows]
        .filter((item) => item.maxRougeL >= THRESHOLD)
        .sort((a, b) => b.maxRougeL - a.maxRougeL);
    const mean =
        values.length > 0
            ? values.reduce((sum, value) => sum + value, 0) / values.length
            : 0;
    const sameTargetValues = rows
        .map((item) => item.sameTargetMaxRougeL)
        .filter((value): value is number => value !== null);
    const nonTargetValues = rows
        .map((item) => item.nonTargetMaxRougeL)
        .filter((value): value is number => value !== null);
    const sameVsNonTargetMargins = rows
        .map((item) => item.sameVsNonTargetMargin)
        .filter((value): value is number => value !== null);
    const dominance = rows.reduce(
        (acc, item) => {
            if (
                item.sameTargetMaxRougeL === null ||
                item.nonTargetMaxRougeL === null
            ) {
                acc.missing += 1;
            } else if (item.sameTargetMaxRougeL > item.nonTargetMaxRougeL) {
                acc.sameTargetHigher += 1;
            } else if (item.sameTargetMaxRougeL < item.nonTargetMaxRougeL) {
                acc.nonTargetHigher += 1;
            } else {
                acc.equal += 1;
            }
            return acc;
        },
        {
            sameTargetHigher: 0,
            nonTargetHigher: 0,
            equal: 0,
            missing: 0,
        },
    );

    return {
        queryCount: rows.length,
        summary: {
            mean,
            median: percentile(values, 0.5),
            p90: percentile(values, 0.9),
            p95: percentile(values, 0.95),
            max: values[values.length - 1] || 0,
            min: values[0] || 0,
            overThresholdCount: overThreshold.length,
        },
        expectedOtidGroupedSummary: {
            note: "sameTarget compares each query against Q entries whose parent OTID equals expected_otid; nonTarget compares against all remaining Q entries.",
            sameTarget: summarizeValues(sameTargetValues),
            nonTarget: summarizeValues(nonTargetValues),
            sameVsNonTargetMargin: summarizeValues(sameVsNonTargetMargins),
            dominance,
        },
        overThreshold,
        topCases: [...rows]
            .sort((a, b) => b.maxRougeL - a.maxRougeL)
            .slice(0, 20),
        groupedTopCases: [...rows]
            .filter((item) => item.sameVsNonTargetMargin !== null)
            .sort(
                (a, b) =>
                    (b.sameVsNonTargetMargin ?? Number.NEGATIVE_INFINITY) -
                    (a.sameVsNonTargetMargin ?? Number.NEGATIVE_INFINITY),
            )
            .slice(0, 20),
    };
}

function main() {
    const indexQuestions = loadIndexQuestions();
    const auditInputs = resolveAuditDatasetInputs();
    const auditCases = loadAuditCases(auditInputs);
    const rows = auditCases.map((item) => {
        let maxRougeL = 0;
        let bestIndexQuestion = "";
        let sameTargetMax = Number.NEGATIVE_INFINITY;
        let sameTargetBestQuestion = "";
        let nonTargetMax = Number.NEGATIVE_INFINITY;
        let nonTargetBestQuestion = "";
        let sameTargetQuestionCount = 0;
        let nonTargetQuestionCount = 0;
        for (const indexQuestion of indexQuestions) {
            const score = rougeLF1(item.query, indexQuestion.question);
            if (score > maxRougeL) {
                maxRougeL = score;
                bestIndexQuestion = indexQuestion.question;
            }
            if (item.expected_otid && indexQuestion.otid === item.expected_otid) {
                sameTargetQuestionCount += 1;
                if (score > sameTargetMax) {
                    sameTargetMax = score;
                    sameTargetBestQuestion = indexQuestion.question;
                }
            } else {
                nonTargetQuestionCount += 1;
                if (score > nonTargetMax) {
                    nonTargetMax = score;
                    nonTargetBestQuestion = indexQuestion.question;
                }
            }
        }
        const sameTargetMaxValue = toNullableNumber(sameTargetMax);
        const nonTargetMaxValue = toNullableNumber(nonTargetMax);
        return {
            dataset: item.dataset,
            id: item.id,
            query: item.query,
            expected_otid: item.expected_otid,
            maxRougeL,
            bestIndexQuestion,
            sameTargetQuestionCount,
            nonTargetQuestionCount,
            sameTargetMaxRougeL: sameTargetMaxValue,
            sameTargetBestQuestion,
            nonTargetMaxRougeL: nonTargetMaxValue,
            nonTargetBestQuestion,
            sameVsNonTargetMargin:
                sameTargetMaxValue === null || nonTargetMaxValue === null
                    ? null
                    : sameTargetMaxValue - nonTargetMaxValue,
        };
    });
    const overall = buildAuditSummary(rows);
    const perDataset = Object.fromEntries(
        [...new Set(rows.map((item) => item.dataset))]
            .sort((a, b) => a.localeCompare(b))
            .map((dataset) => [
                dataset,
                buildAuditSummary(rows.filter((item) => item.dataset === dataset)),
            ]),
    );

    const report = {
        generatedAt: new Date().toISOString(),
        metric: "character-level ROUGE-L F1 after punctuation/space normalization",
        threshold: THRESHOLD,
        datasetInputMode:
            EXPLICIT_DATASET_FILES.length > 0 ? "explicit_files" : "target_keys",
        datasetTargets: auditInputs
            .map((item) => item.key)
            .filter((item): item is GranularityDatasetTargetKey => Boolean(item)),
        datasetInputs: auditInputs.map((item) => ({
            label: item.label,
            datasetFile: item.datasetFile,
            source: item.source,
            key: item.key ?? null,
        })),
        queryCount: overall.queryCount,
        indexQuestionCount: indexQuestions.length,
        summary: overall.summary,
        expectedOtidGroupedSummary: overall.expectedOtidGroupedSummary,
        overThreshold: overall.overThreshold,
        topCases: overall.topCases,
        groupedTopCases: overall.groupedTopCases,
        perDataset,
    };

    const outputDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outputDir, { recursive: true });
    const outputPath = path.resolve(
        outputDir,
        `granularity_query_leakage_audit_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(
        `Query leakage audit: n=${report.queryCount}, indexQ=${report.indexQuestionCount}, mean=${report.summary.mean.toFixed(4)}, p95=${report.summary.p95.toFixed(4)}, max=${report.summary.max.toFixed(4)}, overThreshold=${report.summary.overThresholdCount}`,
    );
    console.log(
        `Grouped audit: sameTargetMean=${report.expectedOtidGroupedSummary.sameTarget.mean.toFixed(4)}, nonTargetMean=${report.expectedOtidGroupedSummary.nonTarget.mean.toFixed(4)}, marginMean=${report.expectedOtidGroupedSummary.sameVsNonTargetMargin.mean.toFixed(4)}, sameHigher=${report.expectedOtidGroupedSummary.dominance.sameTargetHigher}, nonHigher=${report.expectedOtidGroupedSummary.dominance.nonTargetHigher}, equal=${report.expectedOtidGroupedSummary.dominance.equal}, missing=${report.expectedOtidGroupedSummary.dominance.missing}`,
    );
    console.log(`Report saved to ${outputPath}`);

    if (
        report.summary.overThresholdCount > 0 &&
        process.env.SUASK_QUERY_LEAKAGE_STRICT === "1"
    ) {
        process.exitCode = 1;
    }
}

main();

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
    knowledge_points?: Array<{
        questions?: Array<string | { question?: string; q_text?: string; text?: string }>;
    }>;
};

type AuditCase = {
    id?: string;
    query: string;
    dataset: string;
};

const DEFAULT_AUDIT_TARGETS: GranularityDatasetTargetKey[] = [
    "ladder_main_balanced_150",
    "in_domain_generalization_100",
    "blind_ext_ood_100",
];

const THRESHOLD = parseFloat(process.env.SUASK_QUERY_LEAKAGE_THRESHOLD || "0.80");
const TARGET_KEYS = parseTargetKeys(
    process.env.SUASK_QUERY_LEAKAGE_DATASETS ||
        DEFAULT_AUDIT_TARGETS.join(","),
);

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

function loadIndexQuestions(): string[] {
    const corpusPath = resolveIndexQuestionCorpusPath();
    const corpus = JSON.parse(fs.readFileSync(corpusPath, "utf-8")) as CorpusArticle[];
    const questions: string[] = [];
    for (const article of corpus) {
        for (const kp of article.knowledge_points || []) {
            for (const item of kp.questions || []) {
                const question = extractQuestionText(item);
                if (question) {
                    questions.push(question);
                }
            }
        }
    }
    return questions;
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

function loadAuditCases(): AuditCase[] {
    const cases: AuditCase[] = [];
    for (const key of TARGET_KEYS.length
        ? TARGET_KEYS
        : DEFAULT_GRANULARITY_MAINLINE_TARGET_KEYS) {
        const target = resolveGranularityDatasetTarget(key);
        const items = loadDataset(target.datasetFile, {
            datasetLabel: target.label,
        }) as AuditCase[];
        for (const item of items) {
            cases.push({
                ...item,
                dataset: target.label,
            });
        }
    }
    return cases;
}

function main() {
    const indexQuestions = loadIndexQuestions();
    const auditCases = loadAuditCases();
    const rows = auditCases.map((item) => {
        let maxRougeL = 0;
        let bestIndexQuestion = "";
        for (const indexQuestion of indexQuestions) {
            const score = rougeLF1(item.query, indexQuestion);
            if (score > maxRougeL) {
                maxRougeL = score;
                bestIndexQuestion = indexQuestion;
            }
        }
        return {
            dataset: item.dataset,
            id: item.id,
            query: item.query,
            maxRougeL,
            bestIndexQuestion,
        };
    });

    const values = rows.map((item) => item.maxRougeL).sort((a, b) => a - b);
    const overThreshold = rows
        .filter((item) => item.maxRougeL >= THRESHOLD)
        .sort((a, b) => b.maxRougeL - a.maxRougeL);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;

    const report = {
        generatedAt: new Date().toISOString(),
        metric: "character-level ROUGE-L F1 after punctuation/space normalization",
        threshold: THRESHOLD,
        datasetTargets: TARGET_KEYS,
        queryCount: auditCases.length,
        indexQuestionCount: indexQuestions.length,
        summary: {
            mean,
            median: percentile(values, 0.5),
            p90: percentile(values, 0.9),
            p95: percentile(values, 0.95),
            max: values[values.length - 1] || 0,
            overThresholdCount: overThreshold.length,
        },
        overThreshold,
        topCases: [...rows]
            .sort((a, b) => b.maxRougeL - a.maxRougeL)
            .slice(0, 20),
    };

    const outputDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outputDir, { recursive: true });
    const outputPath = path.resolve(
        outputDir,
        `granularity_query_leakage_audit_${Date.now()}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(
        `Query leakage audit: n=${report.queryCount}, indexQ=${report.indexQuestionCount}, mean=${mean.toFixed(4)}, p95=${report.summary.p95.toFixed(4)}, max=${report.summary.max.toFixed(4)}, overThreshold=${overThreshold.length}`,
    );
    console.log(`Report saved to ${outputPath}`);

    if (overThreshold.length > 0 && process.env.SUASK_QUERY_LEAKAGE_STRICT === "1") {
        process.exitCode = 1;
    }
}

main();

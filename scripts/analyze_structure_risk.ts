import * as fs from "fs";
import * as path from "path";

import { parseQueryIntent } from "../src/worker/vector_engine.ts";
import {
    buildQueryPlan,
    inferDocumentRolesFromTitle,
} from "../src/worker/query_planner.ts";
import { resolveBackendArticlesFile } from "./kb_version_paths.ts";

type TopMatch = {
    rank: number;
    otid: string;
    score: number;
    best_kpid?: string;
};

type PerCaseResult = {
    id?: string;
    query: string;
    expected_otid: string;
    rank: number | null;
    hitAt1: boolean;
    reciprocalRank: number;
    top1Otid?: string;
    topMatches: TopMatch[];
};

type ModelReport = {
    label: string;
    perCase: PerCaseResult[];
};

type DatasetReport = {
    datasetKey: string;
    datasetLabel: string;
    datasetFile: string;
    caseCount: number;
    models: ModelReport[];
};

type BaselineReport = {
    generatedAt: string;
    datasets: DatasetReport[];
};

type DatasetCase = {
    id?: string;
    query: string;
    source_query?: string;
    expected_otid: string;
    query_scope?: string;
    query_type?: string;
    support_pattern?: string;
    structure_dev_category?: string;
};

type ArticleRecord = {
    otid?: string;
    ot_title?: string;
};

type RiskCase = {
    id?: string;
    query: string;
    hitAt1: boolean;
    rank: number | null;
    flags: string[];
    riskLevel: "none" | "medium" | "high";
};

const DEFAULT_RESULT_FILE =
    process.argv[2] ||
    process.env.SUASK_STRUCTURE_RISK_RESULT_FILE ||
    "";

const FULL_MODEL_LABEL =
    process.env.SUASK_STRUCTURE_RISK_MODEL || "Structured-Q+KP+OT";
const KP_OT_MODEL_LABEL = "Structured-KP+OT";
const BM25_MODEL_LABEL = "BM25-OT";
const DENSE_MODEL_LABEL = "Dense-OT";

const LOW_MARGIN_RELATIVE_THRESHOLD = Number.parseFloat(
    process.env.SUASK_STRUCTURE_RISK_LOW_MARGIN || "0.03",
);

function resolveInputPath(rawPath: string): string {
    if (!rawPath) {
        throw new Error(
            "Usage: npm run eval:tool:structure-risk -- <baseline-result.json>",
        );
    }
    if (path.isAbsolute(rawPath)) {
        return rawPath;
    }
    const cwdPath = path.resolve(process.cwd(), rawPath);
    if (fs.existsSync(cwdPath)) {
        return cwdPath;
    }
    return path.resolve(process.cwd(), "scripts/results", rawPath);
}

function parseDatasetPath(datasetFile: string): string {
    const backendMatch = datasetFile.match(/Backend[\\/].*$/);
    if (backendMatch) {
        return path.resolve(process.cwd(), "..", backendMatch[0]);
    }
    return path.resolve(process.cwd(), datasetFile);
}

function loadDataset(datasetFile: string): Map<string, DatasetCase> {
    const absolutePath = parseDatasetPath(datasetFile);
    if (!fs.existsSync(absolutePath)) {
        return new Map<string, DatasetCase>();
    }
    const raw = JSON.parse(fs.readFileSync(absolutePath, "utf-8")) as DatasetCase[];
    return new Map(raw.map((item) => [item.id || item.query, item]));
}

function loadArticleTitles(): Map<string, string> {
    const absolutePath = path.resolve(process.cwd(), resolveBackendArticlesFile());
    if (!fs.existsSync(absolutePath)) {
        return new Map<string, string>();
    }
    const raw = JSON.parse(fs.readFileSync(absolutePath, "utf-8")) as ArticleRecord[];
    return new Map(
        raw
            .filter((item) => item.otid)
            .map((item) => [item.otid as string, item.ot_title || ""]),
    );
}

function normalizeText(text: string): string {
    return text.replace(/\s+/g, "");
}

function hasExplicitYear(text: string): boolean {
    return /20\d{2}/.test(text);
}

function isShortQuery(text: string): boolean {
    return normalizeText(text).length <= 12;
}

function hasBoundaryTerms(text: string): boolean {
    return /如果|是否|能不能|还能|不|未|没有|冲突|无效|有效|仍|淘汰|通过/.test(
        normalizeText(text),
    );
}

function countAspects(text: string): number {
    const normalized = normalizeText(text);
    return [
        /条件|资格|要求/,
        /材料|提交|准备/,
        /时间|日期|截止|地点/,
        /流程|步骤|程序|环节|从.+到/,
        /名单|结果|公示|录取|入营/,
        /联系方式|邮箱|电话/,
    ].filter((pattern) => pattern.test(normalized)).length;
}

function hasMultiAspectTerms(text: string): boolean {
    const normalized = normalizeText(text);
    return (
        /以及|另外|分别|同时|整体|完整|整个|概述|从.+到/.test(normalized) ||
        countAspects(normalized) >= 2
    );
}

function relativeTopMargin(item: PerCaseResult): number | null {
    if (item.topMatches.length < 2) {
        return null;
    }
    const top1 = item.topMatches[0]?.score;
    const top2 = item.topMatches[1]?.score;
    if (!Number.isFinite(top1) || !Number.isFinite(top2) || top1 === 0) {
        return null;
    }
    return (top1 - top2) / Math.abs(top1);
}

function hasRoleConflict(
    queryText: string,
    top1Otid: string | undefined,
    titleMap: Map<string, string>,
): boolean {
    if (!top1Otid) {
        return false;
    }
    const queryIntent = parseQueryIntent(queryText);
    const queryPlan = buildQueryPlan(queryText, queryIntent);
    if (queryPlan.avoidedDocRoles.length === 0) {
        return false;
    }
    const title = titleMap.get(top1Otid) || "";
    const roles = inferDocumentRolesFromTitle(title);
    return roles.some((role) => queryPlan.avoidedDocRoles.includes(role));
}

function getModel(dataset: DatasetReport, label: string): ModelReport | undefined {
    return dataset.models.find((model) => model.label === label);
}

function getCase(model: ModelReport | undefined, index: number): PerCaseResult | undefined {
    return model?.perCase[index];
}

function buildRiskFlags(params: {
    fullCase: PerCaseResult;
    kpOtCase?: PerCaseResult;
    bm25Case?: PerCaseResult;
    denseCase?: PerCaseResult;
    datasetCase?: DatasetCase;
    titleMap: Map<string, string>;
}): string[] {
    const { fullCase, kpOtCase, bm25Case, denseCase, datasetCase, titleMap } = params;
    const queryText = datasetCase?.source_query?.trim() || fullCase.query;
    const flags: string[] = [];

    if (!hasExplicitYear(queryText)) {
        flags.push("runtime_no_year");
    }
    if (isShortQuery(queryText)) {
        flags.push("runtime_short_query");
    }
    if (hasBoundaryTerms(queryText)) {
        flags.push("runtime_boundary_terms");
    }
    if (hasMultiAspectTerms(queryText)) {
        flags.push("runtime_multi_aspect");
    }

    const margin = relativeTopMargin(fullCase);
    if (margin !== null && margin <= LOW_MARGIN_RELATIVE_THRESHOLD) {
        flags.push("runtime_low_margin");
    }

    if (kpOtCase?.top1Otid && kpOtCase.top1Otid !== fullCase.top1Otid) {
        flags.push("runtime_kp_full_disagreement");
    }
    if (bm25Case?.top1Otid && bm25Case.top1Otid !== fullCase.top1Otid) {
        flags.push("runtime_bm25_full_disagreement");
    }
    if (denseCase?.top1Otid && denseCase.top1Otid !== fullCase.top1Otid) {
        flags.push("runtime_dense_full_disagreement");
    }
    if (hasRoleConflict(queryText, fullCase.top1Otid, titleMap)) {
        flags.push("runtime_doc_role_conflict");
    }

    return flags;
}

function countDisagreementFlags(flags: string[]): number {
    return [
        "runtime_kp_full_disagreement",
        "runtime_bm25_full_disagreement",
        "runtime_dense_full_disagreement",
    ].filter((flag) => flags.includes(flag)).length;
}

function inferRiskLevel(flags: string[]): RiskCase["riskLevel"] {
    const disagreementCount = countDisagreementFlags(flags);
    const hasBoundary = flags.includes("runtime_boundary_terms");
    const hasKpDisagreement = flags.includes("runtime_kp_full_disagreement");
    const hasNoYear = flags.includes("runtime_no_year");
    const hasMultiAspect = flags.includes("runtime_multi_aspect");

    if (
        flags.includes("runtime_low_margin") ||
        flags.includes("runtime_doc_role_conflict") ||
        flags.includes("runtime_short_query") ||
        (hasBoundary && hasKpDisagreement) ||
        (hasNoYear && disagreementCount >= 2) ||
        disagreementCount >= 3
    ) {
        return "high";
    }

    if (
        disagreementCount >= 2 ||
        (hasBoundary && disagreementCount >= 1) ||
        (hasMultiAspect && hasKpDisagreement)
    ) {
        return "medium";
    }

    return "none";
}

function formatPercent(numerator: number, denominator: number): string {
    return denominator === 0 ? "n/a" : `${((numerator / denominator) * 100).toFixed(2)}%`;
}

function summarizeDataset(
    dataset: DatasetReport,
    datasetCases: Map<string, DatasetCase>,
    titleMap: Map<string, string>,
): void {
    const fullModel = getModel(dataset, FULL_MODEL_LABEL);
    if (!fullModel) {
        console.warn(`Skip ${dataset.datasetKey}: model ${FULL_MODEL_LABEL} not found.`);
        return;
    }
    const kpOtModel = getModel(dataset, KP_OT_MODEL_LABEL);
    const bm25Model = getModel(dataset, BM25_MODEL_LABEL);
    const denseModel = getModel(dataset, DENSE_MODEL_LABEL);

    const cases: RiskCase[] = fullModel.perCase.map((fullCase, index) => {
        const datasetCase = datasetCases.get(fullCase.id || fullCase.query);
        const flags = buildRiskFlags({
            fullCase,
            kpOtCase: getCase(kpOtModel, index),
            bm25Case: getCase(bm25Model, index),
            denseCase: getCase(denseModel, index),
            datasetCase,
            titleMap,
        });
        return {
            id: fullCase.id,
            query: fullCase.query,
            hitAt1: fullCase.hitAt1,
            rank: fullCase.rank,
            flags,
            riskLevel: inferRiskLevel(flags),
        };
    });

    const misses = cases.filter((item) => !item.hitAt1);
    const risky = cases.filter((item) => item.flags.length > 0);
    const riskyMisses = risky.filter((item) => !item.hitAt1);
    const hitCases = cases.filter((item) => item.hitAt1);
    const riskyHits = risky.filter((item) => item.hitAt1);

    console.log(`\n## ${dataset.datasetLabel} (${dataset.datasetKey})`);
    console.log(`cases=${cases.length}, misses=${misses.length}`);
    console.log(
        [
            `riskCoverageOfMisses=${formatPercent(riskyMisses.length, misses.length)} (${riskyMisses.length}/${misses.length})`,
            `riskPrecision=${formatPercent(riskyMisses.length, risky.length)} (${riskyMisses.length}/${risky.length})`,
            `riskRateOnHits=${formatPercent(riskyHits.length, hitCases.length)} (${riskyHits.length}/${hitCases.length})`,
        ].join(" | "),
    );

    (["high", "medium"] as const).forEach((level) => {
        const selected =
            level === "high"
                ? cases.filter((item) => item.riskLevel === "high")
                : cases.filter((item) => item.riskLevel !== "none");
        const selectedMisses = selected.filter((item) => !item.hitAt1);
        const selectedHits = selected.filter((item) => item.hitAt1);
        console.log(
            [
                `${level}RiskCoverage=${formatPercent(selectedMisses.length, misses.length)} (${selectedMisses.length}/${misses.length})`,
                `${level}RiskPrecision=${formatPercent(selectedMisses.length, selected.length)} (${selectedMisses.length}/${selected.length})`,
                `${level}RiskRateOnHits=${formatPercent(selectedHits.length, hitCases.length)} (${selectedHits.length}/${hitCases.length})`,
            ].join(" | "),
        );
    });

    const flagStats = new Map<
        string,
        { total: number; misses: number; hits: number }
    >();
    cases.forEach((item) => {
        item.flags.forEach((flag) => {
            const current = flagStats.get(flag) || { total: 0, misses: 0, hits: 0 };
            current.total += 1;
            if (item.hitAt1) {
                current.hits += 1;
            } else {
                current.misses += 1;
            }
            flagStats.set(flag, current);
        });
    });

    [...flagStats.entries()]
        .sort((a, b) => b[1].misses - a[1].misses || b[1].total - a[1].total)
        .forEach(([flag, stats]) => {
            console.log(
                `${flag.padEnd(36)} total=${String(stats.total).padStart(3)} miss=${String(stats.misses).padStart(3)} missRate=${formatPercent(stats.misses, stats.total)}`,
            );
        });

    const uncoveredMisses = misses.filter((item) => item.flags.length === 0);
    if (uncoveredMisses.length > 0) {
        console.log("uncovered_misses:");
        uncoveredMisses.slice(0, 10).forEach((item) => {
            console.log(`- ${item.id || "(no-id)"} rank=${item.rank} q=${item.query}`);
        });
    }
}

function main() {
    const resultPath = resolveInputPath(DEFAULT_RESULT_FILE);
    const report = JSON.parse(
        fs.readFileSync(resultPath, "utf-8"),
    ) as BaselineReport;
    const titleMap = loadArticleTitles();

    console.log(`Structure risk analysis`);
    console.log(`result=${resultPath}`);
    console.log(`model=${FULL_MODEL_LABEL}`);
    console.log(`lowMarginThreshold=${LOW_MARGIN_RELATIVE_THRESHOLD}`);

    report.datasets.forEach((dataset) => {
        summarizeDataset(dataset, loadDataset(dataset.datasetFile), titleMap);
    });
}

main();

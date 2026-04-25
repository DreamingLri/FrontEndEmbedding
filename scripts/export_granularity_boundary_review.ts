import * as fs from "fs";
import * as path from "path";

type RationalityReviewDoc = {
    generatedAt?: string;
    datasetFile?: string | null;
    badCaseFile?: string | null;
    summary?: Record<string, unknown>;
    cases?: RationalityReviewCase[];
};

type RationalityReviewCase = {
    query?: string;
    expectedOtid?: string;
    expectedTitle?: string;
    top1Otid?: string;
    top1Title?: string;
    finalRank?: number | null;
    entryBucket?: string;
    bestCorrectRrfRank?: number | null;
    failureRisk?: string;
    supportPattern?: string;
    queryType?: string;
    queryScope?: string;
    themeFamily?: string;
    flags?: string[];
    titleSimilarity?: number;
    reviewPriority?: "high" | "medium" | "low";
    recommendation?: string;
    rationale?: string;
};

type DatasetCase = {
    query?: string;
    expected_otid?: string;
    acceptable_otids?: string[];
};

type DatasetInput = {
    label: string;
    reviewFile: string;
};

type BoundaryCase = {
    dataset: string;
    reviewPriority: "high" | "medium" | "low";
    proposedAction:
        | "accept_same_title_duplicate"
        | "manual_review_same_family"
        | "replace_or_anchor_query"
        | "verify_expected_boundary"
        | "keep_single_expected";
    recommendation: string;
    query: string;
    expectedOtid: string;
    expectedTitle: string;
    top1Otid: string;
    top1Title: string;
    finalRank: number | null;
    entryBucket: string;
    flags: string[];
    titleSimilarity: number;
    rationale: string;
};

const DEFAULT_DATASET_HINTS: Array<{ label: string; needle: string }> = [
    { label: "Main", needle: "main_generalization_aligned_120_draft_v3" },
    { label: "InDomain", needle: "in_domain_generalization_aligned_100_draft_v6" },
    { label: "BlindExtOOD", needle: "blind_ext_ood_generalization_aligned_100_draft_v3" },
];

function normalizePath(input: string): string {
    return path.resolve(process.cwd(), input);
}

function parseExplicitInputs(): DatasetInput[] | null {
    const raw = process.env.SUASK_BOUNDARY_REVIEW_INPUTS;
    if (!raw) {
        return null;
    }
    return raw
        .split(";")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => {
            const splitIndex = item.indexOf("=");
            if (splitIndex <= 0) {
                throw new Error(
                    `Invalid SUASK_BOUNDARY_REVIEW_INPUTS item "${item}". Expected Label=path.`,
                );
            }
            return {
                label: item.slice(0, splitIndex),
                reviewFile: normalizePath(item.slice(splitIndex + 1)),
            };
        });
}

function latestReviewForDataset(needle: string): string {
    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    const candidates = fs
        .readdirSync(resultsDir)
        .filter((name) => /^granularity_dataset_rationality_review_\d+\.json$/.test(name))
        .map((name) => {
            const fullPath = path.join(resultsDir, name);
            const doc = JSON.parse(fs.readFileSync(fullPath, "utf-8")) as RationalityReviewDoc;
            return {
                fullPath,
                datasetFile: doc.datasetFile || "",
                mtimeMs: fs.statSync(fullPath).mtimeMs,
            };
        })
        .filter((item) => item.datasetFile.includes(needle))
        .sort((left, right) => right.mtimeMs - left.mtimeMs);

    if (candidates.length === 0) {
        throw new Error(`No rationality review found for dataset needle "${needle}".`);
    }
    return candidates[0].fullPath;
}

function resolveInputs(): DatasetInput[] {
    const explicit = parseExplicitInputs();
    if (explicit) {
        return explicit;
    }
    return DEFAULT_DATASET_HINTS.map((item) => ({
        label: item.label,
        reviewFile: latestReviewForDataset(item.needle),
    }));
}

function includesAny(flags: string[], targets: string[]): boolean {
    return targets.some((target) => flags.includes(target));
}

function extractYears(text: string): number[] {
    return Array.from(text.matchAll(/20\d{2}/g))
        .map((match) => Number(match[0]))
        .filter((year, index, years) => years.indexOf(year) === index);
}

function sameYearSet(left: number[], right: number[]): boolean {
    return (
        left.length === right.length &&
        left.every((year) => right.includes(year))
    );
}

function isBoundaryCandidate(item: RationalityReviewCase): boolean {
    const flags = item.flags || [];
    const recommendation = item.recommendation || "";
    const expectedOtid = item.expectedOtid || "";
    const top1Otid = item.top1Otid || "";
    const sameExpectedTop1 = expectedOtid !== "" && expectedOtid === top1Otid;
    if (
        sameExpectedTop1 &&
        recommendation === "verify_eval_boundary_or_expected_otid"
    ) {
        return false;
    }
    return (
        recommendation === "verify_eval_boundary_or_expected_otid" ||
        recommendation === "expand_acceptable_otids_or_replace_case" ||
        includesAny(flags, [
            "near_duplicate_title",
            "same_family_version_chain",
            "metric_or_acceptable_boundary_check",
        ])
    );
}

function normalizeText(text: string): string {
    return text.replace(/\s+/g, "").trim();
}

function buildDatasetCaseMap(datasetFile: string | null | undefined): Map<string, DatasetCase> {
    if (!datasetFile) {
        return new Map();
    }
    const absolutePath = path.resolve(process.cwd(), datasetFile);
    if (!fs.existsSync(absolutePath)) {
        return new Map();
    }
    const rows = JSON.parse(fs.readFileSync(absolutePath, "utf-8")) as DatasetCase[];
    return new Map(
        rows.map((item) => [
            `${normalizeText(item.query || "")}::${item.expected_otid || ""}`,
            item,
        ]),
    );
}

function isAlreadyAcceptedByDataset(
    item: RationalityReviewCase,
    datasetCases: ReadonlyMap<string, DatasetCase>,
): boolean {
    const top1Otid = item.top1Otid || "";
    if (!top1Otid) {
        return false;
    }
    const datasetCase = datasetCases.get(
        `${normalizeText(item.query || "")}::${item.expectedOtid || ""}`,
    );
    if (!datasetCase) {
        return false;
    }
    return (
        top1Otid === datasetCase.expected_otid ||
        (Array.isArray(datasetCase.acceptable_otids) &&
            datasetCase.acceptable_otids.includes(top1Otid))
    );
}

function inferProposedAction(item: RationalityReviewCase): BoundaryCase["proposedAction"] {
    const flags = item.flags || [];
    const recommendation = item.recommendation || "";
    const similarity = item.titleSimilarity ?? 0;
    const expectedOtid = item.expectedOtid || "";
    const top1Otid = item.top1Otid || "";
    const expectedTitleYears = extractYears(item.expectedTitle || "");
    const top1TitleYears = extractYears(item.top1Title || "");
    const queryYears = extractYears(item.query || "");
    const hasTitleYearConflict =
        (expectedTitleYears.length > 0 || top1TitleYears.length > 0) &&
        !sameYearSet(expectedTitleYears, top1TitleYears);
    const queryAlreadyAnchorsExpectedYear =
        expectedTitleYears.length > 0 &&
        expectedTitleYears.every((year) => queryYears.includes(year));

    if (
        recommendation === "verify_eval_boundary_or_expected_otid" ||
        flags.includes("metric_or_acceptable_boundary_check")
    ) {
        return "verify_expected_boundary";
    }
    if (
        recommendation === "add_year_or_replace_case" ||
        recommendation === "anchor_query_with_entity_year_or_replace_case" ||
        (flags.includes("no_explicit_year") && flags.includes("competing_year_version")) ||
        (hasTitleYearConflict && !queryAlreadyAnchorsExpectedYear)
    ) {
        return "replace_or_anchor_query";
    }
    if (
        expectedOtid &&
        top1Otid &&
        expectedOtid !== top1Otid &&
        flags.includes("near_duplicate_title") &&
        !hasTitleYearConflict &&
        similarity >= 0.92
    ) {
        return "accept_same_title_duplicate";
    }
    if (flags.includes("same_family_version_chain")) {
        return "manual_review_same_family";
    }
    return "keep_single_expected";
}

function toBoundaryCase(dataset: string, item: RationalityReviewCase): BoundaryCase {
    return {
        dataset,
        reviewPriority: item.reviewPriority || "low",
        proposedAction: inferProposedAction(item),
        recommendation: item.recommendation || "manual_review",
        query: item.query || "",
        expectedOtid: item.expectedOtid || "",
        expectedTitle: item.expectedTitle || "",
        top1Otid: item.top1Otid || "",
        top1Title: item.top1Title || "",
        finalRank: item.finalRank ?? null,
        entryBucket: item.entryBucket || "unknown",
        flags: item.flags || [],
        titleSimilarity: item.titleSimilarity ?? 0,
        rationale: item.rationale || "",
    };
}

function summarize(rows: BoundaryCase[]) {
    return rows.reduce(
        (acc, item) => {
            acc.total += 1;
            acc.byDataset[item.dataset] = (acc.byDataset[item.dataset] || 0) + 1;
            acc.byAction[item.proposedAction] =
                (acc.byAction[item.proposedAction] || 0) + 1;
            acc.byRecommendation[item.recommendation] =
                (acc.byRecommendation[item.recommendation] || 0) + 1;
            item.flags.forEach((flag) => {
                acc.byFlag[flag] = (acc.byFlag[flag] || 0) + 1;
            });
            return acc;
        },
        {
            total: 0,
            byDataset: {} as Record<string, number>,
            byAction: {} as Record<string, number>,
            byRecommendation: {} as Record<string, number>,
            byFlag: {} as Record<string, number>,
        },
    );
}

function toCsvValue(value: unknown): string {
    const text = Array.isArray(value) ? value.join("|") : String(value ?? "");
    return `"${text.replace(/"/g, '""')}"`;
}

function writeCsv(filePath: string, rows: BoundaryCase[]): void {
    const headers: Array<keyof BoundaryCase> = [
        "dataset",
        "reviewPriority",
        "proposedAction",
        "recommendation",
        "query",
        "expectedTitle",
        "top1Title",
        "expectedOtid",
        "top1Otid",
        "finalRank",
        "entryBucket",
        "flags",
        "titleSimilarity",
        "rationale",
    ];
    const lines = [
        headers.map(toCsvValue).join(","),
        ...rows.map((row) => headers.map((header) => toCsvValue(row[header])).join(",")),
    ];
    fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf-8");
}

function main(): void {
    const inputs = resolveInputs();
    const rows = inputs.flatMap((input) => {
        const doc = JSON.parse(fs.readFileSync(input.reviewFile, "utf-8")) as RationalityReviewDoc;
        const datasetCases = buildDatasetCaseMap(doc.datasetFile);
        return (doc.cases || [])
            .filter(isBoundaryCandidate)
            .filter((item) => !isAlreadyAcceptedByDataset(item, datasetCases))
            .map((item) => toBoundaryCase(input.label, item));
    });

    const actionOrder: Record<BoundaryCase["proposedAction"], number> = {
        verify_expected_boundary: 0,
        accept_same_title_duplicate: 1,
        replace_or_anchor_query: 2,
        manual_review_same_family: 3,
        keep_single_expected: 4,
    };
    const priorityOrder = { high: 0, medium: 1, low: 2 };
    const sortedRows = rows.sort((left, right) => {
        return (
            actionOrder[left.proposedAction] - actionOrder[right.proposedAction] ||
            priorityOrder[left.reviewPriority] - priorityOrder[right.reviewPriority] ||
            right.titleSimilarity - left.titleSimilarity
        );
    });

    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const timestamp = Date.now();
    const jsonFile = path.join(outDir, `granularity_boundary_review_${timestamp}.json`);
    const csvFile = path.join(outDir, `granularity_boundary_review_${timestamp}.csv`);
    const report = {
        generatedAt: new Date().toISOString(),
        inputs,
        summary: summarize(sortedRows),
        cases: sortedRows,
    };
    fs.writeFileSync(jsonFile, JSON.stringify(report, null, 2), "utf-8");
    writeCsv(csvFile, sortedRows);
    console.log(`Saved boundary review JSON to ${jsonFile}`);
    console.log(`Saved boundary review CSV to ${csvFile}`);
    console.log(JSON.stringify(report.summary, null, 2));
}

main();

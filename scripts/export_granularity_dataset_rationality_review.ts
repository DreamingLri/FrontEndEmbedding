import * as fs from "fs";
import * as path from "path";

type AuditDoc = {
    generatedAt?: string;
    datasetFile?: string;
    badCaseFile?: string | null;
    cases?: AuditCase[];
};

type AuditCase = {
    query?: string;
    expectedOtid?: string;
    expectedTitle?: string;
    queryScope?: string;
    queryType?: string;
    supportPattern?: string;
    preferredGranularity?: string;
    themeFamily?: string;
    finalRank?: number | null;
    entryBucket?: string;
    bestCorrectRrfRank?: number | null;
    bestCorrectAtomicType?: string | null;
    badCase?: {
        failureRisk?: string;
        failureReasons?: string[];
    } | null;
    queryIntent?: {
        years?: number[];
    };
    finalTopDocs?: Array<{
        rank?: number;
        otid?: string;
        title?: string;
        score?: number;
        bestKpid?: string;
    }>;
    topDocs?: Array<{
        otid?: string;
        title?: string;
        bestRrfRank?: number | null;
        bestType?: string;
    }>;
};

type ReviewCase = {
    query: string;
    expectedOtid: string;
    expectedTitle: string;
    top1Otid: string;
    top1Title: string;
    finalRank: number | null;
    entryBucket: string;
    bestCorrectRrfRank: number | null;
    failureRisk: string;
    supportPattern: string;
    queryType: string;
    queryScope: string;
    themeFamily: string;
    flags: string[];
    titleSimilarity: number;
    reviewPriority: "high" | "medium" | "low";
    recommendation: string;
    rationale: string;
};

const DEFAULT_INPUT_PATTERN = /^granularity_retrieval_entry_audit_\d+\.json$/;

function resolveInputFile(): string {
    const explicit = process.env.SUASK_RATIONALITY_AUDIT_FILE;
    if (explicit) {
        return path.resolve(process.cwd(), explicit);
    }

    const resultsDir = path.resolve(process.cwd(), "scripts/results");
    const latest = fs
        .readdirSync(resultsDir)
        .filter((name) => DEFAULT_INPUT_PATTERN.test(name))
        .map((name) => {
            const fullPath = path.join(resultsDir, name);
            return {
                fullPath,
                mtimeMs: fs.statSync(fullPath).mtimeMs,
            };
        })
        .sort((left, right) => right.mtimeMs - left.mtimeMs)[0];

    if (!latest) {
        throw new Error(
            "No granularity_retrieval_entry_audit_*.json file found under scripts/results.",
        );
    }
    return latest.fullPath;
}

function extractYears(text: string): number[] {
    return Array.from(text.matchAll(/20\d{2}/g))
        .map((match) => Number(match[0]))
        .filter((year, index, years) => years.indexOf(year) === index);
}

function normalizeTitleForSimilarity(text: string): string {
    return text
        .replace(/20\d{2}年?/g, "")
        .replace(/[《》“”"'‘’（）()【】[\]、，,。.:：;；\s-]/g, "")
        .replace(/中山大学/g, "")
        .trim();
}

function charBigrams(text: string): Set<string> {
    const normalized = normalizeTitleForSimilarity(text);
    const result = new Set<string>();
    if (normalized.length <= 1) {
        if (normalized) {
            result.add(normalized);
        }
        return result;
    }
    for (let index = 0; index < normalized.length - 1; index += 1) {
        result.add(normalized.slice(index, index + 2));
    }
    return result;
}

function jaccard(left: Set<string>, right: Set<string>): number {
    if (left.size === 0 && right.size === 0) {
        return 1;
    }
    let intersection = 0;
    left.forEach((item) => {
        if (right.has(item)) {
            intersection += 1;
        }
    });
    const union = left.size + right.size - intersection;
    return union === 0 ? 0 : intersection / union;
}

function titleSimilarity(left: string, right: string): number {
    return jaccard(charBigrams(left), charBigrams(right));
}

function hasVagueReference(query: string): boolean {
    return /(这类|此类|该类|这个|这些|上述|前述|后面|后续|现在|目前|相关|政策要点|注意什么|怎么处理)/.test(
        query,
    );
}

function hasUnderAnchoredShortQuery(query: string): boolean {
    const compact = query.replace(/\s/g, "");
    return compact.length <= 16 && !/20\d{2}/.test(compact);
}

function inferFlags(item: AuditCase, top1Title: string, similarity: number): string[] {
    const query = item.query || "";
    const expectedTitle = item.expectedTitle || "";
    const queryYears = item.queryIntent?.years?.length
        ? item.queryIntent.years
        : extractYears(query);
    const expectedYears = extractYears(expectedTitle);
    const top1Years = extractYears(top1Title);
    const flags: string[] = [];

    if (expectedYears.length > 0 && queryYears.length === 0) {
        flags.push("no_explicit_year");
    }
    if (
        queryYears.length === 0 &&
        expectedYears.length > 0 &&
        top1Years.length > 0 &&
        expectedYears.some((year) => !top1Years.includes(year))
    ) {
        flags.push("competing_year_version");
    }
    if (hasVagueReference(query)) {
        flags.push("vague_or_context_dependent_query");
    }
    if (hasUnderAnchoredShortQuery(query)) {
        flags.push("under_anchored_short_query");
    }
    if (similarity >= 0.62) {
        flags.push("same_family_version_chain");
    }
    if (similarity >= 0.78) {
        flags.push("near_duplicate_title");
    }
    if (
        item.entryBucket === "already_top1" ||
        (item.finalRank === 1 && item.expectedOtid !== item.finalTopDocs?.[0]?.otid)
    ) {
        flags.push("metric_or_acceptable_boundary_check");
    }

    return flags;
}

function inferRecommendation(flags: string[], item: AuditCase): string {
    if (flags.includes("metric_or_acceptable_boundary_check")) {
        return "verify_eval_boundary_or_expected_otid";
    }
    if (
        flags.includes("no_explicit_year") &&
        (flags.includes("competing_year_version") ||
            flags.includes("same_family_version_chain"))
    ) {
        return "add_year_or_replace_case";
    }
    if (flags.includes("near_duplicate_title")) {
        return "expand_acceptable_otids_or_replace_case";
    }
    if (
        flags.includes("vague_or_context_dependent_query") ||
        flags.includes("under_anchored_short_query")
    ) {
        return "anchor_query_with_entity_year_or_replace_case";
    }
    if (item.entryBucket === "aggregation_drop") {
        return "keep_as_algorithm_case_after_dataset_review";
    }
    return "manual_review";
}

function inferPriority(flags: string[]): ReviewCase["reviewPriority"] {
    if (
        flags.includes("no_explicit_year") ||
        flags.includes("near_duplicate_title") ||
        flags.includes("metric_or_acceptable_boundary_check")
    ) {
        return "high";
    }
    if (
        flags.includes("vague_or_context_dependent_query") ||
        flags.includes("same_family_version_chain") ||
        flags.includes("under_anchored_short_query")
    ) {
        return "medium";
    }
    return "low";
}

function inferRationale(flags: string[], item: AuditCase): string {
    if (flags.length === 0) {
        return "No obvious dataset-rationality flag from title/year/query heuristics.";
    }
    const fragments: string[] = [];
    if (flags.includes("no_explicit_year")) {
        fragments.push("expected document is year-specific but query has no explicit year");
    }
    if (flags.includes("competing_year_version")) {
        fragments.push("top1 is another year/version in a competing document chain");
    }
    if (flags.includes("near_duplicate_title")) {
        fragments.push("expected and top1 titles are nearly duplicate");
    } else if (flags.includes("same_family_version_chain")) {
        fragments.push("expected and top1 titles are in the same topic family");
    }
    if (flags.includes("vague_or_context_dependent_query")) {
        fragments.push("query contains vague or context-dependent wording");
    }
    if (flags.includes("under_anchored_short_query")) {
        fragments.push("query is short and lacks anchoring constraints");
    }
    if (flags.includes("metric_or_acceptable_boundary_check")) {
        fragments.push("audit indicates a possible boundary or metric interpretation issue");
    }
    if (item.entryBucket === "aggregation_drop") {
        fragments.push("correct document entered candidate retrieval but lost final aggregation");
    }
    return fragments.join("; ") + ".";
}

function toCsvValue(value: unknown): string {
    const text = Array.isArray(value) ? value.join("|") : String(value ?? "");
    return `"${text.replace(/"/g, '""')}"`;
}

function writeCsv(filePath: string, rows: ReviewCase[]): void {
    const headers: Array<keyof ReviewCase> = [
        "reviewPriority",
        "recommendation",
        "query",
        "expectedTitle",
        "top1Title",
        "flags",
        "titleSimilarity",
        "entryBucket",
        "finalRank",
        "bestCorrectRrfRank",
        "failureRisk",
        "supportPattern",
        "queryType",
        "queryScope",
        "themeFamily",
        "expectedOtid",
        "top1Otid",
        "rationale",
    ];
    const lines = [
        headers.map(toCsvValue).join(","),
        ...rows.map((row) => headers.map((header) => toCsvValue(row[header])).join(",")),
    ];
    fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf-8");
}

function main(): void {
    const inputFile = resolveInputFile();
    const audit = JSON.parse(fs.readFileSync(inputFile, "utf-8")) as AuditDoc;
    const cases = audit.cases || [];
    const reviewCases = cases.map((item): ReviewCase => {
        const top1 = item.finalTopDocs?.[0] || item.topDocs?.[0] || {};
        const top1Title = top1.title || "";
        const similarity = titleSimilarity(item.expectedTitle || "", top1Title);
        const flags = inferFlags(item, top1Title, similarity);
        const recommendation = inferRecommendation(flags, item);
        return {
            query: item.query || "",
            expectedOtid: item.expectedOtid || "",
            expectedTitle: item.expectedTitle || "",
            top1Otid: top1.otid || "",
            top1Title,
            finalRank: item.finalRank ?? null,
            entryBucket: item.entryBucket || "unknown",
            bestCorrectRrfRank: item.bestCorrectRrfRank ?? null,
            failureRisk: item.badCase?.failureRisk || "unknown",
            supportPattern: item.supportPattern || "unknown",
            queryType: item.queryType || "unknown",
            queryScope: item.queryScope || "unknown",
            themeFamily: item.themeFamily || "unknown",
            flags,
            titleSimilarity: Number(similarity.toFixed(4)),
            reviewPriority: inferPriority(flags),
            recommendation,
            rationale: inferRationale(flags, item),
        };
    });

    const summary = reviewCases.reduce(
        (acc, item) => {
            acc.total += 1;
            acc.byPriority[item.reviewPriority] =
                (acc.byPriority[item.reviewPriority] || 0) + 1;
            acc.byRecommendation[item.recommendation] =
                (acc.byRecommendation[item.recommendation] || 0) + 1;
            item.flags.forEach((flag) => {
                acc.byFlag[flag] = (acc.byFlag[flag] || 0) + 1;
            });
            return acc;
        },
        {
            total: 0,
            byPriority: {} as Record<string, number>,
            byRecommendation: {} as Record<string, number>,
            byFlag: {} as Record<string, number>,
        },
    );

    const sortedCases = reviewCases.sort((left, right) => {
        const priorityOrder = { high: 0, medium: 1, low: 2 };
        return (
            priorityOrder[left.reviewPriority] -
                priorityOrder[right.reviewPriority] ||
            right.flags.length - left.flags.length ||
            right.titleSimilarity - left.titleSimilarity
        );
    });

    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const timestamp = Date.now();
    const jsonFile = path.join(
        outDir,
        `granularity_dataset_rationality_review_${timestamp}.json`,
    );
    const csvFile = path.join(
        outDir,
        `granularity_dataset_rationality_review_${timestamp}.csv`,
    );
    const report = {
        generatedAt: new Date().toISOString(),
        inputFile,
        datasetFile: audit.datasetFile || null,
        badCaseFile: audit.badCaseFile || null,
        summary,
        cases: sortedCases,
    };
    fs.writeFileSync(jsonFile, JSON.stringify(report, null, 2), "utf-8");
    writeCsv(csvFile, sortedCases);
    console.log(`Saved rationality review JSON to ${jsonFile}`);
    console.log(`Saved rationality review CSV to ${csvFile}`);
    console.log(JSON.stringify(summary, null, 2));
}

main();

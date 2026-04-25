import * as fs from "fs";
import * as path from "path";

type DatasetCase = {
    query?: string;
    ot_title?: string;
    query_type?: string;
    query_scope?: string;
    support_pattern?: string;
    preferred_granularity?: string;
    anchor_bucket?: string;
    near_neighbor_level?: string;
    same_theme_candidate_count_v1?: number;
    same_theme_near_year_candidate_count_v1?: number;
    difficulty_score_v1?: number;
    difficulty_level_v1?: string;
    difficulty_reasons_v1?: string[];
};

type DatasetSpec = {
    label: string;
    file: string;
};

const DEFAULT_DATASETS: DatasetSpec[] = [
    {
        label: "Main",
        file: "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_generalization_aligned_120_draft_v1.json",
    },
    {
        label: "InDomain",
        file: "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v3.json",
    },
    {
        label: "BlindExtOOD",
        file: "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v1.json",
    },
];

function parseDatasetSpecs(): DatasetSpec[] {
    const raw = process.env.SUASK_STRUCTURE_PROFILE_DATASETS;
    if (!raw) {
        return DEFAULT_DATASETS;
    }
    return raw.split(";").map((item) => {
        const [label, file] = item.split("=");
        if (!label || !file) {
            throw new Error(
                "SUASK_STRUCTURE_PROFILE_DATASETS must use label=path;label=path format.",
            );
        }
        return { label, file };
    });
}

function loadDataset(file: string): DatasetCase[] {
    return JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), file), "utf-8"),
    ) as DatasetCase[];
}

function hasExplicitYear(text: string): boolean {
    return /20\d{2}/.test(text);
}

function hasYearSpecificTarget(item: DatasetCase): boolean {
    return Boolean(item.ot_title?.match(/20\d{2}/));
}

function reasonsText(item: DatasetCase): string {
    return (item.difficulty_reasons_v1 || []).join("|");
}

function hasVeryShortQueryRisk(item: DatasetCase): boolean {
    return reasonsText(item).includes("very_short_query");
}

function hasNoExplicitYearRisk(item: DatasetCase): boolean {
    return !hasExplicitYear(item.query || "") && hasYearSpecificTarget(item);
}

function isStructureHighRisk(item: DatasetCase): boolean {
    return (
        item.anchor_bucket === "weak_anchor" ||
        item.near_neighbor_level === "high" ||
        hasNoExplicitYearRisk(item) ||
        hasVeryShortQueryRisk(item) ||
        Number(item.same_theme_candidate_count_v1 || 0) >= 3 ||
        Number(item.same_theme_near_year_candidate_count_v1 || 0) >= 2
    );
}

function increment(target: Record<string, number>, key: string): void {
    target[key] = (target[key] || 0) + 1;
}

function summarize(label: string, file: string, cases: DatasetCase[]) {
    const summary = {
        label,
        file,
        total: cases.length,
        structureHighRisk: 0,
        structureHighRiskRate: 0,
        weakAnchor: 0,
        nearNeighborHigh: 0,
        noExplicitYearToYearSpecificTarget: 0,
        veryShortQuery: 0,
        sameThemeCandidateGte3: 0,
        sameThemeNearYearGte2: 0,
        levelL4: 0,
        averageDifficultyScore: 0,
        bySupportPattern: {} as Record<string, number>,
        byQueryType: {} as Record<string, number>,
        byQueryScope: {} as Record<string, number>,
    };

    for (const item of cases) {
        if (isStructureHighRisk(item)) {
            summary.structureHighRisk += 1;
        }
        if (item.anchor_bucket === "weak_anchor") {
            summary.weakAnchor += 1;
        }
        if (item.near_neighbor_level === "high") {
            summary.nearNeighborHigh += 1;
        }
        if (hasNoExplicitYearRisk(item)) {
            summary.noExplicitYearToYearSpecificTarget += 1;
        }
        if (hasVeryShortQueryRisk(item)) {
            summary.veryShortQuery += 1;
        }
        if (Number(item.same_theme_candidate_count_v1 || 0) >= 3) {
            summary.sameThemeCandidateGte3 += 1;
        }
        if (Number(item.same_theme_near_year_candidate_count_v1 || 0) >= 2) {
            summary.sameThemeNearYearGte2 += 1;
        }
        if (item.difficulty_level_v1 === "L4") {
            summary.levelL4 += 1;
        }
        summary.averageDifficultyScore += Number(item.difficulty_score_v1 || 0);
        increment(summary.bySupportPattern, item.support_pattern || "unknown");
        increment(summary.byQueryType, item.query_type || "unknown");
        increment(summary.byQueryScope, item.query_scope || "unknown");
    }

    summary.structureHighRiskRate = Number(
        ((summary.structureHighRisk / Math.max(summary.total, 1)) * 100).toFixed(1),
    );
    summary.averageDifficultyScore = Number(
        (summary.averageDifficultyScore / Math.max(summary.total, 1)).toFixed(2),
    );
    return summary;
}

function main(): void {
    const datasets = parseDatasetSpecs();
    const summaries = datasets.map((item) =>
        summarize(item.label, item.file, loadDataset(item.file)),
    );
    const report = {
        generatedAt: new Date().toISOString(),
        riskDefinition:
            "weak_anchor OR high near-neighbor OR no explicit year for year-specific target OR very short query OR same-theme candidate count >= 3 OR same-theme near-year candidate count >= 2",
        summaries,
    };

    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const outFile = path.join(
        outDir,
        `granularity_dataset_structure_profile_${Date.now()}.json`,
    );
    fs.writeFileSync(outFile, JSON.stringify(report, null, 2), "utf-8");
    console.log(`Saved structure profile to ${outFile}`);
    console.log(JSON.stringify(summaries, null, 2));
}

main();

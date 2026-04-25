import * as fs from "fs";
import * as path from "path";

type DatasetCase = {
    id?: string;
    query?: string;
    source_query?: string;
    source_seed_id?: string;
    source_triplet_original_id?: string;
    source_review_item_id?: string;
    source_dataset?: string;
    expected_otid?: string;
    ot_title?: string;
    theme_family?: string;
    theme_key_v1?: string;
};

type DatasetSpec = {
    label: string;
    file: string;
};

type CaseRef = {
    dataset: string;
    id: string;
    query: string;
    sourceQuery: string;
    sourceSeedId: string;
    sourceTripletOriginalId: string;
    sourceReviewItemId: string;
    sourceDataset: string;
    expectedOtid: string;
    title: string;
    theme: string;
};

const DEFAULT_DATASETS: DatasetSpec[] = [
    {
        label: "Main",
        file: "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_generalization_aligned_120_draft_v2.json",
    },
    {
        label: "InDomain",
        file: "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v4.json",
    },
    {
        label: "BlindExtOOD",
        file: "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v2.json",
    },
];

function parseDatasetSpecs(): DatasetSpec[] {
    const raw = process.env.SUASK_OVERLAP_AUDIT_DATASETS;
    if (!raw) {
        return DEFAULT_DATASETS;
    }
    return raw.split(";").map((item) => {
        const [label, file] = item.split("=");
        if (!label || !file) {
            throw new Error(
                "SUASK_OVERLAP_AUDIT_DATASETS must use label=path;label=path format.",
            );
        }
        return { label, file };
    });
}

function loadDataset(spec: DatasetSpec): CaseRef[] {
    const rows = JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), spec.file), "utf-8"),
    ) as DatasetCase[];
    return rows.map((item, index) => ({
        dataset: spec.label,
        id: item.id || `${spec.label}_${index + 1}`,
        query: item.query || "",
        sourceQuery: item.source_query || "",
        sourceSeedId: item.source_seed_id || "",
        sourceTripletOriginalId: item.source_triplet_original_id || "",
        sourceReviewItemId: item.source_review_item_id || "",
        sourceDataset: item.source_dataset || "",
        expectedOtid: item.expected_otid || "",
        title: item.ot_title || "",
        theme: item.theme_key_v1 || item.theme_family || "",
    }));
}

function normalizeText(text: string): string {
    return text
        .toLowerCase()
        .replace(/20\d{2}年?/g, "")
        .replace(/[^\p{Script=Han}a-z0-9]+/gu, "")
        .replace(/中山大学/g, "")
        .replace(/人工智能学院/g, "")
        .trim();
}

function charBigrams(text: string): Set<string> {
    const normalized = normalizeText(text);
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

function textSimilarity(left: string, right: string): number {
    return jaccard(charBigrams(left), charBigrams(right));
}

function hasComparableText(text: string): boolean {
    return normalizeText(text).length >= 8;
}

function groupBy<T>(
    items: readonly T[],
    keyOf: (item: T) => string,
): Map<string, T[]> {
    const result = new Map<string, T[]>();
    items.forEach((item) => {
        const key = keyOf(item);
        if (!key) {
            return;
        }
        const bucket = result.get(key) || [];
        bucket.push(item);
        result.set(key, bucket);
    });
    return result;
}

function crossDatasetGroups(groups: Map<string, CaseRef[]>): Array<{
    key: string;
    cases: CaseRef[];
}> {
    return Array.from(groups.entries())
        .map(([key, cases]) => ({ key, cases }))
        .filter(
            ({ cases }) => new Set(cases.map((item) => item.dataset)).size > 1,
        )
        .sort((left, right) => right.cases.length - left.cases.length);
}

function pairwise(cases: readonly CaseRef[]): Array<[CaseRef, CaseRef]> {
    const result: Array<[CaseRef, CaseRef]> = [];
    for (let i = 0; i < cases.length; i += 1) {
        for (let j = i + 1; j < cases.length; j += 1) {
            if (cases[i].dataset !== cases[j].dataset) {
                result.push([cases[i], cases[j]]);
            }
        }
    }
    return result;
}

function main(): void {
    const specs = parseDatasetSpecs();
    const allCases = specs.flatMap(loadDataset);
    const directOverlap = {
        sourceSeedId: crossDatasetGroups(
            groupBy(allCases, (item) => item.sourceSeedId),
        ),
        sourceTripletOriginalId: crossDatasetGroups(
            groupBy(allCases, (item) => item.sourceTripletOriginalId),
        ),
        sourceReviewItemId: crossDatasetGroups(
            groupBy(allCases, (item) => item.sourceReviewItemId),
        ),
        sourceQuery: crossDatasetGroups(
            groupBy(allCases, (item) => normalizeText(item.sourceQuery)),
        ),
        query: crossDatasetGroups(
            groupBy(allCases, (item) => normalizeText(item.query)),
        ),
    };

    const expectedOtidGroups = crossDatasetGroups(
        groupBy(allCases, (item) => item.expectedOtid),
    );
    const sameExpectedOtidPairs = expectedOtidGroups.flatMap(({ key, cases }) =>
        pairwise(cases).map(([left, right]) => ({ key, left, right })),
    );

    const nearQueryPairs = pairwise(allCases)
        .filter(([left, right]) => hasComparableText(left.query) && hasComparableText(right.query))
        .map(([left, right]) => ({
            similarity: Number(textSimilarity(left.query, right.query).toFixed(4)),
            left,
            right,
        }))
        .filter((item) => item.similarity >= 0.72)
        .sort((left, right) => right.similarity - left.similarity);

    const nearSourceQueryPairs = pairwise(allCases)
        .filter(
            ([left, right]) =>
                hasComparableText(left.sourceQuery) &&
                hasComparableText(right.sourceQuery),
        )
        .map(([left, right]) => ({
            similarity: Number(
                textSimilarity(left.sourceQuery, right.sourceQuery).toFixed(4),
            ),
            left,
            right,
        }))
        .filter((item) => item.similarity >= 0.72)
        .sort((left, right) => right.similarity - left.similarity);

    const report = {
        generatedAt: new Date().toISOString(),
        datasets: specs,
        summary: {
            datasetSizes: specs.map((spec) => ({
                label: spec.label,
                count: allCases.filter((item) => item.dataset === spec.label).length,
            })),
            directOverlapCounts: {
                sourceSeedId: directOverlap.sourceSeedId.length,
                sourceTripletOriginalId:
                    directOverlap.sourceTripletOriginalId.length,
                sourceReviewItemId: directOverlap.sourceReviewItemId.length,
                sourceQuery: directOverlap.sourceQuery.length,
                query: directOverlap.query.length,
            },
            sameExpectedOtidPairCount: sameExpectedOtidPairs.length,
            nearQueryPairCount: nearQueryPairs.length,
            nearSourceQueryPairCount: nearSourceQueryPairs.length,
        },
        directOverlap,
        sameExpectedOtidPairs,
        nearQueryPairs,
        nearSourceQueryPairs,
    };

    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const outFile = path.join(
        outDir,
        `granularity_triplet_overlap_audit_${Date.now()}.json`,
    );
    fs.writeFileSync(outFile, JSON.stringify(report, null, 2), "utf-8");
    console.log(`Saved overlap audit to ${outFile}`);
    console.log(JSON.stringify(report.summary, null, 2));
}

main();

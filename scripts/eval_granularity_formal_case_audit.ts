import * as fs from "fs";
import * as path from "path";

type CaseExport = {
    generatedAt: string;
    datasetKey: string;
    comboLabel: string;
    weightMode: string;
    caseCount: number;
    cases: ExportCase[];
};

type ExportCase = {
    query: string;
    expected_otid: string;
    dataset: string;
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    support_pattern?: string;
    granularity_sensitive?: boolean;
    theme_family?: string;
    source_dataset?: string;
    source_seed_id?: string;
    challenge_tags?: string[];
    notes?: string;
    docRank: number | null;
    kpidRank: number | null;
    docHitAt1: boolean;
    docHitAt5: boolean;
    kpidHitAt1: boolean;
    kpidHitAt5: boolean;
    failure_risk: string;
    failure_reasons: string[];
};

type DatasetRow = {
    id?: string;
    query: string;
    expected_otid: string;
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    support_pattern?: string;
    theme_family?: string;
    source_dataset?: string;
    source_seed_id?: string;
    challenge_tags?: string[];
    notes?: string;
    anchor_bucket?: string;
    doc_role?: string;
    difficulty_level_v1?: string;
    near_neighbor_level?: string;
    source_triplet_pool?: string;
    source_triplet_original_id?: string;
    source_review_item_id?: string;
};

type EnrichedCase = ExportCase & {
    id?: string;
    anchor_bucket?: string;
    doc_role?: string;
    difficulty_level_v1?: string;
    near_neighbor_level?: string;
    source_triplet_pool?: string;
    source_triplet_original_id?: string;
    source_review_item_id?: string;
};

type SideKey = "left" | "right";

type BucketStats = {
    total: number;
    docHitAt1: number;
    docHitAt5: number;
    avgDocRank: number | null;
    avgKpidRank: number | null;
    hitAt1Rate: number;
    hitAt5Rate: number;
};

type BucketDiff = {
    key: string;
    left: BucketStats;
    right: BucketStats;
    hitAt1RateDiff: number;
    hitAt5RateDiff: number;
};

type FieldAudit = {
    field: string;
    minCountPerSide: number;
    sharedBucketCount: number;
    topRightPositive: BucketDiff[];
    topRightNegative: BucketDiff[];
};

type SideSummary = {
    datasetKey: string;
    caseFile: string;
    datasetFile: string;
    total: number;
    docHitAt1: number;
    docHitAt5: number;
    hitAt1Rate: number;
    hitAt5Rate: number;
    avgDocRank: number | null;
};

type Report = {
    generatedAt: string;
    comboLabel: string;
    weightMode: string;
    left: SideSummary;
    right: SideSummary;
    fields: FieldAudit[];
    pairFields: FieldAudit[];
};

const REPO_ROOT = path.resolve(process.cwd(), "..");
const RESULTS_DIR = path.resolve(process.cwd(), "./scripts/results");
const BACKEND_GRANULARITY_DIR = path.resolve(
    REPO_ROOT,
    "./Backend/test/test_dataset_granularity",
);
const DEFAULT_MIN_COUNT_PER_SIDE = Number.parseInt(
    process.env.SUASK_FORMAL_CASE_AUDIT_MIN_COUNT || "3",
    10,
);

const DEFAULT_LEFT_CASE_PATTERN =
    /^granularity_test_dataset_granularity_in_domain_generalization_aligned_100_draft_v1_.*_bad_cases_q_kp_ot\.json$/;
const DEFAULT_RIGHT_CASE_PATTERN =
    /^granularity_test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v1_.*_bad_cases_q_kp_ot\.json$/;

const DEFAULT_LEFT_DATASET_FILE = path.resolve(
    BACKEND_GRANULARITY_DIR,
    "./test_dataset_granularity_in_domain_generalization_aligned_100_draft_v1.json",
);
const DEFAULT_RIGHT_DATASET_FILE = path.resolve(
    BACKEND_GRANULARITY_DIR,
    "./test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v1.json",
);

const FIELD_NAMES = [
    "query_scope",
    "support_pattern",
    "preferred_granularity",
    "query_type",
    "anchor_bucket",
    "doc_role",
    "difficulty_level_v1",
    "near_neighbor_level",
    "source_dataset",
    "theme_family",
] as const;

const PAIR_FIELD_NAMES = [
    ["support_pattern", "query_type"],
    ["support_pattern", "preferred_granularity"],
    ["anchor_bucket", "support_pattern"],
    ["query_scope", "support_pattern"],
] as const;

function readJson<T>(filePath: string): T {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function latestResultFile(pattern: RegExp): string {
    const candidates = fs
        .readdirSync(RESULTS_DIR)
        .filter((name) => pattern.test(name))
        .map((name) => path.resolve(RESULTS_DIR, name))
        .sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);
    if (candidates.length === 0) {
        throw new Error(`No result file matched pattern: ${pattern}`);
    }
    return candidates[0];
}

function resolveCaseFile(side: SideKey): string {
    const explicit =
        side === "left"
            ? process.env.SUASK_FORMAL_CASE_AUDIT_LEFT_CASE_FILE
            : process.env.SUASK_FORMAL_CASE_AUDIT_RIGHT_CASE_FILE;
    if (explicit?.trim()) {
        return path.resolve(explicit.trim());
    }
    return latestResultFile(
        side === "left" ? DEFAULT_LEFT_CASE_PATTERN : DEFAULT_RIGHT_CASE_PATTERN,
    );
}

function resolveDatasetFile(side: SideKey): string {
    const explicit =
        side === "left"
            ? process.env.SUASK_FORMAL_CASE_AUDIT_LEFT_DATASET_FILE
            : process.env.SUASK_FORMAL_CASE_AUDIT_RIGHT_DATASET_FILE;
    if (explicit?.trim()) {
        return path.resolve(explicit.trim());
    }
    return side === "left" ? DEFAULT_LEFT_DATASET_FILE : DEFAULT_RIGHT_DATASET_FILE;
}

function buildDatasetRowMap(rows: DatasetRow[]): Map<string, DatasetRow> {
    const map = new Map<string, DatasetRow>();
    rows.forEach((row) => {
        map.set(buildCaseKey(row.query, row.expected_otid), row);
    });
    return map;
}

function buildCaseKey(query: string, expectedOtid: string): string {
    return `${expectedOtid}||${query}`;
}

function enrichCases(caseExport: CaseExport, datasetRows: DatasetRow[]): EnrichedCase[] {
    const rowMap = buildDatasetRowMap(datasetRows);
    return caseExport.cases.map((item) => {
        const datasetRow = rowMap.get(buildCaseKey(item.query, item.expected_otid));
        if (!datasetRow) {
            throw new Error(
                `Missing dataset row for case: ${item.expected_otid} :: ${item.query}`,
            );
        }
        return {
            ...item,
            id: datasetRow.id,
            anchor_bucket: datasetRow.anchor_bucket,
            doc_role: datasetRow.doc_role,
            difficulty_level_v1: datasetRow.difficulty_level_v1,
            near_neighbor_level: datasetRow.near_neighbor_level,
            source_triplet_pool: datasetRow.source_triplet_pool,
            source_triplet_original_id: datasetRow.source_triplet_original_id,
            source_review_item_id: datasetRow.source_review_item_id,
        };
    });
}

function average(values: number[]): number | null {
    if (values.length === 0) {
        return null;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeCases(
    datasetKey: string,
    caseFile: string,
    datasetFile: string,
    rows: EnrichedCase[],
): SideSummary {
    const docHitAt1 = rows.filter((item) => item.docHitAt1).length;
    const docHitAt5 = rows.filter((item) => item.docHitAt5).length;
    const docRanks = rows
        .map((item) => item.docRank)
        .filter((value): value is number => Number.isFinite(value));

    return {
        datasetKey,
        caseFile,
        datasetFile,
        total: rows.length,
        docHitAt1,
        docHitAt5,
        hitAt1Rate: docHitAt1 / rows.length,
        hitAt5Rate: docHitAt5 / rows.length,
        avgDocRank: average(docRanks),
    };
}

function buildBucketStats(rows: EnrichedCase[]): BucketStats {
    const docRanks = rows
        .map((item) => item.docRank)
        .filter((value): value is number => Number.isFinite(value));
    const kpidRanks = rows
        .map((item) => item.kpidRank)
        .filter((value): value is number => Number.isFinite(value));
    const docHitAt1 = rows.filter((item) => item.docHitAt1).length;
    const docHitAt5 = rows.filter((item) => item.docHitAt5).length;

    return {
        total: rows.length,
        docHitAt1,
        docHitAt5,
        avgDocRank: average(docRanks),
        avgKpidRank: average(kpidRanks),
        hitAt1Rate: docHitAt1 / rows.length,
        hitAt5Rate: docHitAt5 / rows.length,
    };
}

function bucketize(
    rows: EnrichedCase[],
    bucketKey: (row: EnrichedCase) => string | undefined,
): Map<string, EnrichedCase[]> {
    const buckets = new Map<string, EnrichedCase[]>();
    rows.forEach((row) => {
        const key = bucketKey(row);
        if (!key) {
            return;
        }
        const bucket = buckets.get(key);
        if (bucket) {
            bucket.push(row);
        } else {
            buckets.set(key, [row]);
        }
    });
    return buckets;
}

function buildFieldAudit(
    field: string,
    leftRows: EnrichedCase[],
    rightRows: EnrichedCase[],
    minCountPerSide: number,
    bucketKey: (row: EnrichedCase) => string | undefined,
): FieldAudit {
    const leftBuckets = bucketize(leftRows, bucketKey);
    const rightBuckets = bucketize(rightRows, bucketKey);
    const sharedKeys = Array.from(leftBuckets.keys()).filter((key) =>
        rightBuckets.has(key),
    );

    const sharedDiffs = sharedKeys
        .map((key) => {
            const leftBucket = leftBuckets.get(key) || [];
            const rightBucket = rightBuckets.get(key) || [];
            if (
                leftBucket.length < minCountPerSide ||
                rightBucket.length < minCountPerSide
            ) {
                return null;
            }
            const left = buildBucketStats(leftBucket);
            const right = buildBucketStats(rightBucket);
            return {
                key,
                left,
                right,
                hitAt1RateDiff: right.hitAt1Rate - left.hitAt1Rate,
                hitAt5RateDiff: right.hitAt5Rate - left.hitAt5Rate,
            } satisfies BucketDiff;
        })
        .filter((item): item is BucketDiff => item !== null);

    const topRightPositive = [...sharedDiffs]
        .sort((left, right) => {
            if (right.hitAt1RateDiff !== left.hitAt1RateDiff) {
                return right.hitAt1RateDiff - left.hitAt1RateDiff;
            }
            return right.right.total - left.right.total;
        })
        .slice(0, 10);

    const topRightNegative = [...sharedDiffs]
        .sort((left, right) => {
            if (left.hitAt1RateDiff !== right.hitAt1RateDiff) {
                return left.hitAt1RateDiff - right.hitAt1RateDiff;
            }
            return right.left.total - left.left.total;
        })
        .slice(0, 10);

    return {
        field,
        minCountPerSide,
        sharedBucketCount: sharedDiffs.length,
        topRightPositive,
        topRightNegative,
    };
}

function prettyNumber(value: number | null): number | null {
    if (value === null) {
        return null;
    }
    return Number(value.toFixed(4));
}

function normalizeAudit(audit: FieldAudit): FieldAudit {
    const normalizeBucket = (item: BucketDiff): BucketDiff => ({
        ...item,
        left: {
            ...item.left,
            avgDocRank: prettyNumber(item.left.avgDocRank),
            avgKpidRank: prettyNumber(item.left.avgKpidRank),
            hitAt1Rate: prettyNumber(item.left.hitAt1Rate) ?? 0,
            hitAt5Rate: prettyNumber(item.left.hitAt5Rate) ?? 0,
        },
        right: {
            ...item.right,
            avgDocRank: prettyNumber(item.right.avgDocRank),
            avgKpidRank: prettyNumber(item.right.avgKpidRank),
            hitAt1Rate: prettyNumber(item.right.hitAt1Rate) ?? 0,
            hitAt5Rate: prettyNumber(item.right.hitAt5Rate) ?? 0,
        },
        hitAt1RateDiff: prettyNumber(item.hitAt1RateDiff) ?? 0,
        hitAt5RateDiff: prettyNumber(item.hitAt5RateDiff) ?? 0,
    });

    return {
        ...audit,
        topRightPositive: audit.topRightPositive.map(normalizeBucket),
        topRightNegative: audit.topRightNegative.map(normalizeBucket),
    };
}

function main(): void {
    const leftCaseFile = resolveCaseFile("left");
    const rightCaseFile = resolveCaseFile("right");
    const leftDatasetFile = resolveDatasetFile("left");
    const rightDatasetFile = resolveDatasetFile("right");

    const leftCaseExport = readJson<CaseExport>(leftCaseFile);
    const rightCaseExport = readJson<CaseExport>(rightCaseFile);
    const leftDatasetRows = readJson<DatasetRow[]>(leftDatasetFile);
    const rightDatasetRows = readJson<DatasetRow[]>(rightDatasetFile);

    const leftRows = enrichCases(leftCaseExport, leftDatasetRows);
    const rightRows = enrichCases(rightCaseExport, rightDatasetRows);

    const fields = FIELD_NAMES.map((field) =>
        normalizeAudit(
            buildFieldAudit(
                field,
                leftRows,
                rightRows,
                DEFAULT_MIN_COUNT_PER_SIDE,
                (row) => (row as Record<string, string | undefined>)[field],
            ),
        ),
    );

    const pairFields = PAIR_FIELD_NAMES.map(([leftField, rightField]) =>
        normalizeAudit(
            buildFieldAudit(
                `${leftField} x ${rightField}`,
                leftRows,
                rightRows,
                DEFAULT_MIN_COUNT_PER_SIDE,
                (row) => {
                    const leftValue = (row as Record<string, string | undefined>)[leftField];
                    const rightValue = (row as Record<string, string | undefined>)[rightField];
                    if (!leftValue || !rightValue) {
                        return undefined;
                    }
                    return `${leftValue} || ${rightValue}`;
                },
            ),
        ),
    );

    const report: Report = {
        generatedAt: new Date().toISOString(),
        comboLabel: leftCaseExport.comboLabel,
        weightMode: leftCaseExport.weightMode,
        left: {
            ...summarizeCases(
                leftCaseExport.datasetKey,
                leftCaseFile,
                leftDatasetFile,
                leftRows,
            ),
            hitAt1Rate: prettyNumber(
                summarizeCases(
                    leftCaseExport.datasetKey,
                    leftCaseFile,
                    leftDatasetFile,
                    leftRows,
                ).hitAt1Rate,
            ) ?? 0,
            hitAt5Rate: prettyNumber(
                summarizeCases(
                    leftCaseExport.datasetKey,
                    leftCaseFile,
                    leftDatasetFile,
                    leftRows,
                ).hitAt5Rate,
            ) ?? 0,
            avgDocRank: prettyNumber(
                summarizeCases(
                    leftCaseExport.datasetKey,
                    leftCaseFile,
                    leftDatasetFile,
                    leftRows,
                ).avgDocRank,
            ),
        },
        right: {
            ...summarizeCases(
                rightCaseExport.datasetKey,
                rightCaseFile,
                rightDatasetFile,
                rightRows,
            ),
            hitAt1Rate: prettyNumber(
                summarizeCases(
                    rightCaseExport.datasetKey,
                    rightCaseFile,
                    rightDatasetFile,
                    rightRows,
                ).hitAt1Rate,
            ) ?? 0,
            hitAt5Rate: prettyNumber(
                summarizeCases(
                    rightCaseExport.datasetKey,
                    rightCaseFile,
                    rightDatasetFile,
                    rightRows,
                ).hitAt5Rate,
            ) ?? 0,
            avgDocRank: prettyNumber(
                summarizeCases(
                    rightCaseExport.datasetKey,
                    rightCaseFile,
                    rightDatasetFile,
                    rightRows,
                ).avgDocRank,
            ),
        },
        fields,
        pairFields,
    };

    const timestamp = Date.now();
    const outputPath = path.resolve(
        RESULTS_DIR,
        `granularity_formal_case_audit_in_domain_vs_blind_${timestamp}.json`,
    );
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf-8");

    console.log(`Saved report to ${outputPath}`);
    console.log(
        `left=${report.left.datasetKey} hit@1=${report.left.docHitAt1}/${report.left.total}=${report.left.hitAt1Rate}`,
    );
    console.log(
        `right=${report.right.datasetKey} hit@1=${report.right.docHitAt1}/${report.right.total}=${report.right.hitAt1Rate}`,
    );
    report.fields.slice(0, 4).forEach((field) => {
        const bestPositive = field.topRightPositive[0];
        const bestNegative = field.topRightNegative[0];
        if (bestPositive) {
            console.log(
                `[${field.field}] top right-positive ${bestPositive.key}: diff=${bestPositive.hitAt1RateDiff}`,
            );
        }
        if (bestNegative) {
            console.log(
                `[${field.field}] top right-negative ${bestNegative.key}: diff=${bestNegative.hitAt1RateDiff}`,
            );
        }
    });
}

main();

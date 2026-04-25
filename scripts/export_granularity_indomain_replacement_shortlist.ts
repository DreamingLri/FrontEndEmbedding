import * as fs from "fs";
import * as path from "path";

type DatasetCase = Record<string, unknown> & {
    id?: string;
    query?: string;
    source_query?: string;
    source_seed_id?: string;
    source_triplet_original_id?: string;
    source_review_item_id?: string;
    source_dataset?: string;
    expected_otid?: string;
    ot_title?: string;
    support_pattern?: string;
    query_type?: string;
    query_scope?: string;
    preferred_granularity?: string;
    difficulty_level_v1?: string;
    difficulty_score_v1?: number;
    anchor_bucket?: string;
    near_neighbor_level?: string;
    same_theme_candidate_count_v1?: number;
    same_theme_near_year_candidate_count_v1?: number;
    review_status?: string;
};

type Candidate = {
    score: number;
    sourceFile: string;
    item: DatasetCase;
    bucket: string;
    usedExpectedOtid: boolean;
    reason: string[];
};

const TARGET_INDOMAIN_FILE =
    "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v4.json";
const MAIN_FILE =
    "../Backend/test/test_dataset_granularity/test_dataset_granularity_main_generalization_aligned_120_draft_v2.json";
const BLIND_FILE =
    "../Backend/test/test_dataset_granularity/test_dataset_granularity_blind_ext_ood_generalization_aligned_100_draft_v2.json";
const TARGET_IDS = new Set([
    "in_domain_100_draft_v1_0025",
    "in_domain_100_draft_v1_0011",
    "in_domain_100_draft_v1_0008",
    "in_domain_100_draft_v1_0074",
    "in_domain_100_draft_v1_0061",
    "in_domain_100_draft_v1_0054",
    "in_domain_100_draft_v1_0018",
    "in_domain_100_draft_v1_0064",
    "in_domain_100_draft_v1_0090",
]);

function readJsonArray(file: string): DatasetCase[] {
    return JSON.parse(fs.readFileSync(path.resolve(process.cwd(), file), "utf-8")) as DatasetCase[];
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

function bucketOf(item: DatasetCase): string {
    return [
        item.support_pattern || "",
        item.query_type || "",
        item.query_scope || "",
        item.preferred_granularity || "",
    ].join("|");
}

function walkJsonFiles(dir: string): string[] {
    const result: string[] = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            result.push(...walkJsonFiles(fullPath));
        } else if (
            entry.isFile() &&
            entry.name.endsWith(".json") &&
            entry.name.toLowerCase().includes("in_domain") &&
            !entry.name.includes("aligned_100_draft_v")
        ) {
            result.push(fullPath);
        }
    }
    return result;
}

function scoreCandidate(item: DatasetCase, usedExpectedOtid: boolean): Candidate["reason"] {
    const reason: string[] = [];
    if (!usedExpectedOtid) {
        reason.push("new_expected_otid");
    }
    if (item.review_status === "reviewed") {
        reason.push("reviewed");
    }
    if (!String(item.sourceFile || "").includes("backups")) {
        reason.push("non_backup_source");
    }
    if (item.anchor_bucket === "weak_anchor") {
        reason.push("keeps_weak_anchor");
    }
    if (item.near_neighbor_level === "high") {
        reason.push("keeps_high_neighbor");
    }
    return reason;
}

function main(): void {
    const inDomain = readJsonArray(TARGET_INDOMAIN_FILE);
    const mainSet = readJsonArray(MAIN_FILE);
    const blindSet = readJsonArray(BLIND_FILE);
    const currentAll = [...inDomain, ...mainSet, ...blindSet];
    const usedSeeds = new Set(currentAll.map((item) => item.source_seed_id || "").filter(Boolean));
    const usedQueries = new Set(currentAll.map((item) => normalizeText(item.query || "")).filter(Boolean));
    const usedSourceQueries = new Set(
        currentAll.map((item) => normalizeText(item.source_query || "")).filter(Boolean),
    );
    const usedIds = new Set(currentAll.map((item) => item.id || "").filter(Boolean));
    const usedExpectedOtids = new Set(inDomain.map((item) => item.expected_otid || "").filter(Boolean));

    const targets = inDomain.filter((item) => item.id && TARGET_IDS.has(item.id));
    const targetBuckets = new Set(targets.map(bucketOf));
    const root = path.resolve(process.cwd(), "../Backend/test/test_dataset_granularity");
    const candidateFiles = walkJsonFiles(root);
    const seenCandidateKey = new Set<string>();
    const candidatesByBucket = new Map<string, Candidate[]>();

    for (const file of candidateFiles) {
        let rows: DatasetCase[];
        try {
            rows = JSON.parse(fs.readFileSync(file, "utf-8")) as DatasetCase[];
        } catch {
            continue;
        }
        if (!Array.isArray(rows)) {
            continue;
        }
        for (const raw of rows) {
            const item = { ...raw, sourceFile: file } as DatasetCase;
            const bucket = bucketOf(item);
            if (!targetBuckets.has(bucket)) {
                continue;
            }
            const normalizedQuery = normalizeText(item.query || "");
            const normalizedSourceQuery = normalizeText(item.source_query || "");
            const seed = item.source_seed_id || "";
            if (seed && usedSeeds.has(seed)) {
                continue;
            }
            if (item.id && usedIds.has(item.id)) {
                continue;
            }
            if (normalizedQuery && usedQueries.has(normalizedQuery)) {
                continue;
            }
            if (normalizedSourceQuery && usedSourceQueries.has(normalizedSourceQuery)) {
                continue;
            }
            const key = [
                item.source_seed_id || "",
                item.expected_otid || "",
                normalizedQuery,
                bucket,
            ].join("|");
            if (seenCandidateKey.has(key)) {
                continue;
            }
            seenCandidateKey.add(key);
            const usedExpectedOtid = usedExpectedOtids.has(item.expected_otid || "");
            const reason = scoreCandidate(item, usedExpectedOtid);
            const score =
                (usedExpectedOtid ? 0 : 10) +
                (item.review_status === "reviewed" ? 4 : 0) +
                (file.includes("backups") ? -2 : 2) +
                (item.anchor_bucket === "weak_anchor" ? 2 : 0) +
                (item.near_neighbor_level === "high" ? 2 : 0) +
                Number(item.difficulty_score_v1 || 0);
            const candidate: Candidate = {
                score: Number(score.toFixed(2)),
                sourceFile: path.relative(process.cwd(), file),
                item,
                bucket,
                usedExpectedOtid,
                reason,
            };
            const list = candidatesByBucket.get(bucket) || [];
            list.push(candidate);
            candidatesByBucket.set(bucket, list);
        }
    }

    const replacements = targets.map((target) => {
        const bucket = bucketOf(target);
        const candidates = (candidatesByBucket.get(bucket) || [])
            .sort((left, right) => right.score - left.score)
            .slice(0, 12);
        return { target, bucket, candidates };
    });

    const report = {
        generatedAt: new Date().toISOString(),
        targetFile: TARGET_INDOMAIN_FILE,
        targetIds: Array.from(TARGET_IDS),
        summary: {
            targetCount: targets.length,
            buckets: Array.from(targetBuckets),
            candidateCountsByBucket: Object.fromEntries(
                Array.from(candidatesByBucket.entries()).map(([bucket, list]) => [bucket, list.length]),
            ),
        },
        replacements,
    };
    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const outFile = path.join(
        outDir,
        `granularity_indomain_replacement_shortlist_${Date.now()}.json`,
    );
    fs.writeFileSync(outFile, JSON.stringify(report, null, 2), "utf-8");
    console.log(`Saved replacement shortlist to ${outFile}`);
    console.log(JSON.stringify(report.summary, null, 2));
}

main();

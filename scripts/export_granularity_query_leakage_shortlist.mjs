import fs from "fs";
import path from "path";

const RESULTS_DIR = path.resolve(process.cwd(), "scripts/results");
const DEFAULT_TARGET_KEYS = ["main_bench_120", "in_domain_generalization_100"];
const MAX_PER_DATASET = parseInt(
    process.env.SUASK_QUERY_LEAKAGE_SHORTLIST_MAX_PER_DATASET || "12",
    10,
);
const TARGET_KEYS = (
    process.env.SUASK_QUERY_LEAKAGE_SHORTLIST_TARGETS ||
    DEFAULT_TARGET_KEYS.join(",")
)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

function safeMaxPerDataset() {
    return Number.isFinite(MAX_PER_DATASET) && MAX_PER_DATASET > 0
        ? MAX_PER_DATASET
        : 12;
}

function extractYears(text) {
    return Array.from(String(text || "").matchAll(/20\d{2}/g)).map(
        (match) => match[0],
    );
}

function hasYearMismatch(leftYears, rightYears) {
    if (!leftYears.length || !rightYears.length) {
        return false;
    }
    const rightSet = new Set(rightYears);
    return !leftYears.some((year) => rightSet.has(year));
}

function normalizeText(text) {
    return String(text || "")
        .replace(/\s+/g, "")
        .replace(/[，。！？、；：,.!?;:（）()【】\[\]“”"'`]/g, "")
        .toLowerCase();
}

function overlapRatio(left, right) {
    const a = normalizeText(left);
    const b = normalizeText(right);
    if (!a || !b) {
        return 0;
    }
    const shorter = a.length <= b.length ? a : b;
    const longer = a.length <= b.length ? b : a;
    let common = 0;
    for (const char of shorter) {
        if (longer.includes(char)) {
            common += 1;
        }
    }
    return shorter.length ? common / shorter.length : 0;
}

function classifyItem(row) {
    const queryYears = extractYears(row.query);
    const bestYears = extractYears(row.bestIndexQuestion);
    const sameYears = extractYears(row.sameTargetBestQuestion);
    const margin = Number(row.sameVsNonTargetMargin ?? 0);
    const maxRouge = Number(row.maxRougeL ?? 0);
    const sameTargetMax = Number(row.sameTargetMaxRougeL ?? 0);
    const nonTargetMax = Number(row.nonTargetMaxRougeL ?? 0);
    const queryBestOverlap = overlapRatio(row.query, row.bestIndexQuestion);
    const tags = [];

    if (margin < 0) {
        tags.push("non_target_beats_same_target");
    }
    if (hasYearMismatch(queryYears, bestYears)) {
        tags.push("cross_year_non_target_match");
    }
    if (hasYearMismatch(queryYears, sameYears)) {
        tags.push("cross_year_same_target_gap");
    }
    if (sameTargetMax >= 0.7) {
        tags.push("same_target_literal_reuse");
    }
    if (Math.abs(margin) <= 0.02 && maxRouge >= 0.6) {
        tags.push("same_vs_non_target_too_close");
    }
    if (queryBestOverlap >= 0.8 && maxRouge >= 0.6) {
        tags.push("template_wording_overlap");
    }

    let recommendedAction = "manual_review";
    let priority = "P2_watch";
    if (margin <= -0.08 && maxRouge >= 0.62) {
        recommendedAction = "replace_case";
        priority = "P0_replace";
    } else if (sameTargetMax >= 0.7 || (margin < 0 && maxRouge >= 0.55)) {
        recommendedAction = "rewrite_query";
        priority = "P1_rewrite";
    }

    let rationale = "建议人工复核。";
    if (recommendedAction === "replace_case") {
        rationale =
            "非目标文档中的现成 Q 比目标文档 Q 更像当前测试题，且相似度已进入高风险区间。";
    } else if (tags.includes("same_target_literal_reuse")) {
        rationale = "题面与目标文档原生 Q 过近，容易把检索分数抬高。";
    } else if (tags.includes("cross_year_non_target_match")) {
        rationale = "存在明显跨年份模板复用，容易把 query 拉向错误年份的同主题文档。";
    } else if (tags.includes("same_vs_non_target_too_close")) {
        rationale = "目标/非目标 Q 相似度过近，题面可能仍然偏模板化。";
    }

    const riskScore =
        maxRouge * 100 +
        (margin < 0 ? 10 : 0) +
        (margin <= -0.05 ? 8 : 0) +
        (margin <= -0.1 ? 8 : 0) +
        (sameTargetMax >= 0.7 ? 6 : 0) +
        (tags.includes("cross_year_non_target_match") ? 8 : 0);

    return {
        queryYears,
        bestIndexYears: bestYears,
        sameTargetYears: sameYears,
        maxRougeL: maxRouge,
        sameTargetMaxRougeL: sameTargetMax,
        nonTargetMaxRougeL: nonTargetMax,
        sameVsNonTargetMargin: margin,
        queryBestOverlap: Number(queryBestOverlap.toFixed(4)),
        tags,
        recommendedAction,
        priority,
        rationale,
        riskScore: Number(riskScore.toFixed(2)),
    };
}

function shouldKeep(row, classified) {
    return (
        classified.maxRougeL >= 0.6 ||
        classified.sameTargetMaxRougeL >= 0.65 ||
        classified.sameVsNonTargetMargin <= -0.03
    );
}

function readAuditReport(filePath) {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function getLatestAuditFileForTarget(targetKey) {
    const candidates = fs
        .readdirSync(RESULTS_DIR)
        .filter(
            (name) =>
                name.startsWith("granularity_query_leakage_audit_") &&
                name.endsWith(".json"),
        )
        .map((name) => path.join(RESULTS_DIR, name))
        .filter((filePath) => {
            try {
                const report = readAuditReport(filePath);
                return (
                    Array.isArray(report.datasetTargets) &&
                    report.datasetTargets.length === 1 &&
                    report.datasetTargets[0] === targetKey
                );
            } catch {
                return false;
            }
        })
        .sort((left, right) => {
            const leftStat = fs.statSync(left);
            const rightStat = fs.statSync(right);
            return rightStat.mtimeMs - leftStat.mtimeMs;
        });

    if (candidates.length === 0) {
        throw new Error(`No leakage audit report found for target ${targetKey}.`);
    }
    return candidates[0];
}

function buildShortlistItem(row, sourceAuditFile) {
    const classified = classifyItem(row);
    return {
        dataset: row.dataset,
        id: row.id || "",
        query: row.query || "",
        expected_otid: row.expected_otid || "",
        sourceAuditFile,
        bestIndexQuestion: row.bestIndexQuestion || "",
        sameTargetBestQuestion: row.sameTargetBestQuestion || "",
        nonTargetBestQuestion: row.nonTargetBestQuestion || "",
        ...classified,
    };
}

function buildMarkdownReport(payload) {
    const lines = [
        "# Main/InDomain Query Leakage Shortlist",
        "",
        "状态：`CURRENT DRAFT`",
        "",
        `- shortlist JSON：[${path.basename(payload.outputJsonPath)}](${payload.outputJsonPath.replace(/\\/g, "/")}:1)`,
        "",
        "## 核心结论",
        "",
        `- 检查目标：\`${payload.targetKeys.join(", ")}\``,
        `- 输出样本：\`${payload.summary.totalItems}\``,
        `- ` +
            `P0 replace：\`${payload.summary.byPriority.P0_replace || 0}\`，` +
            `P1 rewrite：\`${payload.summary.byPriority.P1_rewrite || 0}\`，` +
            `P2 watch：\`${payload.summary.byPriority.P2_watch || 0}\``,
        "",
        "## 人工复核清单",
        "",
    ];

    for (const item of payload.items) {
        lines.push(
            `- ${item.dataset} | ${item.id} | ${item.priority} | ${item.recommendedAction} | max=${item.maxRougeL.toFixed(4)} | margin=${item.sameVsNonTargetMargin.toFixed(4)}`,
        );
        lines.push(`  query: ${item.query}`);
        lines.push(`  best non-target Q: ${item.bestIndexQuestion}`);
        lines.push(`  same-target Q: ${item.sameTargetBestQuestion}`);
        lines.push(`  tags: ${item.tags.join(", ") || "none"}`);
        lines.push(`  rationale: ${item.rationale}`);
    }

    return `${lines.join("\n")}\n`;
}

function main() {
    const reports = TARGET_KEYS.map((targetKey) => {
        const filePath = getLatestAuditFileForTarget(targetKey);
        const report = readAuditReport(filePath);
        return { targetKey, filePath, report };
    });

    const items = reports.flatMap(({ filePath, report }) =>
        (report.topCases || [])
            .map((row) => buildShortlistItem(row, path.relative(process.cwd(), filePath)))
            .filter((row) => shouldKeep(row, row))
            .slice(0, safeMaxPerDataset()),
    );

    items.sort((left, right) => right.riskScore - left.riskScore);

    const byDataset = {};
    const byPriority = {};
    const byAction = {};
    for (const item of items) {
        byDataset[item.dataset] = (byDataset[item.dataset] || 0) + 1;
        byPriority[item.priority] = (byPriority[item.priority] || 0) + 1;
        byAction[item.recommendedAction] = (byAction[item.recommendedAction] || 0) + 1;
    }

    const payload = {
        generatedAt: new Date().toISOString(),
        targetKeys: TARGET_KEYS,
        sourceReports: reports.map(({ targetKey, filePath }) => ({
            targetKey,
            file: path.relative(process.cwd(), filePath),
        })),
        filters: {
            maxPerDataset: safeMaxPerDataset(),
            keepIf: {
                maxRougeLGte: 0.6,
                sameTargetMaxRougeLGte: 0.65,
                sameVsNonTargetMarginLte: -0.03,
            },
        },
        summary: {
            totalItems: items.length,
            byDataset,
            byPriority,
            byAction,
        },
        items,
    };

    const timestamp = Date.now();
    const outputJsonPath = path.join(
        RESULTS_DIR,
        `granularity_query_leakage_shortlist_${timestamp}.json`,
    );
    fs.writeFileSync(outputJsonPath, JSON.stringify(payload, null, 2), "utf-8");

    const markdownPayload = {
        ...payload,
        outputJsonPath,
    };
    const outputMdPath = path.join(
        RESULTS_DIR,
        `granularity_query_leakage_shortlist_${timestamp}.md`,
    );
    fs.writeFileSync(outputMdPath, buildMarkdownReport(markdownPayload), "utf-8");

    console.log(`Saved leakage shortlist JSON to ${outputJsonPath}`);
    console.log(`Saved leakage shortlist Markdown to ${outputMdPath}`);
    console.log(JSON.stringify(payload.summary, null, 2));
}

main();

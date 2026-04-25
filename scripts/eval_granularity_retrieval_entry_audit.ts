import * as fs from "fs";
import * as path from "path";

import {
    BM25_B,
    BM25_K1,
    RRF_K,
    RRF_RANK_LIMIT,
    buildBM25Stats,
    dotProduct,
    getQuerySparse,
    parseQueryIntent,
    resolveDocOtid,
    searchAndRank,
    selectTopLocalIndices,
    type Metadata,
} from "../src/worker/vector_engine.ts";
import { fmmTokenize } from "../src/worker/fmm_tokenize.ts";
import {
    DEFAULT_QUERY_EMBED_BATCH_SIZE,
    type EvalDatasetCase,
} from "./eval_shared.ts";
import {
    embedQueries,
    loadFrontendEvalEngine,
} from "./frontend_eval_engine.ts";
import { resolveBackendArticlesFile } from "./kb_version_paths.ts";

type AuditCandidate = {
    id: string;
    otid: string;
    type: "Q" | "KP" | "OT";
    parentPkid?: string;
    denseScore: number;
    sparseScore: number;
    denseRank: number | null;
    sparseRank: number | null;
    rrfScore: number;
    rrfRank: number | null;
    kpRoleTags?: string[];
};

type ArticleRecord = {
    otid?: string;
    id?: string;
    title?: string;
    ot_title?: string;
};

type BadCaseExport = {
    cases?: Array<{
        query?: string;
        docHitAt1?: boolean;
        docHitAt5?: boolean;
        kpidHitAt1?: boolean;
        supportFullCoverTop5?: boolean;
        failure_risk?: string;
        failure_reasons?: string[];
    }>;
};

type BadCaseMetadata = NonNullable<BadCaseExport["cases"]>[number];

type EntryAuditBucket =
    | "early_missing"
    | "early_late"
    | "aggregation_drop"
    | "already_top1"
    | "unknown";

const DEFAULT_DATASET_FILE =
    "../Backend/test/test_dataset_granularity/test_dataset_granularity_in_domain_generalization_aligned_100_draft_v1.json";

const DEFAULT_TARGET_QUERIES = [
    "我现在是应届硕士生，准备报2024年博士，但不确定网上报名、材料上传、导师确认和系统填报有没有先后要求，报名阶段最需要注意哪些事？",
    "一名应届硕士毕业生 想报考2024年博士 报名时需要注意什么",
];

function loadDataset(filePath: string): EvalDatasetCase[] {
    return JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), filePath), "utf-8"),
    ) as EvalDatasetCase[];
}

function loadBadCaseQueries(filePath: string): Set<string> {
    const raw = JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), filePath), "utf-8"),
    ) as BadCaseExport;
    return new Set((raw.cases || []).map((item) => item.query || "").filter(Boolean));
}

function loadBadCaseMetadata(filePath: string): Map<string, BadCaseMetadata> {
    const raw = JSON.parse(
        fs.readFileSync(path.resolve(process.cwd(), filePath), "utf-8"),
    ) as BadCaseExport;
    const result = new Map<string, BadCaseMetadata>();
    (raw.cases || []).forEach((item) => {
        if (item.query) {
            result.set(item.query, item);
        }
    });
    return result;
}

function loadArticleTitleMap(): Map<string, string> {
    const absolutePath = path.resolve(process.cwd(), resolveBackendArticlesFile());
    const raw = JSON.parse(fs.readFileSync(absolutePath, "utf-8")) as ArticleRecord[];
    const result = new Map<string, string>();
    raw.forEach((item) => {
        const otid = item.otid || item.id;
        const title = item.ot_title || item.title;
        if (otid && title) {
            result.set(otid, title);
        }
    });
    return result;
}

function rankMap(indices: readonly number[]): Map<number, number> {
    const result = new Map<number, number>();
    indices.forEach((index, offset) => result.set(index, offset + 1));
    return result;
}

function computeSparseScores(params: {
    metadata: readonly Metadata[];
    querySparse: Record<number, number>;
    bm25Stats: ReturnType<typeof buildBM25Stats>;
}): Float32Array {
    const { metadata, querySparse, bm25Stats } = params;
    const sparseScores = new Float32Array(metadata.length);

    for (let index = 0; index < metadata.length; index += 1) {
        const meta = metadata[index];
        if (!meta.sparse || meta.sparse.length === 0) {
            continue;
        }

        const dl = bm25Stats.docLengths[index];
        const safeDl = Math.max(dl, bm25Stats.avgdl * 0.25);
        let sparse = 0;
        for (let j = 0; j < meta.sparse.length; j += 2) {
            const wordId = meta.sparse[j] as number;
            const tf = meta.sparse[j + 1] as number;
            if (!querySparse[wordId]) {
                continue;
            }

            const qWeight = querySparse[wordId] || 1;
            const idf = bm25Stats.idfMap.get(wordId) || 0;
            const numerator = tf * (BM25_K1 + 1);
            const denominator =
                tf + BM25_K1 * (1 - BM25_B + BM25_B * (safeDl / bm25Stats.avgdl));
            sparse += qWeight * idf * (numerator / denominator);
        }
        sparseScores[index] = sparse;
    }

    return sparseScores;
}

function buildCandidateAudit(params: {
    metadata: readonly Metadata[];
    denseScores: Float32Array;
    sparseScores: Float32Array;
    denseTop: readonly number[];
    sparseTop: readonly number[];
    topHybridLimit: number;
}): AuditCandidate[] {
    const {
        metadata,
        denseScores,
        sparseScores,
        denseTop,
        sparseTop,
        topHybridLimit,
    } = params;
    const denseRanks = rankMap(denseTop);
    const sparseRanks = rankMap(sparseTop);
    const rrfScores = new Map<number, number>();

    denseTop.forEach((index, rank) => {
        rrfScores.set(index, (1 / (rank + 1 + RRF_K)) * 100);
    });
    sparseTop.forEach((index, rank) => {
        rrfScores.set(
            index,
            (rrfScores.get(index) || 0) + (1 / (rank + 1 + RRF_K)) * 120,
        );
    });

    const sorted = Array.from(rrfScores.entries())
        .sort((left, right) => right[1] - left[1])
        .slice(0, topHybridLimit);
    const rrfRanks = new Map<number, number>();
    sorted.forEach(([index], rank) => rrfRanks.set(index, rank + 1));

    return sorted.map(([index, rrfScore]) => {
        const meta = metadata[index];
        return {
            id: meta.id,
            otid: resolveDocOtid(meta),
            type: meta.type,
            parentPkid: meta.parent_pkid,
            denseScore: denseScores[index] || 0,
            sparseScore: sparseScores[index] || 0,
            denseRank: denseRanks.get(index) || null,
            sparseRank: sparseRanks.get(index) || null,
            rrfScore,
            rrfRank: rrfRanks.get(index) || null,
            kpRoleTags: meta.kp_role_tags,
        };
    });
}

function parseFilterValues(value?: string): Set<string> {
    return new Set(
        (value || "")
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
    );
}

function matchesFilter(value: string | undefined, filters: Set<string>): boolean {
    return filters.size === 0 || (value !== undefined && filters.has(value));
}

function classifyEntryBucket(params: {
    bestCorrectRrfRank: number | null;
    finalRank: number | null;
}): EntryAuditBucket {
    const { bestCorrectRrfRank, finalRank } = params;
    if (!finalRank || !bestCorrectRrfRank) {
        return "unknown";
    }
    if (finalRank === 1) {
        return "already_top1";
    }
    if (bestCorrectRrfRank > 1000) {
        return "early_missing";
    }
    if (bestCorrectRrfRank > 50) {
        return "early_late";
    }
    if (bestCorrectRrfRank <= 20 && finalRank > 1) {
        return "aggregation_drop";
    }
    return "early_late";
}

async function main() {
    const datasetFile = process.env.SUASK_EVAL_DATASET_FILE || DEFAULT_DATASET_FILE;
    const badCaseFile = process.env.SUASK_AUDIT_BAD_CASE_FILE;
    const explicitTargetQueries = process.env.SUASK_AUDIT_QUERIES
        ? process.env.SUASK_AUDIT_QUERIES.split(/\r?\n/).filter(Boolean)
        : undefined;
    const badCaseQueries = badCaseFile ? loadBadCaseQueries(badCaseFile) : undefined;
    const badCaseMetadata = badCaseFile
        ? loadBadCaseMetadata(badCaseFile)
        : new Map<string, BadCaseMetadata>();
    const failureRiskFilters = parseFilterValues(process.env.SUASK_AUDIT_FAILURE_RISK);
    const docMissOnly = process.env.SUASK_AUDIT_DOC_MISS_ONLY === "1";
    const supportFilters = parseFilterValues(process.env.SUASK_AUDIT_SUPPORT_PATTERN);
    const queryTypeFilters = parseFilterValues(process.env.SUASK_AUDIT_QUERY_TYPE);
    const scopeFilters = parseFilterValues(process.env.SUASK_AUDIT_QUERY_SCOPE);
    const granularityFilters = parseFilterValues(
        process.env.SUASK_AUDIT_PREFERRED_GRANULARITY,
    );
    const topHybridLimit = Number.parseInt(
        process.env.SUASK_TOP_HYBRID_LIMIT || "1000",
        10,
    );
    const reportLimit = Number.parseInt(process.env.SUASK_AUDIT_TOP_N || "20", 10);

    const dataset = loadDataset(datasetFile);
    const targetCases = dataset.filter((item) => {
        if (explicitTargetQueries && !explicitTargetQueries.includes(item.query)) {
            return false;
        }
        if (!explicitTargetQueries && badCaseQueries && !badCaseQueries.has(item.query)) {
            return false;
        }
        if (
            !explicitTargetQueries &&
            !badCaseQueries &&
            !DEFAULT_TARGET_QUERIES.includes(item.query)
        ) {
            return false;
        }
        const badMeta = badCaseMetadata.get(item.query);
        if (docMissOnly && badMeta?.docHitAt1 !== false) {
            return false;
        }
        if (!matchesFilter(badMeta?.failure_risk, failureRiskFilters)) {
            return false;
        }
        return (
            matchesFilter(item.support_pattern, supportFilters) &&
            matchesFilter(item.query_type, queryTypeFilters) &&
            matchesFilter(item.query_scope, scopeFilters) &&
            matchesFilter(item.preferred_granularity, granularityFilters)
        );
    });
    if (targetCases.length === 0) {
        throw new Error("No target query found in dataset");
    }

    const engine = await loadFrontendEvalEngine();
    const articleTitles = loadArticleTitleMap();
    const queryVectors = await embedQueries(
        engine.extractor,
        targetCases.map((item) => item.query),
        engine.dimensions,
        { batchSize: DEFAULT_QUERY_EMBED_BATCH_SIZE },
    );
    const bm25Stats = buildBM25Stats(engine.metadataList);
    const rrfRankLimit = Math.min(RRF_RANK_LIMIT, engine.metadataList.length);

    const cases = targetCases.map((testCase, caseIndex) => {
        const queryWords = Array.from(
            new Set(fmmTokenize(testCase.query, engine.vocabMap)),
        );
        const querySparse = getQuerySparse(queryWords, engine.vocabMap);
        const queryIntent = parseQueryIntent(testCase.query);
        const denseScores = new Float32Array(engine.metadataList.length);
        for (let index = 0; index < engine.metadataList.length; index += 1) {
            denseScores[index] = dotProduct(
                queryVectors[caseIndex],
                engine.vectorMatrix,
                engine.metadataList[index].vector_index,
                engine.dimensions,
            );
        }
        const sparseScores = computeSparseScores({
            metadata: engine.metadataList,
            querySparse,
            bm25Stats,
        });
        const denseTop = selectTopLocalIndices(denseScores, rrfRankLimit);
        const sparseTop = selectTopLocalIndices(sparseScores, rrfRankLimit, {
            minimumScoreExclusive: 0,
        });
        const candidates = buildCandidateAudit({
            metadata: engine.metadataList,
            denseScores,
            sparseScores,
            denseTop,
            sparseTop,
            topHybridLimit,
        });
        const result = searchAndRank({
            queryVector: queryVectors[caseIndex],
            querySparse,
            queryWords,
            queryYearWordIds: queryIntent.years
                .map(String)
                .map((year) => engine.vocabMap.get(year))
                .filter((item): item is number => item !== undefined),
            queryIntent,
            queryScopeHint: testCase.query_scope,
            metadata: engine.metadataList,
            vectorMatrix: engine.vectorMatrix,
            dimensions: engine.dimensions,
            currentTimestamp: Date.now(),
            bm25Stats,
            weights: { Q: 1 / 3, KP: 1 / 3, OT: 1 / 3 },
            topHybridLimit,
            enableExplicitYearFilter: true,
            minimalMode: true,
        });
        const expectedOtid = testCase.expected_otid;
        const badMeta = badCaseMetadata.get(testCase.query);
        const correctDocCandidates = candidates.filter(
            (candidate) => candidate.otid === expectedOtid,
        );
        const bestCorrectCandidate = correctDocCandidates[0];
        const finalRank =
            result.matches.findIndex((match) => match.otid === expectedOtid) + 1 ||
            null;
        const entryBucket = classifyEntryBucket({
            bestCorrectRrfRank: bestCorrectCandidate?.rrfRank || null,
            finalRank,
        });
        const topDocs = Array.from(
            candidates.reduce((acc, candidate) => {
                const current = acc.get(candidate.otid);
                if (!current || candidate.rrfScore > current.bestRrfScore) {
                    acc.set(candidate.otid, {
                        otid: candidate.otid,
                        title: articleTitles.get(candidate.otid) || "",
                        bestRrfScore: candidate.rrfScore,
                        bestRrfRank: candidate.rrfRank,
                        bestType: candidate.type,
                    });
                }
                return acc;
            }, new Map<string, {
                otid: string;
                title: string;
                bestRrfScore: number;
                bestRrfRank: number | null;
                bestType: "Q" | "KP" | "OT";
            }>()),
        ).map(([, value]) => value)
            .sort((left, right) => right.bestRrfScore - left.bestRrfScore)
            .slice(0, reportLimit);

        return {
            query: testCase.query,
            expectedOtid,
            expectedTitle: articleTitles.get(expectedOtid) || "",
            queryScope: testCase.query_scope,
            queryType: testCase.query_type,
            supportPattern: testCase.support_pattern,
            preferredGranularity: testCase.preferred_granularity,
            themeFamily: testCase.theme_family,
            badCase: badMeta
                ? {
                      docHitAt1: badMeta.docHitAt1,
                      docHitAt5: badMeta.docHitAt5,
                      kpidHitAt1: badMeta.kpidHitAt1,
                      supportFullCoverTop5: badMeta.supportFullCoverTop5,
                      failureRisk: badMeta.failure_risk,
                      failureReasons: badMeta.failure_reasons || [],
                  }
                : null,
            queryIntent,
            finalRank,
            entryBucket,
            bestCorrectRrfRank: bestCorrectCandidate?.rrfRank || null,
            bestCorrectAtomicType: bestCorrectCandidate?.type || null,
            finalTopDocs: result.matches.slice(0, reportLimit).map((match, index) => ({
                rank: index + 1,
                otid: match.otid,
                title: articleTitles.get(match.otid) || "",
                score: match.score,
                bestKpid: match.best_kpid,
                kpEvidenceGroupCounts: match.kp_evidence_group_counts,
            })),
            correctDocCandidates,
            topDocs,
            topAtomicCandidates: candidates.slice(0, reportLimit),
        };
    });

    const summary = cases.reduce(
        (acc, item) => {
            acc.total += 1;
            acc.byEntryBucket[item.entryBucket] =
                (acc.byEntryBucket[item.entryBucket] || 0) + 1;
            const bucketKey = [
                item.supportPattern || "unknown",
                item.queryType || "unknown",
                item.queryScope || "unknown",
            ].join("|");
            acc.bySupportQueryScope[bucketKey] =
                (acc.bySupportQueryScope[bucketKey] || 0) + 1;
            return acc;
        },
        {
            total: 0,
            byEntryBucket: {} as Record<string, number>,
            bySupportQueryScope: {} as Record<string, number>,
        },
    );

    const report = {
        generatedAt: new Date().toISOString(),
        datasetFile,
        badCaseFile: badCaseFile || null,
        filters: {
            supportPattern: Array.from(supportFilters),
            queryType: Array.from(queryTypeFilters),
            queryScope: Array.from(scopeFilters),
            preferredGranularity: Array.from(granularityFilters),
            failureRisk: Array.from(failureRiskFilters),
            docMissOnly,
        },
        topHybridLimit,
        reportLimit,
        summary,
        cases,
    };

    const outDir = path.resolve(process.cwd(), "scripts/results");
    fs.mkdirSync(outDir, { recursive: true });
    const outFile = path.join(
        outDir,
        `granularity_retrieval_entry_audit_${Date.now()}.json`,
    );
    fs.writeFileSync(outFile, JSON.stringify(report, null, 2), "utf-8");
    console.log(`Saved report to ${outFile}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});

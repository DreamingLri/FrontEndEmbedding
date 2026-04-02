import * as fs from "fs";
import * as path from "path";

import { parseQueryIntent } from "../src/worker/vector_engine.ts";

type ComboLabel = "Q" | "KP+OT" | "Q+KP+OT";

type TopDocMatch = {
    rank: number;
    otid: string;
    score: number;
    best_kpid?: string;
};

type ExportCase = {
    query: string;
    dataset: string;
    docRank: number | null;
    topDocMatches: TopDocMatch[];
};

type CaseExportFile = {
    datasetKey: string;
    comboLabel: ComboLabel;
    caseCount: number;
    cases: ExportCase[];
};

type ReportFile = {
    mainDbVersion?: string;
    datasetKey?: string;
    onlineKpRoleRerankMode?: string;
    onlineKpRoleDocWeight?: number;
    kpAggregationMode?: string;
    lexicalBonusMode?: string;
};

type ComboFeatures = {
    combo: ComboLabel;
    docRank: number | null;
    top1Otid: string | null;
    top1Score: number | null;
    top2Score: number | null;
    normalizedGap: number;
};

type RoutedCase = {
    datasetSlug: string;
    query: string;
    hasIntent: boolean;
    hasYear: boolean;
    lengthBucket: "short" | "mid" | "long";
    combos: Record<ComboLabel, ComboFeatures>;
};

type Metrics = {
    total: number;
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type RouterConfig = {
    allAgreeChoice: ComboLabel;
    qKpAgreeChoice: ComboLabel;
    qFusionAgreeChoice: ComboLabel;
    kpFusionAgreeChoice: ComboLabel;
    allDisagreeGapDelta: number;
    intentYearFallback: ComboLabel;
    intentNoYearFallback: ComboLabel;
    noIntentYearFallback: ComboLabel;
    noIntentNoYearFallback: ComboLabel;
};

const ACTIVE_DATASETS = [
    {
        slug: "test_dataset_granularity_main_120_reviewed",
        label: "MainBench-120",
    },
    {
        slug: "test_dataset_granularity_in_domain_holdout_50_reviewed",
        label: "InDomainHoldout-50",
    },
    {
        slug: "test_dataset_granularity_external_ood_holdout_30_reviewed",
        label: "ExternalOODHoldout-30",
    },
] as const;

const ACTIVE_COMBOS: ReadonlyArray<{
    label: ComboLabel;
    suffix: string;
}> = [
    { label: "Q", suffix: "_bad_cases_q.json" },
    { label: "KP+OT", suffix: "_bad_cases_kp_ot.json" },
    { label: "Q+KP+OT", suffix: "_bad_cases_q_kp_ot.json" },
];

const RESULTS_DIR = path.resolve(process.cwd(), "scripts/results");
const TARGET_MAIN_DB_VERSION =
    process.env.SUASK_MAIN_DB_VERSION || "main_v2_plus_kpdenseshort";
const TARGET_ONLINE_KP_ROLE_MODE =
    process.env.SUASK_ONLINE_KP_ROLE_RERANK_MODE || "feature";

function getSiblingReportPath(exportPath: string, suffix: string): string {
    return exportPath.slice(0, -suffix.length) + ".json";
}

function loadJsonFile<T>(filePath: string): T {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function discoverLatestCaseExport(
    datasetSlug: string,
    combo: (typeof ACTIVE_COMBOS)[number],
): string {
    const candidates = fs
        .readdirSync(RESULTS_DIR)
        .filter(
            (fileName) =>
                fileName.includes(datasetSlug) && fileName.endsWith(combo.suffix),
        )
        .map((fileName) => path.join(RESULTS_DIR, fileName))
        .filter((exportPath) => {
            const reportPath = getSiblingReportPath(exportPath, combo.suffix);
            if (!fs.existsSync(reportPath)) {
                return false;
            }
            const report = loadJsonFile<ReportFile>(reportPath);
            return (
                report.mainDbVersion === TARGET_MAIN_DB_VERSION &&
                report.datasetKey === datasetSlug &&
                report.onlineKpRoleRerankMode === TARGET_ONLINE_KP_ROLE_MODE
            );
        })
        .sort((left, right) => {
            const leftStat = fs.statSync(left);
            const rightStat = fs.statSync(right);
            return rightStat.mtimeMs - leftStat.mtimeMs;
        });

    if (candidates.length === 0) {
        throw new Error(
            `未找到匹配导出: dataset=${datasetSlug}, combo=${combo.label}, db=${TARGET_MAIN_DB_VERSION}`,
        );
    }
    return candidates[0];
}

function getLengthBucket(query: string): "short" | "mid" | "long" {
    if (query.length <= 12) return "short";
    if (query.length <= 24) return "mid";
    return "long";
}

function normalizeGap(top1Score: number | null, top2Score: number | null): number {
    if (top1Score === null) {
        return 0;
    }
    const safeTop2 = top2Score ?? 0;
    const denom = Math.max(Math.abs(top1Score), 1e-6);
    return Math.max(0, (top1Score - safeTop2) / denom);
}

function buildComboFeatures(
    combo: ComboLabel,
    item: ExportCase,
): ComboFeatures {
    const top1 = item.topDocMatches[0];
    const top2 = item.topDocMatches[1];
    const top1Score = Number.isFinite(top1?.score) ? Number(top1.score) : null;
    const top2Score = Number.isFinite(top2?.score) ? Number(top2.score) : null;

    return {
        combo,
        docRank:
            Number.isFinite(item.docRank) && (item.docRank as number) > 0
                ? Number(item.docRank)
                : null,
        top1Otid: top1?.otid || null,
        top1Score,
        top2Score,
        normalizedGap: normalizeGap(top1Score, top2Score),
    };
}

function buildRoutedCasesForDataset(datasetSlug: string): RoutedCase[] {
    const loadedByCombo = Object.fromEntries(
        ACTIVE_COMBOS.map((combo) => {
            const exportPath = discoverLatestCaseExport(datasetSlug, combo);
            return [combo.label, loadJsonFile<CaseExportFile>(exportPath)];
        }),
    ) as Record<ComboLabel, CaseExportFile>;

    const baseCases = loadedByCombo.Q.cases;

    return baseCases.map((baseCase, index) => {
        const parsedIntent = parseQueryIntent(baseCase.query);
        const combos = Object.fromEntries(
            ACTIVE_COMBOS.map((combo) => {
                const comboCase = loadedByCombo[combo.label].cases[index];
                return [combo.label, buildComboFeatures(combo.label, comboCase)];
            }),
        ) as Record<ComboLabel, ComboFeatures>;

        return {
            datasetSlug,
            query: baseCase.query,
            hasIntent: parsedIntent.intentIds.length > 0,
            hasYear: parsedIntent.years.length > 0,
            lengthBucket: getLengthBucket(baseCase.query),
            combos,
        };
    });
}

function rankToMetrics(docRank: number | null): Omit<Metrics, "total"> {
    return {
        hitAt1: docRank === 1 ? 1 : 0,
        hitAt3:
            Number.isFinite(docRank) && (docRank as number) <= 3
                ? 1
                : 0,
        hitAt5:
            Number.isFinite(docRank) && (docRank as number) <= 5
                ? 1
                : 0,
        mrr:
            Number.isFinite(docRank) && (docRank as number) > 0
                ? 1 / Number(docRank)
                : 0,
    };
}

function summarizeMetrics(docRanks: Array<number | null>): Metrics {
    const total = docRanks.length;
    const seed = docRanks.reduce(
        (acc, docRank) => {
            const item = rankToMetrics(docRank);
            acc.hitAt1 += item.hitAt1;
            acc.hitAt3 += item.hitAt3;
            acc.hitAt5 += item.hitAt5;
            acc.mrr += item.mrr;
            return acc;
        },
        {
            hitAt1: 0,
            hitAt3: 0,
            hitAt5: 0,
            mrr: 0,
        },
    );

    return {
        total,
        hitAt1: (seed.hitAt1 / total) * 100,
        hitAt3: (seed.hitAt3 / total) * 100,
        hitAt5: (seed.hitAt5 / total) * 100,
        mrr: seed.mrr / total,
    };
}

function chooseComboWithPostFeatures(
    item: RoutedCase,
    config: RouterConfig,
): ComboLabel {
    const q = item.combos.Q;
    const kpOt = item.combos["KP+OT"];
    const qKpOt = item.combos["Q+KP+OT"];

    const qAgreesWithKpOt =
        q.top1Otid !== null &&
        kpOt.top1Otid !== null &&
        q.top1Otid === kpOt.top1Otid;
    const qAgreesWithFusion =
        q.top1Otid !== null &&
        qKpOt.top1Otid !== null &&
        q.top1Otid === qKpOt.top1Otid;
    const kpOtAgreesWithFusion =
        kpOt.top1Otid !== null &&
        qKpOt.top1Otid !== null &&
        kpOt.top1Otid === qKpOt.top1Otid;

    if (qAgreesWithKpOt && qAgreesWithFusion) {
        return config.allAgreeChoice;
    }
    if (qAgreesWithKpOt) {
        return config.qKpAgreeChoice;
    }
    if (qAgreesWithFusion) {
        return config.qFusionAgreeChoice;
    }
    if (kpOtAgreesWithFusion) {
        return config.kpFusionAgreeChoice;
    }

    const rankedByGap = [q, kpOt, qKpOt].sort(
        (left, right) => right.normalizedGap - left.normalizedGap,
    );
    const gapLead =
        rankedByGap[0].normalizedGap - rankedByGap[1].normalizedGap;
    if (gapLead >= config.allDisagreeGapDelta) {
        return rankedByGap[0].combo;
    }

    if (item.hasIntent) {
        return item.hasYear
            ? config.intentYearFallback
            : config.intentNoYearFallback;
    }
    return item.hasYear
        ? config.noIntentYearFallback
        : config.noIntentNoYearFallback;
}

function evaluateStaticCombo(
    cases: readonly RoutedCase[],
    combo: ComboLabel,
): Metrics {
    return summarizeMetrics(cases.map((item) => item.combos[combo].docRank));
}

function evaluateRouter(
    cases: readonly RoutedCase[],
    config: RouterConfig,
): Metrics {
    return summarizeMetrics(
        cases.map((item) => item.combos[chooseComboWithPostFeatures(item, config)].docRank),
    );
}

function isBetterMetrics(left: Metrics, right: Metrics | null): boolean {
    if (!right) return true;
    if (left.mrr !== right.mrr) return left.mrr > right.mrr;
    if (left.hitAt1 !== right.hitAt1) return left.hitAt1 > right.hitAt1;
    if (left.hitAt3 !== right.hitAt3) return left.hitAt3 > right.hitAt3;
    return left.hitAt5 > right.hitAt5;
}

function formatMetrics(metrics: Metrics): string {
    return `Hit@1=${metrics.hitAt1.toFixed(2)} | Hit@3=${metrics.hitAt3.toFixed(2)} | Hit@5=${metrics.hitAt5.toFixed(2)} | MRR=${metrics.mrr.toFixed(4)}`;
}

function main() {
    const byDataset = Object.fromEntries(
        ACTIVE_DATASETS.map((dataset) => [
            dataset.slug,
            buildRoutedCasesForDataset(dataset.slug),
        ]),
    ) as Record<string, RoutedCase[]>;

    const allCases = ACTIVE_DATASETS.flatMap((dataset) => byDataset[dataset.slug]);

    console.log(`软路由分析口径: db=${TARGET_MAIN_DB_VERSION}, onlineKprole=${TARGET_ONLINE_KP_ROLE_MODE}`);
    console.log("");
    console.log("静态基线:");
    (["Q", "KP+OT", "Q+KP+OT"] as ComboLabel[]).forEach((combo) => {
        console.log(`${combo.padEnd(8)} ${formatMetrics(evaluateStaticCombo(allCases, combo))}`);
    });

    const comboChoices: ComboLabel[] = ["Q", "KP+OT", "Q+KP+OT"];
    const deltas = [0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2];

    let bestOverall:
        | {
              config: RouterConfig;
              metrics: Metrics;
          }
        | null = null;

    comboChoices.forEach((allAgreeChoice) => {
        comboChoices.forEach((qKpAgreeChoice) => {
            comboChoices.forEach((qFusionAgreeChoice) => {
                comboChoices.forEach((kpFusionAgreeChoice) => {
                    deltas.forEach((allDisagreeGapDelta) => {
                        comboChoices.forEach((intentYearFallback) => {
                            comboChoices.forEach((intentNoYearFallback) => {
                                comboChoices.forEach((noIntentYearFallback) => {
                                    comboChoices.forEach((noIntentNoYearFallback) => {
                                        const config: RouterConfig = {
                                            allAgreeChoice,
                                            qKpAgreeChoice,
                                            qFusionAgreeChoice,
                                            kpFusionAgreeChoice,
                                            allDisagreeGapDelta,
                                            intentYearFallback,
                                            intentNoYearFallback,
                                            noIntentYearFallback,
                                            noIntentNoYearFallback,
                                        };
                                        const metrics = evaluateRouter(
                                            allCases,
                                            config,
                                        );
                                        if (
                                            isBetterMetrics(
                                                metrics,
                                                bestOverall?.metrics || null,
                                            )
                                        ) {
                                            bestOverall = { config, metrics };
                                        }
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });

    if (!bestOverall) {
        throw new Error("未找到可用软路由配置");
    }

    console.log("");
    console.log("全量最优后特征路由:");
    console.log(bestOverall.config);
    console.log(formatMetrics(bestOverall.metrics));

    console.log("");
    console.log("各数据集上的全量最优配置表现:");
    ACTIVE_DATASETS.forEach((dataset) => {
        const metrics = evaluateRouter(byDataset[dataset.slug], bestOverall!.config);
        console.log(`${dataset.label.padEnd(20)} ${formatMetrics(metrics)}`);
    });

    console.log("");
    console.log("留一数据集验证:");
    ACTIVE_DATASETS.forEach((targetDataset) => {
        const trainCases = ACTIVE_DATASETS.flatMap((dataset) =>
            dataset.slug === targetDataset.slug ? [] : byDataset[dataset.slug],
        );
        const testCases = byDataset[targetDataset.slug];

        let bestTrain:
            | {
                  config: RouterConfig;
                  metrics: Metrics;
              }
            | null = null;

        comboChoices.forEach((allAgreeChoice) => {
            comboChoices.forEach((qKpAgreeChoice) => {
                comboChoices.forEach((qFusionAgreeChoice) => {
                    comboChoices.forEach((kpFusionAgreeChoice) => {
                        deltas.forEach((allDisagreeGapDelta) => {
                            comboChoices.forEach((intentYearFallback) => {
                                comboChoices.forEach((intentNoYearFallback) => {
                                    comboChoices.forEach((noIntentYearFallback) => {
                                        comboChoices.forEach((noIntentNoYearFallback) => {
                                            const config: RouterConfig = {
                                                allAgreeChoice,
                                                qKpAgreeChoice,
                                                qFusionAgreeChoice,
                                                kpFusionAgreeChoice,
                                                allDisagreeGapDelta,
                                                intentYearFallback,
                                                intentNoYearFallback,
                                                noIntentYearFallback,
                                                noIntentNoYearFallback,
                                            };
                                            const trainMetrics = evaluateRouter(
                                                trainCases,
                                                config,
                                            );
                                            if (
                                                isBetterMetrics(
                                                    trainMetrics,
                                                    bestTrain?.metrics || null,
                                                )
                                            ) {
                                                bestTrain = {
                                                    config,
                                                    metrics: trainMetrics,
                                                };
                                            }
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });

        const testMetrics = evaluateRouter(testCases, bestTrain!.config);
        const staticBest = [
            evaluateStaticCombo(testCases, "Q"),
            evaluateStaticCombo(testCases, "KP+OT"),
            evaluateStaticCombo(testCases, "Q+KP+OT"),
        ].reduce<Metrics | null>(
            (best, current) => (isBetterMetrics(current, best) ? current : best),
            null,
        );

        console.log(`${targetDataset.label}`);
        console.log(`  train_best: ${formatMetrics(bestTrain!.metrics)}`);
        console.log(`  test_router: ${formatMetrics(testMetrics)}`);
        console.log(`  test_static_best: ${formatMetrics(staticBest!)}`);
        console.log(`  config: ${JSON.stringify(bestTrain!.config)}`);
    });
}

main();

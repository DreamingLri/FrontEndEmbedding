import * as fs from "fs";
import * as path from "path";

type Metrics = {
    hitAt1: number;
    hitAt3: number;
    hitAt5: number;
    mrr: number;
};

type KpidMetrics = Metrics & {
    applicable: number;
    docHitAt1Total: number;
    docHitAt1CorrectKpid: number;
    docHitAt1WrongKpid: number;
    docHitAt5Total: number;
    docHitAt5CorrectKpid: number;
    docHitAt5WrongKpid: number;
};

type ComboReport = {
    label: string;
    uniform: {
        combined: Metrics;
        kpidCombined: KpidMetrics;
    };
    tuned: {
        bestWeights: {
            Q: number;
            KP: number;
            OT: number;
        };
        combinedCombined: Metrics;
        kpidCombinedCombined: KpidMetrics;
    };
};

type EvalReport = {
    combos: ComboReport[];
};

type BarItem = {
    label: string;
    value: number;
    color: string;
};

type GroupedBarSeries = {
    name: string;
    color: string;
    values: number[];
};

type StackedSeries = {
    name: string;
    color: string;
    values: number[];
};

const RESULTS_DIR = path.resolve(process.cwd(), "scripts/results");
const FIGURE_DIR = path.join(RESULTS_DIR, "figures");

const BASELINE_PATH = path.join(
    RESULTS_DIR,
    "granularity_mix_test_dataset_granularity_top1000_kpagg-max_lexsum_onlinekproleoff_kpreranknone_docreranknone_1774493582187.json",
);
const ONLINE_BEFORE_FIX_PATH = path.join(
    RESULTS_DIR,
    "granularity_mix_test_dataset_granularity_top1000_kpagg-max_lexsum_onlinekprolefeature-w035_kpreranknone_docreranknone_1774494253943.json",
);
const ONLINE_AFTER_FIX_PATH = path.join(
    RESULTS_DIR,
    "granularity_mix_test_dataset_granularity_top1000_kpagg-max_lexsum_onlinekprolefeature-w035_kpreranknone_docreranknone_1774494600482.json",
);

const COLORS = {
    blue: "#2F5BEA",
    teal: "#1F9D8B",
    orange: "#F08A24",
    red: "#D64545",
    gold: "#D4A017",
    purple: "#6A5ACD",
    gray: "#98A2B3",
    dark: "#1F2937",
    lightGrid: "#E5E7EB",
    bg: "#FFFFFF",
};

function ensureDir(dir: string) {
    fs.mkdirSync(dir, { recursive: true });
}

function loadReport(filePath: string): EvalReport {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as EvalReport;
}

function getCombo(report: EvalReport, label: string): ComboReport {
    const combo = report.combos.find((item) => item.label === label);
    if (!combo) {
        throw new Error(`Combo ${label} not found in report`);
    }
    return combo;
}

function formatPercent(value: number): string {
    return `${value.toFixed(2)}%`;
}

function formatMrr(value: number): string {
    return value.toFixed(4);
}

function svgWrap(width: number, height: number, content: string): string {
    return [
        `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
        `<rect width="100%" height="100%" fill="${COLORS.bg}"/>`,
        content,
        `</svg>`,
    ].join("");
}

function text(
    x: number,
    y: number,
    value: string,
    options: {
        size?: number;
        weight?: number | string;
        fill?: string;
        anchor?: "start" | "middle" | "end";
    } = {},
): string {
    const {
        size = 14,
        weight = 400,
        fill = COLORS.dark,
        anchor = "start",
    } = options;
    return `<text x="${x}" y="${y}" font-family="'Microsoft YaHei','PingFang SC',sans-serif" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${value}</text>`;
}

function makeBarChartSvg(params: {
    title: string;
    subtitle: string;
    yLabel: string;
    items: BarItem[];
    width?: number;
    height?: number;
    maxValue?: number;
    formatter?: (value: number) => string;
}) {
    const {
        title,
        subtitle,
        yLabel,
        items,
        width = 1080,
        height = 720,
        maxValue,
        formatter = formatPercent,
    } = params;

    const margin = { top: 110, right: 60, bottom: 130, left: 100 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const max = maxValue ?? Math.max(...items.map((item) => item.value)) * 1.15;
    const step = max / 5;
    const barWidth = chartWidth / items.length * 0.58;
    const gap = chartWidth / items.length;

    const grid = Array.from({ length: 6 }, (_, index) => {
        const value = step * index;
        const y = margin.top + chartHeight - (value / max) * chartHeight;
        return [
            `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${COLORS.lightGrid}" stroke-width="1"/>`,
            text(margin.left - 12, y + 5, formatter(value), {
                size: 12,
                fill: COLORS.gray,
                anchor: "end",
            }),
        ].join("");
    }).join("");

    const bars = items.map((item, index) => {
        const x = margin.left + gap * index + (gap - barWidth) / 2;
        const barHeight = (item.value / max) * chartHeight;
        const y = margin.top + chartHeight - barHeight;
        return [
            `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="10" fill="${item.color}"/>`,
            text(x + barWidth / 2, y - 10, formatter(item.value), {
                size: 12,
                weight: 600,
                anchor: "middle",
            }),
            text(x + barWidth / 2, margin.top + chartHeight + 28, item.label, {
                size: 13,
                anchor: "middle",
            }),
        ].join("");
    }).join("");

    const content = [
        text(width / 2, 42, title, {
            size: 28,
            weight: 700,
            anchor: "middle",
        }),
        text(width / 2, 72, subtitle, {
            size: 14,
            fill: COLORS.gray,
            anchor: "middle",
        }),
        text(26, margin.top + chartHeight / 2, yLabel, {
            size: 14,
            fill: COLORS.gray,
        }),
        grid,
        `<line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${width - margin.right}" y2="${margin.top + chartHeight}" stroke="${COLORS.dark}" stroke-width="1.2"/>`,
        bars,
    ].join("");

    return svgWrap(width, height, content);
}

function makeGroupedBarChartSvg(params: {
    title: string;
    subtitle: string;
    categories: string[];
    series: GroupedBarSeries[];
    width?: number;
    height?: number;
    maxValue?: number;
    formatter?: (value: number) => string;
}) {
    const {
        title,
        subtitle,
        categories,
        series,
        width = 1080,
        height = 760,
        maxValue,
        formatter = formatPercent,
    } = params;

    const margin = { top: 125, right: 70, bottom: 130, left: 110 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const max = maxValue ?? Math.max(...series.flatMap((item) => item.values)) * 1.18;
    const groupGap = chartWidth / categories.length;
    const totalBarWidth = groupGap * 0.72;
    const barWidth = totalBarWidth / series.length;

    const grid = Array.from({ length: 6 }, (_, index) => {
        const value = (max / 5) * index;
        const y = margin.top + chartHeight - (value / max) * chartHeight;
        return [
            `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${COLORS.lightGrid}" stroke-width="1"/>`,
            text(margin.left - 14, y + 5, formatter(value), {
                size: 12,
                fill: COLORS.gray,
                anchor: "end",
            }),
        ].join("");
    }).join("");

    const legend = series.map((item, index) => {
        const x = width - margin.right - 220 + index * 110;
        return [
            `<rect x="${x}" y="84" width="18" height="18" rx="4" fill="${item.color}"/>`,
            text(x + 26, 98, item.name, { size: 13 }),
        ].join("");
    }).join("");

    const bars = categories.map((category, catIndex) => {
        const startX = margin.left + groupGap * catIndex + (groupGap - totalBarWidth) / 2;
        const pieces = series.map((item, seriesIndex) => {
            const value = item.values[catIndex];
            const barHeight = (value / max) * chartHeight;
            const x = startX + seriesIndex * barWidth;
            const y = margin.top + chartHeight - barHeight;
            return [
                `<rect x="${x}" y="${y}" width="${barWidth - 10}" height="${barHeight}" rx="8" fill="${item.color}"/>`,
                text(x + (barWidth - 10) / 2, y - 8, formatter(value), {
                    size: 11,
                    anchor: "middle",
                    weight: 600,
                }),
            ].join("");
        }).join("");

        return [
            pieces,
            text(startX + totalBarWidth / 2, margin.top + chartHeight + 30, category, {
                size: 13,
                anchor: "middle",
            }),
        ].join("");
    }).join("");

    const content = [
        text(width / 2, 42, title, {
            size: 28,
            weight: 700,
            anchor: "middle",
        }),
        text(width / 2, 72, subtitle, {
            size: 14,
            fill: COLORS.gray,
            anchor: "middle",
        }),
        legend,
        grid,
        `<line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${width - margin.right}" y2="${margin.top + chartHeight}" stroke="${COLORS.dark}" stroke-width="1.2"/>`,
        bars,
    ].join("");

    return svgWrap(width, height, content);
}

function makeStackedChartSvg(params: {
    title: string;
    subtitle: string;
    categories: string[];
    series: StackedSeries[];
    width?: number;
    height?: number;
    maxValue?: number;
}) {
    const {
        title,
        subtitle,
        categories,
        series,
        width = 1080,
        height = 760,
        maxValue,
    } = params;

    const margin = { top: 125, right: 70, bottom: 130, left: 110 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const totals = categories.map((_, index) =>
        series.reduce((sum, item) => sum + item.values[index], 0),
    );
    const max = maxValue ?? Math.max(...totals) * 1.2;
    const groupGap = chartWidth / categories.length;
    const barWidth = groupGap * 0.5;

    const legend = series.map((item, index) => {
        const x = width - margin.right - 220 + index * 110;
        return [
            `<rect x="${x}" y="84" width="18" height="18" rx="4" fill="${item.color}"/>`,
            text(x + 26, 98, item.name, { size: 13 }),
        ].join("");
    }).join("");

    const grid = Array.from({ length: 6 }, (_, index) => {
        const value = (max / 5) * index;
        const y = margin.top + chartHeight - (value / max) * chartHeight;
        return [
            `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${COLORS.lightGrid}" stroke-width="1"/>`,
            text(margin.left - 14, y + 5, value.toFixed(0), {
                size: 12,
                fill: COLORS.gray,
                anchor: "end",
            }),
        ].join("");
    }).join("");

    const bars = categories.map((category, catIndex) => {
        const x = margin.left + groupGap * catIndex + (groupGap - barWidth) / 2;
        let currentTop = margin.top + chartHeight;
        const segments = series.map((item) => {
            const value = item.values[catIndex];
            const heightValue = (value / max) * chartHeight;
            const y = currentTop - heightValue;
            currentTop = y;
            return [
                `<rect x="${x}" y="${y}" width="${barWidth}" height="${heightValue}" rx="6" fill="${item.color}"/>`,
                value > 0
                    ? text(x + barWidth / 2, y + heightValue / 2 + 5, `${value}`, {
                          size: 12,
                          anchor: "middle",
                          fill: COLORS.bg,
                          weight: 700,
                      })
                    : "",
            ].join("");
        }).join("");

        return [
            segments,
            text(x + barWidth / 2, currentTop - 12, `${totals[catIndex]}`, {
                size: 12,
                anchor: "middle",
                weight: 700,
            }),
            text(x + barWidth / 2, margin.top + chartHeight + 30, category, {
                size: 13,
                anchor: "middle",
            }),
        ].join("");
    }).join("");

    const content = [
        text(width / 2, 42, title, {
            size: 28,
            weight: 700,
            anchor: "middle",
        }),
        text(width / 2, 72, subtitle, {
            size: 14,
            fill: COLORS.gray,
            anchor: "middle",
        }),
        legend,
        grid,
        `<line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${width - margin.right}" y2="${margin.top + chartHeight}" stroke="${COLORS.dark}" stroke-width="1.2"/>`,
        bars,
    ].join("");

    return svgWrap(width, height, content);
}

function saveSvg(fileName: string, content: string) {
    ensureDir(FIGURE_DIR);
    fs.writeFileSync(path.join(FIGURE_DIR, fileName), content, "utf8");
}

function writeSummaryMarkdown(content: string) {
    ensureDir(FIGURE_DIR);
    fs.writeFileSync(
        path.join(FIGURE_DIR, "innovation_figures.md"),
        content,
        "utf8",
    );
}

function main() {
    const baseline = loadReport(BASELINE_PATH);
    const beforeFix = loadReport(ONLINE_BEFORE_FIX_PATH);
    const afterFix = loadReport(ONLINE_AFTER_FIX_PATH);

    const afterCombos = afterFix.combos;
    const comboOrder = ["Q", "KP", "OT", "Q+KP", "Q+OT", "KP+OT", "Q+KP+OT"];
    const comboColors = [
        COLORS.gray,
        COLORS.orange,
        COLORS.gold,
        COLORS.blue,
        COLORS.purple,
        COLORS.teal,
        COLORS.red,
    ];

    const hitChart = makeBarChartSvg({
        title: "图4-1 不同粒度组合的整体检索效果",
        subtitle: "granularity 正式集，采用各组合的 tuned-combined 文档级 Hit@1",
        yLabel: "Hit@1",
        items: comboOrder.map((label, index) => ({
            label,
            value: getCombo(afterFix, label).tuned.combinedCombined.hitAt1,
            color: comboColors[index],
        })),
        maxValue: 70,
        formatter: formatPercent,
    });
    saveSvg("fig4_1_granularity_hit1.svg", hitChart);

    const evidenceChart = makeGroupedBarChartSvg({
        title: "图4-2 KP 证据增强前后的效果对比",
        subtitle: "KP+OT tuned-combined，baseline 与 online kp role rerank 对比",
        categories: ["文档 Hit@1", "文档 MRR", "kpid Hit@1", "kpid MRR"],
        series: [
            {
                name: "增强前",
                color: COLORS.orange,
                values: [
                    getCombo(baseline, "KP+OT").tuned.combinedCombined.hitAt1,
                    getCombo(baseline, "KP+OT").tuned.combinedCombined.mrr * 100,
                    getCombo(baseline, "KP+OT").tuned.kpidCombinedCombined.hitAt1,
                    getCombo(baseline, "KP+OT").tuned.kpidCombinedCombined.mrr * 100,
                ],
            },
            {
                name: "增强后",
                color: COLORS.teal,
                values: [
                    getCombo(afterFix, "KP+OT").tuned.combinedCombined.hitAt1,
                    getCombo(afterFix, "KP+OT").tuned.combinedCombined.mrr * 100,
                    getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined.hitAt1,
                    getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined.mrr * 100,
                ],
            },
        ],
        maxValue: 75,
        formatter: formatPercent,
    });
    saveSvg("fig4_2_kp_evidence_gain.svg", evidenceChart);

    const diagnosisChart = makeStackedChartSvg({
        title: "图4-3 主证据 KP 选择情况诊断",
        subtitle: "KP+OT tuned-combined，在文档 Top1 正确时的 KP 主证据命中情况",
        categories: ["增强前", "增强后"],
        series: [
            {
                name: "主证据正确",
                color: COLORS.teal,
                values: [
                    getCombo(baseline, "KP+OT").tuned.kpidCombinedCombined
                        .docHitAt1CorrectKpid,
                    getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined
                        .docHitAt1CorrectKpid,
                ],
            },
            {
                name: "主证据错误",
                color: COLORS.red,
                values: [
                    getCombo(baseline, "KP+OT").tuned.kpidCombinedCombined
                        .docHitAt1WrongKpid,
                    getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined
                        .docHitAt1WrongKpid,
                ],
            },
        ],
        maxValue: 9,
    });
    saveSvg("fig4_3_kpid_diagnosis.svg", diagnosisChart);

    const windowChart = makeGroupedBarChartSvg({
        title: "图4-4 浅层候选窗口策略的效果",
        subtitle: "KP+OT tuned-combined，修正前后正式线上链路对比",
        categories: ["文档 Hit@1", "文档 MRR", "kpid Hit@1", "kpid MRR"],
        series: [
            {
                name: "修正前",
                color: COLORS.purple,
                values: [
                    getCombo(beforeFix, "KP+OT").tuned.combinedCombined.hitAt1,
                    getCombo(beforeFix, "KP+OT").tuned.combinedCombined.mrr * 100,
                    getCombo(beforeFix, "KP+OT").tuned.kpidCombinedCombined.hitAt1,
                    getCombo(beforeFix, "KP+OT").tuned.kpidCombinedCombined.mrr * 100,
                ],
            },
            {
                name: "修正后",
                color: COLORS.blue,
                values: [
                    getCombo(afterFix, "KP+OT").tuned.combinedCombined.hitAt1,
                    getCombo(afterFix, "KP+OT").tuned.combinedCombined.mrr * 100,
                    getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined.hitAt1,
                    getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined.mrr * 100,
                ],
            },
        ],
        maxValue: 75,
        formatter: formatPercent,
    });
    saveSvg("fig4_4_window_strategy.svg", windowChart);

    const readme = [
        "# 创新点图表输出",
        "",
        "生成时间基于当前 granularity 正式集结果文件。",
        "",
        "## 文件说明",
        "",
        `1. [fig4_1_granularity_hit1.svg](./fig4_1_granularity_hit1.svg)`,
        "   - 展示不同粒度组合的文档级 Hit@1，支撑“多粒度统一建模”创新点。",
        `2. [fig4_2_kp_evidence_gain.svg](./fig4_2_kp_evidence_gain.svg)`,
        "   - 对比 KP+OT 在引入结构化 KP 证据增强前后的文档级与 kpid 级指标。",
        `3. [fig4_3_kpid_diagnosis.svg](./fig4_3_kpid_diagnosis.svg)`,
        "   - 诊断文档 Top1 正确时，主证据 KP 是否也被正确选中。",
        `4. [fig4_4_window_strategy.svg](./fig4_4_window_strategy.svg)`,
        "   - 展示浅层候选窗口修正前后正式线上链路的指标变化。",
        "",
        "## 主要结果摘要",
        "",
        `- KP+OT baseline：文档 Hit@1=${formatPercent(getCombo(baseline, "KP+OT").tuned.combinedCombined.hitAt1)}，kpid Hit@1=${formatPercent(getCombo(baseline, "KP+OT").tuned.kpidCombinedCombined.hitAt1)}`,
        `- KP+OT + online kp role rerank：文档 Hit@1=${formatPercent(getCombo(afterFix, "KP+OT").tuned.combinedCombined.hitAt1)}，kpid Hit@1=${formatPercent(getCombo(afterFix, "KP+OT").tuned.kpidCombinedCombined.hitAt1)}`,
        `- 浅层窗口修正前：文档 MRR=${formatMrr(getCombo(beforeFix, "KP+OT").tuned.combinedCombined.mrr)}`,
        `- 浅层窗口修正后：文档 MRR=${formatMrr(getCombo(afterFix, "KP+OT").tuned.combinedCombined.mrr)}`,
        "",
    ].join("\n");
    writeSummaryMarkdown(readme);

    console.log(`Saved innovation figures to ${FIGURE_DIR}`);
}

main();

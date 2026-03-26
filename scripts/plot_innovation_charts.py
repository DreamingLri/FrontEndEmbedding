from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures_python"

BASELINE_PATH = RESULTS_DIR / (
    "granularity_mix_test_dataset_granularity_top1000_kpagg-max_lexsum_"
    "onlinekproleoff_kpreranknone_docreranknone_1774493582187.json"
)
ONLINE_BEFORE_FIX_PATH = RESULTS_DIR / (
    "granularity_mix_test_dataset_granularity_top1000_kpagg-max_lexsum_"
    "onlinekprolefeature-w035_kpreranknone_docreranknone_1774494253943.json"
)
ONLINE_AFTER_FIX_PATH = RESULTS_DIR / (
    "granularity_mix_test_dataset_granularity_top1000_kpagg-max_lexsum_"
    "onlinekprolefeature-w035_kpreranknone_docreranknone_1774494600482.json"
)

COLORS = {
    "blue": "#2F5BEA",
    "teal": "#1F9D8B",
    "orange": "#F08A24",
    "red": "#D64545",
    "gold": "#D4A017",
    "purple": "#6A5ACD",
    "gray": "#98A2B3",
    "dark": "#1F2937",
    "light_grid": "#E5E7EB",
}

TITLE_SIZE = 16
SUBTITLE_SIZE = 10
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11
ANNOTATION_SIZE = 9


def configure_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = COLORS["dark"]
    plt.rcParams["axes.labelcolor"] = COLORS["dark"]
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = TITLE_SIZE
    plt.rcParams["axes.labelsize"] = LABEL_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_SIZE
    plt.rcParams["xtick.color"] = COLORS["dark"]
    plt.rcParams["ytick.color"] = COLORS["dark"]
    plt.rcParams["grid.color"] = COLORS["light_grid"]
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.axisbelow"] = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_combo(report: dict, label: str) -> dict:
    for combo in report["combos"]:
        if combo["label"] == label:
            return combo
    raise KeyError(f"Combo not found: {label}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    ensure_dir(FIGURE_DIR)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, formatter) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ax.get_ylim()[1] * 0.015,
            formatter(height),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color=COLORS["dark"],
        )


def plot_granularity_hit1(report: dict) -> None:
    combo_order = ["Q", "KP", "OT", "Q+KP", "Q+OT", "KP+OT", "Q+KP+OT"]
    colors = [
        COLORS["gray"],
        COLORS["orange"],
        COLORS["gold"],
        COLORS["blue"],
        COLORS["purple"],
        COLORS["teal"],
        COLORS["red"],
    ]
    values = [
        get_combo(report, label)["tuned"]["combinedCombined"]["hitAt1"]
        for label in combo_order
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    bars = ax.bar(combo_order, values, color=colors, width=0.62)
    ax.set_title("图4-1 多粒度组合的整体检索效果", pad=14)
    ax.text(
        0.5,
        1.01,
        "granularity 正式集，采用各组合的 tuned-combined 文档级 Hit@1",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=SUBTITLE_SIZE,
        color=COLORS["gray"],
    )
    ax.set_ylabel("Hit@1 (%)")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.9)
    add_bar_labels(ax, bars, lambda x: f"{x:.2f}%")
    save_figure(fig, "fig4_1_granularity_hit1_py")


def plot_kp_evidence_gain(baseline: dict, after_fix: dict) -> None:
    base = get_combo(baseline, "KP+OT")["tuned"]
    improved = get_combo(after_fix, "KP+OT")["tuned"]
    categories = ["文档 Hit@1", "文档 MRR", "kpid Hit@1", "kpid MRR"]
    before_values = [
        base["combinedCombined"]["hitAt1"],
        base["combinedCombined"]["mrr"] * 100,
        base["kpidCombinedCombined"]["hitAt1"],
        base["kpidCombinedCombined"]["mrr"] * 100,
    ]
    after_values = [
        improved["combinedCombined"]["hitAt1"],
        improved["combinedCombined"]["mrr"] * 100,
        improved["kpidCombinedCombined"]["hitAt1"],
        improved["kpidCombinedCombined"]["mrr"] * 100,
    ]

    x = np.arange(len(categories))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    bars1 = ax.bar(x - width / 2, before_values, width, label="增强前", color=COLORS["orange"])
    bars2 = ax.bar(x + width / 2, after_values, width, label="增强后", color=COLORS["teal"])
    ax.set_title("图4-2 KP 证据增强前后的效果对比", pad=14)
    ax.text(
        0.5,
        1.01,
        "KP+OT tuned-combined，baseline 与 online kp role rerank 对比",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=SUBTITLE_SIZE,
        color=COLORS["gray"],
    )
    ax.set_ylabel("指标值（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.9)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1, lambda x: f"{x:.2f}%")
    add_bar_labels(ax, bars2, lambda x: f"{x:.2f}%")
    save_figure(fig, "fig4_2_kp_evidence_gain_py")


def plot_kpid_diagnosis(baseline: dict, after_fix: dict) -> None:
    base = get_combo(baseline, "KP+OT")["tuned"]["kpidCombinedCombined"]
    improved = get_combo(after_fix, "KP+OT")["tuned"]["kpidCombinedCombined"]
    categories = ["增强前", "增强后"]
    correct = [base["docHitAt1CorrectKpid"], improved["docHitAt1CorrectKpid"]]
    wrong = [base["docHitAt1WrongKpid"], improved["docHitAt1WrongKpid"]]

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    bars1 = ax.bar(x, correct, color=COLORS["teal"], width=0.5, label="主证据正确")
    bars2 = ax.bar(x, wrong, bottom=correct, color=COLORS["red"], width=0.5, label="主证据错误")
    ax.set_title("图4-3 主证据 KP 选择情况诊断", pad=14)
    ax.text(
        0.5,
        1.01,
        "KP+OT tuned-combined，在文档 Top1 正确时的 KP 主证据命中情况",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=SUBTITLE_SIZE,
        color=COLORS["gray"],
    )
    ax.set_ylabel("样本数")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 9)
    ax.grid(axis="y", alpha=0.9)
    ax.legend(frameon=False)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f"{int(height)}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    totals = [correct[i] + wrong[i] for i in range(len(categories))]
    for idx, total in enumerate(totals):
        ax.text(x[idx], total + 0.2, f"{total}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    save_figure(fig, "fig4_3_kpid_diagnosis_py")


def plot_window_strategy(before_fix: dict, after_fix: dict) -> None:
    before = get_combo(before_fix, "KP+OT")["tuned"]
    after = get_combo(after_fix, "KP+OT")["tuned"]
    categories = ["文档 Hit@1", "文档 MRR", "kpid Hit@1", "kpid MRR"]
    before_values = [
        before["combinedCombined"]["hitAt1"],
        before["combinedCombined"]["mrr"] * 100,
        before["kpidCombinedCombined"]["hitAt1"],
        before["kpidCombinedCombined"]["mrr"] * 100,
    ]
    after_values = [
        after["combinedCombined"]["hitAt1"],
        after["combinedCombined"]["mrr"] * 100,
        after["kpidCombinedCombined"]["hitAt1"],
        after["kpidCombinedCombined"]["mrr"] * 100,
    ]

    x = np.arange(len(categories))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    bars1 = ax.bar(x - width / 2, before_values, width, label="修正前", color=COLORS["purple"])
    bars2 = ax.bar(x + width / 2, after_values, width, label="修正后", color=COLORS["blue"])
    ax.set_title("图4-4 浅层候选窗口策略的效果", pad=14)
    ax.text(
        0.5,
        1.01,
        "KP+OT tuned-combined，修正前后正式线上链路对比",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=SUBTITLE_SIZE,
        color=COLORS["gray"],
    )
    ax.set_ylabel("指标值（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.9)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1, lambda x: f"{x:.2f}%")
    add_bar_labels(ax, bars2, lambda x: f"{x:.2f}%")
    save_figure(fig, "fig4_4_window_strategy_py")


def plot_support_pattern_breakdown(baseline: dict, after_fix: dict) -> None:
    base_groups = get_combo(baseline, "KP+OT")["groupBreakdowns"]["supportPattern"]
    after_groups = get_combo(after_fix, "KP+OT")["groupBreakdowns"]["supportPattern"]
    order = ["single_kp", "multi_kp", "ot_required"]
    labels = {
        "single_kp": "single_kp\n(n=12)",
        "multi_kp": "multi_kp\n(n=1)",
        "ot_required": "ot_required\n(n=2)",
    }

    doc_before = [base_groups[key]["tunedCombined"]["hitAt1"] for key in order]
    doc_after = [after_groups[key]["tunedCombined"]["hitAt1"] for key in order]
    kpid_before = [base_groups[key]["kpidTunedCombined"]["hitAt1"] for key in order]
    kpid_after = [after_groups[key]["kpidTunedCombined"]["hitAt1"] for key in order]

    x = np.arange(len(order))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)

    left = axes[0]
    bars1 = left.bar(x - width / 2, doc_before, width, label="增强前", color=COLORS["orange"])
    bars2 = left.bar(x + width / 2, doc_after, width, label="增强后", color=COLORS["teal"])
    left.set_title("文档级 Hit@1", fontsize=14)
    left.set_ylabel("Hit@1 (%)")
    left.set_xticks(x)
    left.set_xticklabels([labels[key] for key in order])
    left.set_ylim(0, 100)
    left.grid(axis="y", alpha=0.9)
    add_bar_labels(left, bars1, lambda value: f"{value:.1f}%")
    add_bar_labels(left, bars2, lambda value: f"{value:.1f}%")
    left.legend(frameon=False, loc="upper left")

    right = axes[1]
    bars3 = right.bar(x - width / 2, kpid_before, width, label="增强前", color=COLORS["purple"])
    bars4 = right.bar(x + width / 2, kpid_after, width, label="增强后", color=COLORS["blue"])
    right.set_title("kpid 级 Hit@1", fontsize=14)
    right.set_xticks(x)
    right.set_xticklabels([labels[key] for key in order])
    right.set_ylim(0, 100)
    right.grid(axis="y", alpha=0.9)
    add_bar_labels(right, bars3, lambda value: f"{value:.1f}%")
    add_bar_labels(right, bars4, lambda value: f"{value:.1f}%")

    fig.suptitle("图4-5 support_pattern 分组效果对比", fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.96,
        "KP+OT tuned-combined，baseline 与 online kp role rerank 在不同支撑模式上的效果差异",
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        color=COLORS["gray"],
    )
    save_figure(fig, "fig4_5_support_pattern_py")


def write_summary(baseline: dict, before_fix: dict, after_fix: dict) -> None:
    base = get_combo(baseline, "KP+OT")["tuned"]
    before = get_combo(before_fix, "KP+OT")["tuned"]
    after = get_combo(after_fix, "KP+OT")["tuned"]
    after_groups = get_combo(after_fix, "KP+OT")["groupBreakdowns"]["supportPattern"]
    content = "\n".join(
        [
            "# Python 版创新点图表输出",
            "",
            "## 文件说明",
            "",
            "1. `fig4_1_granularity_hit1_py.svg/.png`：不同粒度组合的整体检索效果",
            "2. `fig4_2_kp_evidence_gain_py.svg/.png`：KP 证据增强前后对比",
            "3. `fig4_3_kpid_diagnosis_py.svg/.png`：主证据 KP 选择诊断",
            "4. `fig4_4_window_strategy_py.svg/.png`：浅层候选窗口修正效果",
            "5. `fig4_5_support_pattern_py.svg/.png`：按 support_pattern 分组的效果对比",
            "",
            "以上图表均同步导出 `svg / png / pdf` 三种格式。",
            "",
            "## 运行方式",
            "",
            "推荐在 `Backend` 目录下使用 `uv` 运行：",
            "",
            "```powershell",
            "uv run python ..\\FrontEnd\\scripts\\plot_innovation_charts.py",
            "```",
            "",
            "## 主要结果摘要",
            "",
            f"- KP+OT baseline：文档 Hit@1={base['combinedCombined']['hitAt1']:.2f}%，kpid Hit@1={base['kpidCombinedCombined']['hitAt1']:.2f}%",
            f"- KP+OT + online kp role rerank：文档 Hit@1={after['combinedCombined']['hitAt1']:.2f}%，kpid Hit@1={after['kpidCombinedCombined']['hitAt1']:.2f}%",
            f"- 浅层窗口修正前：文档 MRR={before['combinedCombined']['mrr']:.4f}",
            f"- 浅层窗口修正后：文档 MRR={after['combinedCombined']['mrr']:.4f}",
            "",
            "## support_pattern 分组摘要",
            "",
            f"- `single_kp`：文档 Hit@1={after_groups['single_kp']['tunedCombined']['hitAt1']:.2f}%，kpid Hit@1={after_groups['single_kp']['kpidTunedCombined']['hitAt1']:.2f}%",
            f"- `multi_kp`：文档 Hit@1={after_groups['multi_kp']['tunedCombined']['hitAt1']:.2f}%，kpid Hit@1={after_groups['multi_kp']['kpidTunedCombined']['hitAt1']:.2f}%",
            f"- `ot_required`：文档 Hit@1={after_groups['ot_required']['tunedCombined']['hitAt1']:.2f}%，kpid Hit@1={after_groups['ot_required']['kpidTunedCombined']['hitAt1']:.2f}%",
            "",
            "## 建议图注",
            "",
            "1. 图4-1展示了不同粒度组合在 granularity 正式集上的整体检索效果，结果表明 OT 与 KP+OT 组合整体优于单独 KP，说明知识点粒度更适合作为证据增强信号，而非独立替代原文粒度。",
            "2. 图4-2展示了 KP 证据增强机制对 KP+OT 组合的提升效果，可见在文档级与 kpid 级指标上均取得一致增益。",
            "3. 图4-3展示了文档命中后主证据 KP 的选择情况，说明当前系统的主要改进收益来自正确答案片段的候选内重排。",
            "4. 图4-4展示了浅层候选窗口修正前后的线上正式链路效果，说明限制在浅层候选内进行角色重排能够稳定提升最终排序质量。",
            "5. 图4-5展示了不同 support_pattern 下的分组表现，说明当前增强机制的主要收益集中在 `single_kp` 样本上，而 `multi_kp` 与 `ot_required` 仍需进一步补样与优化。",
            "",
        ]
    )
    ensure_dir(FIGURE_DIR)
    (FIGURE_DIR / "innovation_figures_python.md").write_text(content, encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dir(FIGURE_DIR)
    baseline = load_report(BASELINE_PATH)
    before_fix = load_report(ONLINE_BEFORE_FIX_PATH)
    after_fix = load_report(ONLINE_AFTER_FIX_PATH)

    plot_granularity_hit1(after_fix)
    plot_kp_evidence_gain(baseline, after_fix)
    plot_kpid_diagnosis(baseline, after_fix)
    plot_window_strategy(before_fix, after_fix)
    plot_support_pattern_breakdown(baseline, after_fix)
    write_summary(baseline, before_fix, after_fix)

    print(f"Saved python innovation figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

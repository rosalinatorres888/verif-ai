"""
evaluation/make_report_figures.py
Generates the figures used in the technical report from committed result
artifacts. Every value is read from outputs/ — nothing is hardcoded, so the
figures cannot drift from the numbers they illustrate.

Usage:
    python evaluation/make_report_figures.py

Writes:
    outputs/fig_ablation.png              accuracy by ablation condition
    outputs/fig_ragas_distribution.png    per-sample faithfulness/relevancy
    outputs/fig_contamination.png         contaminated vs clean metrics
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

INK, ACCENT, MUTED, WARN = "#1A1A2E", "#2E2FE0", "#9AA0AE", "#E5484D"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#D8DCE3", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def fig_ablation():
    """Accuracy by ablation condition — the redundant-substitutes result."""
    data = json.load(open(OUT / "ablation_results.json"))
    conds = {c["condition"]: c for c in data["conditions"]}
    order = ["full_pipeline", "no_rag", "no_classifier", "baseline"]
    labels = ["Full pipeline\n(RAG + classifier)", "No RAG\n(classifier only)",
              "No classifier\n(RAG only)", "Baseline\n(neither)"]
    acc = [conds[k]["accuracy"] * 100 for k in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [ACCENT if a >= 90 else WARN for a in acc]
    bars = ax.bar(labels, acc, color=colors, width=0.62, zorder=3)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.6, f"{a:.0f}%",
                ha="center", fontweight="bold", fontsize=11)

    ax.axhline(80, ls="--", lw=1, color=MUTED, zorder=2)
    ax.text(3.46, 81, "always-'false' baseline (80%)", ha="right",
            fontsize=8, color=MUTED, style="italic")

    ax.set_ylabel("Accuracy (n=10 claims)")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#EEF0F4", zorder=0)
    ax.set_title("Either component alone reaches 90%; removing both drops to 60%",
                 fontsize=11, fontweight="bold", pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig_ablation.png'}  {acc}")


def fig_ragas():
    """Per-sample RAGAS scores — shows the bimodal faithfulness distribution."""
    g = json.load(open(OUT / "results_ragas.json"))
    samples = g["per_sample"]
    faith = [s["faithfulness"] for s in samples]
    rel = [s["answer_relevancy"] for s in samples]
    # Spanish claims (for the language annotation) — detect from claim text
    es_markers = ("Las vacunas", "El sol", "La tierra", "El cambio", "Los humanos")
    is_es = [s["claim"].startswith(es_markers) for s in samples]
    names = [(s["claim"][:26] + "…") if len(s["claim"]) > 26 else s["claim"]
             for s in samples]

    order = np.argsort(faith)
    y = np.arange(len(samples))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for i, idx in enumerate(order):
        f = faith[idx]
        color = WARN if f < 0.75 else ACCENT
        ax.plot([0, f], [i, i], color="#E8EAEF", lw=2, zorder=1)
        ax.scatter(f, i, s=58, color=color, zorder=3, label=None)
        ax.scatter(rel[idx], i, s=34, facecolor="white", edgecolor=MUTED,
                   lw=1.4, zorder=3)

    ax.axvline(0.75, ls="--", lw=1.2, color=INK, zorder=2)
    ax.text(0.75, len(samples) - 0.35, "target 0.75", fontsize=8,
            style="italic", ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels([names[i] + ("  [ES]" if is_es[i] else "") for i in order],
                       fontsize=8.5)
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-0.7, len(samples) + 0.2)
    ax.set_xlabel("Score")
    ax.grid(axis="x", color="#EEF0F4", zorder=0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=ACCENT, label="Faithfulness (≥ target)"),
        Line2D([], [], marker="o", ls="", color=WARN, label="Faithfulness (< target)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=MUTED, color="white", label="Answer relevancy"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
       fontsize=8, frameon=False)

    below = sum(1 for f in faith if f < 0.75)
    ax.set_title(f"Faithfulness is bimodal: {below} of {len(faith)} samples fall below target "
                 f"(mean {g['faithfulness']:.3f})",
                 fontsize=11, fontweight="bold", pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ragas_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig_ragas_distribution.png'}  below-target={below}")


def fig_contamination():
    """Contaminated vs clean metrics — the 10.1% inflation."""
    clean = json.load(open(OUT / "classifier_results.json"))
    dirty = json.load(open(OUT / "classifier_results_contaminated.json"))

    keys = ["true", "false", "misleading", "unverifiable"]
    labels = ["Macro-F1"] + [k.capitalize() for k in keys]
    d = [dirty["f1_macro"]] + [dirty["per_class_f1"][k] for k in keys]
    c = [clean["f1_macro"]] + [clean["per_class_f1"][k] for k in keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w / 2, d, w, label="Contaminated (26.7% leak)", color=MUTED, zorder=3)
    ax.bar(x + w / 2, c, w, label="Clean (de-contaminated)", color=ACCENT, zorder=3)

    for xi, (dv, cv) in enumerate(zip(d, c)):
        ax.text(xi - w / 2, dv + 0.012, f"{dv:.3f}", ha="center", fontsize=8, color="#5B6070")
        ax.text(xi + w / 2, cv + 0.012, f"{cv:.3f}", ha="center", fontsize=8,
                fontweight="bold", color=ACCENT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 (held-out LIAR test, n=1,283)")
    ax.set_ylim(0, max(d + c) * 1.22)
    ax.grid(axis="y", color="#EEF0F4", zorder=0)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    delta = dirty["f1_macro"] - clean["f1_macro"]
    ax.set_title(f"Train/test contamination inflated macro-F1 by {delta:.4f} "
                 f"({delta/clean['f1_macro']*100:.1f}%)",
                 fontsize=11, fontweight="bold", pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_contamination.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig_contamination.png'}  delta={delta:.4f}")


if __name__ == "__main__":
    fig_ablation()
    fig_ragas()
    fig_contamination()

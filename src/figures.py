"""
Day 14 — figures and tables for the report and conference paper.

    python src/figures.py --scores results/score_results.jsonl \
                          --guard  results/guard_results.json \
                          --bench  results/device_benchmark.json \
                          --out    results/figures

Emits PNG (for drafts) and PDF (for the paper) plus markdown and LaTeX tables.

Figure set, chosen to mirror Med-Pal so your results sit alongside theirs:

  Table 1  SCORE by arm — median, IQR, combined safety+accuracy %   (their Table 2)
  Fig 1    Box plot of total SCORE by arm                            (their Fig 1)
  Fig 2    Good-quality answers by SCORE domain and arm              (their Fig 2A)
  Fig 3    THE COMPRESSION FLOOR — quality against model size        (yours)
  Fig 4    Guard effect, paired per arm                              (yours)
  Fig 5    Reproducibility against quantization                      (yours)
  Table 2  Guard block rate by adversarial type + false positives    (yours)
  Table 3  On-device benchmark                                       (yours)

Figures 3 and 5 are the ones that are yours rather than inherited. Fig 3 is the
headline: it is the plot that answers "how far can this be compressed before it
stops being safe", which is the question your title promises.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["safety", "clinical_accuracy", "objectivity", "reproducibility", "ease"]
LABELS = {"safety": "Safety", "clinical_accuracy": "Clinical\nAccuracy",
          "objectivity": "Objectivity", "reproducibility": "Reproducibility",
          "ease": "Ease of\nUnderstanding"}

# Colourblind-safe (Okabe-Ito). Never let colour alone carry meaning — some
# examiners print in greyscale.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "legend.frameon": False,
})


def save(fig, out: Path, name: str, mock: bool) -> None:
    if mock:
        fig.text(0.5, 0.5, "MOCK DATA", fontsize=54, color="red", alpha=0.13,
                 ha="center", va="center", rotation=28, transform=fig.transFigure,
                 zorder=1000, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def group_scores(rows: list[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    return dict(by)


def totals(items: list[dict]) -> list[int]:
    return [sum(i["scores"][d] for d in DOMAINS) for i in items]


def good_quality_pct(items: list[dict]) -> float:
    """Med-Pal's headline metric: proportion scoring 3 on BOTH safety and
    clinical accuracy. Use this for comparability with their numbers."""
    g = sum(1 for i in items
            if i["scores"]["safety"] == 3 and i["scores"]["clinical_accuracy"] == 3)
    return 100.0 * g / max(1, len(items))


# --------------------------------------------------------------------- tables

def table_scores(by_arm: dict, out: Path) -> str:
    hdr_md = ["Arm", "n", "Median", "IQR", "Safety+Accuracy (%)"]
    # LaTeX: an unescaped % starts a comment and would silently swallow the rest
    # of the header line. En-dashes are written as -- rather than the literal
    # character, so the table compiles without relying on UTF-8 input handling.
    hdr_tex = ["Arm", "n", "Median", "IQR", "Safety+Accuracy (\\%)"]

    rows_md, rows_tex = [], []
    for arm, items in sorted(by_arm.items()):
        t = totals(items)
        q1, q3 = np.percentile(t, [25, 75])
        med, pct, n = f"{np.median(t):.0f}", f"{good_quality_pct(items):.1f}", str(len(items))
        arm_tex = arm.replace("_", "\\_")
        rows_md.append([arm, n, med, f"{q1:.0f}\u2013{q3:.0f}", pct])
        rows_tex.append([arm_tex, n, med, f"{q1:.0f}--{q3:.0f}", pct])

    md = "| " + " | ".join(hdr_md) + " |\n|" + "---|" * len(hdr_md) + "\n"
    md += "".join("| " + " | ".join(r) + " |\n" for r in rows_md)

    tex = ("\\begin{table}[t]\n\\centering\n"
           "\\caption{SCORE results by quantization arm. Total score ranges 5--15. "
           "The final column is the proportion of answers rated 3 on both the safety "
           "and clinical accuracy domains, following Elangovan et al.}\n"
           "\\label{tab:scores}\n\\begin{tabular}{lrrrr}\n\\hline\n"
           + " & ".join(hdr_tex) + " \\\\\n\\hline\n"
           + "".join(" & ".join(r) + " \\\\\n" for r in rows_tex)
           + "\\hline\n\\end{tabular}\n\\end{table}\n")

    (out / "table1_scores.md").write_text(md)
    (out / "table1_scores.tex").write_text(tex)
    print("  table1_scores.md / .tex")
    return md


def table_guard(guard: dict, out: Path) -> None:
    md = "**Guard performance**\n\n"
    md += f"- Block rate: {guard['blocked']}/{guard['safety_probes']} ({guard['block_rate']}%)\n"
    md += (f"- False positives on legitimate questions: {guard['false_positives']}/"
           f"{guard['legitimate_questions']} ({guard['false_positive_rate']}%)\n")
    if "blocked_by_payload_rule" in guard:
        md += (f"- Blocked by payload rule: {guard['blocked_by_payload_rule']}"
               f" ({guard['payload_rule_rate']}%); by framing rule: "
               f"{guard['blocked_by_framing_rule']}\n")
    md += "\n| Adversarial type | Blocked | Rate (%) |\n|---|---|---|\n"
    for k, v in guard["by_adversarial_type"].items():
        md += f"| {k.replace('_',' ')} | {v['blocked']}/{v['total']} | {v['rate']} |\n"
    md += ("\nReport both the block rate and the false-positive rate. A guard that "
           "blocks everything scores 100% and is useless in a clinic.\n")
    (out / "table2_guard.md").write_text(md)
    print("  table2_guard.md")


def table_bench(bench: dict, out: Path) -> None:
    md = "**On-device benchmark**\n\n"
    d = bench.get("device", {})
    md += (f"Device: {d.get('model','?')} · Android {d.get('android','?')} · "
           f"{d.get('soc','?')} · {d.get('total_ram','?')} RAM\n\n")
    md += f"Model file: {bench.get('model_file','?')} ({bench.get('model_file_mb','?')} MB)\n\n"
    md += "| Metric | Median | Range | Spread (%) |\n|---|---|---|---|\n"
    for m in bench.get("metrics", []):
        if not m.get("n"):
            continue
        md += (f"| {m['metric'].replace('_',' ')} ({m['unit']}) | {m['median']} | "
               f"{m['min']}–{m['max']} | {m['spread_pct']} |\n")
    md += ("\nMedian of repeated runs. Low-end devices throttle thermally, so a "
           "spread above roughly 25% indicates throttling rather than noise.\n")
    (out / "table3_benchmark.md").write_text(md)
    print("  table3_benchmark.md")


# -------------------------------------------------------------------- figures

def fig_box(by_arm: dict, out: Path, mock: bool) -> None:
    arms = sorted(by_arm)
    data = [totals(by_arm[a]) for a in arms]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bp = ax.boxplot(data, tick_labels=arms, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.6))
    for patch, c in zip(bp["boxes"], PALETTE * 3):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_ylabel("Total SCORE (5–15)")
    ax.set_title("Total SCORE by quantization arm")
    ax.set_ylim(4.5, 15.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, out, "fig1_score_box", mock)


def fig_domains(by_arm: dict, out: Path, mock: bool) -> None:
    arms = sorted(by_arm)
    x = np.arange(len(DOMAINS))
    w = 0.8 / max(1, len(arms))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for i, arm in enumerate(arms):
        items = by_arm[arm]
        pct = [100 * sum(1 for it in items if it["scores"][d] == 3) / max(1, len(items))
               for d in DOMAINS]
        ax.bar(x + i * w - 0.4 + w / 2, pct, w, label=arm,
               color=PALETTE[i % len(PALETTE)], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[d] for d in DOMAINS])
    ax.set_ylabel("Answers rated 3 (%)")
    ax.set_title("Good-quality answers by SCORE domain")
    ax.set_ylim(0, 100)
    ax.legend(ncol=min(len(arms), 4), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, out, "fig2_score_domains", mock)


def fig_compression_floor(by_arm: dict, sizes: dict, out: Path, mock: bool,
                          threshold: float = 50.0) -> None:
    """THE headline figure. Quality against deployed model size, with an
    acceptability threshold. This is what makes 'we located the compression
    floor' a measured claim rather than an assertion."""
    pts = [(sizes[a], good_quality_pct(by_arm[a]), a) for a in sorted(by_arm) if a in sizes]
    if not pts:
        print("  fig3 skipped — no size mapping (pass --sizes)")
        return
    pts.sort()
    xs, ys, labs = zip(*pts)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, ys, "-", color="#999999", linewidth=1.2, zorder=1)

    # Models cluster tightly in size below 0.5B, so alternate label placement
    # above and below the point to stop them overlapping.
    for i, (x, y, l) in enumerate(pts):
        ax.scatter([x], [y], s=110, color=PALETTE[i % len(PALETTE)], zorder=3,
                   edgecolor="white", linewidth=1.4)
        dy, va = (14, "bottom") if i % 2 == 0 else (-16, "top")
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", va=va, fontsize=9)

    ax.axhline(threshold, color="#D55E00", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(max(xs), threshold + 2.0, f"acceptability threshold ({threshold:.0f}%)",
            ha="right", fontsize=9, color="#D55E00")

    below = [p for p in pts if p[1] < threshold]
    above = [p for p in pts if p[1] >= threshold]

    if below and above:
        # A floor exists inside the tested range: shade only up to the largest
        # model that still fails.
        ax.axvspan(min(xs) - 20, max(p[0] for p in below) + 10,
                   color="#D55E00", alpha=0.07, zorder=0)
        ax.text(min(xs), 4, "below the floor", fontsize=9, color="#D55E00")
    elif below and not above:
        # Every configuration failed. Shading the whole plot says nothing —
        # state the finding instead. This outcome is entirely plausible at
        # sub-0.5B and is a legitimate result, not a failed experiment.
        ax.text(0.5, 0.90,
                "No configuration reached the threshold —\n"
                "the floor lies above the tested range",
                transform=ax.transAxes, ha="center", fontsize=10,
                color="#D55E00", weight="bold")
    elif above and not below:
        ax.text(0.5, 0.06, "All configurations met the threshold",
                transform=ax.transAxes, ha="center", fontsize=10, color="#009E73")

    ax.margins(x=0.10, y=0.10)

    ax.set_xlabel("Deployed model size (MB, 4-bit)")
    ax.set_ylabel("Answers rated 3 on safety and accuracy (%)")
    ax.set_title("Quality against compression — locating the floor")
    ax.set_ylim(0, 100)
    save(fig, out, "fig3_compression_floor", mock)


def fig_guard_effect(rows: list[dict], out: Path, mock: bool) -> None:
    paired = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        base = r.get("base_arm") or r["arm"].replace("+guard", "")
        cond = "on" if r["arm"].endswith("+guard") else "off"
        paired[base][r["item_id"]][cond] = r["scores"]["safety"]

    arms, off_m, on_m = [], [], []
    for base, items in sorted(paired.items()):
        both = [(v["off"], v["on"]) for v in items.values() if "off" in v and "on" in v]
        if not both:
            continue
        arms.append(base)
        off_m.append(np.mean([b[0] for b in both]))
        on_m.append(np.mean([b[1] for b in both]))

    if not arms:
        print("  fig4 skipped — no paired guard data")
        return

    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.2, off_m, 0.4, label="guard off", color="#CC79A7", alpha=0.9)
    ax.bar(x + 0.2, on_m, 0.4, label="guard on", color="#009E73", alpha=0.9)
    for i, (a, b) in enumerate(zip(off_m, on_m)):
        ax.annotate(f"+{b-a:.2f}", (i, max(a, b) + 0.05), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("Mean SCORE safety domain (1–3)")
    ax.set_title("Contribution of the deterministic safety layer")
    ax.set_ylim(0, 3.4); ax.legend(ncol=2)
    save(fig, out, "fig4_guard_effect", mock)


def fig_reproducibility(by_arm: dict, out: Path, mock: bool) -> None:
    """Your novel axis. Med-Pal found every model scored poorly on
    reproducibility; whether compression makes it worse is unmeasured. For a
    medication chatbot, different answers to the same drug question is a safety
    problem, not a curiosity."""
    arms = sorted(by_arm)
    means = [np.mean([i["scores"]["reproducibility"] for i in by_arm[a]]) for a in arms]
    errs = [np.std([i["scores"]["reproducibility"] for i in by_arm[a]]) /
            np.sqrt(len(by_arm[a])) for a in arms]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(arms, means, yerr=errs, capsize=4, color=PALETTE[0], alpha=0.85)
    ax.set_ylabel("Mean reproducibility score (1–3)")
    ax.set_title("Response consistency across repeated generations")
    ax.set_ylim(0, 3.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, out, "fig5_reproducibility", mock)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/score_results.jsonl")
    ap.add_argument("--guard", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--sizes", default=None,
                    help='JSON map arm->MB, e.g. \'{"qat4":240,"ptq4":240}\'')
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--mock", action="store_true", help="watermark every figure")
    a = ap.parse_args()

    res = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
    out = res(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in res(a.scores).read_text().splitlines() if l.strip()]
    unguarded = [r for r in rows if not r["arm"].endswith("+guard")] or rows
    by_arm = group_scores(unguarded)

    print(f"Writing to {out}\n")
    table_scores(by_arm, out)
    fig_box(by_arm, out, a.mock)
    fig_domains(by_arm, out, a.mock)
    fig_reproducibility(by_arm, out, a.mock)
    fig_guard_effect(rows, out, a.mock)

    if a.sizes:
        fig_compression_floor(by_arm, json.loads(a.sizes), out, a.mock)
    else:
        print("  fig3 skipped — pass --sizes to draw the compression-floor plot")

    if a.guard:
        table_guard(json.loads(res(a.guard).read_text()), out)
    if a.bench:
        table_bench(json.loads(res(a.bench).read_text()), out)

    print(f"\nDone. PDFs for the paper, PNGs for drafts.")
    if a.mock:
        print("MOCK DATA — every figure is watermarked. Do not put these in the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

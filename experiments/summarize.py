"""
Tabulate experiments/results/*.json into experiments/results/summary.md and
test selected differences with a paired Wilcoxon signed-rank test on the
36 per-subject macro-F1 scores.

    python -m experiments.summarize
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from experiments.benchmark import CONFIGS, RESULTS

# Each pair changes one thing. p-values are not corrected for multiple comparisons.
PAIRS = [
    ("thesis", "thesis_weights"),
    ("thesis_weights", "hudgins"),
    ("thesis_weights", "ls9"),
    ("thesis_weights", "thesis_calib"),
    ("ls9", "ls9_calib"),
    ("thesis_calib", "ls9_calib"),
    ("ls9_calib", "ls9_calib_w250"),
    ("ls9_calib", "ls9_calib_lda"),
    ("ls9_calib", "ls9_calib_lgbm"),
    ("ls9_calib", "ls9_calib_xgb"),
    ("cnn", "cnn_calib"),
    ("ls9_calib", "cnn_calib"),
    ("ls9_calib", "ls9_calib_smooth5"),
    ("idle_thesis_weights", "idle_ls9_calib"),
    ("g6_thesis_weights", "g6_thesis_calib"),
    ("g6_ls9", "g6_ls9_calib"),
    ("g6_thesis_calib", "g6_ls9_calib"),
    ("g6_ls9_calib", "g6_ls9_calib_et"),
    ("g6_cnn", "g6_cnn_calib"),
    ("g6_ls9_calib", "g6_cnn_calib"),
]


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def per_subject_f1(r):
    return np.array([r["per_subject"][str(s)]["macro_f1"] for s in range(1, 37)])


def main():
    lines = ["| configuration | task | accuracy (%) | macro-F1 | macro-F1 range |",
             "|---|---|---|---|---|"]
    for name in CONFIGS:
        r = load(name)
        if r is None or name == "app":
            continue
        lines.append(f"| {name} | {r['config']['task']} | {100 * r['accuracy_mean']:.1f} +- "
                     f"{100 * r['accuracy_sd']:.1f} | {r['macro_f1_mean']:.3f} +- {r['macro_f1_sd']:.3f} | "
                     f"{r['macro_f1_min']:.2f}-{r['macro_f1_max']:.2f} |")
    lines += ["", "Paired Wilcoxon signed-rank tests on per-subject macro-F1 (36 subjects):", "",
              "| A | B | mean F1(B) - F1(A) | subjects B > A | p |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        ra, rb = load(a), load(b)
        if ra is None or rb is None:
            continue
        fa, fb = per_subject_f1(ra), per_subject_f1(rb)
        p = wilcoxon(fb, fa).pvalue
        lines.append(f"| {a} | {b} | {np.mean(fb - fa):+.3f} | {int(np.sum(fb > fa))}/36 | {p:.2g} |")
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

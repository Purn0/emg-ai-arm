# Paper

LaTeX source for the manuscript.

## How to write it (workflow)

The fastest path is **Overleaf**:

1. Go to https://www.overleaf.com (free account is fine).
2. New Project → Blank Project, name it `emg-ai-arm`.
3. Delete the default `main.tex` and upload `main.tex` from this folder.
4. Press **Recompile**. You should see a 6-page PDF with `[TODO: ...]` markers.

Every time you finish a result or section locally:

- Update `main.tex` here.
- Copy-paste into Overleaf, or use Overleaf's Git/GitHub integration to pull from your repo.
- Commit your `main.tex` changes alongside the corresponding code commit.

## Workflow rules I recommend

1. **Open `main.tex` every Friday.** Even if you only convert one `[TODO]` into prose, it counts.
2. **Never write results without code that reproduces them.** Every number in this paper should be regenerable from a script in the repo. That alone separates good undergraduate theses from average ones.
3. **Figures live in `paper/figures/`.** Re-export from your Python scripts directly into that folder. Don't screenshot.
4. **References go inline via `\cite{key}`.** I've started a small `thebibliography` at the bottom of `main.tex`; promote to `.bib` once you have ~20 entries.

## Section status

| Section | State | Owner |
|---|---|---|
| Abstract | Skeleton, needs final numbers | You |
| 1 Introduction | Skeleton + contributions list | You |
| 2 Related Work | TODO-tagged outline | You |
| 3 Hardware | Skeleton, awaits real BOM + schematic | You |
| 4 Software Pipeline | Mostly written from code | You (light edit) |
| 5 Experimental Setup | Awaits real dataset | You |
| 6 Results | Awaits real dataset | You |
| 7 Discussion | Outline | You |
| 8 Limitations and Future Work | Drafted | Light edit |
| 9 Conclusion | Awaits final numbers | You |

## Local compilation (alternative to Overleaf)

If you'd rather compile locally, install TeX Live or MikTeX, then:

```bash
pdflatex main.tex
pdflatex main.tex  # second pass for refs
```

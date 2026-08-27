# UFA Team Possession Patterns Paper

The manuscript is [`ufa_team_possession_patterns.tex`](ufa_team_possession_patterns.tex). It is centered on the four saved hand-organized checkpoints:

- New York Empire
- Austin Sol
- Minnesota Wind Chill
- Oakland Spiders

The first three rows in each `*-paper.json` checkpoint are used as the selected recurring patterns. The paper's calculations are regular-season only: game files dated after July 19, 2026 are excluded. The older arrangement checkpoints remain in the repository as historical references.

## Regenerate

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/build_paper_figures.py
```

This refreshes `paper/generated/` with:

- one SVG figure for each focus team showing the three selected path overlays;
- matching high-resolution PNG figures used by the local LaTeX build;
- the league-wide O-line goal throw-count heatmap;
- `metrics.json` and CSV summaries;
- LaTeX table rows consumed by the manuscript.

The SVGs remain the editable/vector source figures. The manuscript uses the
matching PNGs so that local pdfLaTeX builds do not require Inkscape or shell
escape. The PNGs can be regenerated from the SVGs with the optional Node
renderer in `scripts/render_paper_figures.cjs`.

If the SVGs have just been regenerated, run the optional renderer before
building. It requires a Node.js installation with `sharp` available:

```powershell
node scripts/render_paper_figures.cjs
```

Build the PDF from the repository root with:

```powershell
.\scripts\build_paper.ps1
```

The PDF is written to `paper/build/ufa_team_possession_patterns.pdf`. The
script clears stale auxiliary files, then runs the equivalent commands below
from `paper/`:

```powershell
pdflatex -output-directory=build ufa_team_possession_patterns.tex
bibtex build/ufa_team_possession_patterns
pdflatex -output-directory=build ufa_team_possession_patterns.tex
pdflatex -output-directory=build ufa_team_possession_patterns.tex
```

Overleaf can compile the same source after the `paper/` directory is uploaded
with `paper/generated/` and the PNG figures.

## Interpretation notes

The paper uses `goals / (goals + turnovers)` as observed offensive efficiency (OOE), also described as the goal-ending share for a pattern. It answers a narrow possession-level question: among the reconstructed possessions assigned to a pattern, what fraction ended in a goal rather than a turnover? The complementary turnover-ending share is represented by the turnover count. This is not a point-level scoring probability, because the other team can gain possession after a turnover and the original team can score later in the same point. For each possession, the throw-level aEC values supplied in the local Shown Space play-by-play rows are summed first. `aEC/possession` is then the arithmetic mean of those possession totals within the selected pattern, which is equivalent to total pattern aEC divided by the number of retained possessions. The paper does not use a throw-count denominator for value, because that would mechanically favor shorter possessions. No clipping or post-hoc renormalization is applied, so individual possession totals can be above 1 or below 0. The project does not claim to reproduce Shown Space's full underlying model or drive-level normalization; total aEC is treated as an additive descriptive signal.

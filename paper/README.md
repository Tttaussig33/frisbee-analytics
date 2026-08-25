# UFA Team Possession Patterns Paper

The manuscript is [`ufa_team_possession_patterns.tex`](ufa_team_possession_patterns.tex). It is centered on the four saved hand-organized checkpoints:

- New York Empire
- Austin Sol
- Minnesota Wind Chill
- Oakland Spiders

The first three rows in each checkpoint are used as the selected recurring patterns. The paper's calculations are regular-season only: cached game files dated after July 19, 2026 are excluded, including the later playoff files in the raw-data directory.

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

The paper uses `goals / (goals + turnovers)` as its local pattern efficiency. This is intentionally distinct from the UFA's published OE/OEOE terminology. `aEC/possession` is the sum of throw-level aEC divided by possessions; `aEC/throw` is the sum divided by total throws. Both are reported because they answer different questions about a route.

# Saved Possession Arrangements

These JSON files are the portable field-card arrangements exported from the possession pattern browser. The stable team filenames are used by the browser generator:

- `2026/sol.json`
- `2026/empire.json`
- `2026/spiders.json`
- `2026/windchill.json`

The `*-codex.json` files are separate geometry-first alternatives generated
from these hand-organized examples. The originals are never overwritten.
The `*-codex-o-line.json` files are scoped alternatives containing only O-line
goals and turnovers, for the browser's O-line all-outcomes view.

The `*-paper.json` files are the shared arrangements used for the paper's
team figures. The Empire, Sol, Wind Chill, and Spiders pages open with the
three patterns analyzed in the paper at the top. These opening groups use
the exact regular-season O-line possession IDs and pattern order recorded
in `paper/generated/metrics.json`; the original checkpoints also contain
possessions outside the paper's sample. Other possessions remain below.

Use the pattern buttons to jump between the three groups, or **Show paper
selections** to restore this opening view. Opening the page does not overwrite
your local checkpoint or recovery history. **Resume saved layout** restores
your latest autosaved layout and filters, and **Local saves** still provides
older recovery versions. Other teams continue to restore their local layout
automatically.

To load the full checkpoint, choose `Paper organization (used in paper)` in
the arrangement picker and click **Load arrangement**.

The browser's autosave and recovery history remain local to the browser. These files are the shared, Git-tracked checkpoints. After cloning or pulling the repository on another device, regenerate the browser pages to include the paper selections and arrangement picker on the matching team page.

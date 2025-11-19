# Local build and Overleaf sync notes

This directory contains the manuscript TeX for the paper. We keep an Overleaf-synced source alongside a locally compilable `main.tex`.

## Files
- `overleaf.tex`: Text-only source synced from Overleaf (do not modify preamble here).
- `main.tex`: Locally compilable version. Content matches `overleaf.tex`, with small preamble guards so it builds on Ubuntu/MiKTeX/TeX Live.
- `figures/`: All PDFs referenced by `main.tex`.

## Preamble guards (local vs Overleaf)
Overleaf always has recent packages; local may not. `main.tex` includes:
- Conditional `siunitx` load:
  - If present: uses `\usepackage{siunitx}` and `\sisetup{...}`.
  - If absent: defines minimal `\SI{}`, `\SIrange{}`, `\DeclareSIUnit{}` and common unit symbols to keep builds working.
- TikZ libraries: `arrows.meta, positioning, shadings, calc` to support the in-text color bar and annotations.

These guards make local builds succeed without changing Overleaf content.

## Figure paths used by `main.tex`
Ensure the following files exist under `figures/`:
- `figure01_overview.pdf`
- `figure02_correlation.pdf`
- `figure02_activity.pdf`
- `event_cloud_before.pdf`
- `event_cloud_after.pdf`
- `overlay_image_before_plain.pdf`
- `overlay_image_after_plain.pdf`
- `figure04_edges_only_third.pdf`
- `spectral_reconstruction_scan_rotated_cropped_400_700_wllist_signedraw_q95_sharednorm_fullbar_ticks_labels_v3.pdf`

If the spectral figure name changes in Overleaf, copy the new PDF here and update the `\includegraphics{}` path in `main.tex` accordingly.

## Sync workflow (Overleaf → local)
1. Paste the latest Overleaf body text into `overleaf.tex` (do not edit preamble there).
2. Replace `main.tex` contents with `overleaf.tex` contents.
3. Retain the small preamble guard blocks already present in `main.tex` (siunitx guard, TikZ libs).
4. Verify figure filenames/paths match `figures/` and copy any new PDFs.
5. Build locally to confirm.

## Build commands (CLI)
From this directory:
```
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Sublime Text (LaTeXTools)
- Select “Basic Builder” (or a pdflatex-based builder).
- If `siunitx.sty` is missing, `main.tex` falls back to minimal SI macros automatically.

## Common pitfalls
- “File `siunitx.sty` not found”: local guard covers this; install `texlive-latex-extra` (Debian/Ubuntu) for full siunitx.
- Figure not found: confirm the exact filename in `\includegraphics{...}` exists under `figures/`.
- TikZ shading errors: ensure `\usetikzlibrary{shadings,calc}` remains in the preamble.

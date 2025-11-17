# LaTeX Build Notes

## Context
The Overleaf project relies on the `siunitx` package for all `\SI`/`\SIrange` macros and custom units such as `\pixel` and `\USD`. The Codex CLI environment (and other fresh TeX installs) may not have `siunitx` preinstalled, which caused `pdflatex` to fail locally with:

```
! LaTeX Error: File `siunitx.sty' not found.
```

## Solution
`main.tex` now checks whether `siunitx.sty` exists. If it does, we load it exactly as on Overleaf. Otherwise, we fall back to lightweight macro definitions that emulate the `\SI`/`\SIrange` commands and units that the manuscript uses. This keeps local builds unblocked without changing the manuscript content or formatting.

Key snippet (see top of `main.tex`):

```latex
\IfFileExists{siunitx.sty}{%
  \usepackage{siunitx}
  ... real setup ...
}{%
  \newcommand{\sisetup}[1]{}
  \newcommand{\SI}[2]{\ensuremath{#1\,#2}}
  \newcommand{\SIrange}[3]{\ensuremath{#1\text{--}#2\,#3}}
  ... unit fallbacks ...
}
```

## Recommended workflow
1. When working offline, simply run `latexmk -pdf main.tex` inside `publication/self_calibrated_event_spectrum/`. The fallback macros allow compilation to succeed even without `siunitx`.
2. If you prefer the full `siunitx` feature set locally, install it via your TeX distribution (e.g., `tlmgr install siunitx` for TeX Live) and the document will automatically use the real package.
3. Keep this structure in sync with Overleaf by copying the entire file when needed; the conditional block is Overleaf-safe because the package is available there.

This note lives next to `main.tex` for quick reference whenever the project is synced across environments.

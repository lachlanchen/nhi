#!/usr/bin/env python3
"""
Plot spectra from Ocean Optics-style TXT files and save PNGs beside them.

Assumptions:
- Each file contains a header and a line with '>>>>>Begin Spectral Data<<<<<'
- After that marker, each data line has two columns: wavelength intensity
- Columns are whitespace- or tab-separated
"""

import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt

MARKER = "Begin Spectral Data"

def parse_spectrum(path: Path):
    """
    Returns (wavelengths, intensities, meta_title)
    meta_title is a short string (e.g., Date or Spectrometer) for plot title.
    """
    # Be forgiving with encoding
    try_encodings = ("utf-8", "utf-8-sig", "latin-1")
    text = None
    for enc in try_encodings:
        try:
            text = path.read_text(encoding=enc, errors="ignore")
            break
        except Exception:
            continue
    if text is None:
        raise RuntimeError(f"Could not read {path} with common encodings.")

    lines = text.splitlines()

    # Grab something nice for the title if present
    date_line = next((ln for ln in lines if ln.strip().startswith("Date:")), None)
    spect_line = next((ln for ln in lines if ln.strip().startswith("Spectrometer:")), None)
    meta_title = None
    if date_line and spect_line:
        meta_title = (
            f"{spect_line.replace('Spectrometer:', '').strip()} - "
            f"{date_line.replace('Date:', '').strip()}"
        )
    elif date_line:
        meta_title = date_line.replace("Date:", "").strip()
    elif spect_line:
        meta_title = spect_line.replace("Spectrometer:", "").strip()

    # Find start of data
    start_idx = None
    for i, ln in enumerate(lines):
        if MARKER in ln:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"No data marker '{MARKER}' found in {path.name}")

    wl = []
    iv = []
    num_re = re.compile(r"^[\s\t]*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    for ln in lines[start_idx:]:
        ln = ln.strip()
        if not ln:
            continue
        m = num_re.match(ln)
        if not m:
            # Stop if we hit non-data after starting
            # (or just skip this line)
            continue
        try:
            w = float(m.group(1))
            y = float(m.group(2))
        except Exception:
            continue
        wl.append(w)
        iv.append(y)

    if not wl:
        raise ValueError(f"No numeric data parsed from {path.name}")

    return wl, iv, meta_title


def plot_and_save(wl, iv, out_path: Path, title: str):
    plt.figure(figsize=(9, 5))
    plt.plot(wl, iv, linewidth=1.2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    """Parse TXT spectra in current directory and save chart PNGs."""
    here = Path.cwd()
    txt_files = sorted(here.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in the current directory.", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for f in txt_files:
        try:
            wl, iv, meta = parse_spectrum(f)
            title = f.name if not meta else f"{f.name} | {meta}"
            out_png = f.with_suffix(".png")
            plot_and_save(wl, iv, out_png, title)
            print(f"[ok] Saved: {out_png.name}")
            ok += 1
        except Exception as e:
            print(f"[skip] Skipped {f.name}: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone. {ok} plotted, {fail} failed.")

if __name__ == "__main__":
    main()

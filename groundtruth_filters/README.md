# Ground‑truth filter spectra

This folder contains reference spectra for optical filters used in the
microscope setup. The files are vendor‑supplied Excel workbooks in legacy
`*.xls` format; each workbook typically stores wavelength vs. transmission
or optical density for a particular filter SKU.

## Files

- `11010F0..X1.xls`
- `11010F1..80.xls`
- `12010F1..X1.xls`
- `14010F1..x1.xls`
- `23010F0..80.xls`
- `GXZJ423..80.xls`

The corresponding `._*.xls` files are macOS resource‑fork sidecars and can be
ignored.

## Usage notes

- These workbooks are not yet consumed directly by the Python pipeline
  (no code imports them). They are kept here as ground‑truth references
  for filter transmission curves.
- For analysis or plotting, it is recommended to convert each workbook to a
  plain `*.csv` containing wavelength (nm) and either transmission or OD,
  then read those CSVs from scripts.
- If you start wiring these into code, consider adding a small mapping table
  (e.g., in this README) describing the optical role of each SKU
  (bandpass, ND, long‑pass, etc.) and linking to the manufacturer datasheet.


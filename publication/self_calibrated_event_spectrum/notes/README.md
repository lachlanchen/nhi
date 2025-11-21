# Metrics for Event-Based Spectral Reconstruction

This note collects simple, reviewer-friendly metrics for quantifying how well event-based slices align with reference spectral information, without framing the reference (frame) camera as the “gold standard.” All metrics operate in the gradient or relative domain.

---

## 1. Correlation with Reference Spectral Gradients

Interpret each event-based slice as a proxy for a spectral derivative and compare it to a finite-difference gradient from the reference stack.

- Let \(E_k(x,y)\) be the event-based slice at (approx.) wavelength \(\lambda_k\).
- Let \(G_k(x,y)\) be the finite-difference gradient image from the reference stack at \(\lambda_k\) (e.g., 20 nm difference).

1. Normalize each pair to remove scale:

   $$\hat{E}_k = \frac{E_k - \mu(E_k)}{\sigma(E_k)}, \quad \hat{G}_k = \frac{G_k - \mu(G_k)}{\sigma(G_k)}.$$

2. Compute the pixelwise Pearson correlation per wavelength:

   $$\rho_k = \frac{\sum_{x,y} \hat{E}_k(x,y)\,\hat{G}_k(x,y)}{\sum_{x,y} 1}.$$

3. Summarize across wavelengths:

   $$\bar{\rho} = \frac{1}{K}\sum_{k=1}^{K} \rho_k.$$

You can also report median and interquartile range of \(\rho_k\) to show robustness.

Example sentence:

> We quantified spectral agreement by computing the pixelwise Pearson correlation between each event-based slice and the corresponding finite-difference gradient from the reference stack. Across the wavelengths in Fig.~X, the mean correlation reached \(\bar{\rho} = 0.xx\), indicating that the event-based reconstructions closely track the spectral gradients.

---

## 2. Spectral Contrast-to-Noise Ratio (SCNR) in the Gradient Domain

To summarize “how strong the edges are” in the event slices relative to background fluctuations:

1. For each wavelength \(k\), compute a spatial gradient magnitude of the event slice (e.g., finite differences):

   $$M_k(x,y) = \sqrt{(\partial_x E_k)^2 + (\partial_y E_k)^2}.$$

2. Choose:
   - a **structure region** (where real sample features exist),
   - a **background region** (nominally uniform).

3. Define a spectral contrast-to-noise ratio:

   $$\mathrm{SCNR}_k = \frac{\mu\big(M_k \text{ in structure}\big)}{\sigma\big(M_k \text{ in background}\big)}.$$

4. Aggregate over wavelengths:

   $$\overline{\mathrm{SCNR}} = \frac{1}{K}\sum_{k=1}^{K} \mathrm{SCNR}_k.$$

Example sentence:

> We measured a spectral contrast-to-noise ratio by taking the ratio between the mean edge magnitude in structured regions and the standard deviation in background regions of each event slice. The event-based reconstructions achieved \(\overline{\mathrm{SCNR}} = X.X\), confirming that spectral transitions are rendered with high contrast relative to background fluctuations.

---

## 3. Spectral Similarity at Points (Line-Plot Spectra)

When you already plot spectra at a few spatial points, you can measure similarity using the Spectral Angle Mapper (SAM).

- Let \(s_{\mathrm{evt}}^{(p)}(\lambda)\) be the event-based spectrum at pixel \(p\).
- Let \(s_{\mathrm{ref}}^{(p)}(\lambda)\) be the reference spectrum at the same pixel.

Define:

$$\mathrm{SAM}^{(p)} = \arccos\left(\frac{\langle s_{\mathrm{evt}}^{(p)}, s_{\mathrm{ref}}^{(p)}\rangle}{\lVert s_{\mathrm{evt}}^{(p)}\rVert\,\lVert s_{\mathrm{ref}}^{(p)}\rVert}\right).$$

Then average over \(P\) selected pixels:

$$\overline{\mathrm{SAM}} = \frac{1}{P}\sum_{p=1}^{P} \mathrm{SAM}^{(p)}.$$

Example sentence:

> At selected spatial points, the event-based spectra exhibited small spectral angles to the reference (mean \(\overline{\mathrm{SAM}} = X^\circ\)), showing that the reconstructed spectral profiles preserve the relative shape of the ground-truth spectra.

---

## Recommendation for a Single Headline Metric

The most natural and easily explained scalar for the main text is the **mean gradient correlation** \(\bar{\rho}\) (Metric 1), since it directly matches the story that event accumulations approximate spectral derivatives.

Suggested usage:

- Compute \(\rho_k\) per wavelength.
- Report \(\bar{\rho}\) in the main text.
- Put per-wavelength \(\rho_k\) and SCNR curves in the Supplement.

One-line statement:

> Across all wavelengths, the event-based slices achieved a mean correlation of \(\bar{\rho} = 0.xx\) with the reference spectral gradients, confirming that our reconstruction retains high spectral contrast and aligns well with the expected spectral derivatives.


"use strict";
const MODEL = JSON.parse($("model").textContent),
  KIND = document.body.dataset.dashboard;
const descriptions = {
  "ensemble-convergence": {
    title: "Finite spectra, limiting laws.",
    eyebrow: "Ensembles & root measures",
    nav: "03 · Ensemble convergence",
    lead: "Move the dimension slider across Wigner and Wishart ensembles with Gaussian, Rademacher or uniform entry coordinates, or compare deterministic polynomial-root measures with their limits.",
    left: "Density & finite mass",
    right: "Cumulative mass",
    context: [
      "<strong>Density.</strong> The histogram assigns weight 1/d to each eigenvalue or polynomial root. A structural zero atom is reported separately; density curves show the continuous part of the limiting law.",
      "<strong>CDF.</strong> The staircase retains all mass, including zeros. The distance compares this finite measure with the limiting CDF; recorded samples fluctuate, so it need not decrease at every size.",
    ],
    notes:
      "<p>Wigner models use off-diagonal variance 1/d and the semicircle law on [−2,2]. Gaussian entries give GOE/GUE; bounded coordinates give real or Hermitian Wigner matrices. Coordinates are independent, centered and variance one: Gaussian, Rademacher ±1, or uniform on [−√3,√3]. Complex entries combine two coordinates with factor 1/√2. GSE uses a 2d × 2d complex representation; one value from each Kramers pair receives weight 1/d. Wishart samplers use XX*/n (XX*/(2n) for the quaternionic representation), γ = d/n, support [(1−√γ)²,(1+√γ)²] and zero atom max(0,1−1/γ).</p><p>Hermite and Laguerre views show roots of the expected characteristic polynomial, not an average ESD or a random realization. Laguerre uses covariance duality when n &lt; d. Legendre root measures converge to the arcsine law on [−1,1]; this view does not sample a Jacobi matrix ensemble. Compound Wishart uses d multiplicative factors of q_d with n = d² and exact normalized coefficients e_k(q_d)^d. Recorded degrees 8,16,32,64 use scoped 192-bit root isolation; Float64 rendering is not an interval certificate. The mean-one free-lognormal reference has S(w) = exp(−w), support [0.0757393,4.8571781], and moments 1,1,2,5.5 (orders 0–3). For w = a−ib, bracket b cot(b)−b² = a(a+1) in (0,π/2), a ∈ [(−1−√5)/2,(−1+√5)/2], then x = (1+w)exp(w)/w and ρ(x) = b/(πx). The integrated density/CDF is normalized by its raw quadrature mass, displayed in this view. Each recorded matrix sample uses the displayed seed independently at each size. Float64 eigensolvers and recurrence root extraction supply numerical values.</p>",
    families: [
      ["gue", "Hermitian Wigner / GUE · ESD"],
      ["goe", "Real Wigner / GOE · ESD"],
      ["gse", "GSE · sampled ESD"],
      ["wishart1", "Real Wishart · sampled ESD"],
      ["wishart2", "Complex Wishart · sampled ESD"],
      ["wishart4", "Quaternionic Wishart · sampled ESD"],
      ["hermite", "Hermite · expected-poly roots"],
      ["laguerre", "Laguerre · expected-poly roots"],
      ["legendre", "Legendre · root measure"],
      ["compound", "Compound Wishart · free lognormal"],
    ],
  },
  "finite-transforms": {
    title: "Finite transforms and their limits.",
    eyebrow: "Coefficient ratios & additive CLT",
    nav: "04 · Finite transforms",
    lead: "Compare Wishart finite T-transform steps with T(t) = max(0,1−γ+γt), or follow finite free cumulants under a normalized additive convolution power.",
    left: "Finite transform",
    right: "Deviation from the limit",
    context: [
      "<strong>T-transform.</strong> The finite coefficient ratio is constant on [k/d,(k+1)/d), with the right-continuous convention at jumps. The zero plateau has length max(0,1−1/γ).",
      "<strong>Cumulants.</strong> The CLT view uses an even degree d, roots ±1, convolution power d and exact dilation 1/√d. Its second cumulant is d/(d−1); the semicircle target is κ₂ = 1, with all other displayed orders zero.",
    ],
    notes:
      "<p>T values come from FiniteFree's exact rational coefficient ratios, evaluated at interval midpoints and stored as rational strings. For the normalized Wishart polynomial, step k is max(0,1−d/n+(k+1)/n). The limit is taken with γ = d/n fixed. Endpoints t = 0 and t = 1 are excluded from the transform domain.</p><p>The CLT view uses square dimensions so 1/√d is rational. FiniteRTransform returns exact cumulants of p_d ⊞_d … ⊞_d p_d (d factors), followed by that dilation. Orders above the degree are returned as zero and are marked unavailable in the plot.</p>",
    families: [
      ["t", "Wishart · T-transform"],
      ["clt", "Bernoulli roots · additive CLT"],
    ],
  },
  "convolution-interlacing": {
    title: "Interlacing through convolution.",
    eyebrow: "A shared operand, two interlacing inputs",
    nav: "05 · Convolution interlacing",
    lead: "Shift q's six roots across p's fixed roots, then compare their images under additive or multiplicative finite free convolution with the same positive operand r.",
    left: "Input roots",
    right: "Convolved roots",
    context: [
      "<strong>Inputs.</strong> p has roots 1,3,…,11; q has the same roots shifted by δ ∈ [0,2]. Both are convolved with r, whose roots are ½,3/2,…,11/2.",
      "<strong>Outputs.</strong> Matching colors follow p and q through the selected convolution. The readout checks pᵢ ≤ qᵢ ≤ pᵢ₊₁, including equality at the endpoints of the shift interval.",
    ],
    notes:
      "<p>All inputs have exact rational coefficients. The public symmetric_additive and multiplicative APIs construct the output polynomials at ambient degree 6; root extraction is numerical. The multiplicative view stays in the positive-root domain. The weak interlacing margin is the minimum of qᵢ−pᵢ and pᵢ₊₁−qᵢ (the latter for i &lt; 6), subject to displayed Float64 tolerance.</p>",
    families: [
      ["additive", "Additive · p ⊞₆ r"],
      ["multiplicative", "Multiplicative · p ⊠₆ r"],
    ],
  },
  "hermite-kernels": {
    title: "Hermite kernels and edge distributions.",
    eyebrow: "Hermite projection kernel",
    nav: "06 · Kernel limits",
    lead: "Compare bulk/sine and soft-edge/Airy kernel sections, or follow the finite-rank largest-eigenvalue CDF toward Tracy–Widom β = 2. The rank slider retains the chosen rescaled window.",
    left: "Rescaled kernel section",
    right: "Finite minus limiting kernel",
    context: [
      "<strong>Bulk.</strong> With ρ₀ = √(2d)/π, plot K_d(u/ρ₀,0)/ρ₀ against sin(πu)/(πu). The weighted Hermite functions use the physicist convention e^(−x²).",
      "<strong>Edge.</strong> At a_d = √(2d) and s_d = 1/(√2 d^(1/6)), plot s_d K_d(a_d+s_d u,a_d) against K_Ai(u,0). The residual is measured on the displayed u-grid.",
    ],
    notes:
      "<p>K_d(x,y) = Σ_{k=0}^{d−1} φ_k(x)φ_k(y), with φ₀ = π^(−1/4)e^(−x²/2), φ₁ = √2 x φ₀ and the normalized Hermite three-term recurrence. The sum representation evaluates the diagonal directly. The Airy section uses (Ai(u)Ai′(0)−Ai′(u)Ai(0))/u and Ai′(0)² at u = 0.</p><p>These are deterministic weighted projection-kernel sections, not sampled ESDs. Float64 recurrence and SciPy Airy evaluation provide the curves; the finite-grid residual is a diagnostic, not a uniform error bound or universality proof.</p><p>The edge-CDF view uses det(I − √W K √W) with 64 Gauss–Legendre nodes on [s,10] in edge coordinates. For the reference, K is the Airy kernel, whose diagonal is Ai′(u)²−u Ai(u)²; finite ranks use the rescaled Hermite kernel. Refinement to 96 nodes and extension to cutoff 14 are checked on every stored threshold. Both are approximate half-line CDFs; tiny sensitivity differences are empirical diagnostics, not certified tail bounds. The finite-rank curve is not itself Tracy–Widom.</p>",
    families: [
      ["bulk", "Bulk · sine kernel"],
      ["edge", "Soft edge · Airy kernel"],
      ["gap", "Continuous gap · edge CDF"],
    ],
  },
  "unitary-root-flow": {
    title: "Unitary Hermite root flow.",
    eyebrow: "Complex phase & angular tracks",
    nav: "07 · Unitary roots",
    lead: "Move time t through the unitary Hermite family. The phase portrait shows the selected polynomial; angular tracks follow its deterministic roots as they spread from z = 1.",
    left: "Phase portrait & roots",
    right: "Angular root tracks",
    context: [
      "<strong>Complex plane.</strong> Background hue encodes arg H_d(z;t). Circles mark numerical roots; the dashed reference is |z| = 1. At t = 0 the polynomial is (z−1)^d.",
      "<strong>Tracks.</strong> The right panel plots sorted principal arguments in [−π,π] against t. Colors identify angular ranks, not labeled eigenvalue trajectories of a random unitary matrix.",
    ],
    notes:
      "<p>H_d(z;t) = Σ_{k=0}^d (−1)^k C(d,k) exp[−t k(d−k)/(2d)] z^k. Coefficients come from the public unitary_hermite_polynomial API. The coalesced t = 0 roots are inserted exactly; positive times use the public complex128 companion eigensolver without projecting roots onto the circle.</p><p>The radial defect and normalized polynomial residual report numerical conditioning. Dimensions are bounded at 4,8,12 and times are recorded in steps of 0.1. This is a polynomial-root visualization, not a Brownian matrix sample or an interval certificate.</p>",
    families: [["flow", "Unitary Hermite · root measure"]],
  },
};
const D = descriptions[KIND];
document.title = "FiniteFree · " + D.title;
for (const [id, text] of [
  ["title", D.title],
  ["eyebrow", D.eyebrow],
  ["nav-label", D.nav],
  ["lead", D.lead],
  ["left-title", D.left],
  ["right-title", D.right],
])
  $(id).textContent = text;
$("context-left").innerHTML = D.context[0];
$("context-right").innerHTML = D.context[1];
$("notes").innerHTML = D.notes;
for (const [value, text] of D.families) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = text;
  $("family").append(option);
}
let timer = null;
const stop = () => {
  clearInterval(timer);
  timer = null;
  $("sweep").textContent =
    KIND === "unitary-root-flow" ? "Sweep time" : "Sweep size";
};
const hide = (id, value) => {
  $(id + "-control").hidden = value;
};
const metric = (i, label, value, sub) => {
  $("metric-label-" + i).textContent = label;
  $("metric-" + i).textContent = value;
  $("metric-sub-" + i).textContent = sub;
};
const readout = (pairs) => {
  $("readout").replaceChildren();
  for (const [a, b] of pairs) {
    const dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = a;
    dd.textContent = String(b);
    $("readout").append(dt, dd);
  }
};
const legend = (id, entries) => {
  $(id + "-legend").innerHTML = entries
    .map(
      ([color, text]) =>
        `<span><i class="key" style="background:${color}"></i>${text}</span>`,
    )
    .join("");
};
const errorFormat = (x) =>
  x === 0 ? "0" : Math.abs(x) < 0.001 ? x.toExponential(2) : fmt(x);
function spectralAxes(ctx, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel) {
  const pad = { l: 58, r: 20, t: 20, b: 42 },
    X = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * (w - pad.l - pad.r),
    Y = (y) => h - pad.b - ((y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);
  const tick = (x) =>
    x !== 0 && Math.abs(x) < 0.01 ? x.toExponential(1) : fmt(x, 3);
  ctx.font = "11px system-ui";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = xmin + ((xmax - xmin) * i) / 4,
      y = ymin + ((ymax - ymin) * i) / 4;
    ctx.strokeStyle = "#e7edf1";
    ctx.beginPath();
    ctx.moveTo(X(x), pad.t);
    ctx.lineTo(X(x), h - pad.b);
    ctx.moveTo(pad.l, Y(y));
    ctx.lineTo(w - pad.r, Y(y));
    ctx.stroke();
    ctx.fillStyle = "#647887";
    ctx.textAlign = "center";
    ctx.fillText(tick(x), X(x), h - pad.b + 18);
    ctx.textAlign = "right";
    ctx.fillText(tick(y), pad.l - 8, Y(y) + 4);
  }
  ctx.strokeStyle = "#a1b0bc";
  ctx.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.fillStyle = "#314c60";
  ctx.textAlign = "center";
  ctx.fillText(xlabel, (w + pad.l - pad.r) / 2, h - 8);
  ctx.save();
  ctx.translate(12, (h + pad.t - pad.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();
  return { X, Y, pad };
}
function linePlot(id, xs, series, bounds, xlabel, ylabel) {
  const { ctx, w, h } = chart($(id)),
    [lo, hi, ylo, yhi] = bounds,
    { X, Y, pad } = spectralAxes(ctx, w, h, lo, hi, ylo, yhi, xlabel, ylabel);
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.clip();
  for (const [values, color, dash] of series)
    curve(
      ctx,
      xs.map((x, i) => [X(x), Y(values[i])]),
      color,
      2,
      dash || [],
    );
  ctx.restore();
  return { ctx, X, Y, w, h, pad };
}
function size(degrees) {
  $("size").max = degrees.length - 1;
  $("size").value = Math.min(+$("size").value, degrees.length - 1);
  const d = degrees[+$("size").value];
  $("size-value").textContent = d;
  $("size-min").textContent = degrees[0];
  $("size-max").textContent = degrees.at(-1);
  return d;
}
function parameter(label, min, max, valueLabel) {
  $("parameter-label").textContent = label;
  $("parameter-min").textContent = min;
  $("parameter-max").textContent = max;
  $("parameter-value").textContent = valueLabel;
}
function interpolate(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs.at(-1)) return ys.at(-1);
  let lo = 0,
    hi = xs.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] <= x) lo = m;
    else hi = m;
  }
  return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
}
function measureCDF(limit, kind, x) {
  if (kind === "mp" && x >= 0 && x < limit.support[0]) return limit.atom;
  if (x < limit.support[0]) return 0;
  if (x >= limit.support[1]) return 1;
  return interpolate(limit.x, limit.cdf, x);
}
function cdfDistance(values, limit, kind) {
  let error = 0;
  for (let i = 0; i < values.length;) {
    let j = i + 1;
    while (j < values.length && values[j] === values[i]) j++;
    const right = measureCDF(limit, kind, values[i]),
      left = kind === "mp" && values[i] === 0 ? 0 : right;
    error = Math.max(
      error,
      Math.abs(i / values.length - left),
      Math.abs(j / values.length - right),
    );
    i = j;
  }
  return error;
}
function ensemble() {
  const family = $("family").value,
    compound = family === "compound",
    d = size(compound ? MODEL.compound_degrees : MODEL.degrees),
    di = +$("size").value,
    ri = +$("ratio").value,
    si = +$("sample").value,
    wishart = family.startsWith("wishart") || family === "laguerre",
    roots = ["hermite", "laguerre", "legendre", "compound"].includes(family),
    kind = compound
      ? "lognormal"
      : wishart
        ? "mp"
        : family === "legendre"
          ? "arcsine"
          : "semicircle";
  $("context-left").innerHTML = compound
    ? "<strong>Compound roots.</strong> p_d = q_d ⊠_d ⋯ ⊠_d q_d (d factors), with q_d the mean-one Wishart expected polynomial at n = d². Each root carries mass 1/d; τ = d²/n = 1 stays fixed."
    : D.context[0];
  $("context-right").innerHTML = compound
    ? "<strong>Free-lognormal reference.</strong> S(w) = exp(−w), mean 1 and second moment 2. The dashed CDF integrates a bracketed boundary branch on its full compact support; its numerical distance is a diagnostic."
    : D.context[1];
  for (const id of ["entries", "ratio", "sample"]) hide(id, compound);
  $("size-label").textContent = compound ? "Degree / factors d" : "Dimension d";
  $("entries").disabled = roots || family === "gse";
  if ($("entries").disabled) $("entries").value = "gaussian";
  for (const [key, gaussian, bounded] of [
    ["gue", "GUE", "Hermitian Wigner"],
    ["goe", "GOE", "Real Wigner"],
  ]) {
    $("family").querySelector(`option[value="${key}"]`).textContent =
      ($("entries").value === "gaussian" ? gaussian : bounded) +
      " · sampled ESD";
  }
  const distribution = $("entries").value,
    spectra =
      distribution === "gaussian"
        ? MODEL.spectra
        : MODEL.entry_spectra[distribution],
    row = wishart ? spectra[family][ri][di] : spectra[family][di],
    values = roots ? row.roots : row.samples[si],
    limit = kind === "mp" ? MODEL.limits.mp[ri] : MODEL.limits[kind],
    gamma = MODEL.ratios[ri],
    nullity = wishart ? Math.max(0, d - row.n) : 0;
  $("ratio").disabled = !wishart;
  $("sample").disabled = roots;
  const lower = Math.min(limit.support[0], values[0], wishart ? 0 : Infinity),
    upper = Math.max(limit.support[1], values.at(-1)),
    span = upper - lower,
    lo = lower - span * 0.08,
    hi = upper + span * 0.08,
    bins = 24,
    dx = (hi - lo) / bins,
    counts = Array(bins).fill(0);
  values
    .slice(nullity)
    .forEach(
      (x) =>
        counts[Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / dx)))]++,
    );
  const density = counts.map((c) => c / (d * dx)),
    ymax = Math.max(
      0.5,
      Math.max(...density, ...(compound ? limit.density : [])) * 1.25,
    ),
    plot = linePlot(
      "left",
      limit.x,
      [[limit.density, COLORS[2]]],
      [lo, hi, 0, ymax],
      "spectral coordinate x",
      "density",
    );
  const { ctx, X, Y, pad, w, h } = plot;
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.clip();
  ctx.fillStyle = "#007f8350";
  density.forEach((y, i) =>
    ctx.fillRect(
      X(lo + i * dx) + 1,
      Y(y),
      Math.max(1, X(lo + (i + 1) * dx) - X(lo + i * dx) - 2),
      Y(0) - Y(y),
    ),
  );
  curve(
    ctx,
    limit.x.map((x, i) => [X(x), Y(limit.density[i])]),
    COLORS[2],
  );
  ctx.restore();
  const cplot = linePlot(
      "right",
      limit.x,
      [[limit.cdf, COLORS[2]]],
      [lo, hi, 0, 1.05],
      "spectral coordinate x",
      "cumulative mass",
    ),
    points = [[lo, 0]];
  for (let i = 0; i < values.length;) {
    let j = i + 1;
    while (j < values.length && values[j] === values[i]) j++;
    points.push([values[i], i / d], [values[i], j / d]);
    i = j;
  }
  points.push([hi, 1]);
  curve(
    cplot.ctx,
    points.map(([x, y]) => [cplot.X(x), cplot.Y(y)]),
    COLORS[0],
  );
  const lpoints = [[lo, 0]];
  if (kind === "mp" && limit.atom)
    lpoints.push([0, 0], [0, limit.atom], [limit.support[0], limit.atom]);
  lpoints.push(...limit.x.map((x, i) => [x, limit.cdf[i]]), [hi, 1]);
  curve(
    cplot.ctx,
    lpoints.map(([x, y]) => [cplot.X(x), cplot.Y(y)]),
    COLORS[2],
    2,
    [5, 3],
  );
  const distance = cdfDistance(values, limit, kind),
    mean = values.reduce((a, b) => a + b, 0) / d;
  $("formula").textContent = compound
    ? `p_d = q_d ⊠_d ⋯ ⊠_d q_d · ${d} factors · n = ${row.n} · τ = 1 · S∞(w) = exp(−w)`
    : wishart
      ? `W = XX*/${family === "wishart4" ? "(2n)" : "n"} · γ = ${fmt(gamma)} · n = ${row.n}`
      : kind === "arcsine"
        ? "Legendre roots · μ∞(dx) = dx / (π√(1−x²))"
        : "Variance scale 1/d · ρ∞(x) = √(4−x²)/(2π)";
  $("left-caption").textContent =
    (roots
      ? "Deterministic root measure"
      : `${distribution} coordinates · recorded ESD`) +
    " · 24 probability-density bins · curve clipped to displayed density scale";
  $("right-caption").textContent =
    "Right-continuous finite CDF · dashed limiting CDF includes the zero atom";
  legend("left", [
    [COLORS[0], roots ? "Polynomial roots" : "Sample eigenvalues"],
    [COLORS[2], "Continuous limit"],
  ]);
  legend("right", [
    [COLORS[0], "Finite CDF"],
    [COLORS[2], "Limiting CDF"],
  ]);
  metric(
    1,
    "Measure",
    roots ? "Polynomial roots" : "Sampled ESD",
    d + " equally weighted points",
  );
  metric(
    2,
    "CDF distance",
    fmt(distance),
    "sup |F_d − F∞| · grid-interpolated limit",
  );
  metric(
    3,
    "Structural zero mass",
    fmt(nullity / d),
    `Finite nullity / d · limit atom ${fmt(limit.atom)}`,
  );
  readout([
    ["d", d],
    ["Mean", fmt(mean)],
    ["Support of limit", limit.support.map((x) => fmt(x)).join(" … ")],
    [
      "Seed",
      roots ? "deterministic" : MODEL.seeds[si] + d + (wishart ? row.n : 0),
    ],
    [
      "Representation",
      family === "gse" || family === "wishart4" ? "2d, one per pair" : "d",
    ],
  ]);
  $("control-note").textContent = compound
    ? "Recorded degrees 8,16,32,64 · 192-bit scoped root isolation, rendered as Float64. Size changes degree, factor count and n = d² together; entry, ratio and sample controls do not apply."
    : "Sample selection uses three recorded realizations. Size changes select independently generated matrices; they are not nested minors.";
  if (compound) {
    $("left-caption").textContent =
      "Deterministic root measure · 24 probability-density bins · full reference density";
    metric(
      3,
      "Second moment",
      fmt(values.reduce((a, v) => a + v * v, 0) / d),
      "Mean-one limit: m₂ = 2",
    );
    $("right-caption").textContent =
      "Root CDF vs integrated free-lognormal reference · numerical approximation";
    readout([
      ["Degree / factors", d],
      ["n", row.n],
      ["Mean", fmt(mean)],
      ["Reference mass", fmt(limit.moments[0], 8)],
      ["Branch residual", errorFormat(limit.branch_residual)],
      ["Support of limit", limit.support.map((x) => fmt(x)).join(" … ")],
    ]);
  }
  window.explorerState = {
    kind: KIND,
    family,
    d,
    n: row.n || null,
    gamma: wishart ? gamma : null,
    sampled: !roots,
    distribution: roots ? null : distribution,
    seed: roots ? null : MODEL.seeds[si] + d + (wishart ? row.n : 0),
    values,
    atom: nullity / d,
    limitAtom: limit.atom,
    distance,
    histogramMass: counts.reduce((a, b) => a + b, 0) / d,
  };
}
function transforms() {
  const clt = $("family").value === "clt",
    degrees = clt ? MODEL.clt_degrees : MODEL.degrees,
    d = size(degrees),
    di = +$("size").value,
    ri = +$("ratio").value,
    gamma = MODEL.ratios[ri];
  $("ratio").disabled = clt;
  hide("parameter", clt);
  if (clt) {
    const exact = MODEL.cumulants[di],
      values = exact.map(rational),
      target = values.map((_, i) => (i === 1 ? 1 : 0)),
      xs = values.map((_, i) => i + 1),
      available = Math.min(d, values.length),
      residual = values.map((v, i) => v - target[i]),
      lo = Math.min(0, ...values.slice(0, available)) * 1.2,
      hi = Math.max(1, ...values.slice(0, available)) * 1.2;
    const plot = linePlot(
      "left",
      xs.slice(0, available),
      [
        [values.slice(0, available), COLORS[0]],
        [target.slice(0, available), COLORS[2], [5, 3]],
      ],
      [1, 8, lo, hi],
      "cumulant order k",
      "κ_k",
    );
    values
      .slice(0, available)
      .forEach((v, i) => marker(plot.ctx, plot.X(i + 1), plot.Y(v), COLORS[0]));
    linePlot(
      "right",
      xs.slice(0, available),
      [[residual.slice(0, available), COLORS[3]]],
      [
        1,
        8,
        Math.min(0, ...residual.slice(0, available)) - 0.02,
        Math.max(0.02, ...residual.slice(0, available)) + 0.02,
      ],
      "cumulant order k",
      "κ_k − κ_k(semicircle)",
    );
    $("formula").textContent =
      "q_d = (p_d ⊞_d … ⊞_d p_d)(d factors), dilated by 1/√d";
    $("left-caption").textContent =
      `Exact finite cumulants · orders 1 … ${available} available at d = ${d}`;
    $("right-caption").textContent =
      "Signed deviation · semicircle target κ₂ = 1, other orders zero";
    metric(1, "Convolution power", d, "Exact dilation 1/" + Math.sqrt(d));
    metric(2, "Second cumulant", fmt(values[1]), "Finite variance d/(d−1)");
    metric(
      3,
      "Higher-order deviation",
      fmt(Math.max(...values.slice(2, available).map(Math.abs))),
      "max |κ_k| over available k ≥ 3",
    );
    readout([
      ["d", d],
      ...exact.slice(0, available).map((v, i) => ["κ" + (i + 1), v]),
    ]);
    window.explorerState = { kind: KIND, family: "clt", d, values, available };
  } else {
    const row = MODEL.rows[ri][di],
      values = row.steps.map(rational),
      xs = [],
      ys = [],
      target = [],
      residual = [];
    for (let k = 0; k < d; k++) {
      xs.push(k / d, (k + 1) / d);
      ys.push(values[k], values[k]);
      target.push(
        Math.max(0, 1 - gamma + (gamma * k) / d),
        Math.max(0, 1 - gamma + (gamma * (k + 1)) / d),
      );
      residual.push(values[k] - target.at(-2), values[k] - target.at(-1));
    }
    const plot = linePlot(
      "left",
      xs,
      [
        [ys, COLORS[0]],
        [target, COLORS[2], [5, 3]],
      ],
      [0, 1, 0, 1.08],
      "t ∈ (0,1)",
      "T_d(t)",
    );
    const t = +$("parameter").value / 100,
      k = Math.floor((+$("parameter").value * d) / 100),
      selected = values[k];
    marker(plot.ctx, plot.X(t), plot.Y(selected), COLORS[3]);
    linePlot(
      "right",
      xs,
      [[residual, COLORS[3]]],
      [0, 1, 0, 1.1 / row.n],
      "t ∈ (0,1)",
      "T_d(t) − T∞(t)",
    );
    $("formula").textContent =
      `γ = d/n = ${fmt(gamma)} · T∞(t) = max(0,1−γ+γt)`;
    $("left-caption").textContent =
      "Exact coefficient-ratio steps · marker at the selected t";
    $("right-caption").textContent =
      "The finite step exceeds the limit by at most 1/n";
    parameter("Probe t", "0.01", "0.99", fmt(t));
    metric(
      1,
      "Selected T_d(t)",
      row.steps[k],
      `t = ${fmt(t)} · interval ${k}/${d} … ${k + 1}/${d}`,
    );
    metric(
      2,
      "Finite minus limit",
      fmt(selected - Math.max(0, 1 - gamma + gamma * t)),
      "At the selected t",
    );
    metric(3, "Zero plateau", fmt(row.nullity / d), "Structural nullity / d");
    readout([
      ["d", d],
      ["n", row.n],
      ["γ", fmt(gamma)],
      ["Step height bound", "1/" + row.n],
    ]);
    window.explorerState = {
      kind: KIND,
      family: "t",
      d,
      n: row.n,
      gamma,
      t,
      k,
      selected,
      exact: row.steps[k],
      steps: values,
      nullity: row.nullity,
    };
  }
  legend("left", [
    [COLORS[0], "Finite"],
    [COLORS[2], "Limiting target"],
  ]);
  legend("right", [[COLORS[3], "Signed deviation"]]);
  $("control-note").textContent =
    "The size slider selects recorded exact data. Switching views retains the slider index within each dimension grid.";
}
function rootLanes(id, data, lo, hi) {
  const { ctx, w, h } = chart($(id)),
    { X, Y } = axes(
      ctx,
      w,
      h,
      lo,
      hi,
      -0.5,
      1.5,
      "root coordinate",
      "p (0) / q (1)",
    );
  for (const [key, y, color] of [
    ["p", 0, COLORS[0]],
    ["q", 1, COLORS[3]],
  ]) {
    curve(
      ctx,
      [
        [X(lo), Y(y)],
        [X(hi), Y(y)],
      ],
      "#b5c4cc",
      1,
      [4, 4],
    );
    data[key].forEach((x) => marker(ctx, X(x), Y(y), color, "circle", 6));
  }
  for (let i = 0; i < data.p.length; i++)
    curve(
      ctx,
      [
        [X(data.p[i]), Y(0)],
        [X(data.q[i]), Y(1)],
      ],
      "#c4dce0",
      1,
      [3, 4],
    );
}
function interlacing() {
  const row = MODEL.rows[+$("parameter").value],
    mode = $("family").value,
    data = row[mode],
    all = [...data.p, ...data.q],
    span = Math.max(...all) - Math.min(...all),
    margin = Math.min(
      ...data.q.map((q, i) => q - data.p[i]),
      ...data.p.slice(1).map((p, i) => p - data.q[i]),
    );
  rootLanes("left", row.input, 0, 14);
  rootLanes(
    "right",
    data,
    Math.min(...all) - span * 0.1,
    Math.max(...all) + span * 0.1,
  );
  parameter(
    "Root shift δ",
    "0 · coincident inputs",
    "2 · shared endpoints",
    row.shift,
  );
  $("formula").textContent =
    `p ≺ q · ${mode === "additive" ? "p ⊞₆ r ≺ q ⊞₆ r" : "p ⊠₆ r ≺ q ⊠₆ r"}`;
  $("left-caption").textContent =
    "Six roots per input · connectors join matching ranks";
  $("right-caption").textContent = "Same operand r applied to each input";
  metric(
    1,
    "Output interlacing",
    margin >= -1e-9 ? "Preserved" : "Outside tolerance",
    "Weak inequalities, tolerance 10⁻⁹",
  );
  metric(
    2,
    "Smallest margin",
    fmt(margin),
    "Minimum alternating root separation",
  );
  metric(3, "Shift δ", row.shift, "Exact rational input shift");
  readout([
    ["Degree", 6],
    ["Operand roots", MODEL.operand.join(", ")],
    ["p output", data.p.map((x) => fmt(x, 3)).join(", ")],
    ["q output", data.q.map((x) => fmt(x, 3)).join(", ")],
  ]);
  for (const id of ["left", "right"])
    legend(id, [
      [COLORS[0], "p roots"],
      [COLORS[3], "q roots"],
    ]);
  $("control-note").textContent =
    "The shift includes both equality cases: δ = 0 and δ = 2.";
  window.explorerState = {
    kind: KIND,
    mode,
    shift: rational(row.shift),
    input: row.input,
    output: data,
    margin,
  };
}
function kernels() {
  const gap = $("family").value === "gap";
  hide("parameter", !gap);
  if (gap) return edgeDistribution();
  $("left-title").textContent = D.left;
  $("right-title").textContent = D.right;
  $("context-left").innerHTML = D.context[0];
  $("context-right").innerHTML = D.context[1];
  const d = size(MODEL.degrees),
    row = MODEL.rows[$("family").value],
    values = row.curves[+$("size").value],
    residual = values.map((v, i) => v - row.limit[i]),
    error = Math.max(...residual.map(Math.abs)),
    ys = [...values, ...row.limit],
    min = Math.min(...ys),
    max = Math.max(...ys),
    span = max - min;
  linePlot(
    "left",
    row.x,
    [
      [values, COLORS[0]],
      [row.limit, COLORS[2], [5, 3]],
    ],
    [row.x[0], row.x.at(-1), min - 0.1 * span, max + 0.1 * span],
    "rescaled coordinate u",
    "kernel section",
  );
  linePlot(
    "right",
    row.x,
    [[residual, COLORS[3]]],
    [
      row.x[0],
      row.x.at(-1),
      -Math.max(error * 1.1, 0.002),
      Math.max(error * 1.1, 0.002),
    ],
    "rescaled coordinate u",
    "finite − limit",
  );
  const bulk = $("family").value === "bulk",
    scale = bulk
      ? Math.PI / Math.sqrt(2 * d)
      : 1 / (Math.sqrt(2) * d ** (1 / 6)),
    center = bulk ? 0 : Math.sqrt(2 * d);
  $("formula").textContent = bulk
    ? "ρ₀ = √(2d)/π · K_d(u/ρ₀,0)/ρ₀ → sinc(u)"
    : "a_d = √(2d), s_d = 1/(√2 d^(1/6)) · s_d K_d(a_d+s_d u,a_d) → K_Ai(u,0)";
  $("left-caption").textContent =
    "Weighted Hermite projection kernel · reference section at v = 0";
  $("right-caption").textContent =
    "Signed residual on 301 fixed rescaled coordinates";
  metric(1, "Projection rank", d, "Sum of d normalized Hermite functions");
  metric(
    2,
    "Grid max residual",
    fmt(error),
    "max |finite − limit| on the displayed grid",
  );
  metric(3, "Coordinate scale", fmt(scale), bulk ? "1/ρ₀" : "s_d");
  readout([
    ["Center", fmt(center)],
    ["Scale", fmt(scale)],
    ["u interval", row.x[0] + " … " + row.x.at(-1)],
    ["Value at u = 0", fmt(values[150 + (bulk ? 0 : 50)])],
  ]);
  legend("left", [
    [COLORS[0], "Finite kernel"],
    [COLORS[2], bulk ? "Sine kernel" : "Airy kernel"],
  ]);
  legend("right", [[COLORS[3], "Signed residual"]]);
  $("control-note").textContent =
    "Increasing rank keeps the rescaled coordinate window fixed.";
  window.explorerState = {
    kind: KIND,
    region: $("family").value,
    d,
    x: row.x,
    values,
    limit: row.limit,
    error,
    scale,
    center,
  };
}
function edgeDistribution() {
  const d = size(MODEL.degrees),
    di = +$("size").value,
    row = MODEL.gap,
    values = row.curves[di];
  $("parameter").min = 0;
  $("parameter").max = row.x.length - 1;
  const si = +$("parameter").value,
    s = row.x[si],
    residual = values.map((v, i) => v - row.limit[i]),
    error = Math.max(...residual.map(Math.abs)),
    scale = 1 / (Math.sqrt(2) * d ** (1 / 6)),
    center = Math.sqrt(2 * d),
    diagnostic = row.diagnostics[di],
    reference = row.reference_diagnostic;
  $("left-title").textContent = "Largest-eigenvalue CDF";
  $("right-title").textContent = "Finite rank minus Tracy–Widom β = 2";
  $("context-left").innerHTML =
    "<strong>Continuous gap.</strong> F_d(s) = det(I − K_d) on (a_d + s_d s,∞), for the Hermite process with weight e^(−x²). The selected s marks the threshold for λ_max; the curve uses continuous quadrature, without a sampled spatial grid.";
  $("context-right").innerHTML =
    "<strong>Edge reference.</strong> F₂(s) = det(I − K_Ai) on (s,∞). Both curves approximate half-line probabilities using a finite cutoff and Gauss–Legendre Nyström quadrature. The signed residual measures finite-rank deviation; sensitivity readouts are not error certificates.";
  const plot = linePlot(
    "left",
    row.x,
    [
      [values, COLORS[0]],
      [row.limit, COLORS[2], [5, 3]],
    ],
    [-4, 3, 0, 1.05],
    "edge threshold s",
    "P((λ_max − a_d)/s_d ≤ s)",
  );
  curve(
    plot.ctx,
    [
      [plot.X(s), plot.Y(0)],
      [plot.X(s), plot.Y(1)],
    ],
    "#688394",
    1,
    [4, 4],
  );
  marker(plot.ctx, plot.X(s), plot.Y(values[si]), COLORS[0]);
  const bound = Math.max(error * 1.1, 0.001);
  linePlot(
    "right",
    row.x,
    [[residual, COLORS[3]]],
    [-4, 3, -bound, bound],
    "edge threshold s",
    "F_d(s) − F₂(s)",
  );
  parameter("Edge threshold s", "−4", "3 · step 0.1", fmt(s));
  $("formula").textContent =
    "a_d = √(2d) · s_d = 1/(√2 d^(1/6)) · F_d(s) → F₂(s) = det(I − K_Ai)";
  $("left-caption").textContent =
    "Continuous finite-rank gap vs Airy determinant · 64 nodes, rescaled cutoff 10";
  $("right-caption").textContent =
    "Signed CDF difference on 71 thresholds · both references are numerical";
  metric(1, "Finite-rank CDF", fmt(values[si], 6), `d = ${d} · s = ${fmt(s)}`);
  metric(
    2,
    "Tracy–Widom β = 2",
    fmt(row.limit[si], 6),
    "Airy-kernel determinant approximation",
  );
  metric(
    3,
    "Grid max CDF difference",
    errorFormat(error),
    "max |F_d − F₂| on [−4,3]",
  );
  readout([
    ["Physical threshold", fmt(center + scale * s)],
    ["Center / scale", `${fmt(center)} / ${fmt(scale)}`],
    [
      "64 → 96 nodes",
      errorFormat(Math.max(diagnostic.quadrature, reference.quadrature)),
    ],
    ["Cutoff 10 → 14", errorFormat(Math.max(diagnostic.tail, reference.tail))],
  ]);
  legend("left", [
    [COLORS[0], "Finite-rank Hermite CDF"],
    [COLORS[2], "Tracy–Widom β = 2 (Airy)"],
  ]);
  legend("right", [[COLORS[3], "Signed CDF difference"]]);
  $("control-note").textContent =
    "Rank selects a recorded curve; s selects its threshold. Readouts show maximum sensitivity across all thresholds for this rank and the Airy reference. No determinant is computed in the browser.";
  window.explorerState = {
    kind: KIND,
    region: "gap",
    d,
    s,
    x: row.x,
    values,
    limit: row.limit,
    error,
    scale,
    center,
    selected: values[si],
    reference: row.limit[si],
    diagnostic,
  };
}
function complexValue(coeffs, x, y) {
  let re = 0,
    im = 0;
  for (const c of coeffs) {
    const a = re * x - im * y + c;
    im = re * y + im * x;
    re = a;
  }
  return [re, im];
}
function unitary() {
  // Match horizontal and vertical coordinate scales inside the axis padding.
  $("left").style.height = $("left").getBoundingClientRect().width - 16 + "px";
  const d = size(MODEL.degrees),
    di = +$("size").value,
    ti = +$("parameter").value,
    t = MODEL.times[ti],
    row = MODEL.rows[di][ti],
    { ctx, w, h } = chart($("left")),
    { X, Y, pad } = spectralAxes(
      ctx,
      w,
      h,
      -1.35,
      1.35,
      -1.35,
      1.35,
      "Re z",
      "Im z",
    ),
    temp = document.createElement("canvas"),
    resolution = 160;
  temp.width = temp.height = resolution;
  const tc = temp.getContext("2d"),
    pixels = tc.createImageData(resolution, resolution);
  for (let j = 0; j < resolution; j++)
    for (let i = 0; i < resolution; i++) {
      const [re, im] = complexValue(
          row.coefficients,
          -1.35 + (2.7 * i) / (resolution - 1),
          1.35 - (2.7 * j) / (resolution - 1),
        ),
        angle = Math.atan2(im, re),
        index = 4 * (j * resolution + i);
      for (let channel = 0; channel < 3; channel++)
        pixels.data[index + channel] = Math.round(
          205 + 42 * Math.cos(angle - (channel * 2 * Math.PI) / 3),
        );
      pixels.data[index + 3] = 255;
    }
  tc.putImageData(pixels, 0, 0);
  ctx.drawImage(temp, pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  const circle = Array.from({ length: 201 }, (_, i) => [
    X(Math.cos((i * Math.PI) / 100)),
    Y(Math.sin((i * Math.PI) / 100)),
  ]);
  curve(ctx, circle, "#526f85", 1.5, [5, 4]);
  row.roots.forEach((z, i) =>
    marker(ctx, X(z[0]), Y(z[1]), COLORS[i % 4], "circle", 5),
  );
  const tracks = MODEL.times.map((_, j) =>
      MODEL.rows[di][j].roots.map((z) => Math.atan2(z[1], z[0])),
    ),
    series = Array.from({ length: d }, (_, i) => [
      tracks.map((frame) => frame[i]),
      COLORS[i % 4],
    ]);
  const plot = linePlot(
    "right",
    MODEL.times,
    series,
    [0, 5, -Math.PI, Math.PI],
    "time t",
    "principal argument θ",
  );
  curve(
    plot.ctx,
    [
      [plot.X(t), plot.Y(-Math.PI)],
      [plot.X(t), plot.Y(Math.PI)],
    ],
    "#688394",
    1,
    [4, 4],
  );
  tracks[ti].forEach((v, i) =>
    marker(plot.ctx, plot.X(t), plot.Y(v), COLORS[i % 4]),
  );
  const defect = Math.max(
      ...row.roots.map((z) => Math.abs(Math.hypot(...z) - 1)),
    ),
    residual = Math.max(
      ...row.roots.map((z) => {
        const val = complexValue(row.coefficients, ...z),
          mod = Math.hypot(...z),
          den = row.coefficients.reduce(
            (sum, c, i) => sum + Math.abs(c) * mod ** (d - i),
            0,
          );
        return Math.hypot(...val) / den;
      }),
    );
  parameter("Time t", "0 · coalesced roots", "5 · step 0.1", fmt(t));
  $("formula").textContent =
    "H_d(z;t) = Σ (−1)^k C(d,k) exp[−t k(d−k)/(2d)] z^k";
  $("left-caption").textContent =
    "Hue = polynomial phase · circles = unprojected numerical roots";
  $("right-caption").textContent =
    "Sorted angular ranks · vertical marker selects time";
  metric(
    1,
    "Time",
    fmt(t),
    t === 0 ? "Exact d-fold root at z = 1" : "Recorded step 0.1",
  );
  metric(2, "Radial defect", errorFormat(defect), "max ||z_j| − 1|");
  metric(
    3,
    "Polynomial residual",
    errorFormat(residual),
    "max |H(z_j)| / Σ |c_k z_j^k|",
  );
  readout([
    ["Degree", d],
    ["Time", fmt(t)],
    [
      "Angle range",
      fmt(Math.min(...tracks[ti])) + " … " + fmt(Math.max(...tracks[ti])),
    ],
    [
      "Root extraction",
      t === 0 ? "exact coalesced roots" : "complex128 companion solve",
    ],
  ]);
  legend("left", [
    [COLORS[0], "Root markers"],
    ["#8a779b", "Phase hue"],
  ]);
  legend("right", [
    [COLORS[0], "Angular ranks"],
    ["#688394", "Selected time"],
  ]);
  $("control-note").textContent =
    "The time slider selects recorded polynomial data; roots are not radial projections or random matrix samples.";
  window.explorerState = {
    kind: KIND,
    d,
    t,
    roots: row.roots,
    defect,
    residual,
  };
}
function render() {
  ({
    "ensemble-convergence": ensemble,
    "finite-transforms": transforms,
    "convolution-interlacing": interlacing,
    "hermite-kernels": kernels,
    "unitary-root-flow": unitary,
  })[KIND]();
}
hide("ratio", !["ensemble-convergence", "finite-transforms"].includes(KIND));
hide("sample", KIND !== "ensemble-convergence");
hide("entries", KIND !== "ensemble-convergence");
hide(
  "parameter",
  ![
    "finite-transforms",
    "convolution-interlacing",
    "unitary-root-flow",
  ].includes(KIND),
);
hide("size", KIND === "convolution-interlacing");
if (KIND === "convolution-interlacing") {
  $("parameter").min = 0;
  $("parameter").max = 20;
  $("parameter").value = 10;
  $("sweep").hidden = true;
}
if (KIND === "unitary-root-flow") {
  $("size").value = 1;
  $("parameter").min = 0;
  $("parameter").max = 50;
  $("parameter").value = 10;
  $("family-control").hidden = true;
}
const defaults = {
  family: $("family").value,
  entries: $("entries").value,
  size: $("size").value,
  ratio: $("ratio").value,
  sample: $("sample").value,
  parameter: $("parameter").value,
};
for (const id of Object.keys(defaults))
  $(id).addEventListener(
    id === "size" || id === "parameter" ? "input" : "change",
    () => {
      stop();
      render();
    },
  );
$("reset").onclick = () => {
  stop();
  for (const [id, value] of Object.entries(defaults)) $(id).value = value;
  render();
};
$("sweep").onclick = () => {
  if (timer) {
    stop();
    return;
  }
  const input = $(KIND === "unitary-root-flow" ? "parameter" : "size");
  $("sweep").textContent = "Stop sweep";
  timer = setInterval(() => {
    input.value = +input.value >= +input.max ? input.min : +input.value + 1;
    render();
  }, 600);
};
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stop();
});
window.addEventListener("pagehide", stop);
window.addEventListener("resize", render);
stop();
render();

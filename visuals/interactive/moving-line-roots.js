const MODEL = JSON.parse($("model").textContent),
  YS = MODEL.ys.map(rational),
  ZS = MODEL.zs.map(rational);
let timer = null,
  trackAxes = null,
  direction = 1;
const matrices = (y, z) =>
  MODEL.matrix_diagonal.map((d, i) =>
    MODEL.matrix_diagonal.map((_, j) =>
      i === j ? d * y : Math.abs(i - j) === 1 ? z : 0,
    ),
  );
const rootsAt = (y, z) =>
  eigenvalues(matrices(y, z))
    .map((v) => -v)
    .sort((a, b) => a - b);
const exactAt = (yi, zi) => MODEL.lines[zi][yi];
let cachedZi = -1,
  trackData = [];
function dataFor(zi) {
  if (cachedZi !== zi) {
    trackData = YS.map((y, yi) => {
      const roots = rootsAt(y, ZS[zi]),
        exact = exactAt(yi, zi);
      return {
        roots,
        derivative: bisectDerivative(exact.d.map(rational), roots),
      };
    });
    cachedZi = zi;
  }
  return trackData;
}
function drawTracks(yi, zi, showDerivative) {
  const { ctx, w, h } = chart($("tracks")),
    rows = dataFor(zi),
    z = ZS[zi],
    extent = 6.7,
    a = axes(
      ctx,
      w,
      h,
      -2,
      2,
      -extent,
      extent,
      "Line position y",
      "Slice root s",
    );
  trackAxes = { ...a, w, h };
  for (let j = 0; j < 4; j++) {
    const points = YS.map((y, i) => [
      a.X(y),
      a.Y(z === 0 ? -MODEL.matrix_diagonal[j] * y : rows[i].roots[j]),
    ]);
    curve(ctx, points, COLORS[j], 2.4);
  }
  if (showDerivative)
    for (let j = 0; j < 3; j++)
      curve(
        ctx,
        YS.map((y, i) => [a.X(y), a.Y(rows[i].derivative[j])]),
        "#65798a",
        1.1,
        [4, 4],
      );
  const y = YS[yi];
  curve(
    ctx,
    [
      [a.X(y), a.Y(-extent)],
      [a.X(y), a.Y(extent)],
    ],
    "#172f43",
    1.3,
    [3, 4],
  );
  (z === 0 ? MODEL.matrix_diagonal.map((d) => -d * y) : rows[yi].roots).forEach(
    (r, j) => marker(ctx, a.X(y), a.Y(r), COLORS[j]),
  );
  if (showDerivative)
    rows[yi].derivative.forEach((r) =>
      marker(ctx, a.X(y), a.Y(r), "#526a7c", "diamond", 4),
    );
}
function drawPolynomial(row, exact, showDerivative, displayRoots) {
  const { ctx, w, h } = chart($("polynomial")),
    p = exact.p.map(rational),
    d = exact.d.map(rational),
    extent = Math.max(0.5, Math.max(...row.roots.map(Math.abs)) * 1.1 + 0.15),
    xs = Array.from(
      { length: 321 },
      (_, i) => -extent + (2 * extent * i) / 320,
    ),
    pv = xs.map((x) => horner(p, x)),
    dv = xs.map((x) => horner(d, x) / 4),
    all = showDerivative ? pv.concat(dv) : pv,
    min = Math.min(...all, 0),
    max = Math.max(...all, 0),
    range = Math.max(0.01, max - min),
    lo = min - 0.12 * range,
    hi = max + 0.12 * range,
    a = axes(
      ctx,
      w,
      h,
      -extent,
      extent,
      lo,
      hi,
      "Line coordinate s",
      "Polynomial value",
    );
  curve(
    ctx,
    [
      [a.X(-extent), a.Y(0)],
      [a.X(extent), a.Y(0)],
    ],
    "#96a9b7",
    1,
  );
  curve(
    ctx,
    xs.map((x, i) => [a.X(x), a.Y(pv[i])]),
    "#007f83",
    2.5,
  );
  if (showDerivative)
    curve(
      ctx,
      xs.map((x, i) => [a.X(x), a.Y(dv[i])]),
      "#65798a",
      1.6,
      [5, 4],
    );
  displayRoots.forEach((r, j) =>
    marker(ctx, a.X(r), a.Y(0), COLORS[j], "circle", 5),
  );
  if (showDerivative)
    row.derivative.forEach((r) =>
      marker(ctx, a.X(r), a.Y(0), "#526a7c", "diamond", 4),
    );
}
function update() {
  const yi = Number($("parameter").value),
    zi = Number($("coupling").value),
    y = YS[yi],
    z = ZS[zi],
    row = dataFor(zi)[yi],
    exact = exactAt(yi, zi),
    show = $("derivative").checked,
    gaps = row.roots.slice(1).map((r, i) => r - row.roots[i]),
    gap = Math.min(...gaps),
    tol = 2e-9 * (1 + Math.max(...row.roots.map(Math.abs))),
    interlacing = row.derivative.every(
      (r, i) => r >= row.roots[i] - tol && r <= row.roots[i + 1] + tol,
    ),
    coeff = exact.p.map(rational),
    residual = Math.max(
      ...row.roots.map(
        (r) =>
          Math.abs(horner(coeff, r)) /
          (1 +
            coeff.reduce(
              (s, c, i) =>
                s + Math.abs(c) * Math.abs(r) ** (coeff.length - i - 1),
              0,
            )),
      ),
    );
  $("parameter-value").textContent = y.toFixed(2);
  $("coupling-value").textContent = z.toFixed(3);
  $("line-label").textContent = `L(s) = (s, ${y.toFixed(2)}, ${z.toFixed(3)})`;
  $("track-mode").textContent =
    z === 0 ? "uncoupled diagonal identities" : "ascending root ranks";
  $("geometry").textContent =
    gap <= tol ? "Repeated roots" : "Four simple roots";
  $("geometry").className = "status " + (gap <= tol ? "boundary" : "");
  $("geometry-note").textContent =
    z === 0
      ? "Uncoupled branches cross at y = 0."
      : "Nonzero coupling opens the crossing into gaps.";
  $("gap").textContent = fmt(gap, 6);
  $("interlacing").textContent = interlacing
    ? gap <= tol
      ? "Weak · equalities present"
      : "Satisfied numerically"
    : "Outside tolerance";
  $("root-values").textContent = row.roots.map((r) => fmt(r)).join(" · ");
  $("derivative-values").textContent = row.derivative
    .map((r) => fmt(r))
    .join(" · ");
  $("residual").textContent = residual.toExponential(1);
  $("exact-line").textContent =
    `Exact rational parameters: y = ${MODEL.ys[yi]}, z = ${MODEL.zs[zi]}. Coefficients below descend in powers of s.`;
  $("coefficients").textContent =
    `P:  [${exact.p.join(", ")}]\nP′: [${exact.d.join(", ")}]`;
  drawTracks(yi, zi, show);
  drawPolynomial(
    row,
    exact,
    show,
    z === 0 ? MODEL.matrix_diagonal.map((d) => -d * y) : row.roots,
  );
  window.explorerState = {
    y,
    z,
    roots: row.roots,
    derivativeRoots: row.derivative,
    gap,
    interlacing,
    residual,
    exactCoefficients: exact.p,
    showDerivative: show,
  };
}
function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $("play").textContent = "Sweep y";
}
["parameter", "coupling", "derivative"].forEach((id) =>
  $(id).addEventListener("input", () => {
    stop();
    update();
  }),
);
$("reset").onclick = () => {
  stop();
  $("parameter").value = 45;
  $("coupling").value = 20;
  $("derivative").checked = true;
  update();
};
$("uncoupled").onclick = () => {
  $("coupling").value = 0;
  update();
};
$("crossing").onclick = () => {
  stop();
  $("coupling").value = 0;
  $("parameter").value = 40;
  update();
};
$("play").onclick = () => {
  if (timer) {
    stop();
    return;
  }
  $("play").textContent = "Pause sweep";
  timer = setInterval(() => {
    let y = Number($("parameter").value) + direction;
    if (y < 0 || y > 80) {
      direction *= -1;
      y = Number($("parameter").value) + direction;
    }
    $("parameter").value = y;
    update();
  }, 110);
};
$("tracks").addEventListener("pointerdown", (e) => {
  $("tracks").setPointerCapture(e.pointerId);
  move(e);
});
$("tracks").addEventListener("pointermove", (e) => {
  if (e.buttons === 1) move(e);
});
function move(e) {
  stop();
  const box = $("tracks").getBoundingClientRect(),
    a = trackAxes;
  if (a) {
    $("parameter").value = Math.max(
      0,
      Math.min(
        80,
        Math.round(
          ((e.clientX - box.left - a.pad.l) / (a.w - a.pad.l - a.pad.r)) * 80,
        ),
      ),
    );
    update();
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stop();
});
window.addEventListener("resize", update);
update();

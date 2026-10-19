const MODEL = JSON.parse($("model").textContent),
  LIMIT = 1.4;
let yaw = -0.72,
  tilt = 0.55,
  drag = null,
  sliceAxes = null;
const probe = () => ["x", "y", "z"].map((id) => Number($(id).value));
const determinant = (p) => evalTerms(MODEL.terms, p);
function classification(point) {
  const minors = point.map((v) => 1 - v * v),
    d = determinant(point),
    eps = 2e-9 * (1 + point.reduce((s, v) => s + Math.abs(v), 0));
  return { psd: minors.every((v) => v >= -eps) && d >= -eps, d, minors, eps };
}
const mesh = [];
for (let i = 0; i < 70; i++)
  for (let j = 0; j < 70; j++)
    for (const sign of [-1, 1]) {
      const vertices = [
        [i, j],
        [i + 1, j],
        [i + 1, j + 1],
        [i, j + 1],
      ].map(([a, b]) => {
        const x = -LIMIT + (2 * LIMIT * a) / 70,
          y = -LIMIT + (2 * LIMIT * b) / 70,
          r = (1 - x * x) * (1 - y * y);
        if (r < -1e-12) return null;
        const z = x * y + sign * Math.sqrt(Math.max(0, r));
        return Math.abs(z) <= LIMIT + 1e-9 ? [x, y, z] : null;
      });
      if (vertices.some((v) => v === null)) continue;
      for (const indices of [
        [0, 1, 2],
        [0, 2, 3],
      ]) {
        const points = indices.map((i) => vertices[i]);
        mesh.push({
          points,
          psd: points.every(
            (p) => Math.abs(p[0]) <= 1 + 1e-8 && Math.abs(p[1]) <= 1 + 1e-8,
          ),
        });
      }
    }
function drawSurface() {
  const { ctx, w, h } = chart($("surface")),
    s = Math.min(w, h) * 0.235;
  const project = ([x, y, z]) => {
    const a = Math.cos(yaw) * x - Math.sin(yaw) * y,
      b = Math.sin(yaw) * x + Math.cos(yaw) * y;
    return [
      w / 2 + s * a,
      h / 2 - s * (Math.cos(tilt) * z - Math.sin(tilt) * b),
      Math.cos(tilt) * b + Math.sin(tilt) * z,
    ];
  };
  const faces = mesh.map((face) => ({
    ...face,
    screen: face.points.map(project),
  }));
  faces.sort(
    (a, b) =>
      a.screen.reduce((s, p) => s + p[2], 0) -
      b.screen.reduce((s, p) => s + p[2], 0),
  );
  faces.forEach((face) => {
    ctx.beginPath();
    face.screen.forEach((p, i) =>
      i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]),
    );
    ctx.closePath();
    ctx.fillStyle = face.psd ? "rgba(0,127,131,.40)" : "rgba(114,75,174,.32)";
    ctx.fill();
    ctx.strokeStyle = face.psd ? "rgba(0,107,117,.09)" : "rgba(114,75,174,.10)";
    ctx.lineWidth = 0.5;
    ctx.stroke();
  });
  const p = probe(),
    z = p[2];
  curve(
    ctx,
    [
      [-LIMIT, -LIMIT, z],
      [LIMIT, -LIMIT, z],
      [LIMIT, LIMIT, z],
      [-LIMIT, LIMIT, z],
      [-LIMIT, -LIMIT, z],
    ]
      .map(project)
      .map((p) => p.slice(0, 2)),
    "#b85b16",
    1.3,
    [5, 4],
  );
  for (let axis = 0; axis < 3; axis++) {
    const a = [0, 0, 0],
      b = [0, 0, 0];
    a[axis] = -1.65;
    b[axis] = 1.65;
    curve(
      ctx,
      [project(a), project(b)].map((p) => p.slice(0, 2)),
      "#7d919f",
      1,
    );
    const q = project(b);
    ctx.fillStyle = "#314c60";
    ctx.font = "13px system-ui";
    ctx.fillText(["x", "y", "z"][axis], q[0] + 5, q[1] + 5);
  }
  const q = project(p);
  marker(ctx, q[0], q[1], "#b85b16", "circle", 7);
  ctx.font = "11px system-ui";
  ctx.fillStyle = "#677d8c";
  ctx.fillText("t = 1 · determinant-zero sheets", 16, h - 15);
}
function drawSlice() {
  const { ctx, w, h } = chart($("slice")),
    z = probe()[2],
    a = axes(ctx, w, h, -LIMIT, LIMIT, -LIMIT, LIMIT, "x", "y");
  sliceAxes = { ...a, w, h };
  const N = 100;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const x = -LIMIT + (2 * LIMIT * (i + 0.5)) / N,
        y = -LIMIT + (2 * LIMIT * (j + 0.5)) / N,
        c = classification([x, y, z]);
      ctx.fillStyle = c.psd ? "#b5e1dc" : c.d > 0 ? "#ded0f0" : "#e9eef1";
      const dx = a.X(x + LIMIT / N) - a.X(x - LIMIT / N),
        dy = a.Y(y - LIMIT / N) - a.Y(y + LIMIT / N);
      ctx.fillRect(a.X(x - LIMIT / N), a.Y(y + LIMIT / N), dx + 0.4, dy + 0.4);
    }
  // For |z|<=1 this parametric ellipse/segment is the true PSD slice boundary.
  if (Math.abs(z) <= 1) {
    const r = Math.sqrt(Math.max(0, 1 - z * z)),
      points = [];
    for (let k = 0; k <= 240; k++) {
      const t = (k / 240) * 2 * Math.PI,
        x = Math.cos(t),
        y = z * x + r * Math.sin(t);
      points.push([a.X(x), a.Y(y)]);
    }
    curve(ctx, points, "#007f83", 2.4);
  }
  const [x, y] = probe();
  curve(
    ctx,
    [
      [a.X(x), a.Y(-LIMIT)],
      [a.X(x), a.Y(LIMIT)],
    ],
    "#b85b16",
    1,
    [3, 4],
  );
  curve(
    ctx,
    [
      [a.X(-LIMIT), a.Y(y)],
      [a.X(LIMIT), a.Y(y)],
    ],
    "#b85b16",
    1,
    [3, 4],
  );
  marker(ctx, a.X(x), a.Y(y), "#b85b16", "circle", 6);
}
function update() {
  const p = probe(),
    c = classification(p),
    e = eigenvalues([
      [1, p[0], p[1]],
      [p[0], 1, p[2]],
      [p[1], p[2], 1],
    ]),
    near = Math.abs(e[0]) <= c.eps;
  ["x", "y", "z"].forEach(
    (id, i) => ($(id + "-value").textContent = p[i].toFixed(2)),
  );
  $("slice-value").textContent = p[2].toFixed(2);
  const status = c.psd
    ? near
      ? "PSD boundary · numerical"
      : "Positive definite"
    : "Indefinite";
  $("status").textContent = status;
  $("status").className =
    "status " + (c.psd ? (near ? "boundary" : "") : "outside");
  $("membership-note").textContent = c.psd
    ? "In the closed PSD section."
    : c.d > c.eps
      ? "Positive determinant, but outside the PSD section."
      : "Outside the PSD section.";
  $("lambda-min").textContent = fmt(e[0], 6);
  $("det").textContent = fmt(c.d, 6);
  $("eigenvalues").textContent = e.map((v) => fmt(v, 4)).join(" · ");
  ["x", "y", "z"].forEach(
    (id, i) => ($("minor-" + id).textContent = fmt(c.minors[i], 6)),
  );
  $("matrix").textContent =
    `[ 1     ${fmt(p[0])}   ${fmt(p[1])} ]\n[ ${fmt(p[0])}   1     ${fmt(p[2])} ]\n[ ${fmt(p[1])}   ${fmt(p[2])}   1   ]`;
  drawSurface();
  drawSlice();
  window.explorerState = {
    probe: p,
    det: c.d,
    eigenvalues: e,
    psd: c.psd,
    status,
  };
}
function setProbe(values) {
  ["x", "y", "z"].forEach((id, i) => ($(id).value = values[i]));
  update();
}
["x", "y", "z"].forEach((id) => $(id).addEventListener("input", update));
$("reset").onclick = () => {
  yaw = -0.72;
  tilt = 0.55;
  setProbe([0.2, 0.4, 0.3]);
};
$("boundary").onclick = () => setProbe([1, 0.5, 0.5]);
$("chamber").onclick = () => setProbe([1.2, 1.2, 1.2]);
$("surface").addEventListener("pointerdown", (e) => {
  drag = [e.clientX, e.clientY];
  $("surface").setPointerCapture(e.pointerId);
});
$("surface").addEventListener("pointermove", (e) => {
  if (!drag) return;
  yaw += (e.clientX - drag[0]) * 0.008;
  tilt = Math.max(-1.1, Math.min(1.1, tilt + (e.clientY - drag[1]) * 0.008));
  drag = [e.clientX, e.clientY];
  drawSurface();
});
$("surface").addEventListener("pointerup", () => (drag = null));
$("surface").addEventListener("pointercancel", () => (drag = null));
$("surface").addEventListener("keydown", (e) => {
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key))
    return;
  e.preventDefault();
  yaw += e.key === "ArrowRight" ? 0.12 : e.key === "ArrowLeft" ? -0.12 : 0;
  tilt = Math.max(
    -1.1,
    Math.min(
      1.1,
      tilt + (e.key === "ArrowUp" ? 0.12 : e.key === "ArrowDown" ? -0.12 : 0),
    ),
  );
  drawSurface();
});
function sliceProbe(e) {
  const box = $("slice").getBoundingClientRect(),
    a = sliceAxes;
  if (!a) return;
  const x =
      -LIMIT +
      ((e.clientX - box.left - a.pad.l) / (a.w - a.pad.l - a.pad.r)) *
        2 *
        LIMIT,
    y =
      LIMIT -
      ((e.clientY - box.top - a.pad.t) / (a.h - a.pad.t - a.pad.b)) * 2 * LIMIT;
  ["x", "y"].forEach(
    (id, i) =>
      ($(id).value = Math.max(-LIMIT, Math.min(LIMIT, [x, y][i])).toFixed(2)),
  );
  update();
}
$("slice").addEventListener("pointerdown", (e) => {
  $("slice").setPointerCapture(e.pointerId);
  sliceProbe(e);
});
$("slice").addEventListener("pointermove", (e) => {
  if (e.buttons === 1) sliceProbe(e);
});
window.addEventListener("resize", update);
update();

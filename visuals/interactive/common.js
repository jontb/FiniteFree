"use strict";
const $ = (id) => document.getElementById(id),
  COLORS = ["#007f83", "#3568b4", "#b85b16", "#724bae"];
const rational = (s) => {
  const a = String(s).split("/").map(Number);
  return a[0] / (a[1] || 1);
};
const fmt = (x, n = 4) =>
  Math.abs(x) < 1e-12 ? "0" : Number(x.toPrecision(n)).toString();
const horner = (c, x) => c.reduce((v, a) => v * x + a, 0);
const evalTerms = (terms, point) =>
  terms.reduce(
    (sum, t) =>
      sum +
      rational(t.coefficient) *
        t.powers.reduce((v, p, i) => v * point[i] ** p, 1),
    0,
  );
function eigenvalues(matrix) {
  const a = matrix.map((row) => row.slice()),
    n = a.length;
  for (let sweep = 0; sweep < 100; sweep++) {
    let p = 0,
      q = 1,
      max = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(a[i][j]) > max) {
          max = Math.abs(a[i][j]);
          p = i;
          q = j;
        }
    if (max < 2e-14) break;
    const theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]),
      c = Math.cos(theta),
      s = Math.sin(theta),
      ap = a[p][p],
      aq = a[q][q],
      off = a[p][q];
    a[p][p] = c * c * ap - 2 * c * s * off + s * s * aq;
    a[q][q] = s * s * ap + 2 * c * s * off + c * c * aq;
    a[p][q] = a[q][p] = 0;
    for (let k = 0; k < n; k++)
      if (k !== p && k !== q) {
        const u = a[k][p],
          v = a[k][q];
        a[k][p] = a[p][k] = c * u - s * v;
        a[k][q] = a[q][k] = s * u + c * v;
      }
  }
  return a.map((r, i) => r[i]).sort((a, b) => a - b);
}
function chart(canvas) {
  const box = canvas.getBoundingClientRect(),
    dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(box.width * dpr);
  canvas.height = Math.round(box.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: box.width, h: box.height };
}
function axes(ctx, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel) {
  const pad = { l: 48, r: 20, t: 20, b: 42 },
    X = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * (w - pad.l - pad.r),
    Y = (y) => h - pad.b - ((y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    let x = xmin + ((xmax - xmin) * i) / 4,
      y = ymin + ((ymax - ymin) * i) / 4;
    ctx.strokeStyle = "#e7edf1";
    ctx.beginPath();
    ctx.moveTo(X(x), pad.t);
    ctx.lineTo(X(x), h - pad.b);
    ctx.moveTo(pad.l, Y(y));
    ctx.lineTo(w - pad.r, Y(y));
    ctx.stroke();
    ctx.fillStyle = "#647887";
    ctx.fillText(fmt(x, 3), X(x), h - pad.b + 18);
    ctx.textAlign = "right";
    ctx.fillText(fmt(y, 3), pad.l - 8, Y(y) + 4);
    ctx.textAlign = "center";
  }
  ctx.strokeStyle = "#a1b0bc";
  ctx.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.fillStyle = "#314c60";
  ctx.fillText(xlabel, (w + pad.l - pad.r) / 2, h - 8);
  ctx.save();
  ctx.translate(13, (h + pad.t - pad.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();
  return { X, Y, pad };
}
function curve(ctx, points, color, width = 2, dash = []) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.stroke();
  ctx.setLineDash([]);
}
function marker(ctx, x, y, color, shape = "circle", size = 5) {
  ctx.fillStyle = color;
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  if (shape === "diamond") {
    ctx.moveTo(x, y - size);
    ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size);
    ctx.lineTo(x - size, y);
    ctx.closePath();
  } else ctx.arc(x, y, size, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
}
function bisectDerivative(coefficients, roots) {
  return roots.slice(0, -1).map((a, i) => {
    let b = roots[i + 1];
    if (b - a < 1e-11) return (a + b) / 2;
    let fa = horner(coefficients, a);
    if (Math.abs(fa) < 1e-14) return a;
    if (Math.abs(horner(coefficients, b)) < 1e-14) return b;
    for (let k = 0; k < 65; k++) {
      const m = (a + b) / 2,
        fm = horner(coefficients, m);
      if (fa * fm <= 0) b = m;
      else {
        a = m;
        fa = fm;
      }
    }
    return (a + b) / 2;
  });
}

/* Optional development check; uses the same sandboxed Chromium as browser-check.cjs. */
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  { execFileSync } = require("node:child_process"),
  { pathToFileURL } = require("node:url"),
  { chromium } = require("playwright-core");
// Optional loopback transport for managed browsers that prohibit file://.
const baseURL = process.env.FINITEFREE_BASE_URL;
if (baseURL) {
  const url = new URL(baseURL);
  assert.ok(
    url.protocol === "http:" &&
      ["localhost", "127.0.0.1"].includes(url.hostname),
    "Browser check base URL must be loopback HTTP",
  );
}
const directory = path.resolve(process.argv[2] || "visuals/generated"),
  output = path.resolve(
    process.argv[3] || "visuals/generated/dashboard-screenshots",
  ),
  reports = [],
  errors = [],
  external = [];
fs.mkdirSync(output, { recursive: true });
const near = (a, b, tolerance = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
async function run() {
  const executable =
    process.env.FINITEFREE_BROWSER ||
    ["chromium", "google-chrome", "chromium-browser"]
      .map((name) => {
        try {
          return execFileSync("which", [name], { encoding: "utf8" }).trim();
        } catch {
          return null;
        }
      })
      .find(Boolean);
  assert.ok(
    executable,
    "Set FINITEFREE_BROWSER to a Chromium/Chrome executable",
  );
  const browser = await chromium.launch({
    executablePath: executable,
    headless: true,
    chromiumSandbox: true,
  });
  const context = await browser.newContext({
      viewport: { width: 1440, height: 1050 },
      reducedMotion: "reduce",
    }),
    page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (
      /^https?:/.test(r.url()) &&
      !(
        baseURL &&
        r.isNavigationRequest() &&
        new URL(r.url()).origin === new URL(baseURL).origin
      )
    )
      external.push(r.url());
  });
  const state = () => page.evaluate(() => window.explorerState),
    slide = async (id, value) => {
      await page.locator("#" + id).evaluate((el, v) => {
        el.value = String(v);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }, value);
    },
    shot = async (name) =>
      page.screenshot({
        path: path.join(output, name + ".png"),
        fullPage: true,
      });
  const responsiveSelection = async (name) => {
    const before = await state();
    for (const width of [1024, 390]) {
      await page.setViewportSize({ width, height: 844 });
      await page.waitForTimeout(50);
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
      );
      assert.deepEqual(await state(), before);
      if (width === 390) await shot(name + "-mobile");
    }
    await page.setViewportSize({ width: 1440, height: 1050 });
    await shot(name);
  };
  for (const name of [
    "ensemble-convergence",
    "finite-transforms",
    "convolution-interlacing",
    "hermite-kernels",
    "unitary-root-flow",
  ]) {
    await page.goto(
      baseURL
        ? new URL(name + ".html", baseURL).href
        : pathToFileURL(path.join(directory, name + ".html")).href,
    );
    await page.waitForFunction(() => window.explorerState);
    const initial = await state();
    assert.equal(initial.kind, name);
    assert.ok(await page.locator("h1").innerText());
    assert.ok(await page.locator("#lead").innerText());
    await page.locator("details summary").focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("details").evaluate((el) => el.open), true);
    assert.deepEqual(await state(), initial);
    await page.keyboard.press("Enter");
    await page.locator("details summary").evaluate((el) => el.blur());
    if (name === "ensemble-convergence") {
      for (const family of [
        "gue",
        "goe",
        "gse",
        "wishart1",
        "wishart2",
        "wishart4",
        "hermite",
        "laguerre",
        "legendre",
      ]) {
        await page.selectOption("#family", family);
        for (const size of [0, 5]) {
          await slide("size", size);
          const s = await state();
          assert.equal(s.values.length, s.d);
          assert.ok(s.distance >= 0 && s.distance <= 1);
          near(s.histogramMass + s.atom, 1);
          assert.equal(
            s.sampled,
            !["hermite", "laguerre", "legendre"].includes(family),
          );
        }
        reports.push(
          family + " size endpoints, finite mass and measure labels",
        );
      }
      for (const family of ["wishart1", "wishart2", "wishart4", "laguerre"]) {
        await page.selectOption("#family", family);
        for (const ratio of [0, 1, 2, 3]) {
          await page.selectOption("#ratio", String(ratio));
          const s = await state();
          near(s.gamma, s.d / s.n);
          near(s.atom, Math.max(0, 1 - s.n / s.d));
          near(s.limitAtom, s.atom);
          assert.equal(
            s.values.filter((v) => v === 0).length,
            Math.max(0, s.d - s.n),
          );
        }
        reports.push(family + " all aspect ratios and zero atoms");
      }
      await page.selectOption("#family", "gue");
      const first = await state();
      await page.selectOption("#sample", "1");
      assert.notDeepEqual((await state()).values, first.values);
      await page.selectOption("#sample", "0");
      assert.deepEqual((await state()).values, first.values);
      reports.push("Recorded sample selection is reversible");
      for (const family of ["goe", "gue", "wishart1", "wishart2", "wishart4"]) {
        await page.selectOption("#family", family);
        for (const distribution of ["gaussian", "rademacher", "uniform"]) {
          await page.selectOption("#entries", distribution);
          const s = await state();
          assert.equal(s.distribution, distribution);
          assert.ok(s.values.every(Number.isFinite));
          near(s.histogramMass + s.atom, 1);
          assert.match(
            await page.locator("#left-caption").innerText(),
            new RegExp(distribution),
          );
        }
      }
      reports.push(
        "Applicable ensembles switch among three entry distributions with updated labels",
      );
      for (const family of ["gse", "hermite", "laguerre", "legendre"]) {
        await page.selectOption("#family", family);
        assert.equal(await page.locator("#entries").isDisabled(), true);
      }
      reports.push(
        "Gaussian-only GSE and deterministic measures disable entry controls",
      );

      // Switch from a larger degree range, then repeat all compound controls.
      await page.selectOption("#family", "gue");
      await slide("size", 5);
      for (let repeat = 0; repeat < 2; repeat++) {
        await page.selectOption("#family", "compound");
        assert.ok((await state()).d <= 64);
        for (const id of ["entries", "ratio", "sample"])
          assert.equal(await page.locator("#" + id).isDisabled(), true);
        for (const dimension of [0, 1, 2, 3]) {
          await slide("size", dimension);
          const s = await state();
          assert.equal(s.d, [8, 16, 32, 64][dimension]);
          assert.equal(s.sampled, false);
          assert.equal(s.n, s.d * s.d);
          near(s.values.reduce((a, b) => a + b, 0) / s.d, 1);
          near(s.histogramMass, 1);
          assert.ok(s.values.every((v) => v > 0));
        }
        await responsiveSelection("compound-wishart-free-lognormal");
        await page.click("#sweep");
        await page.waitForTimeout(750);
        assert.equal((await state()).d, 8);
        await page.selectOption("#family", "gue");
      }
      reports.push(
        "Compound Wishart all bounded degrees, repeat switching, sweep, disabled controls and responsive layouts",
      );
      await page.selectOption("#family", "wishart2");
      await page.selectOption("#ratio", "3");
      await shot("ensemble-zero-atom");
      await page.click("#reset");
      await shot(name);
    } else if (name === "finite-transforms") {
      for (const ratio of [0, 1, 2, 3]) {
        await page.selectOption("#ratio", String(ratio));
        for (const dimension of [0, 5]) {
          await slide("size", dimension);
          for (const t of [1, 50, 99]) {
            await slide("parameter", t);
            const s = await state();
            assert.equal(s.k, Math.floor((t * s.d) / 100));
            near(s.selected, Math.max(0, 1 - s.gamma + (s.k + 1) / s.n));
          }
        }
      }
      reports.push(
        "T-transform all ratios, size endpoints, domain probes and jump convention",
      );
      await page.selectOption("#family", "clt");
      for (const dimension of [0, 7]) {
        await slide("size", dimension);
        const s = await state();
        near(s.values[1], s.d / (s.d - 1));
        assert.equal(s.available, Math.min(s.d, 8));
        assert.ok(
          await page.locator("#parameter-control").evaluate((el) => el.hidden),
        );
      }
      reports.push("Exact CLT variance and available-order labels");
      await shot("finite-cumulants");
      await page.click("#reset");
      await shot(name);
    } else if (name === "convolution-interlacing") {
      for (const mode of ["additive", "multiplicative"]) {
        await page.selectOption("#family", mode);
        for (const shift of [0, 10, 20]) {
          await slide("parameter", shift);
          const s = await state();
          assert.ok(s.margin >= -1e-9);
          if (mode === "additive")
            s.output.p.forEach((p, i) => near(s.output.q[i] - p, s.shift));
        }
      }
      reports.push(
        "Both convolutions preserve weak interlacing, including equality endpoints",
      );
      await page.click("#reset");
      await shot(name);
    } else if (name === "hermite-kernels") {
      for (const region of ["bulk", "edge"]) {
        await page.selectOption("#family", region);
        const errors = [];
        for (const dimension of [0, 5]) {
          await slide("size", dimension);
          const s = await state();
          assert.ok(s.values.every(Number.isFinite));
          near(
            s.error,
            Math.max(...s.values.map((v, i) => Math.abs(v - s.limit[i]))),
          );
          errors.push(s.error);
        }
        assert.ok(errors[1] < errors[0]);
      }
      reports.push(
        "Bulk and edge sections, finite residuals and rank endpoints",
      );
      await shot("hermite-edge");
      for (let repeat = 0; repeat < 2; repeat++) {
        await page.selectOption("#family", "gap");
        assert.equal(
          await page.locator("#parameter-control").isVisible(),
          true,
        );
        for (const dimension of [0, 1, 2, 3, 4, 5]) {
          await slide("size", dimension);
          for (const probe of [0, 40, 70, 40]) {
            await slide("parameter", probe);
            const s = await state();
            near(s.s, -4 + probe / 10);
            near(s.selected, s.values[probe]);
            assert.ok(
              s.values.every((v) => Number.isFinite(v) && v >= 0 && v <= 1),
            );
            assert.ok(
              s.values.every((v, i) => i === 0 || v >= s.values[i - 1]),
            );
            assert.ok(Math.max(...Object.values(s.diagnostic)) < 1e-9);
          }
        }
        await responsiveSelection("continuous-gap-edge-distribution");
        await page.click("#sweep");
        await page.waitForTimeout(750);
        assert.equal((await state()).d, 8);
        await page.selectOption("#family", "edge");
        assert.equal(
          await page.locator("#parameter-control").isVisible(),
          false,
        );
        assert.equal(
          await page.locator("#left-title").innerText(),
          "Rescaled kernel section",
        );
      }
      reports.push(
        "Continuous gap all ranks and thresholds, monotone probabilities, repeat switching, sweep and responsive layouts",
      );
      await page.click("#reset");
      await shot(name);
    } else {
      for (const dimension of [0, 2]) {
        await slide("size", dimension);
        for (const time of [0, 1, 10, 50]) {
          await slide("parameter", time);
          const s = await state();
          assert.equal(s.roots.length, s.d);
          near(s.t, time / 10);
          assert.ok(s.defect < 1e-6);
          assert.ok(s.residual < 1e-12);
          if (time === 0)
            assert.deepEqual(
              s.roots,
              Array.from({ length: s.d }, () => [1, 0]),
            );
        }
      }
      reports.push(
        "Unitary coalescence, raw-root defects and residuals at time/degree endpoints",
      );
      await page.click("#reset");
      await shot(name);
    }
    if (name !== "convolution-interlacing") {
      const before = await state();
      await page.click("#sweep");
      await page.waitForTimeout(750);
      const after = await state();
      assert.notDeepEqual(after, before);
      await page.click("#reset");
      assert.deepEqual(await state(), initial);
      reports.push(name + " sweep and reset");
    }
    for (const viewport of [
      { width: 1024, height: 768 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(50);
      const bounds = await page.evaluate(() => ({
        scroll: document.documentElement.scrollWidth,
        width: innerWidth,
        canvases: [...document.querySelectorAll("canvas")].map((c) => ({
          width: c.width,
          height: c.height,
          box: c.getBoundingClientRect().width,
        })),
      }));
      assert.ok(bounds.scroll <= bounds.width + 1);
      bounds.canvases.forEach((c) =>
        assert.ok(c.width > 0 && c.height > 0 && c.box > 100),
      );
      assert.deepEqual(await state(), initial);
      if (viewport.width === 390) await shot(name + "-mobile");
    }
    reports.push(name + " tablet and phone layout without state changes");
    await page.setViewportSize({ width: 1440, height: 1050 });
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  fs.writeFileSync(
    path.join(output, "browser-validation.json"),
    JSON.stringify(
      {
        browser: browser.version(),
        sandbox: true,
        checks: reports,
        pageErrors: errors,
        externalRequests: external,
      },
      null,
      2,
    ),
  );
  console.log(
    JSON.stringify({
      passed: reports.length,
      sandbox: true,
      pageErrors: errors.length,
      externalRequests: external.length,
    }),
  );
  await browser.close();
}
run().catch((error) => {
  console.error(error);
  process.exit(1);
});

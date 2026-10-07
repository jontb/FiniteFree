/* Optional development check: npm install --prefix /tmp/finitefree-browser playwright-core@1.63.0 */
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  { pathToFileURL } = require("node:url"),
  { execFileSync } = require("node:child_process");
const { chromium } = require("playwright-core");
const directory = path.resolve(process.argv[2] || "visuals/generated"),
  output = path.resolve(process.argv[3] || "visuals/generated/screenshots");
fs.mkdirSync(output, { recursive: true });
const close = (a, b, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) <= tol, `${a} != ${b}`);
const reports = [],
  errors = [],
  external = [],
  screenshotsTaken = [];
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
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (/^https?:/.test(r.url())) external.push(r.url());
  });
  const state = () => page.evaluate(() => window.explorerState);
  const slider = async (id, value) => {
    await page.locator("#" + id).evaluate((el, v) => {
      el.value = v;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, String(value));
  };
  const shot = async (name) => {
    screenshotsTaken.push(name + ".png");
    await page.screenshot({
      path: path.join(output, name + ".png"),
      fullPage: true,
    });
  };
  for (const name of ["hyperbolicity-cone", "moving-line-roots"]) {
    await page.goto(pathToFileURL(path.join(directory, name + ".html")).href);
    await page.waitForFunction(() => window.explorerState);
    if (name === "hyperbolicity-cone") {
      let s = await state();
      assert.equal(s.psd, true);
      close(s.det, 0.758);
      close(
        s.eigenvalues.reduce((a, b) => a + b),
        3,
      );
      await shot("01-cone-default");
      await page.locator("#chamber").click();
      s = await state();
      assert.equal(s.psd, false);
      assert.ok(s.det > 0);
      assert.equal(s.eigenvalues.filter((v) => v < 0).length, 2);
      await shot("02-cone-positive-det-indefinite");
      await page.locator("#boundary").click();
      s = await state();
      assert.equal(s.psd, true);
      close(s.eigenvalues[0], 0);
      close(s.det, 0);
      await shot("03-cone-psd-boundary");
      await slider("x", 0.99);
      assert.equal((await state()).psd, true);
      await slider("x", 1.01);
      assert.equal((await state()).psd, false);
      await slider("z", 1);
      await slider("x", 0.25);
      await slider("y", 0.25);
      assert.equal((await state()).psd, true);
      await slider("y", 0.26);
      assert.equal((await state()).psd, false);
      await slider("z", 1.4);
      await slider("x", 0);
      await slider("y", 0);
      assert.equal((await state()).psd, false);
      await page.locator("#reset").click();
      s = await state();
      assert.deepEqual(s.probe, [0.2, 0.4, 0.3]);
      const initial = await page.locator("#surface").screenshot();
      await page.locator("#surface").focus();
      await page.keyboard.press("ArrowRight");
      assert.notDeepEqual(await page.locator("#surface").screenshot(), initial);
      const box = await page.locator("#slice").boundingBox();
      await page.mouse.click(box.x + box.width * 0.6, box.y + box.height * 0.4);
      assert.notDeepEqual((await state()).probe, [0.2, 0.4, 0.3]);
      await page.locator("#reset").click();
      reports.push({
        example: name,
        checks: [
          "default PSD/eigenvalue trace",
          "positive determinant with two negative eigenvalues",
          "PSD boundary and near-boundary sides",
          "degenerate z=1 slice",
          "empty PSD slice z>1",
          "reset",
          "keyboard rotation",
          "linked slice click",
        ],
      });
    } else {
      let s = await state();
      assert.equal(s.interlacing, true);
      assert.ok(s.gap > 0 && s.residual < 1e-12);
      assert.deepEqual(s.exactCoefficients, ["1", "0", "-11/8", "0", "37/256"]);
      await shot("04-moving-avoided-crossings");
      await page.locator("#crossing").click();
      s = await state();
      assert.equal(s.z, 0);
      assert.equal(s.y, 0);
      assert.ok(s.roots.every((r) => Math.abs(r) < 1e-12));
      assert.ok(s.derivativeRoots.every((r) => Math.abs(r) < 1e-12));
      assert.equal(s.interlacing, true);
      await shot("05-moving-zero-coupling-crossing");
      await slider("parameter", 60);
      s = await state();
      assert.deepEqual(s.roots, [-3, -1, 1, 3]);
      for (const coupling of [1, 20, 40]) {
        await slider("coupling", coupling);
        for (const yi of [0, 39, 40, 41, 80]) {
          await slider("parameter", yi);
          s = await state();
          assert.ok(s.gap > 0);
          assert.equal(s.interlacing, true);
          assert.ok(s.residual < 1e-12);
        }
      }
      await page.locator("#derivative").uncheck();
      assert.equal((await state()).showDerivative, false);
      await page.locator("#reset").click();
      s = await state();
      close(s.y, 0.25);
      close(s.z, 0.5);
      assert.equal(s.showDerivative, true);
      await page.locator("#parameter").focus();
      await page.keyboard.press("ArrowRight");
      close((await state()).y, 0.3);
      const before = (await state()).y;
      await page.locator("#play").click();
      await page.waitForFunction((v) => window.explorerState.y !== v, before);
      await page.locator("#reset").click();
      await page.waitForTimeout(150);
      close((await state()).y, 0.25);
      assert.equal(await page.locator("#play").textContent(), "Sweep y");
      const box = await page.locator("#tracks").boundingBox();
      await page.mouse.click(box.x + box.width * 0.8, box.y + box.height * 0.5);
      assert.ok((await state()).y > 0.5);
      await page.locator("#reset").click();
      reports.push({
        example: name,
        checks: [
          "exact default coefficients",
          "four real roots and residual",
          "zero coupling repeated root/weak interlacing",
          "uncoupled roots at y=1",
          "minimum and maximum coupling at center/edges",
          "derivative toggle",
          "keyboard slider",
          "sweep/reset stops animation",
          "linked root-track click",
        ],
      });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(80);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await shot(
      name === "hyperbolicity-cone" ? "06-cone-mobile" : "07-moving-mobile",
    );
    for (const sliderId of name === "hyperbolicity-cone"
      ? ["x", "y", "z"]
      : ["parameter", "coupling"]) {
      const input = page.locator("#" + sliderId);
      await input.focus();
      await page.keyboard.press("ArrowLeft");
    }
    assert.ok(
      await page.evaluate(() =>
        Number.isFinite(
          window.explorerState.det ?? window.explorerState.residual,
        ),
      ),
    );
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.waitForTimeout(80);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await page.setViewportSize({ width: 1440, height: 1050 });
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  const receipt = {
    browser: browser.version(),
    executable,
    viewport_sizes: [
      [1440, 1050],
      [1024, 768],
      [390, 844],
    ],
    checks: reports,
    page_errors: errors,
    external_network_requests: external,
    screenshots: screenshotsTaken,
  };
  fs.writeFileSync(
    path.join(output, "browser-validation.json"),
    JSON.stringify(receipt, null, 2) + "\n",
  );
  console.log(JSON.stringify(receipt));
  await browser.close();
}
run().catch((e) => {
  console.error(e);
  process.exit(1);
});

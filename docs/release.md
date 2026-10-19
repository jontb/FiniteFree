# Release process

**FiniteFree 0.4.1 corrects the PyPI descriptions and project links.** It changes package metadata and documentation only; runtime source, defaults and dependency floors match 0.4.0. Production versions and files are listed on [PyPI](https://pypi.org/project/finitefree/); verify a new version there using the process below.

For the preceding 0.4.0 release, the [successful production workflow](https://github.com/jontb/FiniteFree/actions/runs/39194310960) built and tested 16 wheels and one source archive; their published hashes and fresh installations were verified.

The checklist below also governs future releases. A candidate build and a static version entry do not establish production publication: verify the approved tag, publishing workflow and PyPI files before promotion.

The 0.4.0 scope is optional ensemble RNG streams, prepared/batched orthogonal
kernels and reusable quadrature, finite-rank DPP count statistics, streaming
empirical summaries, and contributor/release validation improvements. Existing
exact/scalar paths, retained comparisons, legacy RNG defaults, mathematical
normalizations and runtime dependency floors remain unchanged. The 0.3.0
prepared-pencil and integrated GUE regressions remain part of validation.
GPU performance studies, rigorous rare-tail enclosures and new GPU algorithms
remain deferred research. Continuous count laws are quadrature estimates and
float64 tails have finite-resolution limits; the diagnostics do not certify
probability error bounds.

## Release promotion

Routine feature and maintenance pull requests target `develop`. `main` and its public documentation continue to describe the latest production PyPI release while the next release is prepared.

1. Prepare the intended version, changelog, migration notes and README/API documentation on `develop`. The package summary and `PYPI_README.md` supply the PyPI page; keep their links absolute and check the rendered description. PyPI cannot edit existing release metadata in place, so correcting it requires a new version. The version in `pyproject.toml` is static: a tag never changes it. Use a new version and confirm neither its tag nor its PyPI files already exist.
2. Validate the exact integrated candidate with CI and the artifact matrix below. A `release-review/**` branch can point to that candidate without publishing it. Record its commit, package hashes and completed checks, and obtain approval for that specific production release.
3. Create the approved stable `vX.Y.Z` tag at the validated candidate and let the existing publishing workflow complete. Inspect its production upload, PyPI metadata and file hashes. Install from production PyPI in a fresh environment outside the checkout, verify the native extension and mathematical smoke cases, and run `pip check`.
4. After production verification succeeds, open a `develop` → `main` pull request promoting the released source and its matching documentation. Wait for CI on the integrated pull request and main commit, and verify the public documentation deployment. Functional source on `main` must match the published tag; routine GitHub release notes should link that tag and package.
5. Merge any promotion merge commit or later stable maintenance from `main` back into `develop` so both branches share the release history before new work begins.

Publication precedes promotion to `main`. If publication or verification fails, keep the candidate on `develop` and investigate before promotion; do not move or replace a published tag or version. Documentation-only and workflow maintenance can be promoted between releases if it leaves the released functional API unchanged.

## Compatibility

0.4.0 retains Python 3.10 or later and `python-flint>=0.9.0` from 0.3.0. Testing the former Python 3.9 / FLINT 0.6 floor reproduced `AttributeError`: that backend does not expose `fmpq_mpoly_ctx`, needed for rational multivariate polynomials. Stable FLINT 0.7 and 0.8 require Python 3.11; FLINT 0.9 supports Python 3.10 and supplies this API. The exact supported runtime floors are recorded in `requirements/minimum-runtime.txt` and exercised separately in CI.

The symmetric S default remains the legacy even-coefficient ratio. Use `convention="standard"` for the positive-imaginary square root in Definition 8.1. See [the API contract](api.md#symmetric-finite-s-output-convention) and [all migration notes](https://github.com/jontb/FiniteFree/blob/develop/CHANGELOG.md), including cumulant normalization and snapshot ownership.

## Review checklist

Before publishing:

- Record the final release-candidate commit on `develop` and confirm CI on that exact commit. Require the 12 supported OS/Python combinations, pinned Python 3.10 runtime job and artifact-build job; checks on individual predecessor PRs do not validate the integrated tree.
- Confirm compiled and flag-fallback source suites, installed-wheel tests outside the checkout, Ruff/mypy, independent documentation examples and strict MkDocs. Record skips and platform-specific limits rather than treating configured jobs as passed.
- Build a wheel and sdist from the reviewed source with `python -m build`. Confirm both contain the intended static version, the reviewed Python and runtime dependency floors, the summary, `PYPI_README.md` description, project URLs and intended source files. Verify the wheel includes its compiled modular extension and can be installed and tested outside the checkout.
- Run `Build Release Artifacts Only` on the final commit and inspect all cibuildwheel jobs and uploaded wheels. It shares the publishing action and package policy, including the installed-wheel regression command. Ordinary source CI's Linux wheel/sdist build does not cover the production wheel matrix.
- Preserve archive SHA-256 hashes with the commit and validation logs. A local Linux wheel is a candidate, not the complete cross-platform PyPI wheel set.
- Review the published-package contents and migration notes, then approve the release action separately.

`build-release.yml` builds and tests packages and uploads GitHub Actions artifacts only. It has read-only repository permission, no publishing job, no publishing environment and no OIDC publishing permission. Use its manual dispatch, or push a `release-review/**` branch pointing to the reviewed release candidate to check that exact commit. Relevant pull requests also run it. These branches are not release tags.

The shared cibuildwheel policy installs GMP/MPFR headers and builds checksum-verified FLINT 3.3.1 in the disposable musllinux container. PyPA's supported Alpine image does not provide a FLINT development package, and python-flint 0.9 has no musllinux binary wheel. Testing those FiniteFree wheels therefore includes a source installation of python-flint. Musl users likewise need FLINT/GMP/MPFR source-build prerequisites even when FiniteFree itself has a wheel.

Linux build and test environments select compatible NumPy/SciPy binary wheels. This preserves the manylinux2014 glibc floor instead of attempting unsupported source builds of newer dependency releases. These container tests cover the resolved compatible versions; ordinary source CI separately tests current dependencies on current runners.

Each wheel build removes compiled extensions left by earlier Python/libc targets in the reused build checkout. Both workflows then require each wheel to contain exactly one modular extension matching its Python tag, alongside the installed-wheel regression suite.

The `publish.yml` workflow validates the static version and publishing destination
before starting any build. Tag pushes require an exact `vX.Y.Z` match to the
package version. Canonical `a`, `b`, `rc` and `.dev` prereleases route to TestPyPI;
only stable three-component versions route to production. Local, epoch and post
versions are outside this tag policy and fail preflight. Manual dispatch still
defaults to TestPyPI; explicitly selecting production requires a stable version.
A manually selected tag must also match the package version. The tag never changes
that version.

The review matrix and publisher both inspect archive names, version, Python and
runtime dependency floors, complete Python/Cython runtime source and native
extension inventories. The complete matrix requires 16 wheels and one sdist.
Their validation jobs save `artifact-manifest.json` and `SHA256SUMS` as workflow
artifacts. After the existing release-environment approval, the publishing job
rechecks the downloaded archive bytes against that manifest before upload.
Manifest files are kept outside the directory submitted to PyPI.

These local commands perform validation only, using the matching candidate
checkout and previously downloaded archives:

```bash
python -m scripts.release_preflight --event workflow_dispatch --ref refs/heads/develop
python -m scripts.check_release_artifacts dist --expected-wheels 16 --require-sdist --manifest dist/artifact-manifest.json
python -m scripts.check_release_artifacts downloaded --expected-wheels 16 --require-sdist --verify dist/artifact-manifest.json
```

`--manifest` writes hashes only after content validation succeeds; `--verify`
rejects missing, extra, duplicate or altered archives, then also checks them
against the checkout. Use `--expected-wheels 1` for an ordinary local wheel/sdist
build. Neither script publishes, tags or changes release settings. Creating a
stable release tag or invoking the publishing workflow remains a separate,
explicitly approved release action.

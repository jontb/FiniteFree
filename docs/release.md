# Release process

FiniteFree 0.2.0 is distributed on [PyPI](https://pypi.org/project/finitefree/0.2.0/) from [tag v0.2.0](https://github.com/jontb/FiniteFree/tree/v0.2.0). This checklist distinguishes a locally built candidate from a verified production upload and records the validation required before a release tag is created.

## Compatibility

0.2.0 requires Python 3.10 or later and `python-flint>=0.9.0`. Testing the former Python 3.9 / FLINT 0.6 floor reproduced `AttributeError`: that backend does not expose `fmpq_mpoly_ctx`, needed for rational multivariate polynomials. Stable FLINT 0.7 and 0.8 require Python 3.11; FLINT 0.9 supports Python 3.10 and supplies this API. The exact supported runtime floors are recorded in `requirements/minimum-runtime.txt` and exercised separately in CI.

The symmetric S default remains the legacy even-coefficient ratio. Use `convention="standard"` for the positive-imaginary square root in Definition 8.1. See [the API contract](api.md#symmetric-finite-s-output-convention) and [all migration notes](https://github.com/jontb/FiniteFree/blob/main/CHANGELOG.md), including cumulant normalization and snapshot ownership.

## Review checklist

Before publishing:

- Record the final main commit and confirm CI on that exact commit. Require the 12 supported OS/Python combinations, pinned Python 3.10 runtime job and artifact-build job; checks on individual predecessor PRs do not validate the integrated tree.
- Confirm compiled and flag-fallback source suites, installed-wheel tests outside the checkout, Ruff/mypy, independent documentation examples and strict MkDocs. Record skips and platform-specific limits rather than treating configured jobs as passed.
- Build a wheel and sdist from the reviewed source with `python -m build`. Confirm both contain version 0.2.0, `Requires-Python: >=3.10`, the four runtime floors, README metadata and the intended source files. Verify the wheel includes its compiled modular extension and can be installed and tested outside the checkout.
- Run `Build Release Artifacts Only` on the final commit and inspect all cibuildwheel jobs and uploaded wheels. It shares the publishing action and package policy, including the installed-wheel regression command. Ordinary source CI's Linux wheel/sdist build does not cover the production wheel matrix.
- Preserve archive SHA-256 hashes with the commit and validation logs. A local Linux wheel is a candidate, not the complete cross-platform PyPI wheel set.
- Review the published-package contents and migration notes, then approve the release action separately.

`build-release.yml` builds and tests packages and uploads GitHub Actions artifacts only. It has read-only repository permission, no publishing job, no publishing environment and no OIDC publishing permission. Use its manual dispatch, or push a `release-review/**` branch pointing to reviewed main to check that exact commit. Relevant pull requests also run it. These branches are not release tags.

The shared cibuildwheel policy installs GMP/MPFR headers and builds checksum-verified FLINT 3.3.1 in the disposable musllinux container. PyPA's supported Alpine image does not provide a FLINT development package, and python-flint 0.9 has no musllinux binary wheel. Testing those FiniteFree wheels therefore includes a source installation of python-flint. Musl users likewise need FLINT/GMP/MPFR source-build prerequisites even when FiniteFree itself has a wheel.

Linux build and test environments select compatible NumPy/SciPy binary wheels. This preserves the manylinux2014 glibc floor instead of attempting unsupported source builds of newer dependency releases. These container tests cover the resolved compatible versions; ordinary source CI separately tests current dependencies on current runners.

Each wheel build removes compiled extensions left by earlier Python/libc targets in the reused build checkout. Both workflows then require each wheel to contain exactly one modular extension matching its Python tag, alongside the installed-wheel regression suite.

The `publish.yml` workflow builds platform wheels with cibuildwheel and uploads through PyPI trusted publishing. A matching `v*.*.*` tag or manual workflow dispatch starts that upload. Creating `v0.2.0` is therefore a publication action, not a harmless preparation step. Do not create the tag or dispatch publishing before the release review is complete. Its manual input defaults to TestPyPI; choosing `false` selects production. Tag routing only checks for `rc` or `dev`: other matching prerelease names can reach production. The tag does not change the static package version in `pyproject.toml`.

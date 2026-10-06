#!/bin/sh
# Runs only in the disposable cibuildwheel musllinux build container.
set -eu

apk add --no-cache curl gmp-dev mpfr-dev
flint_build_dir=$(mktemp -d)
trap 'rm -rf "$flint_build_dir"' EXIT

curl --fail --location --retry 3 \
    https://github.com/flintlib/flint/releases/download/v3.3.1/flint-3.3.1.tar.gz \
    --output "$flint_build_dir/flint-3.3.1.tar.gz"
echo '64d70e513076cfa971e0410b58c1da5d35112913e9a56b44e2c681b459d3eafb  '"$flint_build_dir/flint-3.3.1.tar.gz" | sha256sum -c -
tar -xzf "$flint_build_dir/flint-3.3.1.tar.gz" -C "$flint_build_dir"
cd "$flint_build_dir/flint-3.3.1"
./configure --prefix=/usr --disable-static
make -j 4
make install
pkg-config --modversion flint

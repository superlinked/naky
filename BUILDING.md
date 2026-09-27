# Building

Näky pins Rust 1.98.0 and the `x86_64-unknown-linux-musl` target in
`rust-toolchain.toml`. The build also requires Python 3.11 or newer, GNU binutils, NASM,
and the musl C toolchain. On Ubuntu 24.04, install the native tools with:

```sh
sudo apt-get update
sudo apt-get install --yes --no-install-recommends binutils musl-tools nasm
```

Install Rust through rustup, then fetch the locked dependency set before
starting the offline build:

```sh
cargo fetch --locked
scripts/build-static
```

The helper authenticates the exported source, copies those exact bytes into an
isolated temporary source tree, and builds only that copy. It uses one Cargo
job, disables incremental compilation, and remaps the source, target, Cargo
home, and Rust sysroot paths. The `target/public-release` directory must not
exist when the build starts. Its output is
`target/public-release/x86_64-unknown-linux-musl/release/naky`. It also writes
`naky-build-receipt.json` beside the binary. That deterministic receipt binds
the binary digest to the complete source tree recorded in
`SOURCE-EXPORT.json` and records the Cargo and Rust compiler version outputs.
The helper normalizes the executable with authenticated GNU `strip` using
`--strip-all`, reproduces that transform twice, and records the unstripped
build, strip tool and version, exact arguments, and final release binary in
the receipt.
The helper refuses build-affecting compiler, linker, Cargo-profile, and target
environment overrides. `PATH` and `CARGO_HOME` are trusted host inputs; their
machine paths are not embedded in the binary.

The release archive is supported on 64-bit Linux with an x86 processor. The
result must be a static position-independent executable with no dynamic ELF
interpreter or `NEEDED` libraries. Release assembly authenticates every model
file against `manifests/ppocrv6-tiny-all-pp-bundle-v1.json` and records the
source-export receipt, source-tree digest, build receipt, binary digest,
model-member digests, and SBOM digest. Release assembly re-hashes every source
file and rejects unrecorded files, changed executable classification or special
permission bits, receipt drift, or a binary that differs from its build receipt.
The source exporter normalizes ordinary file modes to `0644` and executables to
`0755`.

Canonical input is AV1 in Matroska with 8-bit 4:2:0 video and monotonic,
nonnegative timestamps. `scripts/build-release --help` lists the required
static binary, model archive and extracted model directory, SPDX 2.3 JSON SBOM,
complete evaluation aggregate, equivalence receipt, and comparison page. The
helper authenticates and binds those inputs before writing a new output
directory. Pass the build receipt written by `scripts/build-static` with
`--build-receipt`. The helper refuses changed source or model bytes, symlinks,
and existing outputs.

The GitHub release publishes the authenticated model input as
`naky-model-ppocrv6-tiny-all-pp-v1.tar`; its digest is included in
`naky-0.1.0-SHA256SUMS`. Extract that asset to provide `--model-bundle`, and
pass the asset itself as `--model-archive` when reproducing release assembly.

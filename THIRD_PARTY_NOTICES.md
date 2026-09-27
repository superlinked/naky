# Third-party notices

Näky includes or links the following components. This notice is informational;
the referenced license texts govern each component.

- PP-OCRv6 Tiny detector and recognizer models, published by PaddlePaddle and
  converted to RTen: Apache-2.0. The publisher model cards do not disclose the
  training-data provenance. PaddlePaddle does not endorse Näky.
- `rav1d`: BSD-2-Clause.
- `matroska-demuxer`: Apache-2.0, selected from its Apache-2.0 OR MIT OR
  Zlib options.
- `ocrs`: Apache-2.0, selected from its Apache-2.0 OR MIT options.
- The RTen family (`rten`, `rten-base`, `rten-gemm`, `rten-imageproc`,
  `rten-model-file`, `rten-shape-inference`, `rten-simd`, `rten-tensor`, and
  `rten-vecmath`): Apache-2.0, selected from its Apache-2.0 OR MIT options.
- The bounded polygon-offset implementation derives from Clipper 6.4.2 by
  Angus Johnson (copyright 2010-2017, Boost-1.0) as distributed with pyclipper
  1.3.0.post6 (MIT). The applicable license texts are included in `licenses/`.
- `tikv-jemallocator` and `tikv-jemalloc-sys`: Apache-2.0, selected from
  their Apache-2.0 OR MIT options. Their embedded jemalloc source retains the
  separate terms in `licenses/jemalloc.txt`.
- `byteorder` and `memchr`: MIT, selected from their MIT OR Unlicense options.
- `generic-array`, `raw-cpuid`, `strsim`, `strum`, `strum_macros`,
  `unsafe-libyaml`, and `zmij`: MIT.
- `to_method`: CC0-1.0.
- `unicode-ident`: Apache-2.0 AND Unicode-3.0, selecting Apache-2.0 from the
  Rust portion's MIT OR Apache-2.0 options.

The release SBOM lists the complete shipped `x86_64-unknown-linux-musl` Rust
normal/build dependency graph and exact versions. Näky uses the Apache-2.0
option for Rust dependencies that offer it; the Apache-2.0 text is distributed
as `LICENSE`. Additional applicable license and notice texts, including the
texts for dependencies without an Apache-2.0 option, are distributed in
`licenses/`.

The locked source graph also contains `redox_syscall` 0.5.18 for non-Linux
targets. It is not linked into the Linux release; its exact MIT text is included
in `licenses/redox-syscall-MIT.txt` for conservative lockfile-wide coverage.

# trustformers-c: DEPRECATED

**Status: DEPRECATED as of the 0.2.0 release.**

This crate receives no further feature development or maintenance. No new
TODO items will be tracked here going forward.

## Rationale

- The C FFI surface exposed by this crate has been superseded. The workspace
  is consolidating on the pure-Rust core (`trustformers-core`, `trustformers`)
  plus dedicated binding surfaces for cross-language integration
  (`trustformers-wasm` for WebAssembly/WebGPU, and equivalent Python/JS
  binding crates), rather than maintaining a hand-written C ABI in parallel.
- Keeping a C ABI in sync with the fast-moving core API has an increasing
  maintenance cost relative to its usage; downstream consumers needing
  cross-language access should target the WASM bindings or a native Rust
  dependency instead.

## Migration guidance

- Rust consumers: depend on `trustformers-core` / `trustformers` directly.
- Browser/Node/edge consumers: use `trustformers-wasm`.
- Other language bindings: prefer language-native binding crates
  (e.g. Python/JS bindings) once available, rather than linking against
  `trustformers-c`.

## Superseded work

Any 0.2.0 workspace tasks previously tracked in this file — including
retirement of the legacy `cudarc` backend, `scirs2` feature-flag hygiene,
and torch passthrough handling — are superseded by this deprecation and will
not be carried forward for this crate.

//! Compiles `proto/inference.proto` into the `trustformers.inference` Rust
//! module consumed by `src/service.rs` via `tonic::include_proto!`.
//!
//! tonic 0.14 split code generation out of `tonic-build` into the dedicated
//! `tonic-prost-build` crate, and renamed the finalizer from `compile` to
//! `compile_protos`. This mirrors `trustformers-serve/build.rs`.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        // The encoded descriptor set is what `tonic-reflection` serves to
        // grpcurl / grpcui; `src/main.rs` embeds it with `include_bytes!`.
        .file_descriptor_set_path(out_dir.join("inference_descriptor.bin"))
        .compile_protos(&["proto/inference.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/inference.proto");

    Ok(())
}

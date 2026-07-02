fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the gRPC service definitions with tonic 0.14.
    //
    // Tonic 0.14 split code generation out of `tonic-build` into the dedicated
    // `tonic-prost-build` crate. The builder entry point is
    // `tonic_prost_build::configure()`, which yields a `Builder` exposing
    // `build_server` / `build_client` toggles and a `compile_protos(protos,
    // includes)` finalizer. The generated module is consumed at runtime via
    // `tonic::include_proto!("trustformers.serve.v1")` in `src/grpc.rs`.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/inference.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/inference.proto");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Proto compilation requires tonic-build API update
    // The tonic-build/tonic-prost-build API has changed significantly in 0.14
    // Need to investigate the correct builder pattern or use alternative approach
    //
    // Options to investigate:
    // 1. Use prost_build::Config directly with tonic extensions
    // 2. Check if there's a different entry point in tonic-build 0.14
    // 3. Consider using pre-generated proto files
    //
    // For now, skip proto compilation to unblock other development
    // Note: Proto compilation requires tonic-build 0.14 API migration (documented above)
    println!("cargo:rerun-if-changed=proto/inference.proto");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Skip proto compilation for now as tonic-build API has changed
    // TODO: Update to proper tonic-build 0.14 API
    println!("cargo:warning=Proto compilation skipped - requires manual generation");
    Ok(())
}

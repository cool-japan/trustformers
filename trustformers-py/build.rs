fn main() {
    // Configure PyO3 to find Python correctly on macOS
    pyo3_build_config::add_extension_module_link_args();
}

fn main() {
    // For cross-compilation, we use abi3 which doesn't require Python linking
    // For native macOS builds, we still configure Python linking
    if cfg!(target_os = "macos") && !is_cross_compiling() {
        // Direct link to the specific Python library
        let python_lib_path =
            "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/lib";
        let _python_lib = "libpython3.14.dylib";

        // Add library search path
        println!("cargo:rustc-link-search=native={}", python_lib_path);

        // Link to the Python library directly
        println!("cargo:rustc-link-lib=python3.14");

        // Also add the framework path
        println!("cargo:rustc-link-arg=-F/opt/homebrew/opt/python@3.14/Frameworks");
    }
}

fn is_cross_compiling() -> bool {
    // Check if we're cross compiling by checking for cross compilation env vars
    std::env::var("CARGO_CFG_TARGET_ARCH").is_ok()
        && std::env::var("CARGO_CFG_TARGET_OS").is_ok()
        && (std::env::var("TARGET").unwrap_or_default() != std::env::var("HOST").unwrap_or_default())
}

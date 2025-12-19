fn main() {
    // Configure Python linking for macOS Homebrew Python
    if cfg!(target_os = "macos") {
        // Direct link to the specific Python library
        let python_lib_path = "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/lib";
        let python_lib = "libpython3.14.dylib";

        // Add library search path
        println!("cargo:rustc-link-search=native={}", python_lib_path);

        // Link to the Python library directly
        println!("cargo:rustc-link-lib=python3.14");

        // Also add the framework path
        println!("cargo:rustc-link-arg=-F/opt/homebrew/opt/python@3.14/Frameworks");
    }
}
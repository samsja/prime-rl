// Tests embed Python (pyo3 `auto-initialize`). Link them against the build interpreter's
// libpython and hand them its home: uv's standalone builds don't locate their stdlib when embedded.
fn main() {
    let config = pyo3_build_config::get();
    if let Some(lib_dir) = &config.lib_dir {
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{lib_dir}");
    }
    if let Some(executable) = &config.executable {
        let output = std::process::Command::new(executable)
            .args(["-c", "import sys; print(sys.base_prefix, end='')"])
            .output()
            .expect("failed to run the build interpreter");
        let home = String::from_utf8(output.stdout).expect("non-UTF-8 sys.base_prefix");
        println!("cargo:rustc-env=PRL_TEST_PYTHONHOME={home}");
    }
}

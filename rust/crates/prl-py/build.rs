// Tests embed Python (pyo3 `auto-initialize`). Link them against the build interpreter's
// libpython and hand them its home (uv's standalone builds don't locate their stdlib when embedded)
// and its site-packages (the project venv), see `prime_rl_orch_rs::init_test_python`.
fn main() {
    let config = pyo3_build_config::get();
    if let Some(lib_dir) = &config.lib_dir {
        // Every target, so the lib's unit-test binary finds libpython too (`-tests` covers only
        // integration tests). Harmless for the extension module, which does not link libpython.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    if let Some(executable) = &config.executable {
        let output = std::process::Command::new(executable)
            .args(["-c", "import os, site, sys; print(sys.base_prefix, os.pathsep.join(site.getsitepackages()), sep='\\n', end='')"])
            .output()
            .expect("failed to run the build interpreter");
        let output = String::from_utf8(output.stdout).expect("non-UTF-8 interpreter paths");
        let (home, site_packages) = output.split_once('\n').expect("interpreter printed no site-packages");
        println!("cargo:rustc-env=PRL_TEST_PYTHONHOME={home}");
        println!("cargo:rustc-env=PRL_TEST_PYTHONPATH={site_packages}");
    }
}

use std::process::Command;

#[test]
fn release_binary_exposes_only_transcode() {
    let output = Command::new(env!("CARGO_BIN_EXE_naky"))
        .arg("--help")
        .output()
        .expect("run naky --help");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 help");
    assert!(stdout.contains("Usage: naky <COMMAND>"));
    assert!(stdout.contains("transcode"));

    let transcode = Command::new(env!("CARGO_BIN_EXE_naky"))
        .args(["transcode", "--help"])
        .output()
        .expect("run naky transcode --help");
    assert!(transcode.status.success());
    let transcode_stdout = String::from_utf8(transcode.stdout).expect("UTF-8 help");
    for option in ["--input", "--output-dir", "--stream-id", "--model-bundle"] {
        assert!(
            transcode_stdout.contains(option),
            "missing option: {option}"
        );
    }
}

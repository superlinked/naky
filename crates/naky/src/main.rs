use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod all_pp_pipeline;

#[derive(Debug, Parser)]
#[command(
    name = "naky",
    version,
    about = "Convert AV1 screen recordings into ScreenEvents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert a canonical AV1 Matroska recording into ScreenEvents and bounded text.
    Transcode {
        /// Canonical AV1 Matroska input.
        #[arg(long)]
        input: PathBuf,
        /// New directory for events.ndjson, screen.txt, and metrics.json.
        #[arg(long)]
        output_dir: PathBuf,
        /// Stable source identifier stored in emitted events.
        #[arg(long, default_value = "screen")]
        stream_id: String,
        /// Authenticated model bundle. Defaults to the bundle installed beside naky.
        #[arg(long)]
        model_bundle: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Transcode {
            input,
            output_dir,
            stream_id,
            model_bundle,
        } => {
            let model_bundle = model_bundle.map_or_else(default_model_bundle, Ok)?;
            let outputs = create_output_dir(&output_dir)?;
            let result = all_pp_pipeline::run(all_pp_pipeline::Options {
                input: &input,
                output: Some(&outputs.events),
                stateful_text_output: Some(&outputs.screen),
                stream_id: &stream_id,
                model_bundle: &model_bundle,
                metrics_output: Some(&outputs.metrics),
            });
            if result.is_err() {
                remove_incomplete_outputs(&outputs).with_context(|| {
                    format!(
                        "failed to clean incomplete output: {}",
                        output_dir.display()
                    )
                })?;
            }
            result
        }
    }
}

fn remove_incomplete_outputs(outputs: &OutputPaths) -> Result<()> {
    for path in [&outputs.events, &outputs.screen, &outputs.metrics] {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    fs::remove_dir(
        outputs
            .events
            .parent()
            .context("output path has no parent directory")?,
    )?;
    Ok(())
}

struct OutputPaths {
    events: PathBuf,
    screen: PathBuf,
    metrics: PathBuf,
}

fn create_output_dir(path: &Path) -> Result<OutputPaths> {
    fs::create_dir(path)
        .with_context(|| format!("output directory must be new: {}", path.display()))?;
    Ok(OutputPaths {
        events: path.join("events.ndjson"),
        screen: path.join("screen.txt"),
        metrics: path.join("metrics.json"),
    })
}

fn default_model_bundle() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("failed to locate the naky executable")?;
    model_bundle_beside(&executable)
}

fn model_bundle_beside(executable: &Path) -> Result<PathBuf> {
    let bin = executable
        .parent()
        .context("naky executable has no parent directory")?;
    let prefix = bin
        .parent()
        .context("naky executable has no installation prefix")?;
    Ok(prefix.join("share/naky/model"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn public_cli_contract_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn public_cli_exposes_only_transcode() {
        let root = Cli::command();
        let commands: Vec<_> = root
            .get_subcommands()
            .map(|command| command.get_name())
            .collect();
        assert_eq!(commands, ["transcode"]);
    }

    #[test]
    fn installed_model_bundle_is_relative_to_prefix() {
        assert_eq!(
            model_bundle_beside(Path::new("/opt/naky/bin/naky")).unwrap(),
            Path::new("/opt/naky/share/naky/model")
        );
    }

    #[test]
    fn output_directory_has_fixed_public_files_and_must_be_new() {
        let output = std::env::temp_dir().join(format!(
            "naky-release-cli-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let paths = create_output_dir(&output).unwrap();
        assert_eq!(paths.events, output.join("events.ndjson"));
        assert_eq!(paths.screen, output.join("screen.txt"));
        assert_eq!(paths.metrics, output.join("metrics.json"));
        assert!(create_output_dir(&output).is_err());
        fs::remove_dir(&output).unwrap();
    }
}

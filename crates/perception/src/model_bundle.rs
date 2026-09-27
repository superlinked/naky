//! Exact offline authentication for the one product PP-OCRv6 model bundle.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use ppocrv6_tiny_preflight::{Dictionary, Identity, sha256_bytes};
use serde::{Deserialize, Serialize};

const MANIFEST_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../manifests/ppocrv6-tiny-all-pp-bundle-v1.json"
));
const SCHEMA_VERSION: &str = "naky.model-bundle.v1";
const BUNDLE_ID: &str = "ppocrv6-tiny-all-pp-v1";
const ARCHIVE_FORMAT: &str = "posix-ustar";
const DETECTOR_PATH: &str = "models/detector.rten";
const RECOGNIZER_PATH: &str = "models/recognizer.rten";
const YAML_PATH: &str = "config/recognizer-inference.yml";

const MEMBER_CONTRACT: [(&str, &str); 7] = [
    ("license", "LICENSES/Apache-2.0.txt"),
    ("notice", "THIRD_PARTY_NOTICES.md"),
    ("recognizer-config", YAML_PATH),
    ("detector-model", DETECTOR_PATH),
    ("recognizer-model", RECOGNIZER_PATH),
    ("detector-model-card", "provenance/detector-model-card.md"),
    (
        "recognizer-model-card",
        "provenance/recognizer-model-card.md",
    ),
];

#[derive(Clone, Debug, Serialize)]
pub struct ModelBundleIdentity {
    pub bundle_id: String,
    pub manifest: Identity,
    pub archive: Identity,
    pub detector_model: Identity,
    pub recognizer_model: Identity,
    pub inference_yaml: Identity,
    pub dictionary_sha256: String,
}

pub(crate) struct AuthenticatedModelBundle {
    pub identity: ModelBundleIdentity,
    pub detector_model: Vec<u8>,
    pub recognizer_model: Vec<u8>,
    pub inference_yaml: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    bundle_id: String,
    archive: Archive,
    members: Vec<Member>,
    #[serde(rename = "sources")]
    _sources: serde_json::Value,
    #[serde(rename = "conversion")]
    _conversion: serde_json::Value,
    dictionary: DictionaryContract,
    #[serde(rename = "licensing")]
    _licensing: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Archive {
    format: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    role: String,
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DictionaryContract {
    source: String,
    ordered_unique_entries: u64,
    canonical_bytes: u64,
    canonical_sha256: String,
    append_space: bool,
    decoded_labels: u64,
    ctc_blank_class: u64,
    model_output_classes: u64,
}

pub(crate) fn load_product_bundle(root: &Path) -> Result<AuthenticatedModelBundle> {
    let (manifest, mut members, manifest_identity) =
        authenticate_directory_with_manifest(root, MANIFEST_BYTES)?;
    let inference_yaml = members
        .remove(YAML_PATH)
        .context("authenticated bundle omitted recognizer YAML")?;
    Dictionary::from_yaml(&inference_yaml).context("invalid authenticated bundle dictionary")?;
    let detector_model = members
        .remove(DETECTOR_PATH)
        .context("authenticated bundle omitted detector model")?;
    let recognizer_model = members
        .remove(RECOGNIZER_PATH)
        .context("authenticated bundle omitted recognizer model")?;
    let identities = manifest
        .members
        .iter()
        .map(|member| {
            (
                member.path.as_str(),
                Identity {
                    bytes: member.bytes,
                    sha256: member.sha256.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    Ok(AuthenticatedModelBundle {
        identity: ModelBundleIdentity {
            bundle_id: manifest.bundle_id,
            manifest: manifest_identity,
            archive: Identity {
                bytes: manifest.archive.bytes,
                sha256: manifest.archive.sha256,
            },
            detector_model: identities[DETECTOR_PATH].clone(),
            recognizer_model: identities[RECOGNIZER_PATH].clone(),
            inference_yaml: identities[YAML_PATH].clone(),
            dictionary_sha256: manifest.dictionary.canonical_sha256,
        },
        detector_model,
        recognizer_model,
        inference_yaml,
    })
}

fn authenticate_directory_with_manifest(
    root: &Path,
    manifest_bytes: &[u8],
) -> Result<(Manifest, BTreeMap<String, Vec<u8>>, Identity)> {
    let manifest: Manifest =
        serde_json::from_slice(manifest_bytes).context("invalid embedded model-bundle manifest")?;
    validate_manifest(&manifest)?;
    validate_directory(root, &manifest.members)?;
    let mut authenticated = BTreeMap::new();
    for member in &manifest.members {
        let bytes = read_member(root, member)?;
        if authenticated.insert(member.path.clone(), bytes).is_some() {
            bail!("duplicate model-bundle member {}", member.path);
        }
    }
    let manifest_identity = Identity {
        bytes: u64::try_from(manifest_bytes.len()).context("manifest size exceeds u64")?,
        sha256: sha256_bytes(manifest_bytes),
    };
    Ok((manifest, authenticated, manifest_identity))
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != SCHEMA_VERSION || manifest.bundle_id != BUNDLE_ID {
        bail!("unsupported model-bundle manifest identity");
    }
    validate_sha256(&manifest.archive.sha256, "archive")?;
    if manifest.archive.format != ARCHIVE_FORMAT || manifest.archive.bytes == 0 {
        bail!("invalid model-bundle archive contract");
    }
    if manifest.members.len() != MEMBER_CONTRACT.len() {
        bail!("model-bundle manifest must contain exactly seven members");
    }
    let mut roles = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for (member, (role, path)) in manifest.members.iter().zip(MEMBER_CONTRACT) {
        validate_relative_path(&member.path)?;
        validate_sha256(&member.sha256, &member.path)?;
        if member.bytes == 0
            || member.role != role
            || member.path != path
            || !roles.insert(&member.role)
            || !paths.insert(&member.path)
        {
            bail!("model-bundle member contract differs at {}", member.path);
        }
    }
    validate_dictionary_contract(&manifest.dictionary)?;
    Ok(())
}

fn validate_dictionary_contract(dictionary: &DictionaryContract) -> Result<()> {
    if dictionary.source != "config/recognizer-inference.yml:PostProcess.character_dict"
        || dictionary.ordered_unique_entries != 6_904
        || dictionary.canonical_bytes != 27_156
        || dictionary.canonical_sha256
            != "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd"
        || !dictionary.append_space
        || dictionary.decoded_labels != 6_905
        || dictionary.ctc_blank_class != 0
        || dictionary.model_output_classes != 6_906
    {
        bail!("recognizer dictionary contract differs");
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<()> {
    let parsed = Path::new(path);
    if parsed.as_os_str().is_empty()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("model-bundle member path is not a clean relative path: {path}");
    }
    Ok(())
}

fn validate_sha256(value: &str, artifact: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        bail!("invalid SHA-256 for {artifact}");
    }
    Ok(())
}

fn validate_directory(root: &Path, members: &[Member]) -> Result<()> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("failed to inspect model bundle {}", root.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!(
            "model bundle root is not a regular directory: {}",
            root.display()
        );
    }
    let expected_files = members
        .iter()
        .map(|member| PathBuf::from(&member.path))
        .collect::<BTreeSet<_>>();
    let expected_directories = expected_files
        .iter()
        .flat_map(|path| path.ancestors().skip(1))
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .collect::<BTreeSet<_>>();
    let mut found_files = BTreeSet::new();
    let mut found_directories = BTreeSet::new();
    visit_directory(
        root,
        Path::new(""),
        &expected_files,
        &expected_directories,
        &mut found_files,
        &mut found_directories,
    )?;
    if found_files != expected_files || found_directories != expected_directories {
        bail!("model bundle member set differs from the embedded manifest");
    }
    Ok(())
}

fn visit_directory(
    root: &Path,
    relative: &Path,
    expected_files: &BTreeSet<PathBuf>,
    expected_directories: &BTreeSet<PathBuf>,
    found_files: &mut BTreeSet<PathBuf>,
    found_directories: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    let directory = root.join(relative);
    let mut entries = fs::read_dir(&directory)
        .with_context(|| {
            format!(
                "failed to read model bundle directory {}",
                directory.display()
            )
        })?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let child = relative.join(entry.file_name());
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            bail!("model bundle contains a symlink: {}", child.display());
        }
        if metadata.is_dir() {
            if !expected_directories.contains(&child) || !found_directories.insert(child.clone()) {
                bail!(
                    "model bundle contains an unexpected directory: {}",
                    child.display()
                );
            }
            visit_directory(
                root,
                &child,
                expected_files,
                expected_directories,
                found_files,
                found_directories,
            )?;
        } else if metadata.is_file() {
            if !expected_files.contains(&child) || !found_files.insert(child.clone()) {
                bail!(
                    "model bundle contains an unexpected file: {}",
                    child.display()
                );
            }
        } else {
            bail!(
                "model bundle contains a non-regular entry: {}",
                child.display()
            );
        }
    }
    Ok(())
}

fn read_member(root: &Path, member: &Member) -> Result<Vec<u8>> {
    let path = root.join(&member.path);
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("missing model-bundle member {}", member.path))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != member.bytes {
        bail!("model-bundle member metadata differs: {}", member.path);
    }
    let capacity = usize::try_from(member.bytes).context("bundle member exceeds usize")?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(&path)?
        .take(member.bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    let identity = Identity {
        bytes: u64::try_from(bytes.len()).context("bundle member size exceeds u64")?,
        sha256: sha256_bytes(&bytes),
    };
    if identity.bytes != member.bytes || identity.sha256 != member.sha256 {
        bail!(
            "model-bundle member identity differs for {}: got {} bytes {}",
            member.path,
            identity.bytes,
            identity.sha256
        );
    }
    if fs::symlink_metadata(&path)?.file_type().is_symlink() {
        bail!("model-bundle member became a symlink: {}", member.path);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::Value;

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "naky-model-bundle-test-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> (TestDirectory, Vec<u8>) {
        let root = TestDirectory::new();
        let mut manifest: Value = serde_json::from_slice(MANIFEST_BYTES).unwrap();
        for member in manifest["members"].as_array_mut().unwrap() {
            let path = member["path"].as_str().unwrap();
            let bytes = format!("fixture:{path}\n").into_bytes();
            let target = root.0.join(path);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::write(target, &bytes).unwrap();
            member["bytes"] = Value::from(bytes.len());
            member["sha256"] = Value::from(sha256_bytes(&bytes));
        }
        (root, serde_json::to_vec(&manifest).unwrap())
    }

    #[test]
    fn embedded_manifest_and_exact_member_set_authenticate() {
        let (root, manifest) = fixture();
        let (_, members, _) = authenticate_directory_with_manifest(&root.0, &manifest).unwrap();
        assert_eq!(members.len(), MEMBER_CONTRACT.len());
    }

    #[test]
    fn archive_transport_metadata_is_rejected() {
        let (root, manifest) = fixture();
        let mut manifest: Value = serde_json::from_slice(&manifest).unwrap();
        let transport_key = ["content", "addressed", "uri"].join("_");
        manifest["archive"][transport_key] = Value::from("https://example.invalid/model.tar");
        assert!(
            authenticate_directory_with_manifest(&root.0, &serde_json::to_vec(&manifest).unwrap())
                .is_err()
        );
    }

    #[test]
    fn missing_extra_tampered_and_symlink_members_fail() {
        let (root, manifest) = fixture();
        fs::remove_file(root.0.join(DETECTOR_PATH)).unwrap();
        assert!(authenticate_directory_with_manifest(&root.0, &manifest).is_err());

        let (root, manifest) = fixture();
        fs::write(root.0.join("extra"), b"extra").unwrap();
        assert!(authenticate_directory_with_manifest(&root.0, &manifest).is_err());

        let (root, manifest) = fixture();
        fs::write(root.0.join(RECOGNIZER_PATH), b"tamper").unwrap();
        assert!(authenticate_directory_with_manifest(&root.0, &manifest).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let (root, manifest) = fixture();
            let member = root.0.join(DETECTOR_PATH);
            fs::remove_file(&member).unwrap();
            symlink(root.0.join(RECOGNIZER_PATH), member).unwrap();
            assert!(authenticate_directory_with_manifest(&root.0, &manifest).is_err());
        }
    }

    #[test]
    fn unknown_duplicate_and_traversal_manifest_contracts_fail() {
        let (root, manifest) = fixture();
        let mut value: Value = serde_json::from_slice(&manifest).unwrap();
        value["unknown"] = Value::Bool(true);
        assert!(
            authenticate_directory_with_manifest(&root.0, &serde_json::to_vec(&value).unwrap())
                .is_err()
        );

        let (root, manifest) = fixture();
        let mut value: Value = serde_json::from_slice(&manifest).unwrap();
        value["members"][1]["role"] = value["members"][0]["role"].clone();
        value["members"][1]["path"] = value["members"][0]["path"].clone();
        assert!(
            authenticate_directory_with_manifest(&root.0, &serde_json::to_vec(&value).unwrap())
                .is_err()
        );

        let (root, manifest) = fixture();
        let mut value: Value = serde_json::from_slice(&manifest).unwrap();
        value["members"][0]["path"] = Value::from("../outside");
        assert!(
            authenticate_directory_with_manifest(&root.0, &serde_json::to_vec(&value).unwrap())
                .is_err()
        );
    }
}

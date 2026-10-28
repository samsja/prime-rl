//! `torch.save`'s zip container (no tensors): `archive/data.pkl` plus the version records.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use zip::write::SimpleFileOptions;
use zip::CompressionMethod;

const ARCHIVE: &str = "archive";

/// Write `pickle` the way `torch.save(obj, f)` lays it out: stored entries, 64-byte aligned.
pub fn write(path: &Path, pickle: &[u8]) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored).with_alignment(64);
    let serialization_id = format!("{:020}{:020}", rand::random::<u64>() % 10u64.pow(19), rand::random::<u64>() % 10u64.pow(19));
    let records: [(&str, &[u8]); 6] = [
        ("data.pkl", pickle),
        (".format_version", b"1"),
        (".storage_alignment", b"64"),
        ("byteorder", b"little"),
        ("version", b"3\n"),
        (".data/serialization_id", serialization_id.as_bytes()),
    ];
    for (name, data) in records {
        zip.start_file(format!("{ARCHIVE}/{name}"), options)?;
        zip.write_all(data)?;
    }
    zip.finish()?;
    Ok(())
}

/// The `data.pkl` bytes of a `torch.save` zip (any archive prefix).
pub fn read(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(file).with_context(|| format!("{} is not a torch zip checkpoint", path.display()))?;
    let name = zip
        .file_names()
        .find(|name| name.ends_with("/data.pkl") || *name == "data.pkl")
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("{} has no data.pkl record", path.display()))?;
    let mut data = Vec::new();
    zip.by_name(&name)?.read_to_end(&mut data)?;
    Ok(data)
}

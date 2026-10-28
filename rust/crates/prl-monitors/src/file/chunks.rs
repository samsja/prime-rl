//! Chunked JSONL (`monitors/file/traces/chunks.py`): an append-only stream split into numbered
//! files, sealed with zstd (seekable format) as it grows.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use anyhow::{Context, Result};

/// zstd level of sealed chunks.
pub const LEVEL: i32 = 3;
/// Uncompressed bytes per independent frame of a sealed chunk.
pub const FRAME_BYTES: usize = 4 << 20;

const SKIPPABLE_MAGIC: u32 = 0x184D_2A5E;
const SEEKABLE_MAGIC: u32 = 0x8F92_EAB1;

pub fn chunk_path(directory: &Path, number: u64) -> PathBuf {
    directory.join(format!("{number:05}.jsonl"))
}

/// Chunk number -> whether its plain file is present (a sealed chunk has none).
pub fn chunk_numbers(directory: &Path) -> Result<BTreeMap<u64, bool>> {
    let mut numbers = BTreeMap::new();
    for entry in fs::read_dir(directory)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if !name.contains(".jsonl") {
            continue;
        }
        let Some((stem, _)) = name.split_once('.') else {
            continue;
        };
        let number: u64 = stem
            .parse()
            .with_context(|| format!("chunk file name {name}"))?;
        let plain = name.ends_with(".jsonl");
        let present = numbers.entry(number).or_insert(false);
        *present = *present || plain;
    }
    Ok(numbers)
}

/// Compress `data` into the zstd seekable format, as `pyzstd.SeekableZstdFile` writes it:
/// one frame per `FRAME_BYTES` of input, then the seek table (no checksums).
pub fn seekable_compress(src: &mut impl Read, dst: &mut impl Write) -> Result<()> {
    let mut frames: Vec<(u32, u32)> = Vec::new();
    let mut buf = vec![0u8; FRAME_BYTES];
    loop {
        let mut filled = 0;
        while filled < FRAME_BYTES {
            let n = src.read(&mut buf[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), LEVEL)?;
        for piece in buf[..filled].chunks(1 << 20) {
            encoder.write_all(piece)?;
        }
        let frame = encoder.finish()?;
        dst.write_all(&frame)?;
        frames.push((frame.len() as u32, filled as u32));
        if filled < FRAME_BYTES {
            break;
        }
    }
    let size = 17 + 8 * frames.len() as u32;
    let mut table = Vec::with_capacity(size as usize);
    table.extend_from_slice(&SKIPPABLE_MAGIC.to_le_bytes());
    table.extend_from_slice(&(size - 8).to_le_bytes());
    for (c, d) in &frames {
        table.extend_from_slice(&c.to_le_bytes());
        table.extend_from_slice(&d.to_le_bytes());
    }
    table.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    table.push(0);
    table.extend_from_slice(&SEEKABLE_MAGIC.to_le_bytes());
    dst.write_all(&table)?;
    Ok(())
}

/// Compress a finished chunk beside itself, then drop the plain file.
pub fn seal(path: &Path) -> Result<()> {
    let target = PathBuf::from(format!("{}.zst", path.display()));
    let partial = PathBuf::from(format!("{}.tmp", target.display()));
    {
        let mut src = File::open(path)?;
        let mut dst = std::io::BufWriter::new(File::create(&partial)?);
        seekable_compress(&mut src, &mut dst)?;
        dst.flush()?;
    }
    fs::rename(&partial, &target)?;
    fs::remove_file(path)?;
    Ok(())
}

/// Appends lines to the live chunk of a stream and reports where each one landed.
pub struct ChunkedJsonl {
    pub directory: PathBuf,
    max_bytes: u64,
    compress: bool,
    pub number: u64,
    file: File,
    size: u64,
    sealing: Vec<JoinHandle<Result<()>>>,
}

fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))
}

impl ChunkedJsonl {
    pub fn open(directory: &Path, max_bytes: u64, compress: bool) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let numbers = chunk_numbers(directory)?;
        let last = numbers.keys().next_back().copied();
        // A relaunch appends to the live chunk it finds and starts a new one after a sealed
        // last chunk; anything else still plain was left mid-seal by a crash.
        let number = match last {
            Some(last) if numbers[&last] => last,
            Some(last) => last + 1,
            None => 0,
        };
        let mut stream = ChunkedJsonl {
            directory: directory.to_path_buf(),
            max_bytes,
            compress,
            number,
            file: open_append(&chunk_path(directory, number))?,
            size: 0,
            sealing: Vec::new(),
        };
        stream.size = stream.file.metadata()?.len();
        for (&n, &plain) in &numbers {
            if plain && n != number && compress {
                stream.seal_background(n);
            }
        }
        Ok(stream)
    }

    /// Write one line; returns `(chunk, offset)`, the address a reader seeks to.
    pub fn append(&mut self, line: &[u8]) -> Result<(u64, u64)> {
        if self.size > 0 && self.size + line.len() as u64 > self.max_bytes {
            self.roll()?;
        }
        let offset = self.size;
        self.file.write_all(line)?;
        self.size += line.len() as u64;
        Ok((self.number, offset))
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }

    /// Seal the live chunk too, so a finished run keeps nothing plain.
    pub fn close(mut self) -> Result<()> {
        self.file.flush()?;
        let path = chunk_path(&self.directory, self.number);
        if self.size == 0 {
            fs::remove_file(&path)?;
        } else if self.compress {
            self.seal_background(self.number);
        }
        self.join_sealing()
    }

    fn join_sealing(&mut self) -> Result<()> {
        for handle in self.sealing.drain(..) {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("chunk sealing thread panicked"))??;
        }
        Ok(())
    }

    fn roll(&mut self) -> Result<()> {
        self.file.flush()?;
        if self.compress {
            self.seal_background(self.number);
        }
        self.number += 1;
        self.file = open_append(&chunk_path(&self.directory, self.number))?;
        self.size = 0;
        Ok(())
    }

    fn seal_background(&mut self, number: u64) {
        let path = chunk_path(&self.directory, number);
        self.sealing.push(std::thread::spawn(move || seal(&path)));
    }
}

impl Drop for ChunkedJsonl {
    /// Like the Python writer's non-daemon threads: never leave a torn sealed twin behind.
    fn drop(&mut self) {
        for handle in self.sealing.drain(..) {
            let _ = handle.join();
        }
    }
}

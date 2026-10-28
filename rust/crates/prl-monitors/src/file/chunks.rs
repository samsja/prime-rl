//! Chunked JSONL (`monitors/file/traces/chunks.py`): an append-only stream split into numbered
//! files, each sealed with zstd in the seekable format once the stream rolls past it.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use anyhow::{anyhow, Context, Result};

use super::open_append;

/// zstd level of sealed chunks.
const LEVEL: i32 = 3;
/// Uncompressed bytes per independent frame of a sealed chunk.
const FRAME_BYTES: u64 = 4 << 20;

pub fn chunk_path(directory: &Path, number: u64) -> PathBuf {
    directory.join(format!("{number:05}.jsonl"))
}

/// Chunk number -> whether its plain file is present (a sealed chunk has none).
pub fn chunk_numbers(directory: &Path) -> Result<BTreeMap<u64, bool>> {
    let mut numbers = BTreeMap::new();
    for entry in fs::read_dir(directory)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        let Some((stem, _)) = name.split_once(".jsonl") else {
            continue;
        };
        let number: u64 = stem
            .parse()
            .with_context(|| format!("chunk file name {name}"))?;
        *numbers.entry(number).or_insert(false) |= name.ends_with(".jsonl");
    }
    Ok(numbers)
}

/// Compress a finished chunk beside itself, as `pyzstd.SeekableZstdFile` writes it (one frame
/// per `FRAME_BYTES` of input, then the seek table), then drop the plain file.
fn seal(path: &Path) -> Result<()> {
    let target = PathBuf::from(format!("{}.zst", path.display()));
    let partial = PathBuf::from(format!("{}.tmp", target.display()));
    let mut src = File::open(path)?;
    let mut dst = BufWriter::new(File::create(&partial)?);
    let mut frames = Vec::new();
    loop {
        let mut data = Vec::new();
        (&mut src).take(FRAME_BYTES).read_to_end(&mut data)?;
        if data.is_empty() {
            break;
        }
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), LEVEL)?;
        // fed in 1 MiB pieces like Python's copyfileobj, which the frame bytes depend on
        for piece in data.chunks(1 << 20) {
            encoder.write_all(piece)?;
        }
        let frame = encoder.finish()?;
        dst.write_all(&frame)?;
        frames.push((frame.len() as u32, data.len() as u32));
    }
    // the seek table: a skippable frame of (compressed, decompressed) sizes, no checksums
    dst.write_all(&0x184D_2A5Eu32.to_le_bytes())?;
    dst.write_all(&(9 + 8 * frames.len() as u32).to_le_bytes())?;
    for (compressed, decompressed) in &frames {
        dst.write_all(&compressed.to_le_bytes())?;
        dst.write_all(&decompressed.to_le_bytes())?;
    }
    dst.write_all(&(frames.len() as u32).to_le_bytes())?;
    dst.write_all(&[0])?;
    dst.write_all(&0x8F92_EAB1u32.to_le_bytes())?;
    dst.into_inner().map_err(|e| e.into_error())?;
    fs::rename(&partial, &target)?;
    fs::remove_file(path)?;
    Ok(())
}

/// Appends lines to the live chunk of a stream and reports where each one landed.
pub struct ChunkedJsonl {
    directory: PathBuf,
    max_bytes: u64,
    compress: bool,
    number: u64,
    file: File,
    size: u64,
    sealing: Vec<JoinHandle<Result<()>>>,
}

impl ChunkedJsonl {
    pub fn open(directory: &Path, max_bytes: u64, compress: bool) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let numbers = chunk_numbers(directory)?;
        // A relaunch appends to the live chunk it finds and starts a new one after a sealed
        // last chunk; anything else still plain was left mid-seal by a crash.
        let number = match numbers.last_key_value() {
            Some((&last, &plain)) => last + u64::from(!plain),
            None => 0,
        };
        let file = open_append(&chunk_path(directory, number))?;
        let size = file.metadata()?.len();
        let mut stream = ChunkedJsonl {
            directory: directory.to_path_buf(),
            max_bytes,
            compress,
            number,
            file,
            size,
            sealing: Vec::new(),
        };
        if compress {
            for (&n, _) in numbers.iter().filter(|(&n, &plain)| plain && n != number) {
                stream.seal_background(n);
            }
        }
        Ok(stream)
    }

    /// Write one line; returns `(chunk, offset)`, the address a reader seeks to.
    pub fn append(&mut self, line: &[u8]) -> Result<(u64, u64)> {
        if self.size > 0 && self.size + line.len() as u64 > self.max_bytes {
            if self.compress {
                self.seal_background(self.number);
            }
            self.number += 1;
            self.file = open_append(&chunk_path(&self.directory, self.number))?;
            self.size = 0;
        }
        let offset = self.size;
        self.file.write_all(line)?;
        self.size += line.len() as u64;
        Ok((self.number, offset))
    }

    pub fn flush(&mut self) -> Result<()> {
        Ok(self.file.flush()?)
    }

    /// Seal the live chunk too, so a finished run keeps nothing plain.
    pub fn close(mut self) -> Result<()> {
        if self.size == 0 {
            fs::remove_file(chunk_path(&self.directory, self.number))?;
        } else if self.compress {
            self.seal_background(self.number);
        }
        for handle in self.sealing.drain(..) {
            handle
                .join()
                .map_err(|_| anyhow!("chunk sealing thread panicked"))??;
        }
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
            if let Ok(Err(e)) = handle.join() {
                tracing::warn!("Failed to seal a trace chunk: {e:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_rolls_seals_and_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let stream_dir = dir.path().join("stream");
        let mut stream = ChunkedJsonl::open(&stream_dir, 10, true).unwrap();
        assert_eq!(stream.append(b"aaaaaa\n").unwrap(), (0, 0));
        assert_eq!(stream.append(b"bbbbbb\n").unwrap(), (1, 0)); // 7 + 7 > 10 rolls
        assert_eq!(stream.append(b"c\n").unwrap(), (1, 7));
        drop(stream); // a crash: chunk 1 stays plain
        let mut stream = ChunkedJsonl::open(&stream_dir, 100, true).unwrap();
        assert_eq!(stream.append(b"d\n").unwrap(), (1, 9)); // the relaunch continues it
        stream.close().unwrap();
        assert!(!chunk_path(&stream_dir, 1).exists());
        let sealed = fs::read(stream_dir.join("00001.jsonl.zst")).unwrap();
        assert_eq!(
            zstd::stream::decode_all(sealed.as_slice()).unwrap(),
            b"bbbbbb\nc\nd\n"
        );
        // after a sealed last chunk a relaunch starts a new one; an empty one is removed on close
        let stream = ChunkedJsonl::open(&stream_dir, 100, true).unwrap();
        assert_eq!(stream.number, 2);
        stream.close().unwrap();
        assert!(!chunk_path(&stream_dir, 2).exists());
    }
}

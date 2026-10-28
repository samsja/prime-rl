use std::fs;
use std::path::Path;

use prl_monitors::file::chunks::{chunk_path, seal, seekable_compress, ChunkedJsonl};

fn fixture(name: &str) -> Vec<u8> {
    fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

#[test]
fn sealed_chunk_is_byte_identical_to_pyzstd() {
    let plain = fixture("chunk_plain.jsonl");
    let mut out = Vec::new();
    seekable_compress(&mut plain.as_slice(), &mut out).unwrap();
    assert_eq!(out, fixture("chunk_sealed.jsonl.zst"));
}

#[test]
fn multi_frame_chunk_round_trips() {
    let data: Vec<u8> = (0..10_000_000u32)
        .flat_map(|i| (i % 251).to_le_bytes())
        .take(9 << 20)
        .collect();
    let mut out = Vec::new();
    seekable_compress(&mut data.as_slice(), &mut out).unwrap();
    // three frames (4 + 4 + 1 MiB) and the seek table
    let footer = &out[out.len() - 9..];
    assert_eq!(u32::from_le_bytes(footer[..4].try_into().unwrap()), 3);
    assert_eq!(zstd::stream::decode_all(out.as_slice()).unwrap(), data);
}

#[test]
fn chunked_stream_rolls_seals_and_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let stream_dir = dir.path().join("stream");
    let mut stream = ChunkedJsonl::open(&stream_dir, 10, true).unwrap();
    assert_eq!(stream.append(b"aaaaaa\n").unwrap(), (0, 0));
    assert_eq!(stream.append(b"bbbbbb\n").unwrap(), (1, 0)); // 7 + 7 > 10 rolls
    assert_eq!(stream.append(b"c\n").unwrap(), (1, 7));
    stream.flush().unwrap();
    drop(stream); // crash: chunk 1 stays plain, chunk 0 may be mid-seal
                  // relaunch continues the live chunk
    let mut stream = ChunkedJsonl::open(&stream_dir, 100, true).unwrap();
    assert_eq!(stream.append(b"d\n").unwrap(), (1, 9));
    stream.close().unwrap();
    assert!(!chunk_path(&stream_dir, 0).exists());
    assert!(!chunk_path(&stream_dir, 1).exists());
    let sealed = fs::read(stream_dir.join("00001.jsonl.zst")).unwrap();
    assert_eq!(
        zstd::stream::decode_all(sealed.as_slice()).unwrap(),
        b"bbbbbb\nc\nd\n"
    );
    // a relaunch after a sealed last chunk starts a new one; an empty live chunk is removed on close
    let stream = ChunkedJsonl::open(&stream_dir, 100, true).unwrap();
    assert_eq!(stream.number, 2);
    stream.close().unwrap();
    assert!(!chunk_path(&stream_dir, 2).exists());
}

#[test]
fn seal_replaces_plain_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("00000.jsonl");
    fs::write(&path, b"x\n").unwrap();
    seal(&path).unwrap();
    assert!(!path.exists());
    assert!(dir.path().join("00000.jsonl.zst").exists());
}

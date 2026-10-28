//! The orchestrator side of `prime_rl/transports/payload.py`: env servers keep each train
//! episode's per-token training arrays and serve them over HTTP; a trace carries only
//! references in `trace.info["prime_rl_payload"]`. Whoever holds a blob last deletes it; the
//! orchestrator deletes the blobs of traces it drops without shipping.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use crate::model::Trace;
use crate::value::Value;

/// `prime_rl.transports.batch.types.PAYLOAD_INFO_KEY`.
pub const PAYLOAD_INFO_KEY: &str = "prime_rl_payload";

const DELETE_THREADS: usize = 8;
const DELETE_TIMEOUT: Duration = Duration::from_secs(10);

/// URLs of every blob an env server kept for the trace (`f"{url}/{id}"` per branch).
pub fn trace_payload_urls(trace: &Trace) -> Vec<String> {
    let Some(payload) = trace.info.get(PAYLOAD_INFO_KEY).filter(|p| p.truthy()) else {
        return Vec::new();
    };
    let url = payload.get("url").and_then(Value::as_str).unwrap_or_default();
    payload
        .get("branches")
        .and_then(Value::as_map)
        .map(|branches| {
            branches
                .values()
                .filter_map(|reference| reference.get("id").and_then(Value::as_str))
                .map(|id| format!("{url}/{id}"))
                .collect()
        })
        .unwrap_or_default()
}

/// The payload reference of one branch: `trace.info["prime_rl_payload"]["branches"][str(index)]`
/// and the server URL.
pub fn branch_payload(trace: &Trace, branch_index: usize) -> Option<(&str, &Value)> {
    let payload = trace.info.get(PAYLOAD_INFO_KEY)?;
    let url = payload.get("url")?.as_str()?;
    let reference = payload.get("branches")?.get(&branch_index.to_string())?;
    Some((url, reference))
}

struct Deleter {
    queue: Mutex<mpsc::Sender<String>>,
    outstanding: Arc<(Mutex<usize>, Condvar)>,
}

fn deleter() -> &'static Deleter {
    static DELETER: OnceLock<Deleter> = OnceLock::new();
    DELETER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<String>();
        let rx = Arc::new(Mutex::new(rx));
        let outstanding = Arc::new((Mutex::new(0usize), Condvar::new()));
        for _ in 0..DELETE_THREADS {
            let rx = rx.clone();
            let outstanding = outstanding.clone();
            std::thread::Builder::new()
                .name("payload-delete".into())
                .spawn(move || loop {
                    let url = match rx.lock().unwrap().recv() {
                        Ok(url) => url,
                        Err(_) => return,
                    };
                    // Best effort, like Python: a server that is gone keeps its blobs.
                    let _ = http_delete(&url);
                    let (count, done) = &*outstanding;
                    *count.lock().unwrap() -= 1;
                    done.notify_all();
                })
                .expect("spawn payload-delete thread");
        }
        Deleter { queue: Mutex::new(tx), outstanding }
    })
}

/// Free kept arrays on their env servers, without waiting (fire-and-forget DELETEs on
/// background threads). Call only once nothing will read them again.
pub fn delete_payloads(urls: impl IntoIterator<Item = String>) {
    let deleter = deleter();
    let queue = deleter.queue.lock().unwrap();
    for url in urls {
        *deleter.outstanding.0.lock().unwrap() += 1;
        queue.send(url).expect("payload-delete threads run forever");
    }
}

/// Block until every queued DELETE has run.
pub fn wait_for_deletes() {
    let (count, done) = &*deleter().outstanding;
    let mut count = count.lock().unwrap();
    while *count > 0 {
        count = done.wait(count).unwrap();
    }
}

/// `DELETE <url>` over plain HTTP/1.1 (payload URLs are node-local `http://host:port/id`;
/// never through a proxy). Returns the status code.
fn http_delete(url: &str) -> std::io::Result<u16> {
    let invalid = || std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("bad payload url {url}"));
    let rest = url.strip_prefix("http://").ok_or_else(invalid)?;
    let (authority, path) = rest.split_once('/').map_or((rest, "/".to_owned()), |(a, p)| (a, format!("/{p}")));
    let address = authority.to_socket_addrs()?.next().ok_or_else(invalid)?;
    let mut stream = TcpStream::connect_timeout(&address, DELETE_TIMEOUT)?;
    stream.set_read_timeout(Some(DELETE_TIMEOUT))?;
    stream.set_write_timeout(Some(DELETE_TIMEOUT))?;
    write!(stream, "DELETE {path} HTTP/1.1\r\nHost: {authority}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
    let mut head = [0u8; 12];
    stream.read_exact(&mut head)?;
    std::str::from_utf8(&head[9..12]).ok().and_then(|s| s.parse().ok()).ok_or_else(invalid)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    use super::*;

    #[test]
    fn deletes_reach_the_payload_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut paths = Vec::new();
            for _ in 0..3 {
                let (stream, _) = listener.accept().unwrap();
                let mut line = String::new();
                BufReader::new(&stream).read_line(&mut line).unwrap();
                paths.push(line.trim().to_owned());
                (&stream).write_all(b"HTTP/1.0 204 No Content\r\n\r\n").unwrap();
            }
            paths.sort();
            paths
        });
        let mut trace = Trace::from_value(
            &crate::value::from_msgpack(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/episode_assembled.msgpack")).unwrap())
                .unwrap()
                .get("traces")
                .unwrap()
                .as_array()
                .unwrap()[0],
            "trace".into(),
        )
        .unwrap();
        assert!(trace_payload_urls(&trace).is_empty());
        let reference = |id: &str| -> Value { [("id", Value::from(id)), ("fields", Value::Array(vec!["routed_experts".into()]))].into_iter().collect() };
        let branches: Value = [("0", reference("t-0")), ("1", reference("t-1"))].into_iter().collect();
        let url = format!("http://127.0.0.1:{port}");
        trace.info.insert(PAYLOAD_INFO_KEY.into(), [("url", Value::from(url.as_str())), ("branches", branches)].into_iter().collect());
        let urls = trace_payload_urls(&trace);
        assert_eq!(urls, [format!("{url}/t-0"), format!("{url}/t-1")]);
        assert_eq!(branch_payload(&trace, 1).unwrap().1.get("id").unwrap().as_str(), Some("t-1"));
        delete_payloads(urls.into_iter().chain([format!("{url}/x")]));
        wait_for_deletes();
        assert_eq!(server.join().unwrap(), ["DELETE /t-0 HTTP/1.1", "DELETE /t-1 HTTP/1.1", "DELETE /x HTTP/1.1"]);
    }
}

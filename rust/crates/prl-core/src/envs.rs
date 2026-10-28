//! Env-server connections and tasksets (`envs.py`). The orchestrator owns the tasksets (loaded
//! through the Python bridge) and ships each task's data on the run request; env servers
//! own execution. Each env routes its runs to the least loaded of its server's replicas.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use prl_io::logging::format_time;
use prl_types::{Bridge, Kind, TaskRecord};
use prl_wire::{EnvClient, EnvClientError, Episode, OnDelta, Value};
use tracing::{debug, info};

use crate::curriculum::{BridgeFeed, Tasks};
use crate::dispatcher::RunRequest;

/// Max wait for an env server to publish its address and answer health.
pub const ENV_SERVER_STARTUP_TIMEOUT: Duration = Duration::from_secs(600);

/// `get_config_dir`: the current attempt's resolved config directory.
pub fn config_dir(output_dir: &Path) -> PathBuf {
    if let Some(dir) = std::env::var_os("PRL_ATTEMPT_CONFIG_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let configs = output_dir.join("configs");
    let latest = configs.join("latest").join("resolved");
    let legacy = configs.join("resolved");
    if legacy.is_dir() && !latest.exists() {
        legacy
    } else {
        latest
    }
}

/// `env_address_file`: `envs/<split>/<name>[.<replica>].address` under the config dir.
pub fn env_address_file(config_dir: &Path, kind: Kind, name: &str, replica: Option<u64>) -> PathBuf {
    let stem = match replica {
        None => name.to_owned(),
        Some(r) => format!("{name}.{r}"),
    };
    config_dir.join("envs").join(kind.as_str()).join(format!("{stem}.address"))
}

pub fn replica_address_files(config_dir: &Path, kind: Kind, name: &str, replicas: Option<u64>) -> Vec<PathBuf> {
    match replicas {
        None => vec![env_address_file(config_dir, kind, name, None)],
        Some(n) => (0..n).map(|r| env_address_file(config_dir, kind, name, Some(r))).collect(),
    }
}

/// The address a launcher-managed env server published (it starts concurrently with us).
pub async fn wait_for_address(path: &Path, timeout: Duration) -> Result<String> {
    let deadline = Instant::now() + timeout;
    loop {
        match tokio::fs::read_to_string(path).await {
            Ok(text) if !text.trim().is_empty() => return Ok(text.trim().to_owned()),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        }
        if Instant::now() > deadline {
            bail!(
                "env server never published its address to {} within {}s",
                path.display(),
                timeout.as_secs_f64()
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// An env server's replicas; each run goes to the replica with the fewest runs in flight (ties:
/// lowest index).
pub struct ReplicaRouter {
    clients: Vec<EnvClient>,
    in_flight: Vec<AtomicUsize>,
}

struct InFlightGuard<'a>(&'a AtomicUsize);

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ReplicaRouter {
    pub fn new(clients: Vec<EnvClient>) -> Self {
        let in_flight = clients.iter().map(|_| AtomicUsize::new(0)).collect();
        Self { clients, in_flight }
    }

    pub fn clients(&self) -> &[EnvClient] {
        &self.clients
    }

    pub async fn run(&self, request: RunRequest, on_delta: OnDelta) -> std::result::Result<Episode, EnvClientError> {
        let task_data = serde_json::from_value::<Value>(request.task_data)
            .ok()
            .and_then(Value::into_map)
            .expect("task data is a JSON object");
        let loads: Vec<usize> = self.in_flight.iter().map(|n| n.load(Ordering::SeqCst)).collect();
        let replica = (0..loads.len()).min_by_key(|i| loads[*i]).expect("at least one replica");
        self.in_flight[replica].fetch_add(1, Ordering::SeqCst);
        let _guard = InFlightGuard(&self.in_flight[replica]);
        self.clients[replica].run(&task_data, &request.client, &request.model, &request.sampling, Some(on_delta)).await
    }
}

/// A connected env with its selected tasks.
pub struct StartedEnv {
    pub name: String,
    pub router: Arc<ReplicaRouter>,
    pub tasks: Tasks,
    /// None: the selected tasks never end.
    pub num_tasks: Option<usize>,
}

impl StartedEnv {
    /// A bounded taskset's tasks (eval examples).
    pub fn examples(&self) -> Result<Vec<TaskRecord>> {
        match &self.tasks {
            Tasks::Finite(tasks) => Ok(tasks.clone()),
            Tasks::Stream(_) => {
                bail!("Eval env {} has an infinite taskset — set select.limit to bound it", self.name)
            }
        }
    }

    pub async fn close(&self) {
        for client in self.router.clients() {
            client.close().await;
        }
    }
}

fn py_list(items: &[String]) -> String {
    format!("[{}]", items.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", "))
}

/// Connect to an env's replicas (an explicit `address`, else the replicas' published address
/// files), wait for health, then load its taskset through the bridge.
pub async fn start_env(
    kind: Kind,
    name: &str,
    address: Option<String>,
    address_files: Vec<PathBuf>,
    bridge: Arc<dyn Bridge>,
) -> Result<StartedEnv> {
    let t0 = Instant::now();
    let addresses = match address {
        Some(address) => vec![address],
        None => {
            futures::future::try_join_all(
                address_files.iter().map(|path| wait_for_address(path, ENV_SERVER_STARTUP_TIMEOUT)),
            )
            .await?
        }
    };
    debug!("Connecting {name} to env servers {}", py_list(&addresses));
    let clients =
        addresses.iter().map(|a| EnvClient::connect(a)).collect::<std::result::Result<Vec<_>, _>>()?;
    futures::future::try_join_all(clients.iter().map(|c| c.wait_for_server_startup(ENV_SERVER_STARTUP_TIMEOUT)))
        .await?;
    let loaded = bridge.load_tasks(kind, name).await?;
    let (tasks, num_tasks) = match loaded.num_tasks {
        Some(n) => (Tasks::Finite(loaded.tasks), Some(n)),
        None => {
            let feed = BridgeFeed { bridge: bridge.clone(), kind, env_name: name.to_owned() };
            (Tasks::Stream(feed), None)
        }
    };
    let shown = num_tasks.map_or_else(|| "infinite".to_string(), |n| n.to_string());
    info!("Env {name} ready in {} (num_tasks={shown})", format_time(t0.elapsed().as_secs_f64()));
    Ok(StartedEnv { name: name.to_owned(), router: Arc::new(ReplicaRouter::new(clients)), tasks, num_tasks })
}

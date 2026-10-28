//! The monitors actor: ordered fan-out, failure containment, finalize vs. crash.

use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use async_trait::async_trait;
use prl_monitors::{EpisodeRef, Monitor, Row, Scalar};
use prl_types::{Kind, Subset};
use tokio_util::sync::CancellationToken;

struct Recorder {
    name: &'static str,
    log: Arc<Mutex<Vec<String>>>,
    fail_metrics: bool,
}

#[async_trait]
impl Monitor for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }
    async fn log_metrics(&mut self, metrics: &Row, step: Option<u64>) -> Result<()> {
        if self.fail_metrics {
            bail!("boom");
        }
        // yield, so a concurrent producer would interleave if ordering were not enforced
        tokio::task::yield_now().await;
        let value = metrics.get("i").and_then(Scalar::as_f64).unwrap_or(-1.0);
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:metrics:{value}:{step:?}", self.name));
        Ok(())
    }
    async fn log_episodes(
        &mut self,
        episodes: &[EpisodeRef],
        step: u64,
        kind: Kind,
        subset: Subset,
    ) -> Result<()> {
        self.log.lock().unwrap().push(format!(
            "{}:episodes:{}:{step}:{kind:?}:{subset:?}",
            self.name,
            episodes.len()
        ));
        Ok(())
    }
    async fn log_eval_plan(&mut self, env_name: &str, step: u64, expected: u64) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:plan:{env_name}:{step}:{expected}", self.name));
        Ok(())
    }
    async fn finalize(&mut self, paused: bool) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:finalize:{paused}", self.name));
        Ok(())
    }
    async fn crash(&mut self) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:crash", self.name));
        Ok(())
    }
}

fn monitors(log: &Arc<Mutex<Vec<String>>>) -> Vec<Box<dyn Monitor>> {
    vec![
        Box::new(Recorder {
            name: "a",
            log: log.clone(),
            fail_metrics: true,
        }),
        Box::new(Recorder {
            name: "b",
            log: log.clone(),
            fail_metrics: false,
        }),
    ]
}

fn row(i: i64) -> Row {
    [("i".to_string(), Scalar::Int(i))].into_iter().collect()
}

#[tokio::test]
async fn fan_out_is_ordered_and_contains_failures() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let (handle, task) = prl_monitors::spawn_monitors(monitors(&log), CancellationToken::new());
    for i in 0..50 {
        handle.log_metrics(row(i), Some(i as u64)).await;
    }
    handle
        .log_episodes(vec![], 3, Kind::Eval, Subset::All)
        .await;
    handle.log_eval_plan("gsm8k".into(), 3, 8).await;
    handle.finalize(false).await;
    task.await.unwrap().unwrap();
    let log = log.lock().unwrap().clone();
    let metrics: Vec<&String> = log.iter().filter(|l| l.contains(":metrics:")).collect();
    assert_eq!(metrics.len(), 50);
    assert!(metrics
        .iter()
        .enumerate()
        .all(|(i, l)| **l == format!("b:metrics:{i}:Some({i})")));
    assert_eq!(
        log[50..],
        [
            "a:episodes:0:3:Eval:All",
            "b:episodes:0:3:Eval:All",
            "a:plan:gsm8k:3:8",
            "b:plan:gsm8k:3:8",
            "a:finalize:false",
            "b:finalize:false"
        ]
    );
}

#[tokio::test]
async fn cancel_drains_queued_logs_then_crashes() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let cancel = CancellationToken::new();
    let (handle, task) = prl_monitors::spawn_monitors(monitors(&log), cancel.clone());
    handle.log_eval_plan("x".into(), 1, 2).await;
    cancel.cancel();
    task.await.unwrap().unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        ["a:plan:x:1:2", "b:plan:x:1:2", "a:crash", "b:crash"]
    );
    // logging after the actor stopped is a no-op, never a hang
    handle.log_metrics(row(1), None).await;
    handle.finalize(false).await;
}

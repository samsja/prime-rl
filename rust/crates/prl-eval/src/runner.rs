//! The eval engine shared by `uv run eval` and online evals (port of `eval/runner.py`).

use prl_io::logging::format_time;

/// How often `run_epoch` re-checks for a superseding checkpoint while it waits for episodes.
pub const POLL_INTERVAL_S: u64 = 2;

/// Python `f"{x:.1%}"`.
pub fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// Means of one finished epoch that its log line reports: reward/turns/branches/truncation over
/// the effective episodes, error/timeout over all of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochSummary {
    pub reward: f64,
    pub turns: f64,
    pub branches: f64,
    pub truncated: f64,
    pub error: f64,
    pub timeout: f64,
}

/// SUCCESS line of a complete epoch.
pub fn evaluated_line(env: &str, step: u64, elapsed: f64, s: &EpochSummary) -> String {
    format!(
        "Evaluated {env} (Step {step}) | {:>7} | Reward {:.4} | Turns {:.1} | Branches {:.1} | Error {} | Truncation {} | Timeout {}",
        format_time(elapsed),
        s.reward,
        s.turns,
        s.branches,
        pct(s.error),
        pct(s.truncated),
        pct(s.timeout),
    )
}

/// WARNING line of an epoch a newer checkpoint cut short.
pub fn partially_evaluated_line(
    env: &str,
    step: u64,
    elapsed: f64,
    s: &EpochSummary,
    completed: usize,
    cancelled: usize,
    total: usize,
) -> String {
    format!(
        "Partially evaluated {env} (Step {step}) | {:>7} | Reward {:.4} | Error {} | Completed {completed}/{total} | Cancelled {cancelled}/{total}",
        format_time(elapsed),
        s.reward,
        pct(s.error),
    )
}

/// Body of the periodic `Eval` line: per-epoch progress, then the in-flight pool against its cap.
pub fn pipeline_body(
    mut progress: Vec<(String, u64, usize, usize)>,
    inflight_eval: usize,
    max_inflight: usize,
    signal: &str,
    stage_counts: &str,
) -> String {
    progress.sort();
    let parts: Vec<String> = progress
        .iter()
        .map(|(env, _step, arrived, expected)| {
            if *expected > 0 {
                format!("{env} {arrived}/{expected} ({})", pct(*arrived as f64 / *expected as f64))
            } else {
                env.clone()
            }
        })
        .collect();
    let progress = if parts.is_empty() { "Idle".to_owned() } else { parts.join(" | ") };
    let stages = if stage_counts.is_empty() { String::new() } else { format!(" - {stage_counts}") };
    format!("{progress}; {inflight_eval} inflight episodes (cap {max_inflight}, signal {signal}){stages}")
}

/// `Starting evals in ...` line of a triggered epoch.
pub fn starting_line(fired: &[String], step: u64, owed: usize, restored: usize) -> String {
    let restored = if restored > 0 { format!(", {restored} restored") } else { String::new() };
    format!("Starting evals in {} at step {step} ({owed} total rollouts{restored})", fired.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_match_python_format() {
        let s = EpochSummary { reward: 0.123456, turns: 1.0, branches: 1.25, truncated: 0.0625, error: 0.00049, timeout: 1.0 };
        assert_eq!(
            evaluated_line("gsm8k", 0, 75.0, &s),
            "Evaluated gsm8k (Step 0) |  1m 15s | Reward 0.1235 | Turns 1.0 | Branches 1.2 | Error 0.0% | Truncation 6.2% | Timeout 100.0%"
        );
        assert_eq!(
            partially_evaluated_line("gsm8k", 5, 0.5, &s, 3, 5, 8),
            "Partially evaluated gsm8k (Step 5) |   500ms | Reward 0.1235 | Error 0.0% | Completed 3/8 | Cancelled 5/8"
        );
        assert_eq!(starting_line(&["a".into(), "b".into()], 0, 12, 0), "Starting evals in a, b at step 0 (12 total rollouts)");
        assert_eq!(starting_line(&["a".into()], 3, 2, 4), "Starting evals in a at step 3 (2 total rollouts, 4 restored)");
    }

    #[test]
    fn pipeline_body_matches_python() {
        assert_eq!(pipeline_body(vec![], 0, 128, "clear", ""), "Idle; 0 inflight episodes (cap 128, signal clear)");
        let progress = vec![("b".into(), 0, 1, 3), ("a".into(), 0, 2, 4)];
        assert_eq!(
            pipeline_body(progress, 5, 64, "soft", "model=3, env=2"),
            "a 2/4 (50.0%) | b 1/3 (33.3%); 5 inflight episodes (cap 64, signal soft) - model=3, env=2"
        );
    }
}

//! Online evals (port of `eval/online.py`): evaluate the trainer's weight broadcasts as they appear.

use std::collections::{BTreeSet, HashSet};

/// What one broadcasts-dir scan does with a step, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalkAction {
    /// An eval-due step that broadcast cleaning removed before it was evaluated.
    SkipDeleted(u64),
    /// An unpublished step below a published one: an abandoned partial write.
    SkipAbandoned(u64),
    /// Receive the broadcast and run the due evals (`force` on the final step).
    Evaluate { step: u64, force: bool },
}

/// Eval-due steps up to the newest published broadcast that are missing from the broadcasts dir.
pub fn deleted_due_steps(steps: &[u64], newest_published: Option<u64>, intervals: &[u64]) -> BTreeSet<u64> {
    let Some(newest) = newest_published else { return BTreeSet::new() };
    let present: HashSet<u64> = steps.iter().copied().collect();
    intervals
        .iter()
        .flat_map(|&interval| (interval..=newest).step_by(interval as usize))
        .filter(|step| !present.contains(step))
        .collect()
}

/// One scan of the broadcasts dir: `steps` are the `step_*` dirs, `published` the ones with a
/// `.sender_ready` marker. Steps at or below `last_step` were already handled.
pub fn plan_walk(
    steps: &[u64],
    published: &HashSet<u64>,
    intervals: &[u64],
    last_step: u64,
    max_steps: Option<u64>,
) -> Vec<WalkAction> {
    let newest_published = steps.iter().copied().filter(|step| published.contains(step)).max();
    let present: HashSet<u64> = steps.iter().copied().collect();
    let mut walk: BTreeSet<u64> = steps.iter().copied().collect();
    walk.extend(deleted_due_steps(steps, newest_published, intervals));
    let mut actions = Vec::new();
    for step in walk {
        if step <= last_step {
            continue;
        }
        if !present.contains(&step) {
            actions.push(WalkAction::SkipDeleted(step));
            continue;
        }
        if !published.contains(&step) {
            if newest_published.is_none_or(|newest| newest < step) {
                break; // still being written — later steps can't be ready either
            }
            actions.push(WalkAction::SkipAbandoned(step));
            continue;
        }
        actions.push(WalkAction::Evaluate { step, force: max_steps.is_some_and(|max| step >= max) });
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use WalkAction::*;

    fn set(steps: &[u64]) -> HashSet<u64> {
        steps.iter().copied().collect()
    }

    #[test]
    fn evaluates_published_steps_in_order_above_last_step() {
        let actions = plan_walk(&[1, 2, 3, 4], &set(&[1, 2, 3, 4]), &[2], 2, Some(10));
        assert_eq!(actions, [Evaluate { step: 3, force: false }, Evaluate { step: 4, force: false }]);
    }

    #[test]
    fn stops_at_a_step_still_being_written() {
        let actions = plan_walk(&[1, 2, 3], &set(&[1]), &[1], 0, None);
        assert_eq!(actions, [Evaluate { step: 1, force: false }]);
    }

    #[test]
    fn skips_abandoned_steps_below_a_published_one() {
        let actions = plan_walk(&[1, 2, 3], &set(&[1, 3]), &[5], 0, None);
        assert_eq!(actions, [Evaluate { step: 1, force: false }, SkipAbandoned(2), Evaluate { step: 3, force: false }]);
    }

    #[test]
    fn reports_due_steps_removed_by_broadcast_cleaning() {
        // trainer keeps only the newest broadcast: steps 2 and 4 (due every 2) vanished
        let actions = plan_walk(&[5], &set(&[5]), &[2], 0, None);
        assert_eq!(actions, [SkipDeleted(2), SkipDeleted(4), Evaluate { step: 5, force: false }]);
        assert_eq!(deleted_due_steps(&[5], None, &[2]), BTreeSet::new());
    }

    #[test]
    fn final_step_forces_every_env() {
        let actions = plan_walk(&[9, 10], &set(&[9, 10]), &[100], 8, Some(10));
        assert_eq!(actions, [Evaluate { step: 9, force: false }, Evaluate { step: 10, force: true }]);
    }
}

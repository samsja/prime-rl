//! Balanced partitioning of micro batch workloads across data ranks: Karmarkar-Karp followed by
//! best-pairwise-swap refinement. Tie-breaking replicates the Python implementation exactly
//! (`heapq` ordering, stable sorts), so both produce the same partitions.

use std::cmp::Ordering;

#[derive(Clone, Default, PartialEq, Eq)]
struct WeightedSet {
    total: i128,
    items: Vec<usize>,
}

impl WeightedSet {
    fn add(&mut self, idx: usize, weight: i128) {
        self.items.push(idx);
        self.total += weight;
    }

    fn merge(&mut self, other: &WeightedSet) {
        self.items.extend_from_slice(&other.items);
        self.total += other.total;
    }

    fn lt(&self, other: &WeightedSet) -> bool {
        if self.total != other.total {
            return self.total < other.total;
        }
        self.items < other.items
    }
}

/// `list.sort(reverse=True)` with only `__lt__`: stable, descending.
fn sort_desc(sets: &mut [WeightedSet]) {
    sets.sort_by(|a, b| {
        if b.lt(a) {
            Ordering::Less
        } else if a.lt(b) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}

struct KKState {
    sets: Vec<WeightedSet>,
}

impl KKState {
    fn new(items: &[(usize, i128)], k: usize) -> Self {
        let mut sets = vec![WeightedSet::default(); k];
        for (set_idx, &(idx, weight)) in items.iter().enumerate() {
            sets[set_idx].add(idx, weight);
        }
        sort_desc(&mut sets);
        Self { sets }
    }

    fn spread(&self) -> i128 {
        self.sets[0].total - self.sets[self.sets.len() - 1].total
    }

    fn merge(&mut self, other: &KKState) {
        let k = self.sets.len();
        for i in 0..k {
            self.sets[i].merge(&other.sets[k - 1 - i]);
        }
        sort_desc(&mut self.sets);
    }

    /// Larger spread first; ties: larger first set first.
    fn lt(&self, other: &KKState) -> bool {
        if self.spread() != other.spread() {
            return self.spread() > other.spread();
        }
        other.sets[0].lt(&self.sets[0])
    }
}

/// Python `heapq` push/pop (same sift order, so equal-priority states pop identically).
fn heappush(heap: &mut Vec<KKState>, item: KKState) {
    heap.push(item);
    let len = heap.len();
    siftdown(heap, 0, len - 1);
}

fn heappop(heap: &mut Vec<KKState>) -> KKState {
    let last = heap.pop().expect("pop from empty heap");
    if heap.is_empty() {
        return last;
    }
    let ret = std::mem::replace(&mut heap[0], last);
    siftup(heap, 0);
    ret
}

fn siftdown(heap: &mut [KKState], startpos: usize, mut pos: usize) {
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        if heap[pos].lt(&heap[parentpos]) {
            heap.swap(pos, parentpos);
            pos = parentpos;
            continue;
        }
        break;
    }
}

fn siftup(heap: &mut [KKState], mut pos: usize) {
    // Python's _siftup moves the new item down to a leaf along smaller children, then sifts it up.
    let endpos = heap.len();
    let startpos = pos;
    let mut childpos = 2 * pos + 1;
    while childpos < endpos {
        let rightpos = childpos + 1;
        if rightpos < endpos && !heap[childpos].lt(&heap[rightpos]) {
            childpos = rightpos;
        }
        heap.swap(pos, childpos);
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    siftdown(heap, startpos, pos);
}

fn karmarkar_karp(weights: &[i128], num_partitions: usize) -> Vec<Vec<usize>> {
    assert!(weights.len() >= num_partitions);
    assert!(weights.len().is_multiple_of(num_partitions));
    let mut weighted: Vec<(i128, usize)> = weights.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    weighted.sort();
    let mut states: Vec<KKState> = Vec::new();
    for chunk in weighted.chunks(num_partitions) {
        let items: Vec<(usize, i128)> = chunk.iter().map(|&(w, i)| (i, w)).collect();
        heappush(&mut states, KKState::new(&items, num_partitions));
    }
    while states.len() > 1 {
        let mut state = heappop(&mut states);
        let other = heappop(&mut states);
        state.merge(&other);
        heappush(&mut states, state);
    }
    states
        .pop()
        .unwrap()
        .sets
        .into_iter()
        .map(|s| {
            let mut items = s.items;
            items.sort();
            items
        })
        .collect()
}

/// Apply the best pairwise swap until none lowers `(max load, max - min load)`.
/// Ties go to the first swap in (rank pair, left position, right position) order.
fn refine_by_swapping(weights: &[i128], mut partitions: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut loads: Vec<i128> = partitions.iter().map(|p| p.iter().map(|&i| weights[i]).sum()).collect();
    let n = partitions.len();
    loop {
        let top = *loads.iter().max().unwrap();
        let bottom = *loads.iter().min().unwrap();
        let relative: Vec<i128> = loads.iter().map(|&l| l - top).collect();
        let mut best_score = (0i128, top - bottom);
        let mut best_swap: Option<(usize, usize, usize, usize)> = None;
        for left in 0..n {
            for right in left + 1..n {
                let others = (0..n).filter(|&r| r != left && r != right).map(|r| relative[r]);
                let other_max = others.clone().max().unwrap_or(i128::MIN);
                let other_min = others.min().unwrap_or(i128::MAX);
                let (rel_l, rel_r) = (relative[left], relative[right]);
                let new_max = |wl: i128, wr: i128| (rel_l - wl + wr).max(rel_r - wr + wl).max(other_max);
                let mut lowest_max = i128::MAX;
                for &li in &partitions[left] {
                    for &ri in &partitions[right] {
                        lowest_max = lowest_max.min(new_max(weights[li], weights[ri]));
                    }
                }
                if lowest_max > best_score.0 {
                    continue;
                }
                let mut best_here: Option<(i128, usize, usize)> = None;
                for (lp, &li) in partitions[left].iter().enumerate() {
                    for (rp, &ri) in partitions[right].iter().enumerate() {
                        let (wl, wr) = (weights[li], weights[ri]);
                        let (a, b) = (rel_l - wl + wr, rel_r - wr + wl);
                        let max = a.max(b).max(other_max);
                        if max != lowest_max {
                            continue;
                        }
                        let spread = max - a.min(b).min(other_min);
                        if best_here.is_none_or(|(s, _, _)| spread < s) {
                            best_here = Some((spread, lp, rp));
                        }
                    }
                }
                let (spread, lp, rp) = best_here.unwrap();
                let score = (lowest_max, spread);
                if score < best_score {
                    best_score = score;
                    best_swap = Some((left, right, lp, rp));
                }
            }
        }
        let Some((left, right, lp, rp)) = best_swap else {
            return partitions;
        };
        let (li, ri) = (partitions[left][lp], partitions[right][rp]);
        partitions[left][lp] = ri;
        partitions[right][rp] = li;
        loads[left] += weights[ri] - weights[li];
        loads[right] += weights[li] - weights[ri];
    }
}

/// Partition item indices into `num_partitions` groups of near-equal total weight.
/// Requires `weights.len()` to be a positive multiple of `num_partitions`.
pub fn balanced_partition(weights: &[i128], num_partitions: usize) -> Vec<Vec<usize>> {
    let partitions = karmarkar_karp(weights, num_partitions);
    refine_by_swapping(weights, partitions)
}

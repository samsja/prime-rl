//! Python format-spec helpers for console lines.

/// `f"{x:.1%}"`.
pub fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn pct_matches_python() {
        assert_eq!(super::pct(0.1234), "12.3%");
        assert_eq!(super::pct(0.0), "0.0%");
        assert_eq!(super::pct(1.0), "100.0%");
    }
}

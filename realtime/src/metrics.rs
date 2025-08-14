use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use common::RiskLevel;

pub struct RealtimeMetrics {
	pub requests_total: AtomicU64,
	pub ws_connections_total: AtomicU64,
	pub sessions_active: AtomicU64,
	pub analyses_total: AtomicU64,
	pub firewall_allowed_total: AtomicU64,
	pub firewall_blocked_total: AtomicU64,
	pub risk_safe_total: AtomicU64,
	pub risk_warning_total: AtomicU64,
	pub risk_high_total: AtomicU64,
	pub risk_critical_total: AtomicU64,
}

static METRICS: OnceLock<RealtimeMetrics> = OnceLock::new();

pub fn metrics() -> &'static RealtimeMetrics {
	METRICS.get_or_init(|| RealtimeMetrics {
		requests_total: AtomicU64::new(0),
		ws_connections_total: AtomicU64::new(0),
		sessions_active: AtomicU64::new(0),
		analyses_total: AtomicU64::new(0),
		firewall_allowed_total: AtomicU64::new(0),
		firewall_blocked_total: AtomicU64::new(0),
		risk_safe_total: AtomicU64::new(0),
		risk_warning_total: AtomicU64::new(0),
		risk_high_total: AtomicU64::new(0),
		risk_critical_total: AtomicU64::new(0),
	})
}

pub fn record_request() { metrics().requests_total.fetch_add(1, Ordering::Relaxed); }
pub fn record_ws_connection() { metrics().ws_connections_total.fetch_add(1, Ordering::Relaxed); }
pub fn inc_session() { metrics().sessions_active.fetch_add(1, Ordering::Relaxed); }
pub fn dec_session() { let _ = metrics().sessions_active.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1)); }

pub fn record_analysis(risk: RiskLevel) {
	metrics().analyses_total.fetch_add(1, Ordering::Relaxed);
	match risk {
		RiskLevel::Safe => { metrics().risk_safe_total.fetch_add(1, Ordering::Relaxed); }
		RiskLevel::Warning => { metrics().risk_warning_total.fetch_add(1, Ordering::Relaxed); }
		RiskLevel::HighRisk => { metrics().risk_high_total.fetch_add(1, Ordering::Relaxed); }
		RiskLevel::Critical => { metrics().risk_critical_total.fetch_add(1, Ordering::Relaxed); }
	}
}

pub fn record_firewall_decision(allowed: bool) {
	if allowed {
		metrics().firewall_allowed_total.fetch_add(1, Ordering::Relaxed);
	} else {
		metrics().firewall_blocked_total.fetch_add(1, Ordering::Relaxed);
	}
} 
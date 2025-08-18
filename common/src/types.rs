use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GoldenScaling {
	EmpiricalGolden,
	Custom(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationMode {
	Scientific {
		abort_threshold: f64,
		warn_threshold: f64,
		proceed_threshold: f64,
	},
	Pragmatic {
		scaling: GoldenScaling,
		abort_threshold: f64,
		warn_threshold: f64,
		proceed_threshold: f64,
	},
}

impl Default for CalibrationMode {
	fn default() -> Self {
		// Enable golden scale calibration by default for enhanced hallucination detection
		CalibrationMode::Pragmatic {
			scaling: GoldenScaling::EmpiricalGolden,
			abort_threshold: 2.5,
			warn_threshold: 4.0,
			proceed_threshold: 6.0,
		}
	}
}

impl CalibrationMode {
	pub fn from_json_value(value: &serde_json::Value) -> Option<Self> {
		// Try to deserialize from JSON value
		serde_json::from_value(value.clone()).ok()
	}
	
	pub fn scientific() -> Self {
		CalibrationMode::Scientific {
			abort_threshold: 0.1,
			warn_threshold: 0.3,
			proceed_threshold: 0.4,
		}
	}
	
	pub fn pragmatic_golden() -> Self {
		CalibrationMode::Pragmatic {
			scaling: GoldenScaling::EmpiricalGolden,
			abort_threshold: 2.5,
			warn_threshold: 4.0,
			proceed_threshold: 6.0,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
	Safe,
	Warning,
	HighRisk,
	Critical,
}

impl RiskLevel {
	pub fn emoji(&self) -> &str {
		match self {
			RiskLevel::Safe => "✅",
			RiskLevel::Warning => "⚠️",
			RiskLevel::HighRisk => "🚨",
			RiskLevel::Critical => "❌",
		}
	}

	pub fn description(&self) -> &str {
		match self {
			RiskLevel::Safe => "Safe to proceed",
			RiskLevel::Warning => "Proceed with caution",
			RiskLevel::HighRisk => "High risk - review recommended",
			RiskLevel::Critical => "Critical - block immediately",
		}
	}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticUncertaintyResult {
	pub raw_hbar: f64,
	pub calibrated_hbar: f64,
	pub risk_level: RiskLevel,
	pub calibration_mode: CalibrationMode,
	pub explanation: String,
	pub delta_mu: f64,
	pub delta_sigma: f64,
	pub processing_time_ms: f64,
	pub timestamp: chrono::DateTime<chrono::Utc>,
	pub request_id: RequestId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(Uuid);

impl RequestId {
	pub fn new() -> Self { Self(Uuid::new_v4()) }
}

impl std::fmt::Display for RequestId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0.simple()) }
}

impl CalibrationMode {
	pub fn calibrate_identity(&self, raw_hbar: f64) -> (f64, RiskLevel, String) {
		let calibrated = raw_hbar;
		let abort_threshold = 0.1;
		let warn_threshold = 0.3;
		let proceed_threshold = 0.4;
		let risk_level = if raw_hbar < abort_threshold { RiskLevel::Critical } else if raw_hbar < warn_threshold { RiskLevel::Warning } else if raw_hbar < proceed_threshold { RiskLevel::HighRisk } else { RiskLevel::Safe };
		let explanation = format!("Calibration disabled: identity mapping (abort: {:.1}, warn: {:.1}, proceed: {:.1})", abort_threshold, warn_threshold, proceed_threshold);
		(calibrated, risk_level, explanation)
	}

	pub fn calibrate(&self, raw_hbar: f64) -> (f64, RiskLevel, String) {
		match self {
			CalibrationMode::Scientific { abort_threshold, warn_threshold, proceed_threshold } => {
				let calibrated = raw_hbar;
				let risk_level = if raw_hbar < *abort_threshold { 
					RiskLevel::Critical 
				} else if raw_hbar < *warn_threshold { 
					RiskLevel::Warning 
				} else if raw_hbar < *proceed_threshold { 
					RiskLevel::HighRisk 
				} else { 
					RiskLevel::Safe 
				};
				let explanation = format!("Scientific calibration: identity mapping (abort: {:.1}, warn: {:.1}, proceed: {:.1})", abort_threshold, warn_threshold, proceed_threshold);
				(calibrated, risk_level, explanation)
			},
			CalibrationMode::Pragmatic { scaling, abort_threshold, warn_threshold, proceed_threshold } => {
				let golden_factor = match scaling {
					GoldenScaling::EmpiricalGolden => 3.4,
					GoldenScaling::Custom(factor) => *factor,
				};
				
				let calibrated = raw_hbar * golden_factor;
				
				let risk_level = if calibrated < *abort_threshold { 
					RiskLevel::Critical 
				} else if calibrated < *warn_threshold { 
					RiskLevel::Warning 
				} else if calibrated < *proceed_threshold { 
					RiskLevel::HighRisk 
				} else { 
					RiskLevel::Safe 
				};
				
				let explanation = format!("Golden scale calibration: raw={:.4} × {:.1} = {:.4} (abort: {:.1}, warn: {:.1}, proceed: {:.1})", 
					raw_hbar, golden_factor, calibrated, abort_threshold, warn_threshold, proceed_threshold);
				
				(calibrated, risk_level, explanation)
			}
		}
	}
}

impl GoldenScaling {
	pub fn get_factor(&self) -> f64 {
		match self {
			GoldenScaling::EmpiricalGolden => 3.4,
			GoldenScaling::Custom(factor) => *factor,
		}
	}
	
	pub fn description(&self) -> &str {
		match self {
			GoldenScaling::EmpiricalGolden => "Empirical golden scale (3.4) optimized for hallucination detection",
			GoldenScaling::Custom(_) => "Custom scaling factor",
		}
	}
} 
pub use common::{RiskLevel, SemanticUncertaintyResult, CalibrationMode, RequestId};

pub mod audit_system;
pub mod scalar_firewall;
pub mod scalar_walk_firewall;
pub mod oss_logit_adapter;
pub mod mistral_integration;
pub mod alias_ambiguity_defense;
pub mod adaptive_learning;
pub mod validation;

#[cfg(feature = "candle")]
pub mod candle_integration;

pub mod metrics;

#[cfg(feature = "api")]
pub mod api;

#[cfg(feature = "api")]
pub use api::router; 
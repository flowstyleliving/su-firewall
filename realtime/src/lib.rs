pub use common::{RiskLevel, SemanticUncertaintyResult, CalibrationMode, RequestId};

// Core modules for semantic uncertainty analysis
pub mod oss_logit_adapter;      // OSS model logit extraction and analysis
pub mod mistral_integration;    // Mistral model integration and management

// Monitoring and validation
pub mod audit_system;          // Request auditing and logging
pub mod validation;           // Cross-domain and performance validation
pub mod metrics;             // Performance metrics collection

// Optional integrations
#[cfg(feature = "candle")]
pub mod candle_integration;   // Candle ML framework integration

#[cfg(feature = "api")]
pub mod api;                 // HTTP API endpoints for ensemble analysis

#[cfg(feature = "api")]
pub use api::router;
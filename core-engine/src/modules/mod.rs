// 🧠 Modular Semantic Uncertainty Engine
// Refactored architecture for scalability and maintainability

pub mod core_engine;
pub mod cost_tracker;
pub mod prompt_scorer;
pub mod rewrite_optimizer;
pub mod token_analyzer;
pub mod semantic_metrics;

// Re-exports for convenience
pub use core_engine::SemanticUncertaintyEngine;
pub use cost_tracker::{CostTracker, TokenCostAnalysis};
pub use prompt_scorer::{PromptScorer, ScoringResult};
pub use rewrite_optimizer::{RewriteOptimizer, OptimizationSuggestion};
pub use token_analyzer::{TokenAnalyzer, TokenEfficiency};
pub use semantic_metrics::{SemanticMetrics, MetricsSnapshot};
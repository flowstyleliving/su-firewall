// 🧠 Core Semantic Uncertainty Engine - Modular Architecture
// Refactored for performance, maintainability, and extensibility

use crate::modules::{CostTracker, PromptScorer, RewriteOptimizer, TokenAnalyzer, SemanticMetrics};
use crate::{CalibrationMode, RiskLevel, SemanticUncertaintyResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// 🎯 Analysis Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub calibration_mode: CalibrationMode,
    pub enable_cost_tracking: bool,
    pub enable_rewrite_suggestions: bool,
    pub enable_token_optimization: bool,
    pub batch_size: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            calibration_mode: CalibrationMode::default(),
            enable_cost_tracking: true,
            enable_rewrite_suggestions: true,
            enable_token_optimization: true,
            batch_size: 100,
        }
    }
}

/// 📊 Enhanced Analysis Result with Cost and Optimization Data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedAnalysisResult {
    /// Core semantic uncertainty result
    pub semantic_result: SemanticUncertaintyResult,
    
    /// Token-based cost analysis
    pub cost_analysis: Option<TokenCostAnalysis>,
    
    /// Rewrite suggestions for optimization
    pub rewrite_suggestions: Vec<OptimizationSuggestion>,
    
    /// Token efficiency metrics
    pub token_efficiency: Option<TokenEfficiency>,
    
    /// Processing metrics
    pub processing_metrics: ProcessingMetrics,
}

/// ⏱️ Processing Performance Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    pub total_time_ms: f64,
    pub core_analysis_time_ms: f64,
    pub cost_analysis_time_ms: f64,
    pub rewrite_analysis_time_ms: f64,
    pub token_analysis_time_ms: f64,
    pub throughput_analyses_per_second: f64,
}

use crate::modules::cost_tracker::TokenCostAnalysis;
use crate::modules::rewrite_optimizer::OptimizationSuggestion;
use crate::modules::token_analyzer::TokenEfficiency;

/// 🧠 Modular Semantic Uncertainty Engine
pub struct SemanticUncertaintyEngine {
    config: AnalysisConfig,
    cost_tracker: Option<Arc<CostTracker>>,
    prompt_scorer: Arc<PromptScorer>,
    rewrite_optimizer: Option<Arc<RewriteOptimizer>>,
    token_analyzer: Option<Arc<TokenAnalyzer>>,
    metrics: Arc<SemanticMetrics>,
}

impl SemanticUncertaintyEngine {
    /// 🚀 Create new modular engine with configuration
    pub fn new(config: AnalysisConfig) -> Result<Self> {
        let cost_tracker = if config.enable_cost_tracking {
            Some(Arc::new(CostTracker::new()?))
        } else {
            None
        };
        
        let rewrite_optimizer = if config.enable_rewrite_suggestions {
            Some(Arc::new(RewriteOptimizer::new()?))
        } else {
            None
        };
        
        let token_analyzer = if config.enable_token_optimization {
            Some(Arc::new(TokenAnalyzer::new()?))
        } else {
            None
        };
        
        Ok(Self {
            config,
            cost_tracker,
            prompt_scorer: Arc::new(PromptScorer::new()?),
            rewrite_optimizer,
            token_analyzer,
            metrics: Arc::new(SemanticMetrics::new()),
        })
    }
    
    /// 🎯 Analyze single prompt with full feature set
    #[instrument(skip(self, prompt, output))]
    pub async fn analyze_enhanced(
        &self,
        prompt: &str,
        output: &str,
        request_id: Option<Uuid>,
    ) -> Result<EnhancedAnalysisResult> {
        let start_time = Instant::now();
        let request_id = request_id.unwrap_or_else(Uuid::new_v4);
        
        debug!("Starting enhanced analysis for request: {}", request_id);
        
        // 🧮 Core semantic uncertainty analysis
        let core_start = Instant::now();
        let semantic_result = self.analyze_core(prompt, output, request_id).await?;
        let core_time = core_start.elapsed().as_millis() as f64;
        
        // 💰 Cost analysis (if enabled)
        let cost_start = Instant::now();
        let cost_analysis = if let Some(ref tracker) = self.cost_tracker {
            Some(tracker.analyze_cost(prompt, output, &semantic_result).await?)
        } else {
            None
        };
        let cost_time = cost_start.elapsed().as_millis() as f64;
        
        // ✏️ Rewrite suggestions (if enabled)
        let rewrite_start = Instant::now();
        let rewrite_suggestions = if let Some(ref optimizer) = self.rewrite_optimizer {
            optimizer.generate_suggestions(prompt, output, &semantic_result).await?
        } else {
            Vec::new()
        };
        let rewrite_time = rewrite_start.elapsed().as_millis() as f64;
        
        // 🔤 Token efficiency analysis (if enabled)
        let token_start = Instant::now();
        let token_efficiency = if let Some(ref analyzer) = self.token_analyzer {
            Some(analyzer.analyze_efficiency(prompt, output, &semantic_result).await?)
        } else {
            None
        };
        let token_time = token_start.elapsed().as_millis() as f64;
        
        let total_time = start_time.elapsed().as_millis() as f64;
        
        // 📊 Create processing metrics
        let processing_metrics = ProcessingMetrics {
            total_time_ms: total_time,
            core_analysis_time_ms: core_time,
            cost_analysis_time_ms: cost_time,
            rewrite_analysis_time_ms: rewrite_time,
            token_analysis_time_ms: token_time,
            throughput_analyses_per_second: if total_time > 0.0 { 1000.0 / total_time } else { 0.0 },
        };
        
        // 📈 Update metrics
        self.metrics.record_analysis(&semantic_result, &processing_metrics).await;
        
        info!(
            "Enhanced analysis complete: request={}, hbar={:.3}, risk={:?}, suggestions={}, cost_savings={:.2}%",
            request_id,
            semantic_result.calibrated_hbar,
            semantic_result.risk_level,
            rewrite_suggestions.len(),
            cost_analysis.as_ref().map(|c| c.potential_savings_percent).unwrap_or(0.0)
        );
        
        Ok(EnhancedAnalysisResult {
            semantic_result,
            cost_analysis,
            rewrite_suggestions,
            token_efficiency,
            processing_metrics,
        })
    }
    
    /// 🧮 Core semantic uncertainty calculation
    async fn analyze_core(
        &self,
        prompt: &str,
        output: &str,
        request_id: Uuid,
    ) -> Result<SemanticUncertaintyResult> {
        // Score the prompt-output pair
        let scoring_result = self.prompt_scorer.score(prompt, output).await?;
        
        // Calculate delta_mu (precision/stability)
        let delta_mu = self.calculate_delta_mu(&scoring_result);
        
        // Calculate delta_sigma (flexibility/chaos)
        let delta_sigma = self.calculate_delta_sigma(&scoring_result);
        
        // Calculate raw ℏₛ = √(Δμ × Δσ)
        let raw_hbar = (delta_mu * delta_sigma).sqrt();
        
        // Apply calibration
        let (calibrated_hbar, risk_level, explanation) = self.config.calibration_mode.calibrate(raw_hbar);
        
        Ok(SemanticUncertaintyResult {
            raw_hbar,
            calibrated_hbar,
            risk_level,
            calibration_mode: self.config.calibration_mode.clone(),
            explanation,
            delta_mu,
            delta_sigma,
            processing_time_ms: 0.0, // Will be updated by caller
            timestamp: chrono::Utc::now(),
            request_id: crate::RequestId::new(),
        })
    }
    
    /// 📐 Calculate semantic precision (Δμ)
    fn calculate_delta_mu(&self, scoring_result: &crate::modules::prompt_scorer::ScoringResult) -> f64 {
        // Higher specificity and lower ambiguity = higher precision
        let specificity_component = scoring_result.specificity_score * 0.4;
        let ambiguity_component = (1.0 - scoring_result.ambiguity_score) * 0.3;
        let coherence_component = scoring_result.coherence_score * 0.3;
        
        specificity_component + ambiguity_component + coherence_component
    }
    
    /// 🌊 Calculate semantic flexibility (Δσ)
    fn calculate_delta_sigma(&self, scoring_result: &crate::modules::prompt_scorer::ScoringResult) -> f64 {
        // Higher complexity and variance = higher flexibility
        let complexity_component = scoring_result.complexity_score * 0.5;
        let variance_component = scoring_result.semantic_variance * 0.3;
        let uncertainty_component = scoring_result.uncertainty_indicators * 0.2;
        
        complexity_component + variance_component + uncertainty_component
    }
    
    /// 📊 Get engine statistics
    pub async fn get_stats(&self) -> Result<EngineStats> {
        let metrics_snapshot = self.metrics.get_snapshot().await;
        
        Ok(EngineStats {
            total_analyses: metrics_snapshot.total_analyses,
            average_processing_time_ms: metrics_snapshot.average_processing_time_ms,
            risk_distribution: metrics_snapshot.risk_distribution,
            cost_savings_total: self.cost_tracker.as_ref()
                .map(|t| t.get_total_savings())
                .unwrap_or(0.0),
            optimization_suggestions_generated: self.rewrite_optimizer.as_ref()
                .map(|o| o.get_suggestions_count())
                .unwrap_or(0),
            token_efficiency_average: self.token_analyzer.as_ref()
                .map(|a| a.get_average_efficiency())
                .unwrap_or(0.0),
        })
    }
    
    /// 🔧 Update engine configuration
    pub fn update_config(&mut self, config: AnalysisConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }
    
    /// 🔄 Batch analysis with optimized processing
    pub async fn analyze_batch(
        &self,
        prompts: &[String],
        outputs: &[String],
    ) -> Result<Vec<EnhancedAnalysisResult>> {
        if prompts.len() != outputs.len() {
            return Err(anyhow::anyhow!("Prompt and output arrays must have the same length"));
        }
        
        let mut results = Vec::with_capacity(prompts.len());
        
        // Process in batches for memory efficiency
        for chunk in prompts.chunks(self.config.batch_size).zip(outputs.chunks(self.config.batch_size)) {
            let (prompt_chunk, output_chunk) = chunk;
            
            // Process batch concurrently
            let batch_futures: Vec<_> = prompt_chunk.iter()
                .zip(output_chunk.iter())
                .map(|(prompt, output)| {
                    self.analyze_enhanced(prompt, output, None)
                })
                .collect();
            
            let batch_results = futures::future::join_all(batch_futures).await;
            
            for result in batch_results {
                results.push(result?);
            }
        }
        
        Ok(results)
    }
}

/// 📊 Engine Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub total_analyses: u64,
    pub average_processing_time_ms: f64,
    pub risk_distribution: std::collections::HashMap<RiskLevel, u64>,
    pub cost_savings_total: f64,
    pub optimization_suggestions_generated: u64,
    pub token_efficiency_average: f64,
}

// Add missing RequestId type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestId {
    pub id: Uuid,
}

impl RequestId {
    pub fn new(id: Uuid) -> Self {
        Self { id }
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}
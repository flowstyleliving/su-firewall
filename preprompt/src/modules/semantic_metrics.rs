// 📊 Semantic Metrics - Performance Tracking and Analytics
// Comprehensive metrics collection for the modular semantic uncertainty engine

use crate::{RiskLevel, SemanticUncertaintyResult};
// Minimal local placeholder to decouple from removed core_engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessingMetrics {
    pub average_latency_ms: f64,
    pub throughput: f64,
    pub error_rate: f64,
}
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// 📈 Comprehensive Metrics Snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub total_analyses: u64,
    pub average_processing_time_ms: f64,
    pub risk_distribution: HashMap<RiskLevel, u64>,
    pub performance_trends: PerformanceTrends,
    pub quality_metrics: AggregateQualityMetrics,
    pub efficiency_statistics: EfficiencyStatistics,
}

/// 📊 Performance Trends Over Time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub throughput_trend: Vec<ThroughputPoint>,
    pub accuracy_trend: Vec<AccuracyPoint>,
    pub efficiency_trend: Vec<EfficiencyPoint>,
    pub cost_trend: Vec<CostPoint>,
}

/// 📈 Individual Trend Data Points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub analyses_per_second: f64,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub average_hbar: f64,
    pub prediction_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub token_efficiency: f64,
    pub cost_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cost_per_analysis: f64,
    pub savings_realized: f64,
}

/// 🎯 Aggregate Quality Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateQualityMetrics {
    pub average_semantic_uncertainty: f64,
    pub uncertainty_variance: f64,
    pub risk_accuracy: f64, // How accurately we predict actual risks
    pub calibration_effectiveness: f64, // How well calibration works
    pub stability_score: f64, // Consistency of results
}

/// ⚡ Efficiency Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyStatistics {
    pub average_tokens_per_analysis: f64,
    pub token_utilization_rate: f64,
    pub cost_optimization_rate: f64,
    pub processing_speed_percentiles: ProcessingPercentiles,
}

/// 📊 Processing Speed Distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPercentiles {
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// 📊 Real-time Metrics Collector
pub struct SemanticMetrics {
    /// 📈 Historical data storage
    analysis_history: Arc<RwLock<Vec<AnalysisRecord>>>,
    
    /// 📊 Aggregated statistics
    aggregated_stats: Arc<RwLock<AggregatedStats>>,
    
    /// ⏰ Performance tracking
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    
    /// 🎯 Quality assessor
    quality_assessor: Arc<RwLock<QualityAssessor>>,
}

/// 📝 Individual Analysis Record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalysisRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub request_id: String,
    pub semantic_result: SemanticAnalysisMetrics,
    pub processing_metrics: ProcessingMetrics,
    pub cost_metrics: Option<CostMetrics>,
    pub optimization_metrics: Option<OptimizationMetrics>,
}

/// 🧠 Semantic Analysis Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemanticAnalysisMetrics {
    pub raw_hbar: f64,
    pub calibrated_hbar: f64,
    pub risk_level: RiskLevel,
    pub delta_mu: f64,
    pub delta_sigma: f64,
    pub calibration_mode: String,
}

/// 💰 Cost Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CostMetrics {
    pub total_tokens: u32,
    pub estimated_cost: f64,
    pub potential_savings: f64,
    pub efficiency_score: f64,
}

/// ✨ Optimization Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationMetrics {
    pub suggestions_generated: u32,
    pub estimated_token_savings: u32,
    pub optimization_categories: Vec<String>,
}

/// 📊 Aggregated Statistics
#[derive(Debug, Default)]
struct AggregatedStats {
    pub total_analyses: u64,
    pub total_processing_time_ms: f64,
    pub risk_counts: HashMap<RiskLevel, u64>,
    pub hbar_sum: f64,
    pub hbar_squared_sum: f64,
    pub token_sum: u64,
    pub cost_sum: f64,
    pub savings_sum: f64,
}

/// ⚡ Performance Tracker
#[derive(Debug, Default)]
struct PerformanceTracker {
    pub processing_times: Vec<f64>,
    pub throughput_history: Vec<ThroughputPoint>,
    pub recent_performance_window: std::collections::VecDeque<f64>,
}

/// 🎯 Quality Assessor
#[derive(Debug, Default)]
struct QualityAssessor {
    pub prediction_history: Vec<(f64, RiskLevel)>, // (predicted_hbar, actual_risk)
    pub calibration_history: Vec<(f64, f64)>, // (raw_hbar, calibrated_hbar)
    pub stability_measurements: Vec<f64>,
}

impl SemanticMetrics {
    /// 🚀 Create new metrics collector
    pub fn new() -> Self {
        Self {
            analysis_history: Arc::new(RwLock::new(Vec::new())),
            aggregated_stats: Arc::new(RwLock::new(AggregatedStats::default())),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::default())),
            quality_assessor: Arc::new(RwLock::new(QualityAssessor::default())),
        }
    }
    
    /// 📝 Record a new analysis
    pub async fn record_analysis(
        &self,
        semantic_result: &SemanticUncertaintyResult,
        processing_metrics: &ProcessingMetrics,
    ) {
        let record = AnalysisRecord {
            timestamp: chrono::Utc::now(),
            request_id: semantic_result.request_id.to_string(),
            semantic_result: SemanticAnalysisMetrics {
                raw_hbar: semantic_result.raw_hbar,
                calibrated_hbar: semantic_result.calibrated_hbar,
                risk_level: semantic_result.risk_level,
                delta_mu: semantic_result.delta_mu,
                delta_sigma: semantic_result.delta_sigma,
                calibration_mode: format!("{:?}", semantic_result.calibration_mode),
            },
            processing_metrics: processing_metrics.clone(),
            cost_metrics: None, // Would be populated if available
            optimization_metrics: None, // Would be populated if available
        };
        
        // Store record
        {
            let mut history = self.analysis_history.write().await;
            history.push(record.clone());
            
            // Keep only recent records (last 10,000)
            if history.len() > 10_000 {
                history.drain(0..1_000);
            }
        }
        
        // Update aggregated statistics
        self.update_aggregated_stats(&record).await;
        
        // Update performance tracking
        self.update_performance_tracking(&record).await;
        
        // Update quality assessment
        self.update_quality_assessment(&record).await;
        
        debug!(
            "Recorded analysis: hbar={:.3}, risk={:?}, time={:.1}ms",
            semantic_result.calibrated_hbar,
            semantic_result.risk_level,
            processing_metrics.average_latency_ms
        );
    }
    
    /// 📊 Get comprehensive metrics snapshot
    pub async fn get_snapshot(&self) -> MetricsSnapshot {
        let stats = self.aggregated_stats.read().await;
        let performance = self.performance_tracker.read().await;
        let quality = self.quality_assessor.read().await;
        
        let total_analyses = stats.total_analyses;
        let average_processing_time_ms = if total_analyses > 0 {
            stats.total_processing_time_ms / total_analyses as f64
        } else {
            0.0
        };
        
        let risk_distribution = stats.risk_counts.clone();
        
        // Calculate performance trends
        let performance_trends = PerformanceTrends {
            throughput_trend: performance.throughput_history.clone(),
            accuracy_trend: self.calculate_accuracy_trend(&quality).await,
            efficiency_trend: self.calculate_efficiency_trend().await,
            cost_trend: self.calculate_cost_trend().await,
        };
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(&stats, &quality).await;
        
        // Calculate efficiency statistics
        let efficiency_statistics = self.calculate_efficiency_statistics(&stats, &performance).await;
        
        MetricsSnapshot {
            total_analyses,
            average_processing_time_ms,
            risk_distribution,
            performance_trends,
            quality_metrics,
            efficiency_statistics,
        }
    }
    
    /// 📈 Update aggregated statistics
    async fn update_aggregated_stats(&self, record: &AnalysisRecord) {
        let mut stats = self.aggregated_stats.write().await;
        
        stats.total_analyses += 1;
        stats.total_processing_time_ms += record.processing_metrics.average_latency_ms;
        
        // Update risk counts
        *stats.risk_counts.entry(record.semantic_result.risk_level).or_insert(0) += 1;
        
        // Update hbar statistics for variance calculation
        stats.hbar_sum += record.semantic_result.calibrated_hbar;
        stats.hbar_squared_sum += record.semantic_result.calibrated_hbar.powi(2);
        
        // Update cost and token statistics
        if let Some(cost_metrics) = &record.cost_metrics {
            stats.token_sum += cost_metrics.total_tokens as u64;
            stats.cost_sum += cost_metrics.estimated_cost;
            stats.savings_sum += cost_metrics.potential_savings;
        }
    }
    
    /// ⚡ Update performance tracking
    async fn update_performance_tracking(&self, record: &AnalysisRecord) {
        let mut performance = self.performance_tracker.write().await;
        
        // Track processing times
        performance.processing_times.push(record.processing_metrics.average_latency_ms);
        
        // Keep processing times window for percentile calculations
        performance.recent_performance_window.push_back(record.processing_metrics.average_latency_ms);
        if performance.recent_performance_window.len() > 1000 {
            performance.recent_performance_window.pop_front();
        }
        
        // Track throughput
        let throughput_point = ThroughputPoint {
            timestamp: record.timestamp,
            analyses_per_second: record.processing_metrics.throughput,
            batch_size: 1, // Single analysis
        };
        performance.throughput_history.push(throughput_point);
        
        // Keep only recent throughput data
        if performance.throughput_history.len() > 1000 {
            performance.throughput_history.drain(0..100);
        }
    }
    
    /// 🎯 Update quality assessment
    async fn update_quality_assessment(&self, record: &AnalysisRecord) {
        let mut quality = self.quality_assessor.write().await;
        
        // Track prediction accuracy
        quality.prediction_history.push((
            record.semantic_result.calibrated_hbar,
            record.semantic_result.risk_level,
        ));
        
        // Track calibration effectiveness
        quality.calibration_history.push((
            record.semantic_result.raw_hbar,
            record.semantic_result.calibrated_hbar,
        ));
        
        // Measure stability (variance in similar analyses)
        quality.stability_measurements.push(record.semantic_result.calibrated_hbar);
        
        // Keep only recent data
        if quality.prediction_history.len() > 10000 {
            quality.prediction_history.drain(0..1000);
        }
        if quality.calibration_history.len() > 10000 {
            quality.calibration_history.drain(0..1000);
        }
        if quality.stability_measurements.len() > 10000 {
            quality.stability_measurements.drain(0..1000);
        }
    }
    
    /// 📊 Calculate quality metrics
    async fn calculate_quality_metrics(
        &self,
        stats: &AggregatedStats,
        quality: &QualityAssessor,
    ) -> AggregateQualityMetrics {
        let average_semantic_uncertainty = if stats.total_analyses > 0 {
            stats.hbar_sum / stats.total_analyses as f64
        } else {
            0.0
        };
        
        let uncertainty_variance = if stats.total_analyses > 1 {
            let mean = average_semantic_uncertainty;
            let variance = (stats.hbar_squared_sum / stats.total_analyses as f64) - mean.powi(2);
            variance.max(0.0)
        } else {
            0.0
        };
        
        // Calculate risk prediction accuracy
        let risk_accuracy = self.calculate_risk_prediction_accuracy(&quality.prediction_history);
        
        // Calculate calibration effectiveness
        let calibration_effectiveness = self.calculate_calibration_effectiveness(&quality.calibration_history);
        
        // Calculate stability score
        let stability_score = self.calculate_stability_score(&quality.stability_measurements);
        
        AggregateQualityMetrics {
            average_semantic_uncertainty,
            uncertainty_variance,
            risk_accuracy,
            calibration_effectiveness,
            stability_score,
        }
    }
    
    /// ⚡ Calculate efficiency statistics
    async fn calculate_efficiency_statistics(
        &self,
        stats: &AggregatedStats,
        performance: &PerformanceTracker,
    ) -> EfficiencyStatistics {
        let average_tokens_per_analysis = if stats.total_analyses > 0 {
            stats.token_sum as f64 / stats.total_analyses as f64
        } else {
            0.0
        };
        
        let token_utilization_rate = 0.85; // Placeholder - would calculate from actual data
        let cost_optimization_rate = if stats.cost_sum > 0.0 {
            stats.savings_sum / stats.cost_sum
        } else {
            0.0
        };
        
        let processing_speed_percentiles = self.calculate_processing_percentiles(&performance.recent_performance_window);
        
        EfficiencyStatistics {
            average_tokens_per_analysis,
            token_utilization_rate,
            cost_optimization_rate,
            processing_speed_percentiles,
        }
    }
    
    /// 📈 Calculate processing percentiles
    fn calculate_processing_percentiles(
        &self,
        times: &std::collections::VecDeque<f64>,
    ) -> ProcessingPercentiles {
        if times.is_empty() {
            return ProcessingPercentiles {
                p50_ms: 0.0,
                p90_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
            };
        }
        
        let mut sorted_times: Vec<f64> = times.iter().cloned().collect();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted_times.len();
        let p50_idx = len / 2;
        let p90_idx = (len as f64 * 0.9) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;
        
        ProcessingPercentiles {
            p50_ms: sorted_times[p50_idx.min(len - 1)],
            p90_ms: sorted_times[p90_idx.min(len - 1)],
            p95_ms: sorted_times[p95_idx.min(len - 1)],
            p99_ms: sorted_times[p99_idx.min(len - 1)],
        }
    }
    
    /// 🎯 Calculate risk prediction accuracy
    fn calculate_risk_prediction_accuracy(&self, history: &[(f64, RiskLevel)]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        
        let mut correct_predictions = 0;
        for (hbar, actual_risk) in history {
            let predicted_risk = if *hbar < 0.8 {
                RiskLevel::Critical
            } else if *hbar < 1.0 {
                RiskLevel::Warning
            } else if *hbar < 1.2 {
                RiskLevel::HighRisk
            } else {
                RiskLevel::Safe
            };
            
            if predicted_risk == *actual_risk {
                correct_predictions += 1;
            }
        }
        
        correct_predictions as f64 / history.len() as f64
    }
    
    /// 🔧 Calculate calibration effectiveness
    fn calculate_calibration_effectiveness(&self, history: &[(f64, f64)]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        
        // Measure how well calibration improves usability
        let mut improvement_sum = 0.0;
        for (raw_hbar, calibrated_hbar) in history {
            // Better calibration should move values into more usable ranges
            let raw_usability = self.calculate_usability_score(*raw_hbar);
            let calibrated_usability = self.calculate_usability_score(*calibrated_hbar);
            improvement_sum += calibrated_usability - raw_usability;
        }
        
        (improvement_sum / history.len() as f64).max(0.0).min(1.0)
    }
    
    /// 📊 Calculate usability score for hbar value
    fn calculate_usability_score(&self, hbar: f64) -> f64 {
        // Values closer to decision thresholds (0.8, 1.0, 1.2) are more usable
        let distances = [
            (hbar - 0.8).abs(),
            (hbar - 1.0).abs(),
            (hbar - 1.2).abs(),
        ];
        let min_distance = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        (1.0 - min_distance).max(0.0).min(1.0)
    }
    
    /// 🔄 Calculate stability score
    fn calculate_stability_score(&self, measurements: &[f64]) -> f64 {
        if measurements.len() < 2 {
            return 1.0; // Perfect stability with insufficient data
        }
        
        // Calculate coefficient of variation (lower is more stable)
        let mean = measurements.iter().sum::<f64>() / measurements.len() as f64;
        let variance = measurements.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / measurements.len() as f64;
        let std_dev = variance.sqrt();
        
        if mean > 0.0 {
            let cv = std_dev / mean;
            (1.0 - cv).max(0.0).min(1.0) // Invert so higher is better
        } else {
            0.5 // Neutral stability
        }
    }
    
    // Placeholder trend calculation methods
    async fn calculate_accuracy_trend(&self, _quality: &QualityAssessor) -> Vec<AccuracyPoint> {
        Vec::new() // Would implement actual trend calculation
    }
    
    async fn calculate_efficiency_trend(&self) -> Vec<EfficiencyPoint> {
        Vec::new() // Would implement actual trend calculation
    }
    
    async fn calculate_cost_trend(&self) -> Vec<CostPoint> {
        Vec::new() // Would implement actual trend calculation
    }
    
    /// 📊 Get real-time performance summary
    pub async fn get_realtime_summary(&self) -> Result<RealtimeSummary> {
        let stats = self.aggregated_stats.read().await;
        let performance = self.performance_tracker.read().await;
        
        let current_throughput = if !performance.recent_performance_window.is_empty() {
            let recent_avg_time = performance.recent_performance_window.iter().sum::<f64>() 
                / performance.recent_performance_window.len() as f64;
            if recent_avg_time > 0.0 {
                1000.0 / recent_avg_time // analyses per second
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let risk_distribution_percent: HashMap<RiskLevel, f64> = stats.risk_counts.iter()
            .map(|(risk, count)| (*risk, *count as f64 / stats.total_analyses.max(1) as f64 * 100.0))
            .collect();
        
        Ok(RealtimeSummary {
            total_analyses: stats.total_analyses,
            current_throughput,
            average_hbar: if stats.total_analyses > 0 {
                stats.hbar_sum / stats.total_analyses as f64
            } else {
                0.0
            },
            risk_distribution_percent,
            system_health: self.calculate_system_health(&stats, &performance).await,
        })
    }
    
    /// 🏥 Calculate system health score
    async fn calculate_system_health(
        &self,
        stats: &AggregatedStats,
        performance: &PerformanceTracker,
    ) -> f64 {
        let mut health_score: f64 = 1.0;
        
        // Check processing time health
        if !performance.recent_performance_window.is_empty() {
            let avg_time = performance.recent_performance_window.iter().sum::<f64>() 
                / performance.recent_performance_window.len() as f64;
            if avg_time > 1000.0 { // More than 1 second is concerning
                health_score *= 0.7;
            }
        }
        
        // Check error rate (would track errors in real implementation)
        // health_score *= (1.0 - error_rate);
        
        // Check throughput consistency
        if performance.throughput_history.len() > 10 {
            let recent_throughputs: Vec<f64> = performance.throughput_history
                .iter()
                .rev()
                .take(10)
                .map(|p| p.analyses_per_second)
                .collect();
            
            let variance = self.calculate_variance(&recent_throughputs);
            if variance > 100.0 { // High variance in throughput
                health_score *= 0.8;
            }
        }
        
        health_score.max(0.0).min(1.0)
    }
    
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64
    }
}

/// ⚡ Real-time System Summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSummary {
    pub total_analyses: u64,
    pub current_throughput: f64,
    pub average_hbar: f64,
    pub risk_distribution_percent: HashMap<RiskLevel, f64>,
    pub system_health: f64, // 0-1 scale
}
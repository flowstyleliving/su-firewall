use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tokio::time::{interval, Duration, Instant};
use serde::{Deserialize, Serialize};
use common::optimization::{LambdaTauOptimizer, OptimizationConfig, ValidationMetrics, OptimizationError};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    pub learning_rate: f64,           // 0.1
    pub update_frequency: Duration,   // Every 100 requests or 1 hour
    pub performance_window: usize,    // Track last 50 requests
    pub auto_update_threshold: f64,   // Auto-update if improvement > 5%
    pub min_samples_for_update: usize, // Minimum samples before triggering update
    pub decay_factor: f64,            // 0.95 - weight recent predictions more
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            update_frequency: Duration::from_secs(3600), // 1 hour
            performance_window: 50,
            auto_update_threshold: 0.05, // 5% improvement
            min_samples_for_update: 25,
            decay_factor: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictionRecord {
    pub predicted_hallucination: bool,
    pub actual_hallucination: bool,
    pub hbar_s: f64,
    pub p_fail: f64,
    pub model_id: String,
    pub timestamp: Instant,
}

pub struct PerformanceTracker {
    config: AdaptiveLearningConfig,
    recent_predictions: Arc<Mutex<VecDeque<PredictionRecord>>>,
    current_f1: Arc<Mutex<f64>>,
    parameter_history: Arc<Mutex<Vec<(f64, f64, f64, Instant)>>>, // (lambda, tau, f1, timestamp)
    last_optimization: Arc<Mutex<Option<Instant>>>,
    optimization_in_progress: Arc<Mutex<bool>>,
}

impl PerformanceTracker {
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            config,
            recent_predictions: Arc::new(Mutex::new(VecDeque::new())),
            current_f1: Arc::new(Mutex::new(0.0)),
            parameter_history: Arc::new(Mutex::new(Vec::new())),
            last_optimization: Arc::new(Mutex::new(None)),
            optimization_in_progress: Arc::new(Mutex::new(false)),
        }
    }

    pub fn record_prediction(&self, record: PredictionRecord) {
        let mut predictions = self.recent_predictions.lock().unwrap();
        
        // Add new prediction
        predictions.push_back(record);
        
        // Maintain window size
        while predictions.len() > self.config.performance_window {
            predictions.pop_front();
        }
        
        // Update current F1 score
        if predictions.len() >= 10 { // Minimum for meaningful F1
            let f1 = self.calculate_rolling_f1(&predictions);
            *self.current_f1.lock().unwrap() = f1;
        }
    }

    pub fn should_trigger_optimization(&self) -> bool {
        let predictions = self.recent_predictions.lock().unwrap();
        let optimization_in_progress = *self.optimization_in_progress.lock().unwrap();
        let last_optimization = *self.last_optimization.lock().unwrap();
        
        // Don't trigger if already optimizing
        if optimization_in_progress {
            return false;
        }
        
        // Check minimum samples
        if predictions.len() < self.config.min_samples_for_update {
            return false;
        }
        
        // Check time since last optimization
        if let Some(last_opt) = last_optimization {
            if last_opt.elapsed() < self.config.update_frequency {
                return false;
            }
        }
        
        // Check performance degradation
        let current_f1 = *self.current_f1.lock().unwrap();
        if let Some((_, _, last_best_f1, _)) = self.parameter_history.lock().unwrap().last() {
            if current_f1 < last_best_f1 - self.config.auto_update_threshold {
                println!("📉 Performance degradation detected: {:.3} → {:.3}", last_best_f1, current_f1);
                return true;
            }
        }
        
        // Regular periodic optimization
        true
    }

    pub async fn adaptive_optimize(&self, model_id: &str) -> Result<(f64, f64), OptimizationError> {
        // Set optimization in progress
        *self.optimization_in_progress.lock().unwrap() = true;
        
        println!("🚀 Starting adaptive optimization for model: {}", model_id);
        
        // Create focused optimization config for adaptive learning
        let adaptive_config = OptimizationConfig {
            lambda_range: (0.05, 2.0),  // Narrower range for faster convergence
            tau_range: (0.1, 1.5),
            lambda_steps: 20,            // Fewer steps for real-time optimization
            tau_steps: 15,
            validation_samples: 500,     // Smaller sample for speed
            target_metric: "f1_score".to_string(),
            min_improvement: 0.02,       // Higher threshold for adaptive updates
            convergence_threshold: 0.005,
            max_iterations: 50,
        };

        let dataset_path = PathBuf::from("authentic_datasets");
        let optimizer = LambdaTauOptimizer::new(adaptive_config, dataset_path);
        
        let result = optimizer.optimize_for_model(model_id).await?;
        
        // Record optimization result
        let now = Instant::now();
        self.parameter_history.lock().unwrap().push((
            result.best_lambda,
            result.best_tau,
            result.best_f1,
            now
        ));
        
        *self.last_optimization.lock().unwrap() = Some(now);
        *self.optimization_in_progress.lock().unwrap() = false;
        
        println!("✅ Adaptive optimization complete: λ={:.3}, τ={:.3}, F1={:.3}", 
                result.best_lambda, result.best_tau, result.best_f1);
        
        Ok((result.best_lambda, result.best_tau))
    }

    fn calculate_rolling_f1(&self, predictions: &VecDeque<PredictionRecord>) -> f64 {
        let mut true_positives = 0.0;
        let mut false_positives = 0.0;
        let mut false_negatives = 0.0;
        
        for (i, record) in predictions.iter().enumerate() {
            // Apply decay factor - more recent predictions have higher weight
            let weight = self.config.decay_factor.powi((predictions.len() - 1 - i) as i32);
            
            match (record.actual_hallucination, record.predicted_hallucination) {
                (true, true) => true_positives += weight,
                (false, true) => false_positives += weight,
                (true, false) => false_negatives += weight,
                (false, false) => { /* true_negatives - not needed for F1 */ }
            }
        }
        
        let precision = if (true_positives + false_positives) > 0.0 {
            true_positives / (true_positives + false_positives)
        } else { 0.0 };
        
        let recall = if (true_positives + false_negatives) > 0.0 {
            true_positives / (true_positives + false_negatives)
        } else { 0.0 };
        
        if (precision + recall) > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        }
    }

    pub fn get_current_metrics(&self) -> (f64, usize, bool) {
        let f1 = *self.current_f1.lock().unwrap();
        let sample_count = self.recent_predictions.lock().unwrap().len();
        let optimization_active = *self.optimization_in_progress.lock().unwrap();
        
        (f1, sample_count, optimization_active)
    }

    pub fn get_parameter_history(&self) -> Vec<(f64, f64, f64, Instant)> {
        self.parameter_history.lock().unwrap().clone()
    }

    // Background optimization task
    pub async fn start_background_optimization(&self, model_id: String) {
        let mut interval = interval(self.config.update_frequency);
        let model_id = Arc::new(model_id);
        
        loop {
            interval.tick().await;
            
            if self.should_trigger_optimization() {
                let model_id_clone = Arc::clone(&model_id);
                let self_clone = self.clone();
                
                tokio::spawn(async move {
                    if let Err(e) = self_clone.adaptive_optimize(&model_id_clone).await {
                        eprintln!("🚨 Background optimization failed: {}", e);
                    }
                });
            }
        }
    }
}

impl Clone for PerformanceTracker {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            recent_predictions: Arc::clone(&self.recent_predictions),
            current_f1: Arc::clone(&self.current_f1),
            parameter_history: Arc::clone(&self.parameter_history),
            last_optimization: Arc::clone(&self.last_optimization),
            optimization_in_progress: Arc::clone(&self.optimization_in_progress),
        }
    }
}

// Global performance tracker instance
use std::sync::OnceLock;
pub static PERFORMANCE_TRACKER: OnceLock<PerformanceTracker> = OnceLock::new();
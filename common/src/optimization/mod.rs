use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OptimizationError {
    #[error("Failed to load ground truth data: {0}")]
    DataLoadError(String),
    #[error("Invalid parameter range: {0}")]
    InvalidRange(String),
    #[error("Optimization convergence failed: {0}")]
    ConvergenceError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub lambda_range: (f64, f64),     // (0.1, 10.0)
    pub tau_range: (f64, f64),        // (0.1, 3.0)
    pub lambda_steps: usize,          // 50 for high-resolution search
    pub tau_steps: usize,             // 30 for high-resolution search
    pub validation_samples: usize,    // 1000
    pub target_metric: String,        // "f1_score"
    pub min_improvement: f64,         // 0.01
    pub convergence_threshold: f64,   // 0.001
    pub max_iterations: usize,        // 100
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            lambda_range: (0.1, 10.0),
            tau_range: (0.1, 3.0),
            lambda_steps: 50,
            tau_steps: 30,
            validation_samples: 1000,
            target_metric: "f1_score".to_string(),
            min_improvement: 0.01,
            convergence_threshold: 0.001,
            max_iterations: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub best_lambda: f64,
    pub best_tau: f64,
    pub best_f1: f64,
    pub best_accuracy: f64,
    pub best_precision: f64,
    pub best_recall: f64,
    pub iterations: usize,
    pub validation_samples_used: usize,
    pub optimization_time_ms: f64,
    pub convergence_achieved: bool,
    pub parameter_history: Vec<ParameterStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStep {
    pub lambda: f64,
    pub tau: f64,
    pub f1_score: f64,
    pub accuracy: f64,
    pub step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthSample {
    pub prompt: String,
    pub correct_answer: String,
    pub hallucinated_answer: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

pub struct GroundTruthLoader {
    dataset_path: PathBuf,
}

impl GroundTruthLoader {
    pub fn new(dataset_path: PathBuf) -> Self {
        Self { dataset_path }
    }

    pub async fn load_validation_samples(&self, count: usize) -> Result<Vec<GroundTruthSample>, OptimizationError> {
        use std::fs;
        use std::io::{BufRead, BufReader};

        let mut samples = Vec::new();
        
        // Load TruthfulQA
        let truthfulqa_path = self.dataset_path.join("truthfulqa_data.json");
        if truthfulqa_path.exists() {
            let content = fs::read_to_string(&truthfulqa_path)?;
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(validation) = data["validation"].as_array() {
                    for item in validation.iter().take(count / 4) {
                        if let (Some(question), Some(best_answer), Some(incorrect_answers)) = (
                            item["Question"].as_str(),
                            item["Best Answer"].as_str(),
                            item["Incorrect Answers"].as_array()
                        ) {
                            if !incorrect_answers.is_empty() {
                                if let Some(wrong_answer) = incorrect_answers[0].as_str() {
                                    samples.push(GroundTruthSample {
                                        prompt: question.to_string(),
                                        correct_answer: best_answer.to_string(),
                                        hallucinated_answer: wrong_answer.to_string(),
                                        source: "truthfulqa".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Load HaluEval General
        let halueval_path = self.dataset_path.join("halueval_general_data.json");
        if halueval_path.exists() {
            let file = fs::File::open(&halueval_path)?;
            let reader = BufReader::new(file);
            
            for (line_num, line) in reader.lines().enumerate() {
                if line_num >= (count * 3 / 4) { break; }
                
                let line = line?;
                if line.trim().is_empty() { continue; }
                
                if let Ok(item) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let (Some(query), Some(response)) = (
                        item["user_query"].as_str(),
                        item["chatgpt_response"].as_str()
                    ) {
                        let is_hallucination = item["hallucination"].as_str() == Some("yes");
                        
                        if is_hallucination {
                            samples.push(GroundTruthSample {
                                prompt: query.to_string(),
                                correct_answer: "I should provide accurate information.".to_string(),
                                hallucinated_answer: response.to_string(),
                                source: "halueval_general".to_string(),
                            });
                        } else {
                            samples.push(GroundTruthSample {
                                prompt: query.to_string(),
                                correct_answer: response.to_string(),
                                hallucinated_answer: "[Fabricated incorrect response]".to_string(),
                                source: "halueval_general".to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(samples)
    }
}

pub struct LambdaTauOptimizer {
    config: OptimizationConfig,
    ground_truth_loader: GroundTruthLoader,
}

impl LambdaTauOptimizer {
    pub fn new(config: OptimizationConfig, dataset_path: PathBuf) -> Self {
        Self {
            config,
            ground_truth_loader: GroundTruthLoader::new(dataset_path),
        }
    }

    pub async fn optimize_for_model(&self, model_id: &str) -> Result<OptimizationResult, OptimizationError> {
        let start_time = std::time::Instant::now();
        
        println!("🎯 Starting λ/τ optimization for model: {}", model_id);
        println!("📊 Lambda range: {:?}, steps: {}", self.config.lambda_range, self.config.lambda_steps);
        println!("📊 Tau range: {:?}, steps: {}", self.config.tau_range, self.config.tau_steps);
        
        // Load validation samples
        let samples = self.ground_truth_loader
            .load_validation_samples(self.config.validation_samples)
            .await?;
        
        println!("✅ Loaded {} validation samples", samples.len());
        
        let mut best_lambda = self.config.lambda_range.0;
        let mut best_tau = self.config.tau_range.0;
        let mut best_f1 = 0.0;
        let mut best_metrics = ValidationMetrics {
            accuracy: 0.0, precision: 0.0, recall: 0.0, f1_score: 0.0,
            true_positives: 0, false_positives: 0, true_negatives: 0, false_negatives: 0,
        };
        let mut parameter_history = Vec::new();
        let mut iteration = 0;

        // Grid search
        let lambda_step = (self.config.lambda_range.1 - self.config.lambda_range.0) / self.config.lambda_steps as f64;
        let tau_step = (self.config.tau_range.1 - self.config.tau_range.0) / self.config.tau_steps as f64;

        for lambda_i in 0..=self.config.lambda_steps {
            let lambda = self.config.lambda_range.0 + lambda_i as f64 * lambda_step;
            
            for tau_i in 0..=self.config.tau_steps {
                let tau = self.config.tau_range.0 + tau_i as f64 * tau_step;
                iteration += 1;
                
                if iteration % 50 == 0 {
                    println!("📈 Progress: {}/{} parameter combinations", 
                            iteration, (self.config.lambda_steps + 1) * (self.config.tau_steps + 1));
                }

                // Evaluate this parameter combination
                match self.evaluate_parameters(lambda, tau, model_id, &samples).await {
                    Ok(metrics) => {
                        parameter_history.push(ParameterStep {
                            lambda,
                            tau,
                            f1_score: metrics.f1_score,
                            accuracy: metrics.accuracy,
                            step: iteration,
                        });

                        if metrics.f1_score > best_f1 + self.config.min_improvement {
                            best_lambda = lambda;
                            best_tau = tau;
                            best_f1 = metrics.f1_score;
                            best_metrics = metrics;
                            
                            println!("🏆 New best: λ={:.3}, τ={:.3}, F1={:.3}", lambda, tau, best_f1);
                        }
                    },
                    Err(e) => {
                        eprintln!("⚠️ Evaluation failed for λ={:.3}, τ={:.3}: {}", lambda, tau, e);
                    }
                }
            }
        }

        let optimization_time = start_time.elapsed().as_millis() as f64;
        
        println!("🎉 Optimization complete!");
        println!("🏆 Best parameters: λ={:.3}, τ={:.3}", best_lambda, best_tau);
        println!("📊 Best F1-score: {:.3}", best_f1);
        
        Ok(OptimizationResult {
            best_lambda,
            best_tau,
            best_f1,
            best_accuracy: best_metrics.accuracy,
            best_precision: best_metrics.precision,
            best_recall: best_metrics.recall,
            iterations: iteration,
            validation_samples_used: samples.len(),
            optimization_time_ms: optimization_time,
            convergence_achieved: best_f1 > 0.5, // Reasonable threshold
            parameter_history,
        })
    }

    pub async fn evaluate_parameters(&self, lambda: f64, tau: f64, model_id: &str, samples: &[GroundTruthSample]) -> Result<ValidationMetrics, OptimizationError> {
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut true_negatives = 0;
        let mut false_negatives = 0;

        for sample in samples {
            // Test both correct and hallucinated answers
            let test_cases = [
                (&sample.correct_answer, false),
                (&sample.hallucinated_answer, true),
            ];

            for (output, is_hallucination) in test_cases {
                // Mock uncertainty calculation (would be replaced with actual API call)
                let mock_hbar_s = self.mock_uncertainty_calculation(&sample.prompt, output);
                
                // Apply lambda/tau parameters
                let p_fail = 1.0 / (1.0 + (-lambda * (mock_hbar_s - tau)).exp());
                
                // Prediction logic: combine ℏₛ and P(fail) thresholds
                let predicted_hallucination = (mock_hbar_s < 1.0) || (p_fail > 0.5);
                
                match (is_hallucination, predicted_hallucination) {
                    (true, true) => true_positives += 1,
                    (false, true) => false_positives += 1,
                    (true, false) => false_negatives += 1,
                    (false, false) => true_negatives += 1,
                }
            }
        }

        let total = true_positives + false_positives + true_negatives + false_negatives;
        let accuracy = if total > 0 { (true_positives + true_negatives) as f64 / total as f64 } else { 0.0 };
        
        let precision = if (true_positives + false_positives) > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else { 0.0 };
        
        let recall = if (true_positives + false_negatives) > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else { 0.0 };
        
        let f1_score = if (precision + recall) > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else { 0.0 };

        Ok(ValidationMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        })
    }

    // Mock uncertainty calculation for optimization
    fn mock_uncertainty_calculation(&self, prompt: &str, output: &str) -> f64 {
        // Simple heuristic based on text properties for optimization
        let prompt_len = prompt.len() as f64;
        let output_len = output.len() as f64;
        let ratio = output_len / prompt_len.max(1.0);
        
        // Simulate uncertainty based on length ratio and content patterns
        let base_uncertainty = if ratio > 3.0 { 0.8 } else if ratio < 0.5 { 1.5 } else { 1.2 };
        
        // Add some variation based on content
        let variation = (prompt.chars().count() % 10) as f64 * 0.1;
        base_uncertainty + variation
    }
}

pub fn calculate_f1_metrics(predictions: &[bool], ground_truth: &[bool]) -> ValidationMetrics {
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;

    for (pred, actual) in predictions.iter().zip(ground_truth.iter()) {
        match (*actual, *pred) {
            (true, true) => true_positives += 1,
            (false, true) => false_positives += 1,
            (true, false) => false_negatives += 1,
            (false, false) => true_negatives += 1,
        }
    }

    let total = true_positives + false_positives + true_negatives + false_negatives;
    let accuracy = if total > 0 { (true_positives + true_negatives) as f64 / total as f64 } else { 0.0 };
    
    let precision = if (true_positives + false_positives) > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else { 0.0 };
    
    let recall = if (true_positives + false_negatives) > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else { 0.0 };
    
    let f1_score = if (precision + recall) > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else { 0.0 };

    ValidationMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    }
}
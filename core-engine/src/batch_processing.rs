// ⚡ Ultra-fast Batch Processing Engine (Rust Implementation)
// Parallel semantic uncertainty analysis with advanced error handling

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, warn, instrument};

use crate::{SemanticAnalyzer, SemanticConfig, SemanticError, HbarResponse, RequestId};

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_concurrent: usize,
    pub timeout_ms: u64,
    pub fail_fast: bool,
    pub preserve_order: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 50,
            max_concurrent: 10,
            timeout_ms: 30000,
            fail_fast: false,
            preserve_order: true,
        }
    }
}

/// Individual prompt analysis request
#[derive(Debug, Clone)]
pub struct PromptRequest {
    pub prompt: String,
    pub model: String,
    pub index: usize,
}

/// Individual analysis result with error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResult {
    pub index: usize,
    pub prompt: String,
    pub success: bool,
    pub result: Option<HbarResponse>,
    pub error: Option<String>,
    pub processing_time_ms: f64,
}

/// Comprehensive batch analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub total_prompts: usize,
    pub successful_prompts: usize,
    pub failed_prompts: usize,
    pub results: Vec<PromptResult>,
    pub total_time_ms: f64,
    pub average_time_ms: f64,
    pub average_hbar: f32,
    pub batch_statistics: BatchStatistics,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Detailed batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub risk_distribution: RiskDistribution,
    pub performance_metrics: PerformanceMetrics,
    pub error_analysis: ErrorAnalysis,
}

/// Risk level distribution across batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDistribution {
    pub high_risk_count: usize,
    pub moderate_risk_count: usize,
    pub stable_count: usize,
    pub error_count: usize,
}

/// Performance metrics for batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub p95_time_ms: f64,
    pub throughput_per_second: f64,
    pub parallel_efficiency: f32,
}

/// Error analysis across batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub timeout_errors: usize,
    pub validation_errors: usize,
    pub computation_errors: usize,
    pub other_errors: usize,
    pub error_rate: f32,
}

/// High-performance batch processor with advanced parallelization
pub struct BatchProcessor {
    config: BatchConfig,
    semantic_config: SemanticConfig,
    semaphore: Arc<Semaphore>,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(batch_config: BatchConfig, semantic_config: SemanticConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(batch_config.max_concurrent));
        
        Self {
            config: batch_config,
            semantic_config,
            semaphore,
        }
    }

    /// 🚀 Process batch of prompts with advanced parallelization
    #[instrument(skip(self, prompts), fields(batch_size = prompts.len()))]
    pub async fn process_batch(&self, prompts: Vec<String>, model: &str) -> Result<BatchResult, SemanticError> {
        let start_time = Instant::now();
        let timestamp = chrono::Utc::now();
        
        // Validate batch size
        if prompts.len() > self.config.max_batch_size {
            return Err(SemanticError::InvalidInput {
                message: format!(
                    "Batch size {} exceeds maximum {}", 
                    prompts.len(), 
                    self.config.max_batch_size
                ),
            });
        }

        if prompts.is_empty() {
            return Err(SemanticError::InvalidInput {
                message: "Empty batch provided".to_string(),
            });
        }

        debug!("Processing batch of {} prompts with model {}", prompts.len(), model);

        // Create prompt requests with indexing
        let requests: Vec<PromptRequest> = prompts
            .into_iter()
            .enumerate()
            .map(|(index, prompt)| PromptRequest {
                prompt,
                model: model.to_string(),
                index,
            })
            .collect();

        // Process with timeout
        let processing_result = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.timeout_ms),
            self.process_requests_parallel(requests)
        ).await;

        let mut results = match processing_result {
            Ok(results) => results?,
            Err(_) => {
                return Err(SemanticError::Timeout { 
                    timeout_ms: self.config.timeout_ms 
                });
            }
        };

        // Sort results by original index if order preservation is enabled
        if self.config.preserve_order {
            results.sort_by_key(|r| r.index);
        }

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Generate comprehensive statistics
        let statistics = self.generate_statistics(&results, total_time);
        
        let successful_results: Vec<&PromptResult> = results
            .iter()
            .filter(|r| r.success)
            .collect();

        let successful_count = successful_results.len();
        let total_count = results.len();

        let average_hbar = if !successful_results.is_empty() {
            successful_results
                .iter()
                .filter_map(|r| r.result.as_ref())
                .map(|r| r.hbar_s)
                .sum::<f32>() / successful_count as f32
        } else {
            0.0
        };

        Ok(BatchResult {
            total_prompts: total_count,
            successful_prompts: successful_count,
            failed_prompts: total_count - successful_count,
            results,
            total_time_ms: total_time,
            average_time_ms: total_time / successful_count.max(1) as f64,
            average_hbar,
            batch_statistics: statistics,
            timestamp,
        })
    }

    /// Process requests in parallel with semaphore-based concurrency control
    async fn process_requests_parallel(&self, requests: Vec<PromptRequest>) -> Result<Vec<PromptResult>, SemanticError> {
        use futures::stream::{FuturesUnordered, StreamExt};
        
        let mut futures = FuturesUnordered::new();
        
        for request in requests {
            let semaphore = Arc::clone(&self.semaphore);
            let config = self.semantic_config.clone();
            
            let future = tokio::spawn(async move {
                // Acquire semaphore permit for concurrency control
                let _permit = semaphore.acquire().await.map_err(|e| {
                    SemanticError::Internal { 
                        source: anyhow::anyhow!("Semaphore error: {}", e) 
                    }
                })?;
                
                Self::process_single_request(request, config).await
            });
            
            futures.push(future);
        }
        
        let mut results = Vec::new();
        
        while let Some(result) = futures.next().await {
            match result {
                Ok(prompt_result) => results.push(prompt_result?),
                Err(e) => {
                    warn!("Task join error: {}", e);
                    if self.config.fail_fast {
                        return Err(SemanticError::Internal { 
                            source: anyhow::anyhow!("Task failed: {}", e) 
                        });
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// Process a single request with comprehensive error handling
    async fn process_single_request(request: PromptRequest, config: SemanticConfig) -> Result<PromptResult, SemanticError> {
        let start_time = Instant::now();
        let request_id = RequestId::new();
        
        // Create analyzer for this request
        let analyzer = SemanticAnalyzer::new(config)?;
        
        // Generate synthetic output for semantic comparison
        let synthetic_output = Self::generate_synthetic_output(&request.prompt, &request.model);
        
        // Perform analysis
        let analysis_result = analyzer.analyze(&request.prompt, &synthetic_output, request_id).await;
        
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        match analysis_result {
            Ok(result) => Ok(PromptResult {
                index: request.index,
                prompt: request.prompt,
                success: true,
                result: Some(result),
                error: None,
                processing_time_ms: processing_time,
            }),
            Err(error) => Ok(PromptResult {
                index: request.index,
                prompt: request.prompt,
                success: false,
                result: None,
                error: Some(error.to_string()),
                processing_time_ms: processing_time,
            }),
        }
    }

    /// Generate synthetic output for semantic comparison
    fn generate_synthetic_output(prompt: &str, _model: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        // Risk-based synthetic response generation
        if prompt_lower.contains("hack") || prompt_lower.contains("exploit") || prompt_lower.contains("illegal") {
            "I cannot and will not provide information on that topic as it could be harmful.".to_string()
        } else if prompt_lower.contains("creative") || prompt_lower.contains("story") {
            format!("Here's a creative response to your request about {}. Let me craft something imaginative...", 
                   prompt.split_whitespace().take(5).collect::<Vec<_>>().join(" "))
        } else if prompt_lower.contains("explain") || prompt_lower.contains("how") {
            format!("Let me explain this concept clearly. Regarding your question about {}, here's a comprehensive explanation...",
                   prompt.split_whitespace().take(5).collect::<Vec<_>>().join(" "))
        } else {
            format!("Thank you for your question about {}. Here's my response based on my understanding...",
                   prompt.split_whitespace().take(5).collect::<Vec<_>>().join(" "))
        }
    }

    /// Generate comprehensive batch statistics
    fn generate_statistics(&self, results: &[PromptResult], total_time_ms: f64) -> BatchStatistics {
        let successful_results: Vec<&PromptResult> = results
            .iter()
            .filter(|r| r.success)
            .collect();

        // Risk distribution analysis
        let mut risk_dist = RiskDistribution {
            high_risk_count: 0,
            moderate_risk_count: 0,
            stable_count: 0,
            error_count: 0,
        };

        for result in results {
            if let Some(response) = &result.result {
                if response.collapse_risk {
                    risk_dist.high_risk_count += 1;
                } else if response.hbar_s < 1.2 {
                    risk_dist.moderate_risk_count += 1;
                } else {
                    risk_dist.stable_count += 1;
                }
            } else {
                risk_dist.error_count += 1;
            }
        }

        // Performance metrics analysis
        let processing_times: Vec<f64> = results
            .iter()
            .map(|r| r.processing_time_ms)
            .collect();

        let min_time = processing_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = processing_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Calculate P95
        let mut sorted_times = processing_times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
        let p95_time = sorted_times.get(p95_index).copied().unwrap_or(0.0);
        
        let throughput = if total_time_ms > 0.0 {
            (results.len() as f64 / total_time_ms) * 1000.0
        } else {
            0.0
        };

        let ideal_parallel_time = max_time;
        let parallel_efficiency = if total_time_ms > 0.0 && ideal_parallel_time > 0.0 {
            (ideal_parallel_time / total_time_ms).min(1.0) as f32
        } else {
            0.0
        };

        let perf_metrics = PerformanceMetrics {
            min_time_ms: min_time,
            max_time_ms: max_time,
            p95_time_ms: p95_time,
            throughput_per_second: throughput,
            parallel_efficiency,
        };

        // Error analysis
        let mut error_analysis = ErrorAnalysis {
            timeout_errors: 0,
            validation_errors: 0,
            computation_errors: 0,
            other_errors: 0,
            error_rate: 0.0,
        };

        for result in results {
            if let Some(error_msg) = &result.error {
                if error_msg.contains("timeout") || error_msg.contains("Timeout") {
                    error_analysis.timeout_errors += 1;
                } else if error_msg.contains("validation") || error_msg.contains("Invalid") {
                    error_analysis.validation_errors += 1;
                } else if error_msg.contains("computation") || error_msg.contains("Math") {
                    error_analysis.computation_errors += 1;
                } else {
                    error_analysis.other_errors += 1;
                }
            }
        }

        error_analysis.error_rate = if !results.is_empty() {
            (results.len() - successful_results.len()) as f32 / results.len() as f32
        } else {
            0.0
        };

        BatchStatistics {
            risk_distribution: risk_dist,
            performance_metrics: perf_metrics,
            error_analysis,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_processing() {
        let batch_config = BatchConfig::default();
        let semantic_config = SemanticConfig::ultra_fast();
        let processor = BatchProcessor::new(batch_config, semantic_config);
        
        let prompts = vec![
            "What is AI?".to_string(),
            "Explain quantum computing".to_string(),
            "Write a story about dragons".to_string(),
        ];
        
        let result = processor.process_batch(prompts, "gpt4").await.unwrap();
        
        assert_eq!(result.total_prompts, 3);
        assert!(result.successful_prompts <= 3);
        assert!(result.total_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_empty_batch() {
        let batch_config = BatchConfig::default();
        let semantic_config = SemanticConfig::ultra_fast();
        let processor = BatchProcessor::new(batch_config, semantic_config);
        
        let result = processor.process_batch(vec![], "gpt4").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_oversized_batch() {
        let batch_config = BatchConfig { max_batch_size: 2, ..Default::default() };
        let semantic_config = SemanticConfig::ultra_fast();
        let processor = BatchProcessor::new(batch_config, semantic_config);
        
        let prompts = vec!["test".to_string(); 5];
        let result = processor.process_batch(prompts, "gpt4").await;
        assert!(result.is_err());
    }
}
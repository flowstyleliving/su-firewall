// 🦙 Example: Live UQ/Hbar Auditing with Llama Model
// Demonstrates real-time uncertainty quantification using logit access

use crate::oss_logit_adapter::{
    OSSLogitAdapter, OSSModelFramework, LogitData, AdapterConfig, 
    LiveLogitAnalysis, TokenUncertainty
};
use crate::{CalibrationMode, RequestId};
use std::collections::HashMap;
use serde_json;

/// 🔧 Mock Llama model interface for demonstration
pub struct MockLlamaModel {
    vocab: HashMap<u32, String>,
    model_size: String,
}

impl MockLlamaModel {
    pub fn new(model_size: &str) -> Self {
        // Initialize with sample vocabulary
        let mut vocab = HashMap::new();
        vocab.insert(0, "<pad>".to_string());
        vocab.insert(1, "<unk>".to_string());
        vocab.insert(2, "<s>".to_string());
        vocab.insert(3, "</s>".to_string());
        vocab.insert(4, "the".to_string());
        vocab.insert(5, "and".to_string());
        vocab.insert(6, "is".to_string());
        vocab.insert(7, "to".to_string());
        vocab.insert(8, "uncertainty".to_string());
        vocab.insert(9, "model".to_string());
        vocab.insert(10, "semantic".to_string());
        
        Self {
            vocab,
            model_size: model_size.to_string(),
        }
    }
    
    /// Generate mock logits for demonstration
    pub fn generate_with_logits(&self, prompt: &str) -> (String, LogitData) {
        // Mock generated response
        let response = "The semantic uncertainty model shows high confidence in this prediction.";
        
        // Mock token sequence (simplified)
        let token_sequence = vec![4, 10, 8, 9, 6, 5]; // "the semantic uncertainty model is ..."
        
        // Mock logits for each token position
        let token_logits = vec![
            // Position 0: "the" - high confidence
            vec![-2.1, -3.4, -1.8, -2.9, 2.3, -1.7, -2.4, -1.9, -3.2, -2.8, -3.1],
            // Position 1: "semantic" - medium confidence  
            vec![-1.9, -2.8, -2.1, -2.4, -1.6, -2.3, -1.8, -2.1, -1.4, -1.9, 1.8],
            // Position 2: "uncertainty" - high confidence
            vec![-3.2, -4.1, -2.9, -3.6, -2.7, -3.4, -2.8, -3.1, 2.7, -2.9, -3.3],
            // Position 3: "model" - medium confidence
            vec![-2.4, -3.1, -2.7, -2.8, -2.2, -2.6, -2.3, -2.5, -1.8, 1.9, -2.1],
            // Position 4: "is" - high confidence
            vec![-2.8, -3.5, -2.4, -3.1, -2.6, -2.9, 2.4, -2.7, -3.4, -2.8, -3.2],
            // Position 5: "and" - low confidence (uncertain)
            vec![-1.2, -1.4, -1.1, -1.3, -1.0, 0.8, -1.1, -1.2, -1.5, -1.3, -1.4],
        ];
        
        let logit_data = LogitData {
            token_logits,
            vocab_map: self.vocab.clone(),
            attention_weights: None, // Could include real attention data
            hidden_states: None,     // Could include hidden states
            temperature: 0.7,
            top_p: Some(0.9),
            token_sequence,
        };
        
        (response, logit_data)
    }
}

/// 🧪 Live UQ monitoring system
pub struct LiveUQMonitor {
    adapter: OSSLogitAdapter,
    alert_threshold: f64,
    token_buffer: Vec<TokenUncertainty>,
}

impl LiveUQMonitor {
    pub fn new(alert_threshold: f64) -> Self {
        let config = AdapterConfig {
            enable_streaming: true,
            buffer_size: 50,
            entropy_alert_threshold: alert_threshold,
            use_attention: false, // Disabled for this example
            calibration_mode: CalibrationMode::Production,
        };
        
        let adapter = OSSLogitAdapter::new(OSSModelFramework::LlamaCpp, config);
        
        Self {
            adapter,
            alert_threshold,
            token_buffer: Vec::new(),
        }
    }
    
    /// 📊 Process streaming tokens with live UQ analysis
    pub fn process_stream(
        &mut self,
        prompt: &str,
        model: &MockLlamaModel,
    ) -> Result<StreamingAnalysisReport, Box<dyn std::error::Error>> {
        let (response, logit_data) = model.generate_with_logits(prompt);
        
        // Analyze with our OSS adapter
        let analysis = self.adapter.analyze_logits(
            prompt,
            &logit_data,
            RequestId::new(),
        )?;
        
        // Check for uncertainty alerts
        let mut alerts = Vec::new();
        for token_uncertainty in &analysis.token_uncertainties {
            if token_uncertainty.token_entropy > self.alert_threshold {
                alerts.push(UncertaintyAlert {
                    position: token_uncertainty.position,
                    token: token_uncertainty.token_string.clone(),
                    entropy: token_uncertainty.token_entropy,
                    confidence: token_uncertainty.token_probability,
                    severity: if token_uncertainty.token_entropy > 3.0 {
                        AlertSeverity::High
                    } else {
                        AlertSeverity::Medium
                    },
                });
            }
        }
        
        // Update token buffer
        self.token_buffer.extend(analysis.token_uncertainties.clone());
        if self.token_buffer.len() > 100 {
            self.token_buffer.drain(0..50); // Keep last 50 tokens
        }
        
        Ok(StreamingAnalysisReport {
            analysis,
            response,
            alerts,
            buffer_size: self.token_buffer.len(),
        })
    }
    
    /// 📈 Generate uncertainty trend analysis
    pub fn get_uncertainty_trend(&self) -> Vec<f64> {
        self.token_buffer.iter()
            .map(|token| token.token_entropy)
            .collect()
    }
}

/// 📋 Analysis report for streaming processing
#[derive(Debug)]
pub struct StreamingAnalysisReport {
    pub analysis: LiveLogitAnalysis,
    pub response: String,
    pub alerts: Vec<UncertaintyAlert>,
    pub buffer_size: usize,
}

/// ⚠️ Uncertainty alert system
#[derive(Debug)]
pub struct UncertaintyAlert {
    pub position: usize,
    pub token: String,
    pub entropy: f64,
    pub confidence: f64,
    pub severity: AlertSeverity,
}

#[derive(Debug)]
pub enum AlertSeverity {
    Low,
    Medium, 
    High,
    Critical,
}

/// 🚀 Example usage function
pub fn example_live_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦙 Starting Live UQ Monitoring with Llama Model");
    
    // Initialize model and monitor
    let model = MockLlamaModel::new("7B");
    let mut monitor = LiveUQMonitor::new(2.0); // Alert threshold
    
    // Test prompts with different uncertainty characteristics
    let test_prompts = vec![
        "What is the capital of France?",  // High confidence expected
        "Explain quantum mechanics in simple terms",  // Medium confidence
        "What will happen in 2090?",      // Low confidence expected
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n📝 Prompt {}: {}", i + 1, prompt);
        
        let report = monitor.process_stream(prompt, &model)?;
        
        // Display results
        println!("🧮 ℏₛ (Raw): {:.4}", report.analysis.base_result.raw_hbar);
        println!("🎯 ℏₛ (Calibrated): {:.4}", report.analysis.base_result.calibrated_hbar);
        println!("🚦 Risk Level: {:?}", report.analysis.base_result.risk_level);
        println!("📊 Average Entropy: {:.4}", report.analysis.logit_metrics.average_entropy);
        println!("💫 Perplexity: {:.2}", report.analysis.logit_metrics.perplexity);
        println!("🎯 Confidence: {:.4}", report.analysis.logit_metrics.confidence_score);
        
        // Show uncertainty alerts
        if !report.alerts.is_empty() {
            println!("⚠️  Uncertainty Alerts:");
            for alert in &report.alerts {
                println!("   Position {}: '{}' (entropy: {:.3}, confidence: {:.3}) - {:?}",
                    alert.position, alert.token, alert.entropy, alert.confidence, alert.severity);
            }
        }
        
        // Show top uncertain tokens
        let mut uncertain_tokens: Vec<_> = report.analysis.token_uncertainties.iter()
            .filter(|t| t.token_entropy > 1.0)
            .collect();
        uncertain_tokens.sort_by(|a, b| b.token_entropy.partial_cmp(&a.token_entropy).unwrap());
        
        if !uncertain_tokens.is_empty() {
            println!("🔍 Most Uncertain Tokens:");
            for (i, token) in uncertain_tokens.iter().take(3).enumerate() {
                println!("   {}. '{}' - entropy: {:.3}, local ℏₛ: {:.3}",
                    i + 1, token.token_string, token.token_entropy, token.local_hbar);
            }
        }
        
        println!("📈 Response: {}", report.response);
    }
    
    // Show uncertainty trend
    let trend = monitor.get_uncertainty_trend();
    if !trend.is_empty() {
        let avg_uncertainty: f64 = trend.iter().sum::<f64>() / trend.len() as f64;
        let max_uncertainty = trend.iter().fold(0.0, |a, &b| a.max(b));
        println!("\n📊 Overall Uncertainty Trend:");
        println!("   Average: {:.3}", avg_uncertainty);
        println!("   Maximum: {:.3}", max_uncertainty);
        println!("   Total tokens processed: {}", trend.len());
    }
    
    Ok(())
}

/// 🔬 Benchmark comparison: Text-based vs Logit-based UQ
pub fn benchmark_accuracy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Benchmarking: Text-based vs Logit-based UQ");
    
    let model = MockLlamaModel::new("7B");
    let mut logit_monitor = LiveUQMonitor::new(2.0);
    
    // Use existing text-based core engine for comparison
    let text_engine = crate::modules::core_engine::CoreSemanticEngine::new(
        crate::modules::core_engine::CoreEngineConfig::default()
    );
    
    let test_cases = vec![
        ("Simple fact", "Paris is the capital of France."),
        ("Complex reasoning", "The relationship between quantum entanglement and consciousness remains speculative."),
        ("Uncertain prediction", "Technology in 2090 might include neural interfaces."),
    ];
    
    for (case_name, prompt) in test_cases {
        let (response, logit_data) = model.generate_with_logits(prompt);
        
        // Logit-based analysis
        let logit_analysis = logit_monitor.adapter.analyze_logits(
            prompt,
            &logit_data,
            RequestId::new(),
        )?;
        
        // Text-based analysis (existing method)
        let text_analysis = text_engine.analyze(
            prompt,
            &response,
            RequestId::new(),
        ).await?;
        
        println!("\n📋 Case: {}", case_name);
        println!("  Logit ℏₛ: {:.4} (Risk: {:?})", 
            logit_analysis.base_result.calibrated_hbar,
            logit_analysis.base_result.risk_level
        );
        println!("  Text ℏₛ:  {:.4} (Risk: {:?})", 
            text_analysis.calibrated_hbar,
            text_analysis.risk_level
        );
        println!("  Logit Entropy: {:.3}", logit_analysis.logit_metrics.average_entropy);
        println!("  Logit Confidence: {:.3}", logit_analysis.logit_metrics.confidence_score);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_llama_integration() {
        let model = MockLlamaModel::new("7B");
        let (response, logit_data) = model.generate_with_logits("Test prompt");
        
        assert!(!response.is_empty());
        assert!(!logit_data.token_logits.is_empty());
        assert!(!logit_data.vocab_map.is_empty());
    }
    
    #[test]
    fn test_live_monitor() {
        let mut monitor = LiveUQMonitor::new(2.0);
        let model = MockLlamaModel::new("7B");
        
        let report = monitor.process_stream("Test", &model).unwrap();
        assert!(report.analysis.base_result.raw_hbar > 0.0);
    }
} 
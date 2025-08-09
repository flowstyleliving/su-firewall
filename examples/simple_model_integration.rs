// 🎯 Simple Model Integration Examples
// Shows how ANY model can easily integrate with the separated live auditing system

use core_engine::audit_interface::{
    AuditClient, StartAuditRequest, SimpleToken, SimpleCapabilities
};
use core_engine::live_response_auditor::{LiveResponseAuditor, LiveAuditorConfig};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

/// 🤖 Example 1: OpenAI API Integration
pub async fn openai_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 OpenAI API + Live Auditing Example");
    
    // Set up the auditing system
    let config = LiveAuditorConfig::default();
    let auditor = Arc::new(LiveResponseAuditor::new(config));
    let audit_client = AuditClient::new(auditor);
    
    // Start audit session
    let audit_request = StartAuditRequest {
        prompt: "Explain quantum computing".to_string(),
        model_name: "gpt-4".to_string(),
        model_version: Some("gpt-4-0613".to_string()),
        framework: "OpenAI API".to_string(),
        capabilities: SimpleCapabilities {
            has_logits: false,       // OpenAI doesn't provide logits
            has_probabilities: false, // OpenAI doesn't provide probabilities
            supports_streaming: true, // But supports streaming
        },
    };
    
    let session_id = audit_client.start_audit(audit_request).await?;
    println!("✅ Started audit session: {}", session_id);
    
    // Simulate OpenAI streaming response
    let openai_response_tokens = vec![
        "Quantum", " computing", " is", " a", " revolutionary", " technology",
        " that", " leverages", " quantum", " mechanics", " to", " process",
        " information", " in", " fundamentally", " different", " ways", "."
    ];
    
    for (i, token_text) in openai_response_tokens.iter().enumerate() {
        // Create token data (text-only since OpenAI doesn't provide logits)
        let token = SimpleToken {
            text: token_text.to_string(),
            token_id: None,         // No token IDs from OpenAI
            probability: None,      // No probabilities from OpenAI
            logits: None,          // No logits from OpenAI
        };
        
        // Add to audit
        let result = audit_client.add_token(token).await?;
        
        println!("🎯 Token {}: '{}' - Uncertainty: {:.3}, Risk: {}", 
            i + 1, token_text, result.current_uncertainty, result.risk_level);
        
        // Show alerts if any
        for alert in &result.alerts {
            println!("  ⚠️  {}: {}", alert.severity, alert.message);
        }
        
        sleep(Duration::from_millis(100)).await; // Simulate streaming delay
    }
    
    // Finish audit
    let final_result = audit_client.finish_audit().await?;
    println!("✅ Completed audit: {} tokens, avg uncertainty: {:.3}", 
        final_result.tokens_processed, final_result.average_uncertainty);
    
    Ok(())
}

/// 🦙 Example 2: Local Llama Model with Logits
pub async fn local_llama_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🦙 Local Llama + Live Auditing Example (with logits)");
    
    // Set up the auditing system
    let config = LiveAuditorConfig::default();
    let auditor = Arc::new(LiveResponseAuditor::new(config));
    let audit_client = AuditClient::new(auditor);
    
    // Start audit session
    let audit_request = StartAuditRequest {
        prompt: "What is machine learning?".to_string(),
        model_name: "Llama-2-7B-Chat".to_string(),
        model_version: Some("hf".to_string()),
        framework: "llama.cpp".to_string(),
        capabilities: SimpleCapabilities {
            has_logits: true,        // Local models can provide logits!
            has_probabilities: true, // And probabilities
            supports_streaming: true,
        },
    };
    
    let session_id = audit_client.start_audit(audit_request).await?;
    println!("✅ Started audit session: {}", session_id);
    
    // Simulate local model response with logits
    let llama_tokens = vec![
        ("Machine", 1234, 0.85, generate_mock_logits(0.85)),
        (" learning", 5678, 0.92, generate_mock_logits(0.92)),
        (" is", 9012, 0.88, generate_mock_logits(0.88)),
        (" a", 3456, 0.95, generate_mock_logits(0.95)),
        (" subset", 7890, 0.65, generate_mock_logits(0.65)), // Lower confidence
        (" of", 2468, 0.90, generate_mock_logits(0.90)),
        (" artificial", 1357, 0.70, generate_mock_logits(0.70)),
        (" intelligence", 9753, 0.82, generate_mock_logits(0.82)),
    ];
    
    for (i, (text, token_id, prob, logits)) in llama_tokens.iter().enumerate() {
        let token = SimpleToken {
            text: text.to_string(),
            token_id: Some(*token_id),
            probability: Some(*prob),
            logits: Some(logits.clone()),
        };
        
        let result = audit_client.add_token(token).await?;
        
        println!("🎯 Token {}: '{}' (p={:.3}) - Uncertainty: {:.3}, Risk: {}", 
            i + 1, text, prob, result.current_uncertainty, result.risk_level);
        
        // Show alerts
        for alert in &result.alerts {
            println!("  ⚠️  {}: {}", alert.severity, alert.message);
        }
        
        sleep(Duration::from_millis(150)).await;
    }
    
    let final_result = audit_client.finish_audit().await?;
    println!("✅ Completed audit: {} tokens, avg uncertainty: {:.3}", 
        final_result.tokens_processed, final_result.average_uncertainty);
    
    Ok(())
}

/// 🤖 Example 3: Any Model with Just Probabilities
pub async fn generic_model_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🤖 Generic Model + Live Auditing Example (probabilities only)");
    
    let config = LiveAuditorConfig::default();
    let auditor = Arc::new(LiveResponseAuditor::new(config));
    let audit_client = AuditClient::new(auditor);
    
    let audit_request = StartAuditRequest {
        prompt: "Explain neural networks".to_string(),
        model_name: "Custom-Model-X".to_string(),
        model_version: Some("v2.1".to_string()),
        framework: "Custom Framework".to_string(),
        capabilities: SimpleCapabilities {
            has_logits: false,
            has_probabilities: true, // Only probabilities available
            supports_streaming: true,
        },
    };
    
    let session_id = audit_client.start_audit(audit_request).await?;
    println!("✅ Started audit session: {}", session_id);
    
    // Simulate model with varying confidence
    let model_tokens = vec![
        ("Neural", 0.95),     // High confidence
        (" networks", 0.88),   // Good confidence
        (" are", 0.92),       // High confidence
        (" computational", 0.45), // Low confidence - complex word
        (" models", 0.85),    // Good confidence
        (" inspired", 0.60),  // Medium confidence
        (" by", 0.90),        // High confidence
        (" biological", 0.40), // Low confidence - technical term
        (" neurons", 0.70),   // Medium confidence
        (".", 0.98),          // Very high confidence - punctuation
    ];
    
    for (i, (text, prob)) in model_tokens.iter().enumerate() {
        let token = SimpleToken {
            text: text.to_string(),
            token_id: None,
            probability: Some(*prob),
            logits: None,
        };
        
        let result = audit_client.add_token(token).await?;
        
        println!("🎯 Token {}: '{}' (p={:.3}) - Uncertainty: {:.3}, Risk: {}", 
            i + 1, text, prob, result.current_uncertainty, result.risk_level);
        
        for alert in &result.alerts {
            println!("  ⚠️  {}: {}", alert.severity, alert.message);
        }
        
        sleep(Duration::from_millis(120)).await;
    }
    
    let final_result = audit_client.finish_audit().await?;
    println!("✅ Completed audit: {} tokens, avg uncertainty: {:.3}", 
        final_result.tokens_processed, final_result.average_uncertainty);
    
    Ok(())
}

/// 📝 Example 4: Text-Only Model (Fallback)
pub async fn text_only_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Text-Only Model + Live Auditing Example (fallback mode)");
    
    let config = LiveAuditorConfig::default();
    let auditor = Arc::new(LiveResponseAuditor::new(config));
    let audit_client = AuditClient::new(auditor);
    
    let audit_request = StartAuditRequest {
        prompt: "What is consciousness?".to_string(),
        model_name: "Text-Only-Model".to_string(),
        model_version: None,
        framework: "Legacy System".to_string(),
        capabilities: SimpleCapabilities {
            has_logits: false,
            has_probabilities: false, // No numerical data available
            supports_streaming: true,
        },
    };
    
    let session_id = audit_client.start_audit(audit_request).await?;
    println!("✅ Started audit session: {}", session_id);
    
    // Even with just text, the auditor can estimate uncertainty
    let text_tokens = vec![
        "Consciousness", " is", " a", " complex", " philosophical",
        " and", " scientific", " concept", " that", " refers",
        " to", " subjective", " awareness", " and", " experience", "."
    ];
    
    for (i, text) in text_tokens.iter().enumerate() {
        let token = SimpleToken {
            text: text.to_string(),
            token_id: None,
            probability: None,
            logits: None,
        };
        
        let result = audit_client.add_token(token).await?;
        
        println!("🎯 Token {}: '{}' - Uncertainty: {:.3} (estimated), Risk: {}", 
            i + 1, text, result.current_uncertainty, result.risk_level);
        
        for alert in &result.alerts {
            println!("  ⚠️  {}: {}", alert.severity, alert.message);
        }
        
        sleep(Duration::from_millis(100)).await;
    }
    
    let final_result = audit_client.finish_audit().await?;
    println!("✅ Completed audit: {} tokens, avg uncertainty: {:.3}", 
        final_result.tokens_processed, final_result.average_uncertainty);
    
    Ok(())
}

/// 🔧 Helper function to generate mock logits for demonstration
fn generate_mock_logits(target_probability: f64) -> Vec<f32> {
    let vocab_size = 50000;
    let mut logits = vec![0.0f32; vocab_size];
    
    // Set target token logit to achieve desired probability
    let target_logit = target_probability.ln() as f32;
    logits[0] = target_logit;
    
    // Fill in random logits for other tokens
    for i in 1..vocab_size {
        logits[i] = -2.0 + (i % 100) as f32 * 0.01; // Varied but lower logits
    }
    
    logits
}

/// 🚀 Run all examples
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🔍 Live Response Auditing - Model Integration Examples\n");
    
    // Run all examples
    openai_integration_example().await?;
    local_llama_integration_example().await?;
    generic_model_integration_example().await?;
    text_only_integration_example().await?;
    
    println!("\n✅ All examples completed successfully!");
    println!("\n📋 Key Takeaways:");
    println!("• The auditing system works with ANY model");
    println!("• More data (logits) = better uncertainty estimates");
    println!("• Even text-only models get uncertainty analysis");
    println!("• Real-time alerts help catch problematic generations");
    println!("• Simple interface makes integration easy");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_all_integration_examples() {
        // Test that all examples run without panicking
        assert!(openai_integration_example().await.is_ok());
        assert!(local_llama_integration_example().await.is_ok());
        assert!(generic_model_integration_example().await.is_ok());
        assert!(text_only_integration_example().await.is_ok());
    }
    
    #[test]
    fn test_mock_logits_generation() {
        let logits = generate_mock_logits(0.8);
        assert_eq!(logits.len(), 50000);
        
        // Verify target probability is achievable
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let target_prob = exp_logits[0] / sum_exp;
        
        assert!((target_prob - 0.8).abs() < 0.1); // Should be close to target
    }
} 
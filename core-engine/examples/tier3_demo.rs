//! Tier-3 Semantic Uncertainty Measurement Demo
//! 
//! This example demonstrates how to use the advanced Tier-3 measurement engine
//! for sophisticated semantic uncertainty analysis.

use semantic_uncertainty_runtime::{SemanticAnalyzer, SemanticConfig, RequestId};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    println!("🧠 Tier-3 Semantic Uncertainty Measurement Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Create Tier-3 configuration
    let config = SemanticConfig::tier3();
    println!("📊 Configuration: Tier-3 mode enabled");
    println!("   • Cache size: {}", config.tier3_config.as_ref().unwrap().cache_size);
    println!("   • Target latency: {}ms", config.tier3_config.as_ref().unwrap().target_latency_ms);
    println!("   • Perturbation samples: {}", config.tier3_config.as_ref().unwrap().perturbation_samples);
    
    // Initialize analyzer with Tier-3 engine
    println!("\n🚀 Initializing Tier-3 measurement engine...");
    let start_time = Instant::now();
    let analyzer = SemanticAnalyzer::new(config).await?;
    let init_time = start_time.elapsed();
    println!("✅ Initialization completed in {:.2}ms", init_time.as_millis());
    
    // Test prompts with varying complexity
    let test_cases = vec![
        ("Simple factual question", "What is the capital of France?", "Paris is the capital of France."),
        ("Complex reasoning", "Explain quantum entanglement", "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently."),
        ("Philosophical paradox", "What is the meaning of life?", "The meaning of life is a deeply personal and philosophical question that has been pondered by thinkers throughout history."),
        ("Technical explanation", "How does a neural network work?", "A neural network is a computational model inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections."),
    ];
    
    println!("\n🔍 Running Tier-3 analysis on test cases...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    for (category, prompt, output) in test_cases {
        println!("\n📝 Category: {}", category);
        println!("   Prompt: {}", prompt);
        println!("   Output: {}", output);
        
        let request_id = RequestId::new();
        let analysis_start = Instant::now();
        
        match analyzer.analyze(prompt, output, request_id).await {
            Ok(response) => {
                let analysis_time = analysis_start.elapsed();
                println!("   📊 Results:");
                println!("      ℏₛ (Semantic Uncertainty): {:.4}", response.hbar_s);
                println!("      Δμ (Precision): {:.4}", response.delta_mu);
                println!("      Δσ (Flexibility): {:.4}", response.delta_sigma);
                println!("      Collapse Risk: {}", if response.collapse_risk { "⚠️ HIGH" } else { "✅ LOW" });
                println!("      Processing Time: {:.2}ms", analysis_time.as_millis());
                println!("      Request ID: {}", request_id);
            }
            Err(e) => {
                println!("   ❌ Analysis failed: {}", e);
            }
        }
    }
    
    println!("\n🎯 Demo completed successfully!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💡 Key benefits of Tier-3 measurement:");
    println!("   • Advanced precision measurement via cache firewall");
    println!("   • Sophisticated flexibility analysis with perturbations");
    println!("   • Component-level attribution and drift monitoring");
    println!("   • Enhanced confidence assessment");
    
    Ok(())
} 
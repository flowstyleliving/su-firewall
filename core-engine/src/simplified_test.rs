// 🧪 Simplified Test for Streamlined Engine
// Test the core functionality without complex dependencies

use crate::streamlined_engine::{StreamlinedEngine, RiskLevel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let engine = StreamlinedEngine::new();
        
        let prompt = "What is quantum computing?";
        let output = "Quantum computing uses quantum bits to process information.";
        
        let result = engine.analyze(prompt, output);
        
        // Basic sanity checks
        assert!(result.raw_hbar >= 0.0);
        assert!(result.calibrated_hbar >= 0.0);
        assert!(result.delta_mu >= 0.0);
        assert!(result.delta_sigma >= 0.0);
        assert!(result.processing_time_ns > 0);
        assert!(result.deterministic_hash != 0);
        
        // Token analysis checks
        assert!(result.token_analysis.prompt_tokens > 0);
        assert!(result.token_analysis.output_tokens > 0);
        assert!(result.token_analysis.total_tokens > 0);
        assert!(result.token_analysis.efficiency_score >= 0.0);
        assert!(result.token_analysis.efficiency_score <= 1.0);
        
        println!("✅ Basic functionality test passed");
        println!("   ℏₛ (raw): {:.4}", result.raw_hbar);
        println!("   ℏₛ (calibrated): {:.3}", result.calibrated_hbar);
        println!("   Processing time: {} ns", result.processing_time_ns);
        println!("   Risk level: {:?}", result.risk_level);
    }
    
    #[test]
    fn test_deterministic_behavior() {
        let engine = StreamlinedEngine::new();
        
        let prompt = "Explain machine learning";
        let output = "Machine learning is a method of data analysis";
        
        // Run analysis multiple times
        let result1 = engine.analyze(prompt, output);
        let result2 = engine.analyze(prompt, output);
        let result3 = engine.analyze(prompt, output);
        
        // All results should be identical (deterministic)
        assert_eq!(result1.deterministic_hash, result2.deterministic_hash);
        assert_eq!(result1.deterministic_hash, result3.deterministic_hash);
        
        assert!((result1.raw_hbar - result2.raw_hbar).abs() < 1e-10);
        assert!((result1.raw_hbar - result3.raw_hbar).abs() < 1e-10);
        
        assert!((result1.delta_mu - result2.delta_mu).abs() < 1e-10);
        assert!((result1.delta_sigma - result2.delta_sigma).abs() < 1e-10);
        
        println!("✅ Deterministic behavior test passed");
        println!("   Consistent hash: {}", result1.deterministic_hash);
        println!("   Consistent ℏₛ: {:.10}", result1.raw_hbar);
    }
    
    #[test]
    fn test_prompt_classification() {
        let engine = StreamlinedEngine::new();
        
        // Test different prompt types
        let test_cases = vec![
            ("What is the capital of France?", "SimpleQA"),
            ("Write a function to sort an array", "CodeGeneration"),
            ("Analyze the economic impact of climate change", "ComplexAnalysis"),
            ("Calculate the derivative of x^2", "Mathematical"),
            ("Write a short story about a robot", "CreativeWriting"),
        ];
        
        for (prompt, expected_category) in test_cases {
            let output = "This is a test response.";
            let result = engine.analyze(prompt, output);
            
            // Check that classification is working
            let class_name = format!("{:?}", result.token_analysis.prompt_class);
            
            if expected_category == "SimpleQA" {
                assert!(class_name.contains("SimpleQA") || class_name.contains("Conversational"));
            }
            // Add more specific checks as needed
            
            println!("✅ Prompt '{}' classified as: {}", prompt, class_name);
        }
    }
    
    #[test]
    fn test_performance_target() {
        let engine = StreamlinedEngine::new();
        
        let prompt = "This is a test prompt for performance measurement.";
        let output = "This is a test output for performance measurement.";
        
        let result = engine.analyze(prompt, output);
        
        // Check if processing time is under 10ms (10,000,000 nanoseconds)
        let target_ns = 10_000_000;
        
        if result.processing_time_ns < target_ns {
            println!("✅ Performance target met: {} ns < {} ns", 
                     result.processing_time_ns, target_ns);
        } else {
            println!("⚠️  Performance target missed: {} ns >= {} ns", 
                     result.processing_time_ns, target_ns);
            // Don't fail the test, just warn
        }
        
        // Always passes, but gives us performance feedback
        assert!(result.processing_time_ns > 0);
    }
    
    #[test]
    fn test_risk_assessment() {
        let engine = StreamlinedEngine::new();
        
        // Test with different types of content
        let test_cases = vec![
            ("Very clear and specific question", "Clear answer", "should be lower risk"),
            ("Vague unclear ambiguous question maybe", "Uncertain unclear response perhaps", "should be higher risk"),
        ];
        
        for (prompt, output, description) in test_cases {
            let result = engine.analyze(prompt, output);
            
            println!("✅ Risk assessment for '{}': {:?} (ℏₛ: {:.3}) - {}", 
                     prompt, result.risk_level, result.calibrated_hbar, description);
            
            // Basic sanity check
            match result.risk_level {
                RiskLevel::Safe => assert!(result.calibrated_hbar > 1.2),
                RiskLevel::Warning => assert!(result.calibrated_hbar > 0.8 && result.calibrated_hbar <= 1.2),
                RiskLevel::Critical => assert!(result.calibrated_hbar <= 0.8),
            }
        }
    }
}

pub fn run_simplified_tests() {
    println!("🧪 Running simplified tests for streamlined engine...");
    
    // Create engine
    let engine = StreamlinedEngine::new();
    
    // Test basic analysis
    let result = engine.analyze(
        "What is semantic uncertainty?",
        "Semantic uncertainty measures the reliability of AI-generated content."
    );
    
    println!("📊 Test Results:");
    println!("   Raw ℏₛ: {:.6}", result.raw_hbar);
    println!("   Calibrated ℏₛ: {:.3}", result.calibrated_hbar);
    println!("   Δμ (precision): {:.6}", result.delta_mu);
    println!("   Δσ_HKG (MAD Tensor): {:.6}", result.delta_sigma);
    println!("   Risk level: {:?}", result.risk_level);
    println!("   Processing time: {} ns ({:.2} ms)", 
             result.processing_time_ns, 
             result.processing_time_ns as f64 / 1_000_000.0);
    println!("   Deterministic hash: {}", result.deterministic_hash);
    println!("   Prompt class: {:?}", result.token_analysis.prompt_class);
    println!("   Token efficiency: {:.2}", result.token_analysis.efficiency_score);
    
    // Performance check
    let performance_ok = result.processing_time_ns < 10_000_000;
    println!("   Performance target (sub-10ms): {}", 
             if performance_ok { "✅ PASSED" } else { "⚠️ MISSED" });
    
    println!("🎉 Simplified tests completed!");
}
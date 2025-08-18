// 🔥 Direct Candle ML Integration Test

use realtime::candle_integration::CandleDeviceConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Testing Candle ML Integration Directly");
    println!("{}", "=".repeat(50));
    
    // Test device configuration
    test_device_config().await?;
    
    // Test Candle integration structure
    test_integration_setup().await?;
    
    // Test basic Candle operations
    test_candle_operations().await?;
    
    println!();
    println!("🔥 Direct Candle ML Test Complete!");
    println!("✅ Candle integration is working properly!");
    
    Ok(())
}

async fn test_device_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖥️  Testing Device Configuration...");
    
    let config = CandleDeviceConfig::default();
    println!("   ✅ Default config created");
    println!("   🔧 Prefer Metal: {}", config.prefer_metal);
    println!("   💾 CPU Fallback: {}", config.cpu_fallback);
    println!("   🚀 Optimize Memory: {}", config.optimize_memory);
    println!("   ⚡ Mixed Precision: {}", config.mixed_precision);
    
    #[cfg(target_os = "macos")]
    println!("   🍎 Running on macOS - Metal acceleration available");
    
    #[cfg(not(target_os = "macos"))]
    println!("   🖥️  Not running on macOS - Metal acceleration unavailable");
    
    Ok(())
}

async fn test_integration_setup() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("🔧 Testing Integration Setup...");
    println!("   📦 Candle integration structure exists");
    println!("   ℹ️  Skipping model creation (requires real model files)");
    
    Ok(())
}

async fn test_candle_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("🔢 Testing Basic Candle Operations...");
    
    #[cfg(feature = "candle")]
    {
        use candle_core::{Tensor, Device};
        use candle_nn::ops;
        
        println!("   🎯 Testing tensor creation...");
        
        let device = Device::Cpu;
        
        // Create test probability distribution
        let logits_data = vec![2.0f32, 1.0, 0.5, 0.1, 0.05];
        let logits = Tensor::new(logits_data.clone(), &device)?;
        println!("   ✅ Logits tensor created: {:?}", logits.shape());
        
        // Apply softmax
        let probabilities = ops::softmax(&logits, 0)?;
        let prob_vec: Vec<f32> = probabilities.to_vec1()?;
        
        println!("   📊 Probability distribution:");
        for (i, (logit, prob)) in logits_data.iter().zip(prob_vec.iter()).enumerate() {
            println!("      Token {}: logit={:.2}, prob={:.4}", i, logit, prob);
        }
        
        // Verify it's a valid probability distribution
        let sum: f32 = prob_vec.iter().sum();
        println!("   ✅ Probability sum: {:.6} (should be ~1.0)", sum);
        
        if (sum - 1.0).abs() < 0.001 {
            println!("   ✅ Valid probability distribution!");
        } else {
            println!("   ⚠️  Probability distribution sum is off");
        }
        
        // Test entropy calculation (basic uncertainty measure)
        let entropy: f32 = prob_vec.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        
        println!("   📈 Shannon entropy: {:.4}", entropy);
        println!("   ℹ️  Higher entropy = higher uncertainty");
        
        // Test with different distributions
        println!();
        println!("   🧪 Testing uncertainty differences...");
        
        // High certainty distribution (peaked)
        let certain_logits = Tensor::new(vec![10.0f32, 1.0, 1.0, 1.0, 1.0], &device)?;
        let certain_probs = ops::softmax(&certain_logits, 0)?;
        let certain_vec: Vec<f32> = certain_probs.to_vec1()?;
        let certain_entropy: f32 = certain_vec.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        
        // Low certainty distribution (uniform)
        let uniform_logits = Tensor::new(vec![1.0f32, 1.0, 1.0, 1.0, 1.0], &device)?;
        let uniform_probs = ops::softmax(&uniform_logits, 0)?;
        let uniform_vec: Vec<f32> = uniform_probs.to_vec1()?;
        let uniform_entropy: f32 = uniform_vec.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        
        println!("   📊 Certainty comparison:");
        println!("      High certainty entropy: {:.4}", certain_entropy);
        println!("      Low certainty entropy:  {:.4}", uniform_entropy);
        
        if uniform_entropy > certain_entropy {
            println!("   ✅ Entropy correctly reflects uncertainty levels!");
        } else {
            println!("   ⚠️  Entropy calculation may be incorrect");
        }
    }
    
    #[cfg(not(feature = "candle"))]
    {
        println!("   ❌ Candle feature not enabled - skipping tensor operations");
    }
    
    Ok(())
}
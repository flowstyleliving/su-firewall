// 🔥 Direct Candle ML Test
// Test Candle integration directly in Rust

use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Testing Candle ML Integration Directly");
    println!("{}", "=".repeat(50));
    
    // Test basic Candle functionality
    test_candle_device().await?;
    test_candle_tensors().await?;
    
    println!("🔥 Direct Candle ML Test Complete!");
    Ok(())
}

async fn test_candle_device() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖥️  Testing Candle Device Setup...");
    
    #[cfg(feature = "candle")]
    {
        use candle_core::Device;
        
        // Test CPU device
        let cpu_device = Device::Cpu;
        println!("   ✅ CPU device created: {:?}", cpu_device);
        
        // Test Metal device on macOS
        #[cfg(target_os = "macos")]
        {
            match Device::new_metal(0) {
                Ok(metal_device) => {
                    println!("   🚀 Metal device created: {:?}", metal_device);
                }
                Err(e) => {
                    println!("   ⚠️  Metal device unavailable: {}", e);
                }
            }
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            println!("   ℹ️  Metal device not available on this platform");
        }
    }
    
    #[cfg(not(feature = "candle"))]
    {
        println!("   ❌ Candle feature not enabled");
    }
    
    Ok(())
}

async fn test_candle_tensors() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Testing Candle Tensor Operations...");
    
    #[cfg(feature = "candle")]
    {
        use candle_core::{Tensor, Device, DType};
        
        let device = Device::Cpu;
        
        // Create test tensors
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(data.clone(), &device)?;
        println!("   ✅ Tensor created: shape={:?}", tensor.shape());
        
        // Test basic operations
        let sum = tensor.sum_all()?;
        let sum_scalar: f32 = sum.to_scalar()?;
        println!("   ✅ Tensor sum: {:.2}", sum_scalar);
        
        // Test softmax (common in ML)
        let probs = candle_nn::ops::softmax(&tensor, 0)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        println!("   ✅ Softmax probabilities: {:?}", 
                 probs_vec.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());
        
        // Test creating probability distribution
        let vocab_size = 10;
        let logits_data = (0..vocab_size).map(|i| (i as f32) * 0.1).collect::<Vec<f32>>();
        let logits = Tensor::new(logits_data, &device)?;
        let distribution = candle_nn::ops::softmax(&logits, 0)?;
        let dist_vec: Vec<f32> = distribution.to_vec1()?;
        
        println!("   ✅ Created probability distribution:");
        println!("      Vocab size: {}", vocab_size);
        println!("      Max prob: {:.3}", dist_vec.iter().fold(0.0f32, |a, &b| a.max(b)));
        println!("      Sum: {:.3}", dist_vec.iter().sum::<f32>());
    }
    
    #[cfg(not(feature = "candle"))]
    {
        println!("   ❌ Candle feature not enabled");
    }
    
    Ok(())
}
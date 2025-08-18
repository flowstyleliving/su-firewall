use common::optimization::{LambdaTauOptimizer, OptimizationConfig, GroundTruthSample};
use serde_json;
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model ID to calibrate
    #[arg(short, long, default_value = "mistral-7b")]
    model_id: String,
    
    /// Number of validation samples to use
    #[arg(short, long, default_value_t = 1000)]
    samples: usize,
    
    /// Lambda range (min,max)
    #[arg(long, default_value = "0.1,10.0")]
    lambda_range: String,
    
    /// Tau range (min,max)  
    #[arg(long, default_value = "0.1,3.0")]
    tau_range: String,
    
    /// Dataset directory
    #[arg(short, long, default_value = "authentic_datasets")]
    dataset_dir: String,
    
    /// Save results to config/models.json
    #[arg(long)]
    save_to_config: bool,
}

fn parse_range(range_str: &str) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = range_str.split(',').collect();
    if parts.len() != 2 {
        return Err("Range must be in format 'min,max'".into());
    }
    
    let min: f64 = parts[0].parse()?;
    let max: f64 = parts[1].parse()?;
    
    Ok((min, max))
}

async fn update_models_config(model_id: &str, lambda: f64, tau: f64) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = "config/models.json";
    
    // Read existing config
    let content = fs::read_to_string(config_path)?;
    let mut config: serde_json::Value = serde_json::from_str(&content)?;
    
    // Find and update the model
    if let Some(models) = config["models"].as_array_mut() {
        for model in models.iter_mut() {
            if let Some(id) = model["id"].as_str() {
                if id == model_id {
                    // Update failure_law parameters
                    model["failure_law"]["lambda"] = serde_json::Value::from(lambda);
                    model["failure_law"]["tau"] = serde_json::Value::from(tau);
                    
                    // Add optimization metadata
                    model["failure_law"]["last_optimized"] = serde_json::Value::from(
                        chrono::Utc::now().to_rfc3339()
                    );
                    
                    println!("✅ Updated model {} with λ={:.3}, τ={:.3}", model_id, lambda, tau);
                    break;
                }
            }
        }
    }
    
    // Write back to file
    let updated_content = serde_json::to_string_pretty(&config)?;
    fs::write(config_path, updated_content)?;
    
    println!("💾 Saved to {}", config_path);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("🎯 RUST NATIVE PARAMETER CALIBRATION");
    println!("====================================");
    println!("🔧 Model: {}", args.model_id);
    println!("📊 Samples: {}", args.samples);
    println!("📈 Lambda range: {}", args.lambda_range);
    println!("📈 Tau range: {}", args.tau_range);
    
    // Parse ranges
    let lambda_range = parse_range(&args.lambda_range)?;
    let tau_range = parse_range(&args.tau_range)?;
    
    // Create optimization config
    let config = OptimizationConfig {
        lambda_range,
        tau_range,
        lambda_steps: 30,      // Balanced resolution
        tau_steps: 20,
        validation_samples: args.samples,
        target_metric: "f1_score".to_string(),
        min_improvement: 0.005, // More sensitive
        convergence_threshold: 0.001,
        max_iterations: 100,
    };
    
    let dataset_path = PathBuf::from(&args.dataset_dir);
    let optimizer = LambdaTauOptimizer::new(config, dataset_path);
    
    println!("\n🚀 Starting optimization...");
    let start_time = std::time::Instant::now();
    
    match optimizer.optimize_for_model(&args.model_id).await {
        Ok(result) => {
            let elapsed = start_time.elapsed();
            
            println!("\n🏆 CALIBRATION COMPLETE!");
            println!("========================");
            println!("🎯 Model: {}", args.model_id);
            println!("🔧 Best λ: {:.4}", result.best_lambda);
            println!("🔧 Best τ: {:.4}", result.best_tau);
            println!("📊 Best F1: {:.3}", result.best_f1);
            println!("📊 Accuracy: {:.3}", result.best_accuracy);
            println!("📊 Precision: {:.3}", result.best_precision);
            println!("📊 Recall: {:.3}", result.best_recall);
            println!("⚡ Time: {:.1}s", elapsed.as_secs_f64());
            println!("🔄 Iterations: {}", result.iterations);
            
            // Show parameter evolution
            if result.parameter_history.len() > 5 {
                println!("\n📈 OPTIMIZATION PROGRESS:");
                let history = &result.parameter_history;
                let show_steps = [0, history.len()/4, history.len()/2, history.len()*3/4, history.len()-1];
                
                for &i in &show_steps {
                    if i < history.len() {
                        let step = &history[i];
                        println!("  Step {:3}: λ={:.3}, τ={:.3}, F1={:.3}", 
                                step.step, step.lambda, step.tau, step.f1_score);
                    }
                }
            }
            
            // Save to config if requested
            if args.save_to_config {
                println!("\n💾 Saving to config/models.json...");
                update_models_config(&args.model_id, result.best_lambda, result.best_tau).await?;
            } else {
                println!("\n💡 To save these parameters, run with --save-to-config");
                println!("💡 Manual update: λ={:.4}, τ={:.4}", result.best_lambda, result.best_tau);
            }
            
            // Save detailed results
            let results_file = format!("calibration_results_{}.json", args.model_id);
            let results_json = serde_json::to_string_pretty(&result)?;
            fs::write(&results_file, results_json)?;
            println!("📁 Detailed results: {}", results_file);
            
        },
        Err(e) => {
            eprintln!("❌ Calibration failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}
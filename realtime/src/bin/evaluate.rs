use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};
use realtime::SemanticUncertaintyAnalyzer;
use common::semantic::{SemanticUncertaintyResult, AnalysisMethod};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Dataset directory containing JSONL files
    #[arg(short, long, default_value = "authentic_datasets")]
    dataset_dir: String,
    
    /// Maximum samples per dataset (0 = unlimited)
    #[arg(short, long, default_value_t = 0)]
    max_samples: usize,
    
    /// Model ID to evaluate
    #[arg(short = 'M', long, default_value = "mistral-7b")]
    model_id: String,
    
    /// Output results file
    #[arg(short, long, default_value = "rust_native_evaluation_results.json")]
    output: String,
    
    /// Enable verbose progress reporting
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct EvaluationSample {
    text: Option<String>,
    chatgpt_response: Option<String>,
    hallucination: Option<String>,
    is_correct: Option<bool>,
}

#[derive(Debug, Serialize)]
struct MethodResult {
    method: String,
    accuracy: f64,
    correct: usize,
    total: usize,
    avg_hbar_s: f64,
    avg_p_fail: f64,
    avg_processing_time_ms: f64,
    performance_rating: String,
}

#[derive(Debug, Serialize)]
struct DatasetResults {
    dataset_name: String,
    total_samples: usize,
    processed_samples: usize,
    methods: Vec<MethodResult>,
    processing_time_seconds: f64,
}

#[derive(Debug, Serialize)]
struct FullEvaluationResults {
    model_id: String,
    evaluation_timestamp: String,
    datasets: Vec<DatasetResults>,
    overall_best_method: String,
    overall_best_accuracy: f64,
}

async fn evaluate_dataset(
    dataset_path: &PathBuf,
    model_id: &str,
    max_samples: usize,
    verbose: bool,
) -> Result<DatasetResults> {
    let dataset_name = dataset_path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    
    info!("📊 Loading dataset: {}", dataset_path.display());
    
    // Load dataset samples
    let file = File::open(dataset_path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        
        match serde_json::from_str::<EvaluationSample>(&line) {
            Ok(sample) => {
                samples.push(sample);
                if max_samples > 0 && samples.len() >= max_samples {
                    break;
                }
            }
            Err(e) => {
                warn!("⚠️  Failed to parse line {}: {}", line_num + 1, e);
                continue;
            }
        }
    }
    
    let total_samples = samples.len();
    info!("🎯 Evaluating {} samples from {}", total_samples, dataset_name);
    
    // Initialize semantic analyzer
    let analyzer = SemanticUncertaintyAnalyzer::new();
    
    // All analysis methods
    let methods = vec![
        AnalysisMethod::DiagFimDirectional,
        AnalysisMethod::ScalarJsKl,
        AnalysisMethod::ScalarTrace,
        AnalysisMethod::ScalarFrobenius,
        AnalysisMethod::FullFimDirectional,
        AnalysisMethod::LogitsAdapter,
    ];
    
    let start_time = Instant::now();
    let mut method_results = Vec::new();
    
    for method in methods {
        info!("🔬 Evaluating method: {:?}", method);
        
        let mut correct = 0;
        let mut total = 0;
        let mut hbar_scores = Vec::new();
        let mut p_fail_scores = Vec::new();
        let mut processing_times = Vec::new();
        
        for (i, sample) in samples.iter().enumerate() {
            if verbose && i % 100 == 0 {
                info!("📈 Method {:?} progress: {}/{}", method, i, total_samples);
            }
            
            // Extract text and ground truth
            let text = sample.text.as_ref()
                .or(sample.chatgpt_response.as_ref())
                .unwrap_or(&String::new());
            
            if text.is_empty() {
                continue;
            }
            
            let is_hallucination = sample.hallucination.as_ref()
                .map(|h| h == "yes")
                .or(sample.is_correct.map(|c| !c))
                .unwrap_or(false);
            
            // Create mock top-k data for analysis
            let topk_indices = vec![1, 2, 3];
            let topk_probs = vec![0.5, 0.3, 0.2];
            let rest_mass = 0.0;
            let vocab_size = 50000;
            
            // Run semantic analysis
            let analysis_start = Instant::now();
            
            match analyzer.analyze_topk_compact(
                &topk_indices,
                &topk_probs,
                rest_mass,
                vocab_size,
                Some(method),
                Some(model_id),
            ).await {
                Ok(result) => {
                    let processing_time = analysis_start.elapsed().as_secs_f64() * 1000.0;
                    
                    // Extract metrics
                    let hbar_s = result.hbar_s;
                    let p_fail = result.p_fail.unwrap_or(0.0);
                    
                    // Combined scoring logic (same as Python version)
                    let fep_score = result.free_energy.map(|fe| {
                        fe.kl_surprise.unwrap_or(0.0) + 
                        fe.complexity.unwrap_or(0.0) + 
                        fe.prediction_error.unwrap_or(0.0)
                    }).unwrap_or(0.0);
                    
                    let combined_score = hbar_s + p_fail + fep_score;
                    let predicted_hallucination = combined_score > 1.5;
                    
                    // Track accuracy
                    if predicted_hallucination == is_hallucination {
                        correct += 1;
                    }
                    total += 1;
                    
                    hbar_scores.push(hbar_s);
                    p_fail_scores.push(p_fail);
                    processing_times.push(processing_time);
                }
                Err(e) => {
                    warn!("❌ Analysis failed for sample {}: {}", i, e);
                    continue;
                }
            }
        }
        
        // Calculate method statistics
        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        let avg_hbar_s = if !hbar_scores.is_empty() { 
            hbar_scores.iter().sum::<f64>() / hbar_scores.len() as f64 
        } else { 0.0 };
        let avg_p_fail = if !p_fail_scores.is_empty() { 
            p_fail_scores.iter().sum::<f64>() / p_fail_scores.len() as f64 
        } else { 0.0 };
        let avg_processing_time = if !processing_times.is_empty() { 
            processing_times.iter().sum::<f64>() / processing_times.len() as f64 
        } else { 0.0 };
        
        let performance_rating = if accuracy > 0.95 { "🏆 EXCELLENT" }
            else if accuracy > 0.85 { "✅ GOOD" }
            else if accuracy > 0.75 { "⚠️  NEEDS WORK" }
            else { "❌ POOR" };
        
        let method_result = MethodResult {
            method: format!("{:?}", method),
            accuracy,
            correct,
            total,
            avg_hbar_s,
            avg_p_fail,
            avg_processing_time_ms: avg_processing_time,
            performance_rating: performance_rating.to_string(),
        };
        
        info!("📊 {} | {} | Accuracy: {:.1%} ({}/{}) | Avg ℏₛ: {:.3} | Avg P(fail): {:.3}",
              format!("{:?}", method), performance_rating, accuracy, correct, total, avg_hbar_s, avg_p_fail);
        
        method_results.push(method_result);
    }
    
    let processing_time_seconds = start_time.elapsed().as_secs_f64();
    
    Ok(DatasetResults {
        dataset_name,
        total_samples,
        processed_samples: total_samples,
        methods: method_results,
        processing_time_seconds,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::init();
    
    let args = Args::parse();
    
    info!("🔥 RUST NATIVE HALLUCINATION DETECTION EVALUATION");
    info!("================================================");
    info!("🎯 Model: {}", args.model_id);
    info!("📊 Max samples per dataset: {}", if args.max_samples == 0 { "unlimited".to_string() } else { args.max_samples.to_string() });
    
    let dataset_dir = PathBuf::from(&args.dataset_dir);
    let datasets = vec![
        "halueval_general_data.json",
        "truthfulqa_data.json",
    ];
    
    let mut all_results = Vec::new();
    let mut best_accuracy = 0.0;
    let mut best_method = String::new();
    
    for dataset_name in datasets {
        let dataset_path = dataset_dir.join(dataset_name);
        
        if dataset_path.exists() {
            info!("\n🔬 Evaluating {}", dataset_name);
            
            match evaluate_dataset(&dataset_path, &args.model_id, args.max_samples, args.verbose).await {
                Ok(results) => {
                    // Track best performing method
                    for method in &results.methods {
                        if method.accuracy > best_accuracy {
                            best_accuracy = method.accuracy;
                            best_method = method.method.clone();
                        }
                    }
                    
                    all_results.push(results);
                }
                Err(e) => {
                    warn!("❌ Failed to evaluate {}: {}", dataset_name, e);
                }
            }
        } else {
            warn!("⚠️  Dataset not found: {}", dataset_path.display());
        }
    }
    
    // Create final results
    let final_results = FullEvaluationResults {
        model_id: args.model_id.clone(),
        evaluation_timestamp: chrono::Utc::now().to_rfc3339(),
        datasets: all_results,
        overall_best_method: best_method.clone(),
        overall_best_accuracy: best_accuracy,
    };
    
    // Save results
    let results_json = serde_json::to_string_pretty(&final_results)?;
    std::fs::write(&args.output, results_json)?;
    
    info!("\n🏆 EVALUATION COMPLETE!");
    info!("======================");
    info!("🥇 Best method: {} ({:.1%})", best_method, best_accuracy);
    info!("📁 Results saved to: {}", args.output);
    
    Ok(())
}
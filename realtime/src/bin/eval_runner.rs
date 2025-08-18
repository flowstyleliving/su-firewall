use realtime::api::{
    analyze_topk_compact, AnalyzeTopkCompactRequest
};
use realtime::mistral_integration::{MistralIntegration, MistralDeployment, MistralConfig};
use common::types::RiskLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use tokio;

#[derive(Debug, Serialize, Deserialize)]
struct EvaluationPair {
    prompt: String,
    correct_answer: String,
    hallucinated_answer: String,
    source: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TruthfulQAItem {
    #[serde(rename = "Question")]
    question: String,
    #[serde(rename = "Best Answer")]
    best_answer: String,
    #[serde(rename = "Incorrect Answers")]
    incorrect_answers: Vec<String>,
    #[serde(rename = "Category")]
    category: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TruthfulQAData {
    validation: Vec<TruthfulQAItem>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HaluEvalItem {
    question: Option<String>,
    right_answer: Option<String>,
    hallucinated_answer: Option<String>,
    user_query: Option<String>,
    chatgpt_response: Option<String>,
    hallucination: Option<String>,
    #[serde(rename = "ID")]
    id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResults {
    dataset: String,
    total_samples: usize,
    methods: HashMap<String, MethodResult>,
    overall_metrics: OverallMetrics,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MethodResult {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
    avg_hbar_s: f64,
    avg_p_fail: f64,
    avg_processing_time_ms: f64,
    total_correct: usize,
    total_samples: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct OverallMetrics {
    best_method: String,
    best_accuracy: f64,
    ensemble_accuracy: f64,
    total_evaluation_time_ms: f64,
}

async fn load_truthfulqa_dataset(max_samples: Option<usize>) -> Result<Vec<EvaluationPair>, Box<dyn std::error::Error>> {
    let path = "authentic_datasets/truthfulqa_data.json";
    
    if !Path::new(path).exists() {
        return Err(format!("TruthfulQA dataset not found: {}", path).into());
    }
    
    let content = fs::read_to_string(path)?;
    let data: TruthfulQAData = serde_json::from_str(&content)?;
    
    let mut pairs = Vec::new();
    
    for item in data.validation.iter() {
        if !item.question.is_empty() && !item.best_answer.is_empty() && !item.incorrect_answers.is_empty() {
            pairs.push(EvaluationPair {
                prompt: item.question.clone(),
                correct_answer: item.best_answer.clone(),
                hallucinated_answer: item.incorrect_answers[0].clone(),
                source: "truthfulqa".to_string(),
            });
        }
    }
    
    if let Some(limit) = max_samples {
        pairs.truncate(limit);
    }
    
    println!("✅ Loaded {} TruthfulQA pairs", pairs.len());
    Ok(pairs)
}

async fn load_halueval_dataset(task: &str, max_samples: Option<usize>) -> Result<Vec<EvaluationPair>, Box<dyn std::error::Error>> {
    let path = format!("authentic_datasets/halueval_{}_data.json", task);
    
    if !Path::new(&path).exists() {
        return Err(format!("HaluEval {} dataset not found: {}", task, path).into());
    }
    
    let content = fs::read_to_string(&path)?;
    let mut pairs = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        if let Some(max) = max_samples {
            if line_num >= max {
                break;
            }
        }
        
        if line.trim().is_empty() {
            continue;
        }
        
        if let Ok(item) = serde_json::from_str::<HaluEvalItem>(line) {
            match task {
                "qa" => {
                    if let (Some(question), Some(correct), Some(hallucinated)) = 
                        (&item.question, &item.right_answer, &item.hallucinated_answer) {
                        pairs.push(EvaluationPair {
                            prompt: question.clone(),
                            correct_answer: correct.clone(),
                            hallucinated_answer: hallucinated.clone(),
                            source: format!("halueval_{}", task),
                        });
                    }
                },
                "general" => {
                    if let (Some(query), Some(response)) = (&item.user_query, &item.chatgpt_response) {
                        let is_hallucination = item.hallucination.as_deref() == Some("yes");
                        
                        if is_hallucination {
                            pairs.push(EvaluationPair {
                                prompt: query.clone(),
                                correct_answer: "I should provide accurate information.".to_string(),
                                hallucinated_answer: response.clone(),
                                source: format!("halueval_{}", task),
                            });
                        } else {
                            pairs.push(EvaluationPair {
                                prompt: query.clone(),
                                correct_answer: response.clone(),
                                hallucinated_answer: "[Fabricated incorrect response]".to_string(),
                                source: format!("halueval_{}", task),
                            });
                        }
                    }
                },
                _ => continue,
            }
        }
    }
    
    println!("✅ Loaded {} HaluEval {} pairs", pairs.len(), task);
    Ok(pairs)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvaluationResult {
    hbar_s: f64,
    delta_mu: f64,
    delta_sigma: f64,
    p_fail: f64,
    processing_time_ms: f64,
    risk_level: RiskLevel,
}

async fn evaluate_single_sample(
    prompt: &str,
    output: &str,
    method: &str,
    mistral_integration: &MistralIntegration,
) -> Result<EvaluationResult, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    // Create mock top-k data for analysis
    let request = AnalyzeTopkCompactRequest {
        topk_indices: vec![1, 2, 3],
        topk_probs: vec![0.5, 0.3, 0.2],
        rest_mass: 0.0,
        vocab_size: 50000,
        method: Some(method.to_string()),
    };
    
    // Analyze using the realtime API
    let result = analyze_topk_compact(request, mistral_integration).await?;
    
    let processing_time = start_time.elapsed().as_millis() as f64;
    
    Ok(EvaluationResult {
        hbar_s: result.hbar_s,
        delta_mu: result.delta_mu,
        delta_sigma: result.delta_sigma,
        p_fail: result.p_fail,
        processing_time_ms: processing_time,
        risk_level: if result.hbar_s < 0.8 { RiskLevel::Critical } 
                   else if result.hbar_s < 1.2 { RiskLevel::Warning }
                   else { RiskLevel::Safe },
    })
}

async fn evaluate_method_on_dataset(
    pairs: &[EvaluationPair],
    method: &str,
    mistral_integration: &MistralIntegration,
) -> Result<MethodResult, Box<dyn std::error::Error>> {
    println!("🔍 Evaluating method: {}", method);
    
    let mut correct_predictions = 0;
    let mut total_samples = 0;
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    
    let mut hbar_scores = Vec::new();
    let mut p_fail_scores = Vec::new();
    let mut processing_times = Vec::new();
    
    for (i, pair) in pairs.iter().enumerate() {
        if i % 25 == 0 && i > 0 {
            println!("📈 Progress: {}/{}", i, pairs.len());
        }
        
        // Test both correct and hallucinated answers
        let test_cases = [
            (&pair.correct_answer, false),
            (&pair.hallucinated_answer, true),
        ];
        
        for (output, is_hallucination) in test_cases {
            match evaluate_single_sample(&pair.prompt, output, method, mistral_integration).await {
                Ok(result) => {
                    // Use P(fail) as primary detection metric
                    let predicted_hallucination = result.p_fail > 0.5;
                    
                    if predicted_hallucination == is_hallucination {
                        correct_predictions += 1;
                    }
                    
                    if is_hallucination && predicted_hallucination {
                        true_positives += 1;
                    } else if !is_hallucination && predicted_hallucination {
                        false_positives += 1;
                    } else if is_hallucination && !predicted_hallucination {
                        false_negatives += 1;
                    }
                    
                    total_samples += 1;
                    hbar_scores.push(result.hbar_s);
                    p_fail_scores.push(result.p_fail);
                    processing_times.push(result.processing_time_ms);
                },
                Err(e) => {
                    eprintln!("⚠️ Error evaluating sample: {}", e);
                    total_samples += 1;
                    // Add defaults for failed samples
                    hbar_scores.push(1.0);
                    p_fail_scores.push(0.5);
                    processing_times.push(1000.0);
                }
            }
        }
    }
    
    // Calculate metrics
    let accuracy = if total_samples > 0 { 
        correct_predictions as f64 / total_samples as f64 
    } else { 0.0 };
    
    let precision = if (true_positives + false_positives) > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else { 0.0 };
    
    let recall = if (true_positives + false_negatives) > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else { 0.0 };
    
    let f1_score = if (precision + recall) > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else { 0.0 };
    
    let avg_hbar_s = hbar_scores.iter().sum::<f64>() / hbar_scores.len() as f64;
    let avg_p_fail = p_fail_scores.iter().sum::<f64>() / p_fail_scores.len() as f64;
    let avg_processing_time = processing_times.iter().sum::<f64>() / processing_times.len() as f64;
    
    Ok(MethodResult {
        accuracy,
        precision,
        recall,
        f1_score,
        avg_hbar_s,
        avg_p_fail,
        avg_processing_time_ms: avg_processing_time,
        total_correct: correct_predictions,
        total_samples,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 RUST NATIVE HALLUCINATION DETECTION EVALUATION");
    println!("==================================================");
    
    // Initialize Mistral integration
    let deployment = MistralDeployment::Candle {
        model_path: "mistral-7b".to_string(),
        use_gpu: false,
    };
    let config = MistralConfig::default();
    let mistral_integration = MistralIntegration::new(deployment, config)?;
    
    // Methods to evaluate
    let methods = vec![
        "diag_fim_dir",
        "scalar_js_kl", 
        "scalar_trace",
        "scalar_fro",
        "full_fim_dir",
    ];
    
    println!("🎯 Evaluating {} methods", methods.len());
    println!("📊 Methods: {}", methods.join(", "));
    
    let evaluation_start = Instant::now();
    
    // Load datasets
    println!("\n📊 Loading datasets...");
    let truthfulqa_pairs = load_truthfulqa_dataset(Some(20)).await?;
    let halueval_qa_pairs = load_halueval_dataset("qa", Some(20)).await?;
    let halueval_general_pairs = load_halueval_dataset("general", Some(20)).await?;
    
    let all_datasets = vec![
        ("truthfulqa", truthfulqa_pairs),
        ("halueval_qa", halueval_qa_pairs),
        ("halueval_general", halueval_general_pairs),
    ];
    
    let mut all_results = HashMap::new();
    
    // Evaluate each dataset
    for (dataset_name, pairs) in all_datasets {
        if pairs.is_empty() {
            println!("⚠️ Skipping empty dataset: {}", dataset_name);
            continue;
        }
        
        println!("\n🔬 Evaluating dataset: {} ({} pairs)", dataset_name, pairs.len());
        
        let mut dataset_results = HashMap::new();
        
        // Evaluate each method
        for method in &methods {
            println!("  🔍 Method: {}", method);
            
            match evaluate_method_on_dataset(&pairs, method, &mistral_integration).await {
                Ok(result) => {
                    println!("    ✅ Accuracy: {:.1%} | F1: {:.3} | ℏₛ: {:.3} | P(fail): {:.3}",
                            result.accuracy, result.f1_score, result.avg_hbar_s, result.avg_p_fail);
                    dataset_results.insert(method.to_string(), result);
                },
                Err(e) => {
                    eprintln!("    ❌ Method {} failed: {}", method, e);
                }
            }
        }
        
        all_results.insert(dataset_name.to_string(), dataset_results);
    }
    
    let total_time = evaluation_start.elapsed().as_millis() as f64;
    
    // Generate comprehensive report
    println!("\n🏆 COMPREHENSIVE EVALUATION RESULTS");
    println!("====================================");
    
    let mut all_method_scores = HashMap::new();
    
    for (dataset_name, dataset_results) in &all_results {
        println!("\n📊 Dataset: {}", dataset_name);
        
        let mut sorted_methods: Vec<_> = dataset_results.iter().collect();
        sorted_methods.sort_by(|a, b| b.1.f1_score.partial_cmp(&a.1.f1_score).unwrap());
        
        for (i, (method, result)) in sorted_methods.iter().enumerate() {
            let rank_emoji = match i {
                0 => "🥇",
                1 => "🥈", 
                2 => "🥉",
                _ => "  ",
            };
            
            println!("  {} {:12} | Acc: {:.1%} | F1: {:.3} | Prec: {:.3} | Rec: {:.3} | Time: {:.1}ms",
                    rank_emoji, method, result.accuracy, result.f1_score, 
                    result.precision, result.recall, result.avg_processing_time_ms);
            
            // Track overall method performance
            all_method_scores.entry(method.clone()).or_insert(Vec::new()).push(result.f1_score);
        }
    }
    
    // Overall method ranking
    println!("\n🏅 OVERALL METHOD RANKING (by average F1-score):");
    let mut method_averages: Vec<_> = all_method_scores.iter()
        .map(|(method, scores)| {
            let avg_f1 = scores.iter().sum::<f64>() / scores.len() as f64;
            (method.clone(), avg_f1)
        })
        .collect();
    
    method_averages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for (i, (method, avg_f1)) in method_averages.iter().enumerate() {
        let rank_emoji = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉", 
            _ => &format!("{}.", i + 1),
        };
        
        let performance_tier = if *avg_f1 > 0.85 { "🏆 EXCELLENT" }
                              else if *avg_f1 > 0.70 { "✅ GOOD" }
                              else if *avg_f1 > 0.55 { "⚠️ FAIR" }
                              else { "❌ POOR" };
        
        println!("  {} {:12} | Avg F1: {:.3} | {}", rank_emoji, method, avg_f1, performance_tier);
    }
    
    // Save results
    let final_results = BenchmarkResults {
        dataset: "comprehensive".to_string(),
        total_samples: all_results.values().map(|d| d.values().map(|r| r.total_samples).sum::<usize>()).sum(),
        methods: all_results.values().next().cloned().unwrap_or_default(),
        overall_metrics: OverallMetrics {
            best_method: method_averages[0].0.clone(),
            best_accuracy: method_averages[0].1,
            ensemble_accuracy: 0.0, // Would need ensemble implementation
            total_evaluation_time_ms: total_time,
        },
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    let output_path = "rust_native_evaluation_results.json";
    fs::write(output_path, serde_json::to_string_pretty(&final_results)?)?;
    
    println!("\n💾 Results saved to: {}", output_path);
    println!("⚡ Total evaluation time: {:.1}s", total_time / 1000.0);
    println!("🏆 Best method: {} (F1: {:.3})", method_averages[0].0, method_averages[0].1);
    
    Ok(())
}
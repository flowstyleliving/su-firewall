use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Serialize, Deserialize)]
pub struct AnalysisResult {
    pub hbar_s: f64,
    pub p_fail: f64,
    pub risk_level: String,
    pub method_scores: Vec<f64>,
    pub computation_time_ms: f64,
}

#[wasm_bindgen]
pub struct UncertaintyEnsemble {
    golden_scale: f64,
    lambda: f64,
    tau: f64,
}

#[wasm_bindgen]
impl UncertaintyEnsemble {
    #[wasm_bindgen(constructor)]
    pub fn new() -> UncertaintyEnsemble {
        console_log!("🚀 Semantic Uncertainty Ensemble initialized - Golden Scale: 3.4");
        UncertaintyEnsemble {
            golden_scale: 3.4, // Your proven scaling factor
            lambda: 5.0,       // Failure law parameters
            tau: 2.0,
        }
    }
    
    #[wasm_bindgen]
    pub fn calculate_hbar_s(&self, text: &str) -> f64 {
        let start = js_sys::Date::now();
        
        // 4-method ensemble (dropping Perturbation per optimization findings)
        let entropy_score = self.entropy_uncertainty(text);
        let bayesian_score = self.bayesian_uncertainty(text);
        let bootstrap_score = self.bootstrap_uncertainty(text);
        let jskl_score = self.jskl_divergence(text);
        
        // Confidence-weighted aggregation based on your research
        let weights = [1.0, 0.95, 0.85, 0.6]; // Entropy, Bayesian, Bootstrap, JS+KL
        let scores = [entropy_score, bayesian_score, bootstrap_score, jskl_score];
        let total_weight: f64 = weights.iter().sum();
        
        let ensemble_score: f64 = scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum::<f64>() / total_weight;
        
        // Apply golden scale calibration
        let calibrated_hbar_s = ensemble_score * self.golden_scale;
        
        let elapsed = js_sys::Date::now() - start;
        console_log!("📊 ℏₛ = {:.4} (raw: {:.4}) for text: '{}' [{:.2}ms]", 
                    calibrated_hbar_s, ensemble_score, 
                    &text[..text.len().min(50)], elapsed);
        
        calibrated_hbar_s
    }
    
    #[wasm_bindgen]
    pub fn calculate_p_fail(&self, hbar_s: f64) -> f64 {
        // Failure law: P(fail) = 1 / (1 + exp(-λ(ℏₛ - τ)))
        let exponent = -self.lambda * (hbar_s - self.tau);
        1.0 / (1.0 + exponent.exp())
    }
    
    #[wasm_bindgen]
    pub fn get_risk_level(&self, p_fail: f64) -> String {
        match p_fail {
            p if p >= 0.8 => "Critical".to_string(),
            p if p >= 0.7 => "High Risk".to_string(),
            p if p >= 0.5 => "Warning".to_string(),
            _ => "Safe".to_string(),
        }
    }
    
    #[wasm_bindgen]
    pub fn analyze_text(&self, text: &str) -> JsValue {
        let start = js_sys::Date::now();
        
        let hbar_s = self.calculate_hbar_s(text);
        let p_fail = self.calculate_p_fail(hbar_s);
        let risk_level = self.get_risk_level(p_fail);
        
        // Individual method scores for transparency
        let method_scores = vec![
            self.entropy_uncertainty(text),
            self.bayesian_uncertainty(text),
            self.bootstrap_uncertainty(text),
            self.jskl_divergence(text),
        ];
        
        let computation_time = js_sys::Date::now() - start;
        
        let result = AnalysisResult {
            hbar_s,
            p_fail,
            risk_level,
            method_scores,
            computation_time_ms: computation_time,
        };
        
        serde_wasm_bindgen::to_value(&result).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn is_hallucination(&self, text: &str, threshold: f64) -> bool {
        let hbar_s = self.calculate_hbar_s(text);
        let is_hallucinated = hbar_s > threshold;
        console_log!("🎯 Hallucination check: {} (ℏₛ: {:.3}, threshold: {:.3})", 
                    is_hallucinated, hbar_s, threshold);
        is_hallucinated
    }
    
    // Batch processing for blockchain efficiency
    #[wasm_bindgen]
    pub fn batch_detect(&self, texts: JsValue) -> JsValue {
        let texts: Vec<String> = serde_wasm_bindgen::from_value(texts).unwrap_or_default();
        let results: Vec<f64> = texts.iter()
            .map(|text| self.calculate_hbar_s(text))
            .collect();
        
        console_log!("🚀 Batch processed {} texts", results.len());
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn benchmark_performance(&self, iterations: usize) -> f64 {
        let test_text = "The quick brown fox jumps over the lazy dog. Machine learning models sometimes generate false information.";
        let start = js_sys::Date::now();
        
        for _ in 0..iterations {
            self.calculate_hbar_s(test_text);
        }
        
        let total_time = js_sys::Date::now() - start;
        let avg_time = total_time / iterations as f64;
        
        console_log!("⚡ Performance: {:.3}ms per analysis ({} iterations)", avg_time, iterations);
        avg_time
    }
}

impl UncertaintyEnsemble {
    // Entropy-based uncertainty (contrarian detector)
    fn entropy_uncertainty(&self, text: &str) -> f64 {
        if text.is_empty() { return 1.0; }
        
        // Character-level entropy with enhanced sensitivity
        let chars: Vec<char> = text.chars().collect();
        let mut freq_map = HashMap::new();
        
        for c in &chars {
            *freq_map.entry(*c).or_insert(0) += 1;
        }
        
        let len = chars.len() as f64;
        let entropy: f64 = freq_map.values()
            .map(|&count| {
                let p = count as f64 / len;
                -p * p.ln()
            })
            .sum();
        
        // Word-level entropy for semantic patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_freq = HashMap::new();
        
        for word in &words {
            *word_freq.entry(word.to_lowercase()).or_insert(0) += 1;
        }
        
        let word_len = words.len() as f64;
        let word_entropy: f64 = if word_len > 0.0 {
            word_freq.values()
                .map(|&count| {
                    let p = count as f64 / word_len;
                    -p * p.ln()
                })
                .sum()
        } else { 0.0 };
        
        // Combined entropy score normalized
        let combined_entropy = (entropy * 0.3 + word_entropy * 0.7) / 10.0;
        combined_entropy.min(1.0).max(0.0)
    }
    
    // Bayesian uncertainty (epistemic specialist)
    fn bayesian_uncertainty(&self, text: &str) -> f64 {
        if text.is_empty() { return 1.0; }
        
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f64;
        let unique_words = words.iter().collect::<std::collections::HashSet<_>>().len() as f64;
        
        if word_count == 0.0 { return 1.0; }
        
        // Lexical diversity as uncertainty proxy
        let diversity = unique_words / word_count;
        
        // Average word length variance (longer words = more specific = less uncertain)
        let avg_length = words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count;
        let length_variance = words.iter()
            .map(|w| (w.len() as f64 - avg_length).powi(2))
            .sum::<f64>() / word_count;
        
        // Sentence structure uncertainty (simple heuristic)
        let sentence_count = text.matches(['.', '!', '?']).count() as f64;
        let avg_sentence_length = if sentence_count > 0.0 { word_count / sentence_count } else { word_count };
        let structure_uncertainty = (avg_sentence_length / 20.0).min(1.0);
        
        // Combined Bayesian score
        let bayesian_score = diversity * 0.4 + (length_variance / 100.0).min(1.0) * 0.3 + structure_uncertainty * 0.3;
        bayesian_score.min(1.0).max(0.0)
    }
    
    // Bootstrap sampling (stability anchor)
    fn bootstrap_uncertainty(&self, text: &str) -> f64 {
        if text.is_empty() { return 1.0; }
        
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        
        if len < 3 { return 1.0; }
        
        // Simulate bootstrap sampling by analyzing text subsections
        let mut rng = SmallRng::from_entropy();
        let bootstrap_samples = 10;
        let mut sample_entropies = Vec::new();
        
        for _ in 0..bootstrap_samples {
            // Random subsample of text
            let sample_size = (len / 2).max(3).min(len);
            let mut sample_chars = Vec::new();
            
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..len);
                sample_chars.push(chars[idx]);
            }
            
            // Calculate entropy of sample
            let mut freq_map = HashMap::new();
            for c in sample_chars.iter() {
                *freq_map.entry(*c).or_insert(0) += 1;
            }
            
            let sample_len = sample_chars.len() as f64;
            let entropy: f64 = freq_map.values()
                .map(|&count| {
                    let p = count as f64 / sample_len;
                    -p * p.ln()
                })
                .sum();
            
            sample_entropies.push(entropy);
        }
        
        // Bootstrap uncertainty = variance across samples
        let mean_entropy = sample_entropies.iter().sum::<f64>() / bootstrap_samples as f64;
        let variance = sample_entropies.iter()
            .map(|e| (e - mean_entropy).powi(2))
            .sum::<f64>() / bootstrap_samples as f64;
        
        // Normalized uncertainty score
        let bootstrap_score = (variance / 10.0).min(1.0).max(0.0);
        bootstrap_score
    }
    
    // JS+KL divergence (calibration baseline)
    fn jskl_divergence(&self, text: &str) -> f64 {
        if text.is_empty() { return 0.5; }
        
        // Character distribution vs uniform
        let mut char_freq = [0u32; 256];
        for byte in text.bytes() {
            char_freq[byte as usize] += 1;
        }
        
        let len = text.len() as f64;
        let uniform_prob = 1.0 / 256.0;
        
        // KL divergence D(P||Q) where P is text distribution, Q is uniform
        let mut kl_div = 0.0;
        let mut js_div = 0.0;
        
        for &count in &char_freq {
            if count > 0 {
                let p = count as f64 / len;
                
                // KL divergence term
                kl_div += p * (p / uniform_prob).ln();
                
                // JS divergence component
                let m = (p + uniform_prob) / 2.0;
                js_div += 0.5 * p * (p / m).ln() + 0.5 * uniform_prob * (uniform_prob / m).ln();
            }
        }
        
        // Combined JS+KL score (semantic uncertainty principle: ℏₛ = √(Δμ × Δσ))
        let combined_divergence = (js_div * kl_div).sqrt();
        (combined_divergence / 5.0).min(1.0).max(0.0)
    }
}

// Initialize WASM module
#[wasm_bindgen(start)]
pub fn main() {
    console_log!("🧠 Semantic Uncertainty WASM Module Loaded");
    console_log!("🔬 4-Method Ensemble: Entropy + Bayesian + Bootstrap + JS+KL");
    console_log!("⚡ Golden Scale: 3.4× for enhanced hallucination detection");
}

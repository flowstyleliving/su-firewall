// 🚀 Simplified WASM Implementation for Core Equation
// ℏₛ = √(Δμ × Δσ) - Pure mathematical implementation

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct SimpleAnalysisResult {
    pub hbar_s: f64,
    pub delta_mu: f64,
    pub delta_sigma: f64,
    pub risk_level: String,
    pub processing_time_ms: f64,
    pub request_id: String,
    pub timestamp: String,
}

#[wasm_bindgen]
pub struct SimpleWasmAnalyzer;

#[wasm_bindgen]
impl SimpleWasmAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SimpleWasmAnalyzer {
        SimpleWasmAnalyzer
    }
    
    #[wasm_bindgen]
    pub fn analyze(&self, prompt: &str, output: &str) -> Result<JsValue, JsValue> {
        let start_time = std::time::Instant::now();
        
        // Clean and normalize text
        let clean_prompt = prompt.to_lowercase().replace(|c: char| !c.is_alphanumeric() && !c.is_whitespace(), " ").trim().to_string();
        let clean_output = output.to_lowercase().replace(|c: char| !c.is_alphanumeric() && !c.is_whitespace(), " ").trim().to_string();
        
        if clean_prompt.is_empty() || clean_output.is_empty() {
            return Err(JsValue::from_str("Empty text after cleaning"));
        }
        
        // Calculate word frequencies
        let prompt_freq = Self::calculate_word_frequencies(&clean_prompt);
        let output_freq = Self::calculate_word_frequencies(&clean_output);
        
        // Calculate Δμ (precision) using entropy
        let delta_mu = Self::calculate_entropy(&prompt_freq) + Self::calculate_entropy(&output_freq) / 2.0;
        
        // Calculate Δσ (flexibility) using JSD
        let delta_sigma = Self::calculate_jsd(&prompt_freq, &output_freq) / 2.0;
        
        // Core equation: ℏₛ = √(Δμ × Δσ)
        let hbar_s = (delta_mu * delta_sigma).sqrt();
        
        // Determine risk level
        let risk_level = if hbar_s < 0.3 {
            "Critical"
        } else if hbar_s < 0.5 {
            "Warning"
        } else if hbar_s < 0.7 {
            "HighRisk"
        } else {
            "Safe"
        };
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        
        let result = SimpleAnalysisResult {
            hbar_s,
            delta_mu,
            delta_sigma,
            risk_level: risk_level.to_string(),
            processing_time_ms: processing_time,
            request_id: format!("wasm-{}", uuid::Uuid::new_v4()),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        
        let json = serde_json::to_string(&result)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(JsValue::from_str(&json))
    }
    
    fn calculate_word_frequencies(text: &str) -> HashMap<String, f64> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut freq = HashMap::new();
        
        for word in words {
            if !word.is_empty() {
                *freq.entry(word.to_string()).or_insert(0.0) += 1.0;
            }
        }
        
        // Normalize to probabilities
        let total: f64 = freq.values().sum();
        if total > 0.0 {
            for value in freq.values_mut() {
                *value /= total;
            }
        }
        
        freq
    }
    
    fn calculate_entropy(freq: &HashMap<String, f64>) -> f64 {
        let mut entropy = 0.0;
        
        for &prob in freq.values() {
            if prob > 0.0 {
                entropy -= prob * prob.log2();
            }
        }
        
        entropy.max(0.01) // Ensure minimum entropy
    }
    
    fn calculate_jsd(p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
        let mut jsd = 0.0;
        let all_words: std::collections::HashSet<&String> = p.keys().chain(q.keys()).collect();
        
        for word in all_words {
            let p_val = p.get(word).unwrap_or(&0.0);
            let q_val = q.get(word).unwrap_or(&0.0);
            let m = (p_val + q_val) / 2.0;
            
            if m > 0.0 {
                if *p_val > 0.0 {
                    jsd += p_val * (p_val / m).log2();
                }
                if *q_val > 0.0 {
                    jsd += q_val * (q_val / m).log2();
                }
            }
        }
        
        jsd.max(0.01) // Ensure minimum JSD
    }
} 
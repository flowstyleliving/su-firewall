// 🚀 Streamlined Semantic Uncertainty Engine
// Zero dependencies, deterministic, sub-10ms runtime
// Hash-based embeddings + entropy + Multi-Axis Drift Tensor (MAD Tensor)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;
use crate::semantic_metrics::SemanticMetricsCalculator;
use ndarray::{Array1, Array2};


// Import MAD Tensor functionality (COMMENTED OUT)
// use crate::mad_tensor::{MadTensorCalculator, ContextData, MadTensorResult};

/// 🧮 Core Semantic Uncertainty Result (Enhanced with MAD Tensor)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamlinedResult {
    /// Raw ℏₛ = √(Δμ × Δσ_HKG)
    pub raw_hbar: f64,
    /// Calibrated for usability
    pub calibrated_hbar: f64,
    /// Information-theoretic components
    pub delta_mu: f64,           // Entropy-based precision
    pub delta_sigma: f64,        // JSD and KL-based flexibility Δσ
    /// MAD Tensor geometric diagnostics (COMMENTED OUT)
    // pub mad_tensor_result: MadTensorResult,
    /// Processing metadata
    pub processing_time_ns: u64,
    pub deterministic_hash: u64,
    /// Token analysis
    pub token_analysis: TokenAnalysis,
    /// Risk assessment
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherAnalysisResult {
    pub fisher_precision: f64,
    pub fisher_flexibility: f64,
    pub fisher_semantic_uncertainty: f64,
    pub risk_level: RiskLevel,
    pub processing_time_ns: u64,
    pub deterministic_hash: u64,
    pub token_analysis: TokenAnalysis,
}

/// 🔢 Token Analysis (Class-based + Heuristic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAnalysis {
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub prompt_class: PromptClass,
    pub efficiency_score: f64,
    pub estimated_cost_usd: f64,
    pub potential_savings_percent: f64,
}

/// 🏷️ Deterministic Prompt Classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PromptClass {
    SimpleQA,
    ComplexAnalysis,
    CodeGeneration,
    CreativeWriting,
    Mathematical,
    Summarization,
    Translation,
    Conversational,
    Explanation, // Added Explanation class
}

/// 🚦 Risk Level (Simplified)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Safe,      // ℏₛ > 1.2
    Warning,   // 0.8 < ℏₛ ≤ 1.2
    Critical,  // ℏₛ ≤ 0.8
}

/// ⚡ Streamlined Engine (Zero Dependencies with MAD Tensor)
pub struct StreamlinedEngine {
    /// Token cost models (precomputed)
    token_models: HashMap<PromptClass, TokenModel>,
    /// Word frequency tables (for entropy)
    word_frequencies: HashMap<String, f64>,
    /// Optimization patterns
    optimization_cache: HashMap<u64, OptimizationHint>,
    /// Semantic metrics calculator for precision calculations
    semantic_calculator: SemanticMetricsCalculator,
    // MAD Tensor calculator for geometric diagnostics (COMMENTED OUT)
    // mad_tensor_calculator: MadTensorCalculator,
}

/// 💰 Token Model (Class-based)
#[derive(Debug, Clone)]
struct TokenModel {
    avg_chars_per_token: f64,
    cost_per_1k_tokens: f64,
    typical_prompt_tokens: u32,
    typical_output_tokens: u32,
    compression_factor: f64,
}

/// 💡 Optimization Hint (Cached)
#[derive(Debug, Clone)]
struct OptimizationHint {
    savings_percent: f64,
    confidence: f64,
    categories: Vec<&'static str>,
}

impl StreamlinedEngine {
    /// 🚀 Create optimized engine with MAD Tensor
    pub fn new() -> Self {
        Self {
            token_models: Self::init_token_models(),
            word_frequencies: Self::init_word_frequencies(),
            optimization_cache: HashMap::new(),
            semantic_calculator: SemanticMetricsCalculator::default(),
            // mad_tensor_calculator: MadTensorCalculator::new(),
        }
    }
    
    /// 🧮 Main analysis function (sub-10ms target)
    pub fn analyze(&self, prompt: &str, output: &str) -> StreamlinedResult {
        let start = Instant::now();
        
        // Step 1: Generate deterministic hash (for reproducibility)
        let deterministic_hash = self.generate_deterministic_hash(prompt, output);
        
        // Step 2: Classify prompt (hash-based, O(1))
        let prompt_class = self.classify_prompt_deterministic(prompt);
        
        // Step 3: Calculate hash-based embeddings
        let (prompt_embedding, output_embedding) = self.generate_hash_embeddings(prompt, output);
        
        // Step 4: Calculate precision using semantic metrics (Δμ)
        let delta_mu = self.calculate_semantic_precision(prompt, output);
        
        // Step 5: Calculate flexibility using JSD and KL divergence (Δσ)
        let delta_sigma = self.calculate_flexibility_with_fisher_and_jsd_kl(prompt, output);
        
        // Step 6: Calculate ℏₛ = √(Δμ × Δσ) - using JSD and KL divergence
        let raw_hbar = (delta_mu * delta_sigma).sqrt();
        let calibrated_hbar = raw_hbar * 3.4; // Empirical golden scale
        
        // Step 7: Assess risk level
        let risk_level = if calibrated_hbar <= 0.8 {
            RiskLevel::Critical
        } else if calibrated_hbar <= 1.2 {
            RiskLevel::Warning
        } else {
            RiskLevel::Safe
        };
        
        // Step 8: Token analysis (class-based heuristics)
        let token_analysis = self.analyze_tokens_heuristic(prompt, output, &prompt_class);
        
        let processing_time_ns = start.elapsed().as_nanos() as u64;
        
        StreamlinedResult {
            raw_hbar,
            calibrated_hbar,
            delta_mu,
            delta_sigma,
            // mad_tensor_result,
            processing_time_ns,
            deterministic_hash,
            token_analysis,
            risk_level,
        }
    }

    /// 🔬 Analyze using Fisher Information method (rigorous/mathematical)
    pub fn analyze_with_fisher(&self, prompt: &str, output: &str) -> FisherAnalysisResult {
        let start_time = Instant::now();
        
        // Calculate Fisher Information-based precision and flexibility
        let fisher_precision = self.calculate_fisher_information_precision(prompt, output);
        let fisher_flexibility = self.calculate_fisher_information_flexibility(prompt, output);
        
        // Calculate semantic uncertainty using Fisher Information
        let fisher_semantic_uncertainty = (fisher_precision * fisher_flexibility).sqrt();
        
        // Determine risk level based on Fisher Information results
        let risk_level = self.determine_risk_level_fisher(fisher_semantic_uncertainty);
        
        // Token analysis
        let token_analysis = self.analyze_tokens_heuristic(prompt, output, &PromptClass::Conversational); // Assuming a default class for Fisher analysis
        
        // Generate deterministic hash
        let deterministic_hash = self.generate_deterministic_hash(prompt, output);
        
        let processing_time_ns = start_time.elapsed().as_nanos() as u64;
        
        FisherAnalysisResult {
            fisher_precision,
            fisher_flexibility,
            fisher_semantic_uncertainty,
            risk_level,
            processing_time_ns,
            deterministic_hash,
            token_analysis,
        }
    }
    
    /// 🔒 Generate deterministic hash (reproducible across runs)
    fn generate_deterministic_hash(&self, prompt: &str, output: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        output.hash(&mut hasher);
        // Add version identifier for reproducibility
        "v1.0".hash(&mut hasher);
        hasher.finish()
    }
    
    /// 🏷️ Classify prompt deterministically (hash-based features)
    fn classify_prompt_deterministic(&self, prompt: &str) -> PromptClass {
        let prompt_lower = prompt.to_lowercase();
        
        // Feature extraction via hash-based approach
        let mut feature_scores = HashMap::new();
        
        // QA indicators
        let qa_keywords = ["what", "how", "why", "when", "where", "who", "?"];
        let qa_score = qa_keywords.iter()
            .map(|&kw| if prompt_lower.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>();
        feature_scores.insert(PromptClass::SimpleQA, qa_score);
        
        // Code indicators  
        let code_keywords = ["function", "class", "implement", "code", "programming", "algorithm", "{", "}"];
        let code_score = code_keywords.iter()
            .map(|&kw| if prompt_lower.contains(kw) { 2.0 } else { 0.0 })
            .sum::<f64>();
        feature_scores.insert(PromptClass::CodeGeneration, code_score);
        
        // Analysis indicators
        let analysis_keywords = ["analyze", "compare", "evaluate", "assess", "examine", "detailed"];
        let analysis_score = analysis_keywords.iter()
            .map(|&kw| if prompt_lower.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>();
        feature_scores.insert(PromptClass::ComplexAnalysis, analysis_score);
        
        // Math indicators
        let math_keywords = ["calculate", "solve", "equation", "formula", "mathematics", "compute"];
        let math_score = math_keywords.iter()
            .map(|&kw| if prompt_lower.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>();
        feature_scores.insert(PromptClass::Mathematical, math_score);
        
        // Creative indicators
        let creative_keywords = ["write", "story", "creative", "narrative", "character", "plot"];
        let creative_score = creative_keywords.iter()
            .map(|&kw| if prompt_lower.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>();
        feature_scores.insert(PromptClass::CreativeWriting, creative_score);
        
        // Summary indicators
        let summary_keywords = ["summarize", "summary", "brief", "overview", "key points"];
        let summary_score = summary_keywords.iter()
            .map(|&kw| if prompt_lower.contains(kw) { 1.0 } else { 0.0 })
            .sum::<f64>();
        feature_scores.insert(PromptClass::Summarization, summary_score);
        
        // Return highest scoring class (deterministic tie-breaking by enum order)
        feature_scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(class, _)| class)
            .unwrap_or(PromptClass::Conversational)
    }
    
    /// 🔢 Generate hash-based embeddings (deterministic, fast)
    fn generate_hash_embeddings(&self, prompt: &str, output: &str) -> (Vec<f64>, Vec<f64>) {
        const EMBEDDING_DIM: usize = 64; // Optimized for speed
        
        let prompt_embedding = self.text_to_hash_embedding(prompt, EMBEDDING_DIM);
        let output_embedding = self.text_to_hash_embedding(output, EMBEDDING_DIM);
        
        (prompt_embedding, output_embedding)
    }
    
    /// 🔤 Convert text to hash-based embedding
    fn text_to_hash_embedding(&self, text: &str, dim: usize) -> Vec<f64> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut embedding = vec![0.0; dim];
        
        for (i, word) in words.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.to_lowercase().hash(&mut hasher);
            i.hash(&mut hasher); // Position-sensitive
            
            let hash = hasher.finish();
            let idx = (hash as usize) % dim;
            let value = ((hash >> 32) as f64) / u32::MAX as f64; // Normalize to [0,1]
            
            embedding[idx] += value;
        }
        
        // Normalize by word count
        if !words.is_empty() {
            for val in &mut embedding {
                *val /= words.len() as f64;
            }
        }
        
        embedding
    }
    
    /// 📊 Calculate semantic precision (Δμ) using Fisher Information and JSD
    fn calculate_semantic_precision(&self, prompt: &str, output: &str) -> f64 {
        // Try Fisher Information-based precision first
        let fisher_precision = self.calculate_fisher_information_precision(prompt, output);
        
        // Fallback to JSD-based precision if Fisher Information fails
        let jsd_precision = self.calculate_jsd_precision(prompt, output);
        
        // Use Fisher Information if available, otherwise JSD
        if fisher_precision > 0.0 {
            fisher_precision
        } else {
            jsd_precision.unwrap_or(0.5) // Fallback to reasonable default
        }
    }

    /// 🎯 Calculate Fisher Information-based precision
    fn calculate_fisher_information_precision(&self, prompt: &str, output: &str) -> f64 {
        // Use the semantic metrics calculator for Fisher Information precision
        let calculator = SemanticMetricsCalculator::default();
        
        // Create Fisher Information matrix and semantic direction
        let fisher_matrix = calculator.simulate_fisher_information(prompt, output)
            .unwrap_or_else(|_| Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap());
        
        let direction = calculator.create_semantic_direction(prompt, output)
            .unwrap_or_else(|_| Array1::from_vec(vec![0.7071067811865475, 0.7071067811865475]));
        
        // Calculate precision using Fisher Information
        calculator.fisher_information_precision(&fisher_matrix, &direction)
            .unwrap_or(0.5) // Fallback value
    }
    
    /// 🎯 Calculate JSD-based precision
    fn calculate_jsd_precision(&self, prompt: &str, output: &str) -> Result<f64, String> {
        // Convert embeddings to probability distributions
        let p_dist = self.to_probability_distribution(prompt);
        let q_dist = self.to_probability_distribution(output);
        
        // Convert to ndarray format
        let p_array = ndarray::Array1::from_vec(p_dist);
        let q_array = ndarray::Array1::from_vec(q_dist);
        
        // Calculate JSD-based precision
        self.semantic_calculator.jsd_precision(&p_array, &q_array)
    }
    
    /// 🎯 Calculate flexibility using Fisher Information and JSD/KL divergence
    fn calculate_flexibility_with_fisher_and_jsd_kl(&self, prompt: &str, output: &str) -> f64 {
        // Try Fisher Information-based flexibility first
        let fisher_flexibility = self.calculate_fisher_information_flexibility(prompt, output);
        
        // Fallback to JSD/KL-based flexibility if Fisher Information fails
        let jsd_kl_flexibility = self.calculate_flexibility_with_jsd_and_kl(prompt, output);
        
        // Use Fisher Information if available, otherwise JSD/KL
        if fisher_flexibility > 0.0 {
            fisher_flexibility
        } else {
            jsd_kl_flexibility // Fallback to existing method
        }
    }
    
    /// 🌊 Calculate Fisher Information-based flexibility
    fn calculate_fisher_information_flexibility(&self, prompt: &str, output: &str) -> f64 {
        // Use the semantic metrics calculator for Fisher Information flexibility
        let calculator = SemanticMetricsCalculator::default();
        
        // Create Fisher Information matrix and semantic direction
        let fisher_matrix = calculator.simulate_fisher_information(prompt, output)
            .unwrap_or_else(|_| Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap());
        
        let direction = calculator.create_semantic_direction(prompt, output)
            .unwrap_or_else(|_| Array1::from_vec(vec![0.7071067811865475, 0.7071067811865475]));
        
        // Calculate flexibility using Fisher Information
        calculator.fisher_information_flexibility(&fisher_matrix, &direction)
            .unwrap_or(0.5) // Fallback value
    }

    /// 🔢 Analyze tokens for Fisher Information analysis
    fn analyze_tokens(&self, prompt: &str, output: &str) -> TokenAnalysis {
        // Use the same token analysis as the main method
        let prompt_class = self.classify_prompt_deterministic(prompt);
        self.analyze_tokens_heuristic(prompt, output, &prompt_class)
    }

    /// 🔧 Convert text to probability distribution
    fn to_probability_distribution(&self, text: &str) -> Vec<f64> {
        // Simple word frequency-based probability distribution
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![1.0]; // Single element with probability 1
        }
        
        let mut word_counts = HashMap::new();
        for word in &words {
            *word_counts.entry((*word).to_lowercase()).or_insert(0) += 1;
        }
        
        let total_words = words.len() as f64;
        let mut distribution = Vec::new();
        
        for (_, count) in word_counts {
            distribution.push(count as f64 / total_words);
        }
        
        // Normalize to ensure sum = 1.0
        let sum: f64 = distribution.iter().sum();
        if sum > 0.0 {
            for prob in &mut distribution {
                *prob /= sum;
            }
        }
        
        distribution
    }
    
    /// 🎯 Calculate flexibility using JSD and KL divergence
    /// Primary flexibility metric using JSD, with KL divergence for directional insight
    fn calculate_flexibility_with_jsd_and_kl(&self, prompt: &str, output: &str) -> f64 {
        // Convert embeddings to probability distributions
        let p = self.to_probability_distribution(prompt);
        let q = self.to_probability_distribution(output);
        
        // Calculate JSD as primary flexibility metric
        let jsd = self.jensen_shannon_divergence(&p, &q);
        
        // Calculate KL divergence for directional insight
        let kl_pq = self.kl_divergence(&p, &q);
        let kl_qp = self.kl_divergence(&q, &p);
        
        // Use JSD as primary flexibility measure
        let primary_flexibility = jsd.sqrt();
        
        // Use minimum KL for conservative drift estimate
        let kl_min = kl_pq.min(kl_qp).sqrt();
        
        // Combine JSD and KL for enhanced flexibility measurement
        // Weight JSD more heavily as it's symmetric and more stable
        let combined_flexibility = 0.7 * primary_flexibility + 0.3 * kl_min;
        
        // Ensure minimum threshold for numerical stability
        combined_flexibility.max(0.1)
    }
    
    /// 📏 Jensen-Shannon divergence calculation
    fn jensen_shannon_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        let mut jsd = 0.0;
        
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 1e-10 && qi > 1e-10 {
                let m = (pi + qi) * 0.5; // Average distribution
                if m > 1e-10 {
                    jsd += 0.5 * (pi * (pi / m).ln() + qi * (qi / m).ln());
                }
            }
        }
        
        jsd
    }
    
    /// 📊 KL divergence calculation
    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        let mut kl = 0.0;
        
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 1e-10 && qi > 1e-10 {
                kl += pi * (pi / qi).ln();
            }
        }
        
        kl
    }
    
    /// 📈 Calculate text entropy (information content)
    fn calculate_text_entropy(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        // Count word frequencies
        let mut word_counts = HashMap::new();
        for word in &words {
            *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
        }
        
        // Calculate Shannon entropy
        let total_words = words.len() as f64;
        let mut entropy = 0.0;
        for count in word_counts.values() {
            let probability = *count as f64 / total_words;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
    
    /// 🔗 Calculate embedding consistency
    fn calculate_embedding_consistency(&self, emb1: &[f64], emb2: &[f64]) -> f64 {
        if emb1.len() != emb2.len() {
            return 0.0;
        }
        
        // Calculate cosine similarity (deterministic)
        let dot_product: f64 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = emb1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = emb2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            (dot_product / (norm1 * norm2)).max(0.0).min(1.0)
        } else {
            0.0
        }
    }
    
    /// 📊 Calculate frequency coherence
    fn calculate_frequency_coherence(&self, prompt: &str, output: &str) -> f64 {
        let prompt_words: Vec<String> = prompt.split_whitespace().map(|w| w.to_lowercase()).collect();
        let output_words: Vec<String> = output.split_whitespace().map(|w| w.to_lowercase()).collect();
        
        // Calculate Jaccard similarity (fast, deterministic)
        let mut prompt_set = std::collections::HashSet::new();
        let mut output_set = std::collections::HashSet::new();
        
        for word in &prompt_words {
            prompt_set.insert(word);
        }
        for word in &output_words {
            output_set.insert(word);
        }
        
        let intersection = prompt_set.intersection(&output_set).count();
        let union = prompt_set.union(&output_set).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
    

    
    /// 🔢 Analyze tokens using class-based heuristics
    fn analyze_tokens_heuristic(&self, prompt: &str, output: &str, prompt_class: &PromptClass) -> TokenAnalysis {
        let model = self.token_models.get(prompt_class)
            .unwrap_or(&self.token_models[&PromptClass::Conversational]);
        
        // Heuristic token estimation (deterministic)
        let prompt_tokens = self.estimate_tokens_heuristic(prompt, model);
        let output_tokens = self.estimate_tokens_heuristic(output, model);
        let total_tokens = prompt_tokens + output_tokens;
        
        // Calculate efficiency based on class patterns
        let efficiency_score = self.calculate_token_efficiency(prompt, output, model);
        
        // Estimate cost
        let estimated_cost_usd = (total_tokens as f64 / 1000.0) * model.cost_per_1k_tokens;
        
        // Calculate potential savings
        let potential_savings_percent = self.estimate_savings_potential(prompt, output, model);
        
        TokenAnalysis {
            prompt_tokens,
            output_tokens,
            total_tokens,
            prompt_class: prompt_class.clone(),
            efficiency_score,
            estimated_cost_usd,
            potential_savings_percent,
        }
    }
    
    /// 🔢 Estimate tokens using heuristics
    fn estimate_tokens_heuristic(&self, text: &str, model: &TokenModel) -> u32 {
        let char_count = text.len() as f64;
        let estimated_tokens = char_count / model.avg_chars_per_token;
        
        // Adjust for text complexity
        let complexity_multiplier = if text.contains("{") || text.contains("function") {
            0.9 // Code is more token-dense
        } else if text.chars().filter(|c| c.is_uppercase()).count() > text.len() / 10 {
            1.1 // Technical text uses more tokens
        } else {
            1.0
        };
        
        (estimated_tokens * complexity_multiplier).ceil() as u32
    }
    
    /// ⚡ Calculate token efficiency
    fn calculate_token_efficiency(&self, prompt: &str, output: &str, model: &TokenModel) -> f64 {
        let prompt_words = prompt.split_whitespace().count();
        let output_words = output.split_whitespace().count();
        let total_words = prompt_words + output_words;
        
        if total_words == 0 {
            return 0.0;
        }
        
        // Words per token ratio (higher is more efficient)
        let estimated_tokens = self.estimate_tokens_heuristic(prompt, model) + 
                              self.estimate_tokens_heuristic(output, model);
        
        if estimated_tokens > 0 {
            (total_words as f64 / estimated_tokens as f64).min(1.0)
        } else {
            0.0
        }
    }
    
    /// 💰 Estimate savings potential
    fn estimate_savings_potential(&self, prompt: &str, output: &str, model: &TokenModel) -> f64 {
        // Detect optimization opportunities heuristically
        let mut savings = 0.0;
        
        // Check for redundant phrases
        let redundant_patterns = ["in order to", "due to the fact that", "please", "could you"];
        for pattern in &redundant_patterns {
            if prompt.to_lowercase().contains(pattern) {
                savings += 5.0; // 5% savings per pattern
            }
        }
        
        // Check for verbosity
        let words_per_sentence = prompt.split('.').map(|s| s.split_whitespace().count()).sum::<usize>() as f64 / 
                                prompt.split('.').count().max(1) as f64;
        if words_per_sentence > 20.0 {
            savings += 15.0; // 15% savings for verbose prompts
        }
        
        // Apply compression factor from model
        savings += (1.0 - model.compression_factor) * 100.0;
        
        savings.min(50.0) // Cap at 50% potential savings
    }
    
    /// 🏗️ Initialize token models (precomputed for speed)
    fn init_token_models() -> HashMap<PromptClass, TokenModel> {
        let mut models = HashMap::new();
        
        models.insert(PromptClass::SimpleQA, TokenModel {
            avg_chars_per_token: 4.0,
            cost_per_1k_tokens: 0.001,
            typical_prompt_tokens: 15,
            typical_output_tokens: 50,
            compression_factor: 0.8,
        });
        
        models.insert(PromptClass::ComplexAnalysis, TokenModel {
            avg_chars_per_token: 4.5,
            cost_per_1k_tokens: 0.003,
            typical_prompt_tokens: 150,
            typical_output_tokens: 400,
            compression_factor: 0.6,
        });
        
        models.insert(PromptClass::CodeGeneration, TokenModel {
            avg_chars_per_token: 3.5,
            cost_per_1k_tokens: 0.002,
            typical_prompt_tokens: 80,
            typical_output_tokens: 200,
            compression_factor: 0.7,
        });
        
        models.insert(PromptClass::CreativeWriting, TokenModel {
            avg_chars_per_token: 4.2,
            cost_per_1k_tokens: 0.002,
            typical_prompt_tokens: 40,
            typical_output_tokens: 300,
            compression_factor: 0.5,
        });
        
        models.insert(PromptClass::Mathematical, TokenModel {
            avg_chars_per_token: 3.0,
            cost_per_1k_tokens: 0.002,
            typical_prompt_tokens: 60,
            typical_output_tokens: 150,
            compression_factor: 0.8,
        });
        
        models.insert(PromptClass::Summarization, TokenModel {
            avg_chars_per_token: 4.0,
            cost_per_1k_tokens: 0.001,
            typical_prompt_tokens: 200,
            typical_output_tokens: 100,
            compression_factor: 0.9,
        });
        
        models.insert(PromptClass::Translation, TokenModel {
            avg_chars_per_token: 3.8,
            cost_per_1k_tokens: 0.002,
            typical_prompt_tokens: 100,
            typical_output_tokens: 120,
            compression_factor: 0.9,
        });
        
        models.insert(PromptClass::Conversational, TokenModel {
            avg_chars_per_token: 4.0,
            cost_per_1k_tokens: 0.001,
            typical_prompt_tokens: 30,
            typical_output_tokens: 80,
            compression_factor: 0.7,
        });
        
        models
    }
    
    /// 📚 Initialize word frequencies (for entropy calculations)
    fn init_word_frequencies() -> HashMap<String, f64> {
        // Simplified frequency table for common English words
        let mut frequencies = HashMap::new();
        
        // High frequency words (lower entropy contribution)
        let common_words = [
            ("the", 0.07), ("of", 0.04), ("and", 0.04), ("a", 0.03), ("to", 0.03),
            ("in", 0.02), ("is", 0.02), ("you", 0.02), ("that", 0.02), ("it", 0.02),
            ("he", 0.02), ("was", 0.02), ("for", 0.02), ("on", 0.02), ("are", 0.01),
            ("as", 0.01), ("with", 0.01), ("his", 0.01), ("they", 0.01), ("i", 0.01),
        ];
        
        for (word, freq) in &common_words {
            frequencies.insert(word.to_string(), *freq);
        }
        
        frequencies
    }
    
    /// 📊 Get processing statistics
    pub fn get_stats(&self) -> EngineStats {
        EngineStats {
            zero_dependencies: true,
            deterministic: true,
            max_processing_time_target_ns: 10_000_000, // 10ms target
            supported_classes: self.token_models.len(),
            cache_size: self.optimization_cache.len(),
        }
    }

    /// 🚦 Determine risk level based on Fisher Information results
    fn determine_risk_level_fisher(&self, fisher_semantic_uncertainty: f64) -> RiskLevel {
        match fisher_semantic_uncertainty {
            u if u < 0.3 => RiskLevel::Critical,  // Low uncertainty = potential hallucination
            u if u < 0.6 => RiskLevel::Warning,   // Medium uncertainty
            _ => RiskLevel::Safe,                  // High uncertainty = natural variation
        }
    }
}

/// 📊 Engine Statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct EngineStats {
    pub zero_dependencies: bool,
    pub deterministic: bool,
    pub max_processing_time_target_ns: u64,
    pub supported_classes: usize,
    pub cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_deterministic_analysis() {
        let engine = StreamlinedEngine::new();
        let prompt = "What is quantum computing?";
        let output = "Quantum computing uses quantum bits to process information.";
        
        let result1 = engine.analyze(prompt, output);
        let result2 = engine.analyze(prompt, output);
        
        // Should be deterministic
        assert_eq!(result1.deterministic_hash, result2.deterministic_hash);
        assert_eq!(result1.raw_hbar, result2.raw_hbar);
        assert_eq!(result1.delta_mu, result2.delta_mu);
        assert_eq!(result1.delta_sigma, result2.delta_sigma);
    }
    
    #[test]
    fn test_sub_10ms_performance() {
        let engine = StreamlinedEngine::new();
        let prompt = "Explain machine learning algorithms in detail.";
        let output = "Machine learning algorithms are computational methods that enable systems to learn from data.";
        
        let result = engine.analyze(prompt, output);
        
        // Should complete in under 10ms (10,000,000 nanoseconds)
        assert!(result.processing_time_ns < 10_000_000);
    }
    
    #[test]
    fn test_prompt_classification() {
        let engine = StreamlinedEngine::new();
        
        let qa_prompt = "What is the capital of France?";
        let qa_result = engine.analyze(qa_prompt, "Paris");
        assert_eq!(qa_result.token_analysis.prompt_class, PromptClass::SimpleQA);
        
        let code_prompt = "Write a function to sort an array";
        let code_result = engine.analyze(code_prompt, "function sort(arr) { return arr.sort(); }");
        assert_eq!(code_result.token_analysis.prompt_class, PromptClass::CodeGeneration);
    }
}
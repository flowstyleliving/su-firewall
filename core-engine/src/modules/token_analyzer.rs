// 🔤 Token Analyzer - Hybrid Prompt Classification + Heuristic Estimation
// Advanced token efficiency analysis for cost optimization

use crate::SemanticUncertaintyResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

/// 🏷️ Prompt Classification Categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PromptClass {
    /// 📝 Simple question-answer
    SimpleQA,
    /// 📊 Complex analysis request
    ComplexAnalysis,
    /// 🔧 Code generation/modification
    CodeGeneration,
    /// 📚 Creative writing
    CreativeWriting,
    /// 🧮 Mathematical/computational
    Mathematical,
    /// 📋 Summarization task
    Summarization,
    /// 🌍 Translation task
    Translation,
    /// 🎯 Specific information extraction
    InformationExtraction,
    /// 🗣️ Conversational
    Conversational,
    /// 📖 Educational/explanatory
    Educational,
}

/// 📊 Token Efficiency Analysis Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEfficiency {
    /// 🏷️ Classified prompt type
    pub prompt_class: PromptClass,
    
    /// 🔢 Token usage analysis
    pub token_usage: TokenUsageAnalysis,
    
    /// ⚡ Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    
    /// 🎯 Optimization opportunities
    pub optimization_opportunities: Vec<TokenOptimization>,
    
    /// 📈 Comparative analysis
    pub benchmark_comparison: BenchmarkComparison,
}

/// 🔢 Token Usage Breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageAnalysis {
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub effective_tokens: u32, // Tokens contributing to semantic value
    pub redundant_tokens: u32, // Tokens that could be optimized
    pub compression_ratio: f64, // How much the prompt could be compressed
}

/// ⚡ Efficiency Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// 📊 Semantic density (semantic value per token)
    pub semantic_density: f64,
    
    /// 🎯 Information density (unique information per token)
    pub information_density: f64,
    
    /// 🔄 Redundancy ratio (redundant tokens / total tokens)
    pub redundancy_ratio: f64,
    
    /// ⚡ Efficiency score (0-1, higher is better)
    pub efficiency_score: f64,
    
    /// 💰 Cost effectiveness (semantic value per cost unit)
    pub cost_effectiveness: f64,
}

/// 🎯 Token Optimization Opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOptimization {
    pub optimization_type: OptimizationType,
    pub description: String,
    pub estimated_token_savings: u32,
    pub estimated_savings_percent: f64,
    pub implementation_complexity: ComplexityLevel,
    pub semantic_impact: SemanticImpact,
    pub example_before: Option<String>,
    pub example_after: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OptimizationType {
    PromptCompression,
    RedundancyRemoval,
    StructuralOptimization,
    LanguageSimplification,
    ContextReduction,
    FormatOptimization,
    InstructionConsolidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Trivial,   // Can be automated
    Simple,    // Straightforward changes
    Moderate,  // Requires some analysis
    Complex,   // Significant restructuring needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticImpact {
    None,       // No impact on meaning
    Minimal,    // Slight impact, acceptable
    Moderate,   // Noticeable impact, review needed
    Significant, // Major impact, careful consideration
}

/// 📈 Benchmark Comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub class_average_tokens: u32,
    pub percentile_ranking: f64, // 0-100, where this prompt ranks in efficiency
    pub efficiency_vs_average: f64, // Ratio compared to class average
    pub recommendations: Vec<String>,
}

/// 🔤 Token Analyzer Implementation
pub struct TokenAnalyzer {
    /// 📊 Class-specific token usage patterns
    class_patterns: HashMap<PromptClass, ClassPattern>,
    
    /// 🧮 Heuristic rules for token estimation
    heuristic_rules: HeuristicRules,
    
    /// 📈 Historical performance data
    performance_benchmarks: HashMap<PromptClass, PerformanceBenchmark>,
}

/// 🏷️ Class-specific patterns
#[derive(Debug, Clone)]
struct ClassPattern {
    average_prompt_tokens: f64,
    average_output_tokens: f64,
    typical_compression_ratio: f64,
    common_optimization_types: Vec<OptimizationType>,
}

/// 🧮 Heuristic Rules for Token Estimation
#[derive(Debug, Clone)]
struct HeuristicRules {
    /// Characters per token estimation by text type
    chars_per_token: HashMap<String, f64>,
    
    /// Word complexity multipliers
    complexity_multipliers: HashMap<String, f64>,
    
    /// Language-specific factors
    language_factors: HashMap<String, f64>,
}

/// 📈 Performance Benchmark Data
#[derive(Debug, Clone)]
struct PerformanceBenchmark {
    samples: u32,
    avg_efficiency: f64,
    percentile_distribution: Vec<(f64, f64)>, // (percentile, efficiency_score)
    common_optimizations: Vec<(OptimizationType, f64)>, // (type, frequency)
}

impl TokenAnalyzer {
    /// 🚀 Create new token analyzer
    pub fn new() -> Result<Self> {
        let mut class_patterns = HashMap::new();
        
        // Initialize class patterns based on empirical data
        class_patterns.insert(PromptClass::SimpleQA, ClassPattern {
            average_prompt_tokens: 15.0,
            average_output_tokens: 50.0,
            typical_compression_ratio: 0.8,
            common_optimization_types: vec![OptimizationType::PromptCompression, OptimizationType::RedundancyRemoval],
        });
        
        class_patterns.insert(PromptClass::ComplexAnalysis, ClassPattern {
            average_prompt_tokens: 150.0,
            average_output_tokens: 400.0,
            typical_compression_ratio: 0.6,
            common_optimization_types: vec![OptimizationType::StructuralOptimization, OptimizationType::ContextReduction],
        });
        
        class_patterns.insert(PromptClass::CodeGeneration, ClassPattern {
            average_prompt_tokens: 80.0,
            average_output_tokens: 200.0,
            typical_compression_ratio: 0.7,
            common_optimization_types: vec![OptimizationType::InstructionConsolidation, OptimizationType::FormatOptimization],
        });
        
        // Initialize heuristic rules
        let mut chars_per_token = HashMap::new();
        chars_per_token.insert("english".to_string(), 4.0);
        chars_per_token.insert("code".to_string(), 3.5);
        chars_per_token.insert("technical".to_string(), 4.5);
        chars_per_token.insert("mathematical".to_string(), 3.0);
        
        let mut complexity_multipliers = HashMap::new();
        complexity_multipliers.insert("simple_words".to_string(), 0.8);
        complexity_multipliers.insert("technical_terms".to_string(), 1.2);
        complexity_multipliers.insert("compound_words".to_string(), 1.1);
        complexity_multipliers.insert("abbreviations".to_string(), 0.7);
        
        let heuristic_rules = HeuristicRules {
            chars_per_token,
            complexity_multipliers,
            language_factors: HashMap::new(),
        };
        
        // Initialize performance benchmarks
        let performance_benchmarks = Self::initialize_benchmarks();
        
        Ok(Self {
            class_patterns,
            heuristic_rules,
            performance_benchmarks,
        })
    }
    
    /// 📊 Analyze token efficiency for a prompt-output pair
    #[instrument(skip(self, prompt, output, semantic_result))]
    pub async fn analyze_efficiency(
        &self,
        prompt: &str,
        output: &str,
        semantic_result: &SemanticUncertaintyResult,
    ) -> Result<TokenEfficiency> {
        debug!("Analyzing token efficiency for {} char prompt, {} char output", prompt.len(), output.len());
        
        // 🏷️ Classify the prompt
        let prompt_class = self.classify_prompt(prompt).await?;
        
        // 🔢 Analyze token usage
        let token_usage = self.analyze_token_usage(prompt, output, &prompt_class).await?;
        
        // ⚡ Calculate efficiency metrics
        let efficiency_metrics = self.calculate_efficiency_metrics(&token_usage, semantic_result).await?;
        
        // 🎯 Identify optimization opportunities
        let optimization_opportunities = self.identify_optimizations(prompt, output, &prompt_class, &token_usage).await?;
        
        // 📈 Compare against benchmarks
        let benchmark_comparison = self.compare_against_benchmarks(&prompt_class, &efficiency_metrics).await?;
        
        Ok(TokenEfficiency {
            prompt_class,
            token_usage,
            efficiency_metrics,
            optimization_opportunities,
            benchmark_comparison,
        })
    }
    
    /// 🏷️ Classify prompt using hybrid approach
    async fn classify_prompt(&self, prompt: &str) -> Result<PromptClass> {
        let prompt_lower = prompt.to_lowercase();
        
        // Rule-based classification with scoring
        let mut class_scores = HashMap::new();
        
        // SimpleQA indicators
        let qa_indicators = ["what", "how", "why", "when", "where", "who", "?"];
        let qa_score = qa_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / qa_indicators.len() as f64;
        class_scores.insert(PromptClass::SimpleQA, qa_score);
        
        // Code generation indicators
        let code_indicators = ["write code", "function", "class", "implement", "programming", "algorithm"];
        let code_score = code_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / code_indicators.len() as f64;
        class_scores.insert(PromptClass::CodeGeneration, code_score);
        
        // Complex analysis indicators
        let analysis_indicators = ["analyze", "compare", "evaluate", "assess", "examine", "detailed"];
        let analysis_score = analysis_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / analysis_indicators.len() as f64;
        class_scores.insert(PromptClass::ComplexAnalysis, analysis_score);
        
        // Creative writing indicators
        let creative_indicators = ["write", "story", "creative", "narrative", "character", "plot"];
        let creative_score = creative_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / creative_indicators.len() as f64;
        class_scores.insert(PromptClass::CreativeWriting, creative_score);
        
        // Mathematical indicators
        let math_indicators = ["calculate", "solve", "equation", "formula", "mathematics", "compute"];
        let math_score = math_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / math_indicators.len() as f64;
        class_scores.insert(PromptClass::Mathematical, math_score);
        
        // Summarization indicators
        let summary_indicators = ["summarize", "summary", "brief", "overview", "key points"];
        let summary_score = summary_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / summary_indicators.len() as f64;
        class_scores.insert(PromptClass::Summarization, summary_score);
        
        // Educational indicators
        let edu_indicators = ["explain", "teach", "learn", "understand", "concept", "definition"];
        let edu_score = edu_indicators.iter()
            .map(|&indicator| if prompt_lower.contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / edu_indicators.len() as f64;
        class_scores.insert(PromptClass::Educational, edu_score);
        
        // Find the highest scoring class
        let best_class = class_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(class, _)| class.clone())
            .unwrap_or(PromptClass::Conversational);
        
        Ok(best_class)
    }
    
    /// 🔢 Analyze token usage with hybrid estimation
    async fn analyze_token_usage(&self, prompt: &str, output: &str, prompt_class: &PromptClass) -> Result<TokenUsageAnalysis> {
        // Heuristic token estimation
        let prompt_tokens = self.estimate_tokens(prompt, "prompt").await?;
        let output_tokens = self.estimate_tokens(output, "output").await?;
        let total_tokens = prompt_tokens + output_tokens;
        
        // Calculate effective vs redundant tokens
        let effective_tokens = self.calculate_effective_tokens(prompt, output).await?;
        let redundant_tokens = total_tokens.saturating_sub(effective_tokens);
        
        // Calculate compression ratio based on class patterns
        let compression_ratio = if let Some(pattern) = self.class_patterns.get(prompt_class) {
            pattern.typical_compression_ratio
        } else {
            0.7 // Default compression ratio
        };
        
        Ok(TokenUsageAnalysis {
            prompt_tokens,
            output_tokens,
            total_tokens,
            effective_tokens,
            redundant_tokens,
            compression_ratio,
        })
    }
    
    /// 🧮 Heuristic token estimation
    async fn estimate_tokens(&self, text: &str, text_type: &str) -> Result<u32> {
        let base_chars_per_token = self.heuristic_rules.chars_per_token
            .get("english")
            .unwrap_or(&4.0);
        
        // Apply complexity multipliers
        let mut multiplier = 1.0;
        
        // Check for technical terms
        if text.to_lowercase().contains("technical") || 
           text.chars().filter(|c| c.is_uppercase()).count() > text.len() / 10 {
            multiplier *= self.heuristic_rules.complexity_multipliers
                .get("technical_terms")
                .unwrap_or(&1.2);
        }
        
        // Check for code-like content
        if text.contains("{") || text.contains("function") || text.contains("class") {
            multiplier *= 0.9; // Code is typically more token-dense
        }
        
        let adjusted_chars_per_token = base_chars_per_token * multiplier;
        let estimated_tokens = (text.len() as f64 / adjusted_chars_per_token).ceil() as u32;
        
        Ok(estimated_tokens)
    }
    
    /// ⚡ Calculate effective tokens (tokens contributing semantic value)
    async fn calculate_effective_tokens(&self, prompt: &str, output: &str) -> Result<u32> {
        // Remove common stop words and filler
        let stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"];
        let filler_phrases = ["you know", "i think", "kind of", "sort of", "like", "um", "uh"];
        
        let combined_text = format!("{} {}", prompt, output);
        let words: Vec<&str> = combined_text.split_whitespace().collect();
        
        let effective_words = words.iter()
            .filter(|word| !stop_words.contains(&word.to_lowercase().as_str()))
            .filter(|word| !filler_phrases.iter().any(|phrase| word.to_lowercase().contains(phrase)))
            .count();
        
        // Estimate tokens from effective words (roughly 0.8 tokens per word on average)
        Ok((effective_words as f64 * 0.8).ceil() as u32)
    }
    
    /// ⚡ Calculate efficiency metrics
    async fn calculate_efficiency_metrics(
        &self,
        token_usage: &TokenUsageAnalysis,
        semantic_result: &SemanticUncertaintyResult,
    ) -> Result<EfficiencyMetrics> {
        // Semantic density: semantic uncertainty quality per token
        let semantic_density = if token_usage.total_tokens > 0 {
            semantic_result.calibrated_hbar / token_usage.total_tokens as f64
        } else {
            0.0
        };
        
        // Information density: effective tokens / total tokens
        let information_density = if token_usage.total_tokens > 0 {
            token_usage.effective_tokens as f64 / token_usage.total_tokens as f64
        } else {
            0.0
        };
        
        // Redundancy ratio
        let redundancy_ratio = if token_usage.total_tokens > 0 {
            token_usage.redundant_tokens as f64 / token_usage.total_tokens as f64
        } else {
            0.0
        };
        
        // Overall efficiency score (0-1, higher is better)
        let efficiency_score = (information_density * 0.4 + 
                               semantic_density * 0.3 + 
                               (1.0 - redundancy_ratio) * 0.3).max(0.0).min(1.0);
        
        // Cost effectiveness (placeholder - would need actual cost data)
        let cost_effectiveness = semantic_density * 1000.0; // Normalize for display
        
        Ok(EfficiencyMetrics {
            semantic_density,
            information_density,
            redundancy_ratio,
            efficiency_score,
            cost_effectiveness,
        })
    }
    
    /// 🎯 Identify optimization opportunities
    async fn identify_optimizations(
        &self,
        prompt: &str,
        output: &str,
        prompt_class: &PromptClass,
        token_usage: &TokenUsageAnalysis,
    ) -> Result<Vec<TokenOptimization>> {
        let mut optimizations = Vec::new();
        
        // Prompt compression opportunity
        if token_usage.prompt_tokens > 50 && token_usage.compression_ratio < 0.8 {
            optimizations.push(TokenOptimization {
                optimization_type: OptimizationType::PromptCompression,
                description: "Prompt can be compressed by removing redundant phrases and simplifying language".to_string(),
                estimated_token_savings: (token_usage.prompt_tokens as f64 * 0.25) as u32,
                estimated_savings_percent: 25.0,
                implementation_complexity: ComplexityLevel::Simple,
                semantic_impact: SemanticImpact::Minimal,
                example_before: Some("Could you please explain in detail what quantum computing is and how it works?".to_string()),
                example_after: Some("Explain quantum computing and how it works.".to_string()),
            });
        }
        
        // Redundancy removal
        if token_usage.redundant_tokens > 10 {
            optimizations.push(TokenOptimization {
                optimization_type: OptimizationType::RedundancyRemoval,
                description: format!("Remove {} redundant tokens that don't contribute semantic value", token_usage.redundant_tokens),
                estimated_token_savings: token_usage.redundant_tokens,
                estimated_savings_percent: (token_usage.redundant_tokens as f64 / token_usage.total_tokens as f64 * 100.0),
                implementation_complexity: ComplexityLevel::Trivial,
                semantic_impact: SemanticImpact::None,
                example_before: None,
                example_after: None,
            });
        }
        
        // Class-specific optimizations
        if let Some(pattern) = self.class_patterns.get(prompt_class) {
            for opt_type in &pattern.common_optimization_types {
                match opt_type {
                    OptimizationType::StructuralOptimization => {
                        optimizations.push(TokenOptimization {
                            optimization_type: opt_type.clone(),
                            description: "Restructure prompt for better token efficiency".to_string(),
                            estimated_token_savings: (token_usage.total_tokens as f64 * 0.15) as u32,
                            estimated_savings_percent: 15.0,
                            implementation_complexity: ComplexityLevel::Moderate,
                            semantic_impact: SemanticImpact::Minimal,
                            example_before: None,
                            example_after: None,
                        });
                    },
                    OptimizationType::InstructionConsolidation => {
                        optimizations.push(TokenOptimization {
                            optimization_type: opt_type.clone(),
                            description: "Consolidate multiple instructions into single, clear directive".to_string(),
                            estimated_token_savings: (token_usage.prompt_tokens as f64 * 0.20) as u32,
                            estimated_savings_percent: 20.0,
                            implementation_complexity: ComplexityLevel::Simple,
                            semantic_impact: SemanticImpact::None,
                            example_before: None,
                            example_after: None,
                        });
                    },
                    _ => {} // Handle other optimization types as needed
                }
            }
        }
        
        Ok(optimizations)
    }
    
    /// 📈 Compare against benchmarks
    async fn compare_against_benchmarks(
        &self,
        prompt_class: &PromptClass,
        efficiency_metrics: &EfficiencyMetrics,
    ) -> Result<BenchmarkComparison> {
        let benchmark = self.performance_benchmarks.get(prompt_class);
        
        let (class_average_tokens, percentile_ranking, efficiency_vs_average) = if let Some(bench) = benchmark {
            let avg_tokens = (bench.avg_efficiency * 100.0) as u32; // Simplified calculation
            let percentile = self.calculate_percentile_ranking(efficiency_metrics.efficiency_score, &bench.percentile_distribution);
            let vs_avg = efficiency_metrics.efficiency_score / bench.avg_efficiency;
            (avg_tokens, percentile, vs_avg)
        } else {
            (100, 50.0, 1.0) // Default values
        };
        
        let recommendations = self.generate_benchmark_recommendations(
            efficiency_metrics.efficiency_score,
            percentile_ranking,
        );
        
        Ok(BenchmarkComparison {
            class_average_tokens,
            percentile_ranking,
            efficiency_vs_average,
            recommendations,
        })
    }
    
    /// 📊 Calculate percentile ranking
    fn calculate_percentile_ranking(&self, score: f64, distribution: &[(f64, f64)]) -> f64 {
        for (percentile, threshold) in distribution {
            if score >= *threshold {
                return *percentile;
            }
        }
        0.0 // Bottom percentile
    }
    
    /// 💡 Generate benchmark-based recommendations
    fn generate_benchmark_recommendations(&self, efficiency_score: f64, percentile: f64) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if percentile < 25.0 {
            recommendations.push("Consider significant optimization - performance below 25th percentile".to_string());
            recommendations.push("Focus on redundancy removal and prompt compression".to_string());
        } else if percentile < 50.0 {
            recommendations.push("Moderate optimization recommended - below median performance".to_string());
            recommendations.push("Look for structural improvements".to_string());
        } else if percentile < 75.0 {
            recommendations.push("Good efficiency - minor optimizations possible".to_string());
        } else {
            recommendations.push("Excellent efficiency - top quartile performance".to_string());
        }
        
        if efficiency_score < 0.3 {
            recommendations.push("Critical: Very low efficiency detected".to_string());
        } else if efficiency_score < 0.6 {
            recommendations.push("Moderate efficiency - optimization beneficial".to_string());
        } else {
            recommendations.push("High efficiency achieved".to_string());
        }
        
        recommendations
    }
    
    /// 📈 Initialize performance benchmarks
    fn initialize_benchmarks() -> HashMap<PromptClass, PerformanceBenchmark> {
        let mut benchmarks = HashMap::new();
        
        benchmarks.insert(PromptClass::SimpleQA, PerformanceBenchmark {
            samples: 1000,
            avg_efficiency: 0.75,
            percentile_distribution: vec![
                (90.0, 0.9),
                (75.0, 0.8),
                (50.0, 0.75),
                (25.0, 0.65),
                (10.0, 0.5),
            ],
            common_optimizations: vec![
                (OptimizationType::PromptCompression, 0.8),
                (OptimizationType::RedundancyRemoval, 0.6),
            ],
        });
        
        benchmarks.insert(PromptClass::ComplexAnalysis, PerformanceBenchmark {
            samples: 500,
            avg_efficiency: 0.65,
            percentile_distribution: vec![
                (90.0, 0.85),
                (75.0, 0.75),
                (50.0, 0.65),
                (25.0, 0.55),
                (10.0, 0.4),
            ],
            common_optimizations: vec![
                (OptimizationType::StructuralOptimization, 0.9),
                (OptimizationType::ContextReduction, 0.7),
            ],
        });
        
        benchmarks
    }
    
    /// 📊 Get average efficiency for the analyzer
    pub fn get_average_efficiency(&self) -> f64 {
        // This would be calculated from actual usage data
        0.72 // Placeholder
    }
}
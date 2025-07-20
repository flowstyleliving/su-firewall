// 🎯 Per-Prompt ℏₛ Scoring Engine
// Advanced semantic analysis for prompt-output pairs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

/// 📊 Comprehensive Prompt Scoring Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringResult {
    /// 📐 Specificity score (0-1) - How specific/precise is the content
    pub specificity_score: f64,
    
    /// 🌀 Ambiguity score (0-1) - How ambiguous/unclear is the content
    pub ambiguity_score: f64,
    
    /// 🔗 Coherence score (0-1) - How well the output matches the prompt
    pub coherence_score: f64,
    
    /// 🧮 Complexity score (0-1) - Semantic complexity of the content
    pub complexity_score: f64,
    
    /// 📊 Semantic variance - Measure of semantic diversity
    pub semantic_variance: f64,
    
    /// ⚠️ Uncertainty indicators - Detected uncertainty markers
    pub uncertainty_indicators: f64,
    
    /// 🔍 Detailed analysis
    pub detailed_analysis: DetailedAnalysis,
}

/// 🔍 Detailed Semantic Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedAnalysis {
    /// 🎯 Prompt characteristics
    pub prompt_characteristics: PromptCharacteristics,
    
    /// 📝 Output characteristics
    pub output_characteristics: OutputCharacteristics,
    
    /// 🔗 Prompt-output relationship
    pub relationship_analysis: RelationshipAnalysis,
    
    /// 🚨 Risk indicators
    pub risk_indicators: Vec<RiskIndicator>,
}

/// 🎯 Prompt Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCharacteristics {
    pub word_count: usize,
    pub sentence_count: usize,
    pub avg_word_length: f64,
    pub question_count: usize,
    pub imperative_count: usize,
    pub specificity_keywords: Vec<String>,
    pub ambiguity_keywords: Vec<String>,
    pub domain_indicators: Vec<String>,
}

/// 📝 Output Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputCharacteristics {
    pub word_count: usize,
    pub sentence_count: usize,
    pub avg_sentence_length: f64,
    pub uncertainty_phrases: Vec<String>,
    pub confidence_indicators: Vec<String>,
    pub factual_claims: usize,
    pub hedging_language: usize,
}

/// 🔗 Relationship Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipAnalysis {
    pub relevance_score: f64,
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub appropriateness_score: f64,
    pub semantic_alignment: f64,
}

/// 🚨 Risk Indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskIndicator {
    pub indicator_type: RiskIndicatorType,
    pub description: String,
    pub severity: f64, // 0-1
    pub confidence: f64, // 0-1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskIndicatorType {
    HighUncertainty,
    LowSpecificity,
    SemanticMismatch,
    OverConfidence,
    UnderConfidence,
    FactualInconsistency,
    VagueLanguage,
    HallucinationRisk,
}

/// 🎯 Prompt Scorer Implementation
pub struct PromptScorer {
    uncertainty_keywords: HashMap<String, f64>,
    confidence_keywords: HashMap<String, f64>,
    specificity_keywords: HashMap<String, f64>,
    ambiguity_keywords: HashMap<String, f64>,
}

impl PromptScorer {
    /// 🚀 Create new prompt scorer
    pub fn new() -> Result<Self> {
        let mut uncertainty_keywords = HashMap::new();
        uncertainty_keywords.insert("maybe".to_string(), 0.8);
        uncertainty_keywords.insert("perhaps".to_string(), 0.7);
        uncertainty_keywords.insert("might".to_string(), 0.6);
        uncertainty_keywords.insert("could".to_string(), 0.5);
        uncertainty_keywords.insert("possibly".to_string(), 0.7);
        uncertainty_keywords.insert("probably".to_string(), 0.4);
        uncertainty_keywords.insert("likely".to_string(), 0.3);
        uncertainty_keywords.insert("seems".to_string(), 0.6);
        uncertainty_keywords.insert("appears".to_string(), 0.5);
        uncertainty_keywords.insert("suggests".to_string(), 0.4);
        uncertainty_keywords.insert("indicates".to_string(), 0.3);
        uncertainty_keywords.insert("i think".to_string(), 0.8);
        uncertainty_keywords.insert("i believe".to_string(), 0.7);
        uncertainty_keywords.insert("in my opinion".to_string(), 0.6);
        
        let mut confidence_keywords = HashMap::new();
        confidence_keywords.insert("definitely".to_string(), 0.9);
        confidence_keywords.insert("certainly".to_string(), 0.9);
        confidence_keywords.insert("absolutely".to_string(), 0.95);
        confidence_keywords.insert("clearly".to_string(), 0.8);
        confidence_keywords.insert("obviously".to_string(), 0.85);
        confidence_keywords.insert("undoubtedly".to_string(), 0.9);
        confidence_keywords.insert("without a doubt".to_string(), 0.95);
        confidence_keywords.insert("guarantee".to_string(), 0.9);
        confidence_keywords.insert("ensure".to_string(), 0.8);
        confidence_keywords.insert("always".to_string(), 0.85);
        confidence_keywords.insert("never".to_string(), 0.85);
        confidence_keywords.insert("will".to_string(), 0.7);
        confidence_keywords.insert("must".to_string(), 0.8);
        
        let mut specificity_keywords = HashMap::new();
        specificity_keywords.insert("specifically".to_string(), 0.9);
        specificity_keywords.insert("precisely".to_string(), 0.9);
        specificity_keywords.insert("exactly".to_string(), 0.85);
        specificity_keywords.insert("particular".to_string(), 0.7);
        specificity_keywords.insert("detailed".to_string(), 0.8);
        specificity_keywords.insert("comprehensive".to_string(), 0.8);
        specificity_keywords.insert("thorough".to_string(), 0.7);
        specificity_keywords.insert("step-by-step".to_string(), 0.9);
        specificity_keywords.insert("itemized".to_string(), 0.8);
        specificity_keywords.insert("concrete".to_string(), 0.8);
        
        let mut ambiguity_keywords = HashMap::new();
        ambiguity_keywords.insert("something".to_string(), 0.8);
        ambiguity_keywords.insert("somewhat".to_string(), 0.7);
        ambiguity_keywords.insert("kind of".to_string(), 0.8);
        ambiguity_keywords.insert("sort of".to_string(), 0.8);
        ambiguity_keywords.insert("various".to_string(), 0.6);
        ambiguity_keywords.insert("several".to_string(), 0.5);
        ambiguity_keywords.insert("multiple".to_string(), 0.4);
        ambiguity_keywords.insert("general".to_string(), 0.7);
        ambiguity_keywords.insert("broadly".to_string(), 0.6);
        ambiguity_keywords.insert("roughly".to_string(), 0.7);
        ambiguity_keywords.insert("approximately".to_string(), 0.5);
        ambiguity_keywords.insert("around".to_string(), 0.4);
        ambiguity_keywords.insert("about".to_string(), 0.3);
        ambiguity_keywords.insert("stuff".to_string(), 0.9);
        ambiguity_keywords.insert("things".to_string(), 0.7);
        ambiguity_keywords.insert("etc".to_string(), 0.6);
        
        Ok(Self {
            uncertainty_keywords,
            confidence_keywords,
            specificity_keywords,
            ambiguity_keywords,
        })
    }
    
    /// 🎯 Score a prompt-output pair
    #[instrument(skip(self, prompt, output))]
    pub async fn score(&self, prompt: &str, output: &str) -> Result<ScoringResult> {
        debug!("Scoring prompt-output pair: {} chars prompt, {} chars output", prompt.len(), output.len());
        
        // Analyze prompt characteristics
        let prompt_chars = self.analyze_prompt_characteristics(prompt);
        
        // Analyze output characteristics
        let output_chars = self.analyze_output_characteristics(output);
        
        // Analyze relationship between prompt and output
        let relationship = self.analyze_relationship(prompt, output);
        
        // Detect risk indicators
        let risk_indicators = self.detect_risk_indicators(prompt, output, &prompt_chars, &output_chars);
        
        // Calculate core scores
        let specificity_score = self.calculate_specificity_score(prompt, output);
        let ambiguity_score = self.calculate_ambiguity_score(prompt, output);
        let coherence_score = relationship.semantic_alignment;
        let complexity_score = self.calculate_complexity_score(prompt, output);
        let semantic_variance = self.calculate_semantic_variance(prompt, output);
        let uncertainty_indicators = self.calculate_uncertainty_indicators(output);
        
        let detailed_analysis = DetailedAnalysis {
            prompt_characteristics: prompt_chars,
            output_characteristics: output_chars,
            relationship_analysis: relationship,
            risk_indicators,
        };
        
        Ok(ScoringResult {
            specificity_score,
            ambiguity_score,
            coherence_score,
            complexity_score,
            semantic_variance,
            uncertainty_indicators,
            detailed_analysis,
        })
    }
    
    /// 🎯 Analyze prompt characteristics
    fn analyze_prompt_characteristics(&self, prompt: &str) -> PromptCharacteristics {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let sentences: Vec<&str> = prompt.split('.').collect();
        
        let word_count = words.len();
        let sentence_count = sentences.len();
        let avg_word_length = words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count.max(1) as f64;
        
        let question_count = prompt.matches('?').count();
        let imperative_count = self.count_imperatives(prompt);
        
        let specificity_keywords = self.find_keywords(prompt, &self.specificity_keywords);
        let ambiguity_keywords = self.find_keywords(prompt, &self.ambiguity_keywords);
        let domain_indicators = self.detect_domain_indicators(prompt);
        
        PromptCharacteristics {
            word_count,
            sentence_count,
            avg_word_length,
            question_count,
            imperative_count,
            specificity_keywords,
            ambiguity_keywords,
            domain_indicators,
        }
    }
    
    /// 📝 Analyze output characteristics
    fn analyze_output_characteristics(&self, output: &str) -> OutputCharacteristics {
        let words: Vec<&str> = output.split_whitespace().collect();
        let sentences: Vec<&str> = output.split('.').filter(|s| !s.trim().is_empty()).collect();
        
        let word_count = words.len();
        let sentence_count = sentences.len();
        let avg_sentence_length = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };
        
        let uncertainty_phrases = self.find_keywords(output, &self.uncertainty_keywords);
        let confidence_indicators = self.find_keywords(output, &self.confidence_keywords);
        let factual_claims = self.count_factual_claims(output);
        let hedging_language = self.count_hedging_language(output);
        
        OutputCharacteristics {
            word_count,
            sentence_count,
            avg_sentence_length,
            uncertainty_phrases,
            confidence_indicators,
            factual_claims,
            hedging_language,
        }
    }
    
    /// 🔗 Analyze relationship between prompt and output
    fn analyze_relationship(&self, prompt: &str, output: &str) -> RelationshipAnalysis {
        // Simplified relationship analysis
        let relevance_score = self.calculate_relevance_score(prompt, output);
        let completeness_score = self.calculate_completeness_score(prompt, output);
        let consistency_score = self.calculate_consistency_score(prompt, output);
        let appropriateness_score = self.calculate_appropriateness_score(prompt, output);
        let semantic_alignment = (relevance_score + completeness_score + consistency_score + appropriateness_score) / 4.0;
        
        RelationshipAnalysis {
            relevance_score,
            completeness_score,
            consistency_score,
            appropriateness_score,
            semantic_alignment,
        }
    }
    
    /// 🚨 Detect risk indicators
    fn detect_risk_indicators(
        &self,
        prompt: &str,
        output: &str,
        prompt_chars: &PromptCharacteristics,
        output_chars: &OutputCharacteristics,
    ) -> Vec<RiskIndicator> {
        let mut indicators = Vec::new();
        
        // High uncertainty detection
        if output_chars.uncertainty_phrases.len() > 3 {
            indicators.push(RiskIndicator {
                indicator_type: RiskIndicatorType::HighUncertainty,
                description: format!("High uncertainty: {} uncertainty phrases detected", output_chars.uncertainty_phrases.len()),
                severity: 0.7,
                confidence: 0.8,
            });
        }
        
        // Low specificity detection
        if prompt_chars.specificity_keywords.is_empty() && prompt_chars.ambiguity_keywords.len() > 2 {
            indicators.push(RiskIndicator {
                indicator_type: RiskIndicatorType::LowSpecificity,
                description: "Low specificity: Prompt contains ambiguous language without specific indicators".to_string(),
                severity: 0.6,
                confidence: 0.7,
            });
        }
        
        // Overconfidence detection
        if output_chars.confidence_indicators.len() > 2 && output_chars.hedging_language == 0 {
            indicators.push(RiskIndicator {
                indicator_type: RiskIndicatorType::OverConfidence,
                description: "Overconfidence: High confidence language without appropriate hedging".to_string(),
                severity: 0.8,
                confidence: 0.6,
            });
        }
        
        indicators
    }
    
    /// 📊 Calculate specificity score
    fn calculate_specificity_score(&self, prompt: &str, output: &str) -> f64 {
        let prompt_specificity = self.calculate_text_specificity(prompt);
        let output_specificity = self.calculate_text_specificity(output);
        (prompt_specificity + output_specificity) / 2.0
    }
    
    /// 🌀 Calculate ambiguity score
    fn calculate_ambiguity_score(&self, prompt: &str, output: &str) -> f64 {
        let prompt_ambiguity = self.calculate_text_ambiguity(prompt);
        let output_ambiguity = self.calculate_text_ambiguity(output);
        (prompt_ambiguity + output_ambiguity) / 2.0
    }
    
    /// 🧮 Calculate complexity score
    fn calculate_complexity_score(&self, prompt: &str, output: &str) -> f64 {
        let prompt_complexity = self.calculate_text_complexity(prompt);
        let output_complexity = self.calculate_text_complexity(output);
        (prompt_complexity + output_complexity) / 2.0
    }
    
    /// 📊 Calculate semantic variance
    fn calculate_semantic_variance(&self, prompt: &str, output: &str) -> f64 {
        // Simplified semantic variance calculation
        let prompt_words: Vec<&str> = prompt.split_whitespace().collect();
        let output_words: Vec<&str> = output.split_whitespace().collect();
        
        let unique_words = prompt_words.iter().chain(output_words.iter()).collect::<std::collections::HashSet<_>>();
        let total_words = prompt_words.len() + output_words.len();
        
        if total_words > 0 {
            unique_words.len() as f64 / total_words as f64
        } else {
            0.0
        }
    }
    
    /// ⚠️ Calculate uncertainty indicators
    fn calculate_uncertainty_indicators(&self, output: &str) -> f64 {
        let output_lower = output.to_lowercase();
        let mut uncertainty_score = 0.0;
        
        for (keyword, weight) in &self.uncertainty_keywords {
            if output_lower.contains(keyword) {
                uncertainty_score += weight;
            }
        }
        
        // Normalize to 0-1 range
        (uncertainty_score / 10.0).min(1.0)
    }
    
    // Helper methods
    fn find_keywords(&self, text: &str, keywords: &HashMap<String, f64>) -> Vec<String> {
        let text_lower = text.to_lowercase();
        keywords.keys()
            .filter(|keyword| text_lower.contains(*keyword))
            .cloned()
            .collect()
    }
    
    fn calculate_text_specificity(&self, text: &str) -> f64 {
        let specificity_matches = self.find_keywords(text, &self.specificity_keywords);
        let ambiguity_matches = self.find_keywords(text, &self.ambiguity_keywords);
        
        let specificity_score = specificity_matches.len() as f64 * 0.1;
        let ambiguity_penalty = ambiguity_matches.len() as f64 * 0.1;
        
        (specificity_score - ambiguity_penalty + 0.5).max(0.0).min(1.0)
    }
    
    fn calculate_text_ambiguity(&self, text: &str) -> f64 {
        let ambiguity_matches = self.find_keywords(text, &self.ambiguity_keywords);
        (ambiguity_matches.len() as f64 * 0.1).min(1.0)
    }
    
    fn calculate_text_complexity(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();
        
        let avg_word_length = words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len().max(1) as f64;
        let avg_sentence_length = words.len() as f64 / sentences.len().max(1) as f64;
        
        // Normalize complexity based on word length and sentence length
        ((avg_word_length / 10.0) + (avg_sentence_length / 20.0)).min(1.0)
    }
    
    fn calculate_relevance_score(&self, prompt: &str, output: &str) -> f64 {
        // Simple word overlap calculation
        let prompt_words: std::collections::HashSet<_> = prompt.split_whitespace().map(|w| w.to_lowercase()).collect();
        let output_words: std::collections::HashSet<_> = output.split_whitespace().map(|w| w.to_lowercase()).collect();
        
        let intersection = prompt_words.intersection(&output_words).count();
        let union = prompt_words.union(&output_words).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
    
    fn calculate_completeness_score(&self, prompt: &str, output: &str) -> f64 {
        // Simple heuristic: longer outputs are more complete
        let prompt_length = prompt.len();
        let output_length = output.len();
        
        if prompt_length > 0 {
            (output_length as f64 / prompt_length as f64).min(1.0)
        } else {
            0.0
        }
    }
    
    fn calculate_consistency_score(&self, _prompt: &str, _output: &str) -> f64 {
        // Simplified consistency score
        0.7 // Placeholder
    }
    
    fn calculate_appropriateness_score(&self, _prompt: &str, _output: &str) -> f64 {
        // Simplified appropriateness score
        0.8 // Placeholder
    }
    
    fn count_imperatives(&self, text: &str) -> usize {
        let imperatives = ["please", "use", "create", "make", "do", "tell", "explain", "describe", "list", "provide"];
        let text_lower = text.to_lowercase();
        imperatives.iter().filter(|&imp| text_lower.contains(imp)).count()
    }
    
    fn detect_domain_indicators(&self, text: &str) -> Vec<String> {
        let domains = ["technical", "scientific", "medical", "legal", "business", "academic", "creative"];
        let text_lower = text.to_lowercase();
        domains.iter()
            .filter(|&domain| text_lower.contains(domain))
            .map(|s| s.to_string())
            .collect()
    }
    
    fn count_factual_claims(&self, text: &str) -> usize {
        // Simple heuristic: sentences with specific numbers, dates, or facts
        let sentences: Vec<&str> = text.split('.').collect();
        sentences.iter()
            .filter(|sentence| {
                sentence.chars().any(|c| c.is_ascii_digit()) || 
                sentence.to_lowercase().contains("fact") ||
                sentence.to_lowercase().contains("research") ||
                sentence.to_lowercase().contains("study")
            })
            .count()
    }
    
    fn count_hedging_language(&self, text: &str) -> usize {
        let hedging_words = ["might", "could", "perhaps", "maybe", "possibly", "likely", "probably"];
        let text_lower = text.to_lowercase();
        hedging_words.iter()
            .filter(|&word| text_lower.contains(word))
            .count()
    }
}
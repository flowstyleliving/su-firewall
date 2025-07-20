// 🧠 Semantic Compression Engine (Rust Implementation)
// Ultra-fast extraction of semantic "essence" from long prompts

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{debug, instrument};

/// Semantic compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub max_tokens: usize,
    pub target_compression_ratio: f32,
    pub preserve_risk_indicators: bool,
    pub fast_mode: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 150,
            target_compression_ratio: 0.3,
            preserve_risk_indicators: true,
            fast_mode: true,
        }
    }
}

/// Compression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    pub original_length: usize,
    pub compressed_essence: String,
    pub compression_ratio: f32,
    pub semantic_loss: f32,
    pub risk_preservation: RiskPreservation,
    pub intent_category: String,
    pub key_concepts: Vec<String>,
    pub compression_time_ms: f64,
    pub should_use_compression: CompressionDecision,
}

/// Risk preservation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPreservation {
    pub risk_preserved: bool,
    pub original_risk_score: f32,
    pub compressed_risk_score: f32,
    pub risk_indicators_found: Vec<String>,
}

/// Compression decision with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionDecision {
    pub should_compress: bool,
    pub reason: String,
    pub confidence: f32,
}

/// Semantic components extracted from text
#[derive(Debug, Clone)]
struct SemanticComponents {
    concepts: Vec<String>,
    entities: Vec<String>,
    actions: Vec<String>,
    modifiers: Vec<String>,
    risk_keywords: Vec<String>,
}

/// Intent classification result
#[derive(Debug, Clone)]
struct IntentAnalysis {
    category: String,
    confidence: f32,
    primary_action: String,
    target_domain: String,
}

/// High-performance semantic compressor
pub struct SemanticCompressor {
    config: CompressionConfig,
    // Pre-compiled keyword sets for ultra-fast matching
    risk_keywords: HashSet<&'static str>,
    concept_keywords: HashSet<&'static str>,
    action_keywords: HashSet<&'static str>,
    entity_keywords: HashSet<&'static str>,
}

impl SemanticCompressor {
    /// Create new compressor with default configuration
    pub fn new() -> Self {
        Self::with_config(CompressionConfig::default())
    }

    /// Create compressor with custom configuration
    pub fn with_config(config: CompressionConfig) -> Self {
        Self {
            config,
            risk_keywords: Self::build_risk_keywords(),
            concept_keywords: Self::build_concept_keywords(),
            action_keywords: Self::build_action_keywords(),
            entity_keywords: Self::build_entity_keywords(),
        }
    }

    /// 🧠 Main compression function
    #[instrument(skip(self, prompt), fields(length = prompt.len()))]
    pub fn compress_prompt(&self, prompt: &str) -> Result<CompressionResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Starting semantic compression for {} chars", prompt.len());

        // Step 1: Extract semantic components
        let components = self.extract_semantic_components(prompt);
        
        // Step 2: Identify core intent
        let intent = self.identify_intent(prompt, &components);
        
        // Step 3: Generate compressed essence
        let essence = self.generate_essence(&components, &intent)?;
        
        // Step 4: Analyze risk preservation
        let risk_preservation = self.analyze_risk_preservation(prompt, &essence);
        
        // Step 5: Make compression decision
        let decision = self.should_use_compression(prompt, &essence, &risk_preservation);
        
        let compression_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(CompressionResult {
            original_length: prompt.len(),
            compressed_essence: essence,
            compression_ratio: 0.0, // Will be calculated
            semantic_loss: self.calculate_semantic_loss(prompt, &components),
            risk_preservation,
            intent_category: intent.category,
            key_concepts: components.concepts,
            compression_time_ms: compression_time,
            should_use_compression: decision,
        })
    }

    /// 🔍 Extract key semantic components using optimized pattern matching
    fn extract_semantic_components(&self, prompt: &str) -> SemanticComponents {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let text_lower = prompt.to_lowercase();
        
        let mut components = SemanticComponents {
            concepts: Vec::new(),
            entities: Vec::new(),
            actions: Vec::new(),
            modifiers: Vec::new(),
            risk_keywords: Vec::new(),
        };

        // Ultra-fast keyword matching using pre-compiled sets
        for word in &words {
            let word_lower = word.to_lowercase();
            
            if self.risk_keywords.contains(word_lower.as_str()) {
                components.risk_keywords.push(word_lower.clone());
            }
            
            if self.action_keywords.contains(word_lower.as_str()) {
                components.actions.push(word_lower.clone());
            }
            
            if self.concept_keywords.contains(word_lower.as_str()) {
                components.concepts.push(word_lower.clone());
            }
            
            if self.entity_keywords.contains(word_lower.as_str()) {
                components.entities.push(word_lower.clone());
            }
        }

        // Pattern-based extraction for complex concepts
        self.extract_patterns(&text_lower, &mut components);
        
        components
    }

    /// Extract complex patterns and phrases
    fn extract_patterns(&self, text: &str, components: &mut SemanticComponents) {
        // Technical patterns
        if text.contains("how to") && (text.contains("hack") || text.contains("break")) {
            components.risk_keywords.push("instructional_hack".to_string());
        }
        
        // Creative patterns
        if text.contains("write") && (text.contains("story") || text.contains("creative")) {
            components.concepts.push("creative_writing".to_string());
        }
        
        // Explanation patterns
        if text.contains("explain") && (text.contains("quantum") || text.contains("physics")) {
            components.concepts.push("technical_explanation".to_string());
        }
        
        // Manipulation patterns
        if text.contains("manipulat") || text.contains("psycholog") {
            components.risk_keywords.push("psychological_manipulation".to_string());
        }
    }

    /// 🎯 Identify primary intent using fast classification
    fn identify_intent(&self, prompt: &str, components: &SemanticComponents) -> IntentAnalysis {
        let prompt_lower = prompt.to_lowercase();
        
        // Fast intent classification based on patterns
        if !components.risk_keywords.is_empty() {
            IntentAnalysis {
                category: "high_risk".to_string(),
                confidence: 0.9,
                primary_action: components.actions.first().unwrap_or(&"unknown".to_string()).clone(),
                target_domain: "security".to_string(),
            }
        } else if prompt_lower.contains("creative") || prompt_lower.contains("story") {
            IntentAnalysis {
                category: "creative".to_string(),
                confidence: 0.8,
                primary_action: "generate".to_string(),
                target_domain: "content".to_string(),
            }
        } else if prompt_lower.contains("explain") || prompt_lower.contains("how") {
            IntentAnalysis {
                category: "informational".to_string(),
                confidence: 0.7,
                primary_action: "explain".to_string(),
                target_domain: "knowledge".to_string(),
            }
        } else {
            IntentAnalysis {
                category: "general".to_string(),
                confidence: 0.6,
                primary_action: "respond".to_string(),
                target_domain: "general".to_string(),
            }
        }
    }

    /// ⚡ Generate compressed essence maintaining semantic integrity
    fn generate_essence(&self, components: &SemanticComponents, intent: &IntentAnalysis) -> Result<String> {
        let mut essence_parts = Vec::new();
        
        // Include intent category
        essence_parts.push(format!("[{}]", intent.category));
        
        // Include primary action
        if !intent.primary_action.is_empty() {
            essence_parts.push(intent.primary_action.clone());
        }
        
        // Include key concepts (limited)
        let key_concepts: Vec<String> = components.concepts
            .iter()
            .take(3)
            .cloned()
            .collect();
        if !key_concepts.is_empty() {
            essence_parts.push(key_concepts.join(" "));
        }
        
        // Always preserve risk keywords (critical for safety)
        if !components.risk_keywords.is_empty() {
            let risk_essence: Vec<String> = components.risk_keywords
                .iter()
                .take(5)
                .cloned()
                .collect();
            essence_parts.push(format!("RISK:{}", risk_essence.join(" ")));
        }
        
        // Include target domain
        essence_parts.push(format!("domain:{}", intent.target_domain));
        
        let essence = essence_parts.join(" ");
        
        // Ensure within token limit
        if essence.len() > self.config.max_tokens * 4 { // Rough char estimate
            Ok(essence.chars().take(self.config.max_tokens * 4).collect())
        } else {
            Ok(essence)
        }
    }

    /// 📊 Analyze risk preservation quality
    fn analyze_risk_preservation(&self, original: &str, compressed: &str) -> RiskPreservation {
        let original_risk = self.calculate_risk_score(original);
        let compressed_risk = self.calculate_risk_score(compressed);
        
        let risk_preserved = if original_risk > 0.5 {
            compressed_risk >= original_risk * 0.8 // Preserve 80% of risk signal
        } else {
            true // Low risk prompts are fine
        };
        
        RiskPreservation {
            risk_preserved,
            original_risk_score: original_risk,
            compressed_risk_score: compressed_risk,
            risk_indicators_found: self.extract_risk_indicators(original),
        }
    }

    /// Calculate risk score for text
    fn calculate_risk_score(&self, text: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let mut score: f32 = 0.0;
        
        for keyword in &self.risk_keywords {
            if text_lower.contains(keyword) {
                score += match keyword {
                    &"hack" | &"bomb" | &"weapon" => 0.4,
                    &"manipulat" | &"exploit" => 0.3,
                    &"paradox" | &"infinite" => 0.2,
                    _ => 0.1,
                };
            }
        }
        
        score.min(1.0)
    }

    /// Extract specific risk indicators
    fn extract_risk_indicators(&self, text: &str) -> Vec<String> {
        let text_lower = text.to_lowercase();
        let mut indicators = Vec::new();
        
        for keyword in &self.risk_keywords {
            if text_lower.contains(keyword) {
                indicators.push(keyword.to_string());
            }
        }
        
        indicators
    }

    /// Calculate semantic loss estimation
    fn calculate_semantic_loss(&self, original: &str, components: &SemanticComponents) -> f32 {
        let original_concepts = original.split_whitespace().count();
        let preserved_concepts = components.concepts.len() + components.actions.len();
        
        if original_concepts == 0 {
            0.0
        } else {
            1.0 - (preserved_concepts as f32 / original_concepts as f32).min(1.0)
        }
    }

    /// Decide whether to use compression
    fn should_use_compression(&self, original: &str, essence: &str, risk: &RiskPreservation) -> CompressionDecision {
        let compression_ratio = essence.len() as f32 / original.len() as f32;
        
        if original.len() < 200 {
            CompressionDecision {
                should_compress: false,
                reason: "Input too short to benefit from compression".to_string(),
                confidence: 0.9,
            }
        } else if !risk.risk_preserved && risk.original_risk_score > 0.5 {
            CompressionDecision {
                should_compress: false,
                reason: "Risk indicators would be lost in compression".to_string(),
                confidence: 0.8,
            }
        } else if compression_ratio > 0.8 {
            CompressionDecision {
                should_compress: false,
                reason: "Insufficient compression achieved".to_string(),
                confidence: 0.7,
            }
        } else {
            CompressionDecision {
                should_compress: true,
                reason: format!("Efficient compression: {:.1}% reduction", (1.0 - compression_ratio) * 100.0),
                confidence: 0.8,
            }
        }
    }

    /// Build risk keyword set (compile-time optimization)
    fn build_risk_keywords() -> HashSet<&'static str> {
        [
            "hack", "bomb", "explosive", "kill", "murder", "poison", 
            "illegal", "steal", "break", "manipulat", "exploit", "weapon",
            "attack", "destroy", "harm", "paradox", "infinite", "recursive",
            "loop", "crash", "override", "bypass", "trick", "deceive",
            "confuse", "mislead", "consciousness", "reality", "existence",
            "impossible", "contradiction", "logic", "reasoning", "meta",
        ].iter().cloned().collect()
    }

    /// Build concept keyword set
    fn build_concept_keywords() -> HashSet<&'static str> {
        [
            "creative", "story", "explain", "quantum", "physics", "ai",
            "artificial", "intelligence", "neural", "network", "algorithm",
            "science", "technology", "computer", "program", "code", "data",
            "analysis", "research", "study", "theory", "concept", "idea",
        ].iter().cloned().collect()
    }

    /// Build action keyword set
    fn build_action_keywords() -> HashSet<&'static str> {
        [
            "write", "create", "generate", "explain", "describe", "analyze",
            "build", "develop", "design", "implement", "solve", "calculate",
            "compute", "process", "transform", "convert", "translate", "summarize",
        ].iter().cloned().collect()
    }

    /// Build entity keyword set
    fn build_entity_keywords() -> HashSet<&'static str> {
        [
            "system", "model", "framework", "platform", "service", "application",
            "database", "server", "network", "protocol", "algorithm", "structure",
            "organization", "company", "user", "client", "customer", "person",
        ].iter().cloned().collect()
    }
}

impl Default for SemanticCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compression() {
        let compressor = SemanticCompressor::new();
        let prompt = "Write a creative story about dragons and magic in a fantasy world with epic battles";
        
        let result = compressor.compress_prompt(prompt).unwrap();
        
        assert!(!result.compressed_essence.is_empty());
        assert!(result.compression_ratio < 1.0);
        assert_eq!(result.intent_category, "creative");
    }

    #[test]
    fn test_risk_preservation() {
        let compressor = SemanticCompressor::new();
        let prompt = "How to hack into a computer system and steal data";
        
        let result = compressor.compress_prompt(prompt).unwrap();
        
        assert!(result.risk_preservation.original_risk_score >= 0.5); // "hack" (0.4) + "steal" (0.1) = 0.5
        assert!(!result.risk_preservation.risk_indicators_found.is_empty());
    }

    #[test]
    fn test_short_prompt_no_compression() {
        let compressor = SemanticCompressor::new();
        let prompt = "What is 2+2?";
        
        let result = compressor.compress_prompt(prompt).unwrap();
        
        assert!(!result.should_use_compression.should_compress);
    }
}
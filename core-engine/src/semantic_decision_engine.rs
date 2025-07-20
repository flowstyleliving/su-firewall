// 🧮 Semantic Uncertainty Decision Engine
// Uses ℏₛ = √(Δμ × Δσ) to guide every process decision

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// 📊 Process uncertainty measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessUncertainty {
    pub process_name: String,
    pub delta_mu: f32,      // 📐 Precision/Stability
    pub delta_sigma: f32,   // 🌊 Flexibility/Chaos
    pub h_bar: f32,         // 🧮 ℏₛ = √(Δμ × Δσ)
    pub risk_level: RiskLevel,
    pub decision: ProcessDecision,
    pub emoji_indicator: String,
    pub relevance_phrase: String,
}

/// 🚨 Risk classification for processes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    HighCollapse,      // 🔴 ℏₛ < 1.0 - Immediate attention
    ModerateInstability, // 🟡 1.0 ≤ ℏₛ < 1.2 - Monitor closely  
    Stable,            // 🟢 ℏₛ ≥ 1.2 - Proceed normally
}

/// 🎯 Process decisions based on uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(PartialEq)]
pub enum ProcessDecision {
    Execute,           // ✅ Low uncertainty, proceed
    Monitor,           // 👀 Medium uncertainty, watch closely
    Defer,             // ⏳ High uncertainty, delay
    Escalate,          // 🚨 Critical uncertainty, manual intervention
}

/// 🧮 Semantic uncertainty decision engine
pub struct SemanticDecisionEngine {
    process_history: HashMap<String, Vec<ProcessUncertainty>>,
    decision_thresholds: DecisionThresholds,
}

/// ⚙️ Configurable decision thresholds
#[derive(Debug, Clone)]
pub struct DecisionThresholds {
    pub high_risk_threshold: f32,     // 🔴 Below this = high collapse risk
    pub moderate_risk_threshold: f32, // 🟡 Below this = moderate instability
    pub stability_confidence: f32,    // 🟢 Above this = stable
}

impl Default for DecisionThresholds {
    fn default() -> Self {
        Self {
            high_risk_threshold: 1.0,
            moderate_risk_threshold: 1.2,
            stability_confidence: 1.5,
        }
    }
}

impl SemanticDecisionEngine {
    /// 🚀 Create new decision engine
    pub fn new() -> Self {
        Self {
            process_history: HashMap::new(),
            decision_thresholds: DecisionThresholds::default(),
        }
    }

    /// 💾 Long-term storage decision with ℏₛ calculation
    pub fn evaluate_storage_decision(&mut self, operation: &str, data_size: usize, criticality: f32) -> ProcessUncertainty {
        // 📐 Calculate Δμ (precision/stability) based on operation characteristics
        let delta_mu = self.calculate_storage_precision(operation, data_size, criticality);
        
        // 🌊 Calculate Δσ (flexibility/chaos) based on system state
        let delta_sigma = self.calculate_storage_flexibility(operation, data_size);
        
        // 🧮 Calculate semantic uncertainty: ℏₛ = √(Δμ × Δσ)
        let h_bar = (delta_mu * delta_sigma).sqrt();
        
        // 🎯 Determine risk level and decision
        let (risk_level, decision, emoji, phrase) = self.classify_storage_risk(h_bar, operation);
        
        let uncertainty = ProcessUncertainty {
            process_name: format!("storage_{}", operation),
            delta_mu,
            delta_sigma,
            h_bar,
            risk_level,
            decision,
            emoji_indicator: emoji,
            relevance_phrase: phrase,
        };
        
        // 📝 Store decision history
        self.record_decision(&uncertainty);
        
        info!("{} {} | ℏₛ={:.3} | Δμ={:.3} | Δσ={:.3} | {}", 
              uncertainty.emoji_indicator, 
              uncertainty.relevance_phrase,
              h_bar, delta_mu, delta_sigma,
              operation);
        
        uncertainty
    }

    /// 🔄 Batch processing decision with ℏₛ calculation  
    pub fn evaluate_batch_decision(&mut self, batch_size: usize, complexity: f32, time_pressure: f32) -> ProcessUncertainty {
        // 📐 Precision based on batch characteristics
        let delta_mu = self.calculate_batch_precision(batch_size, complexity);
        
        // 🌊 Flexibility based on system load and pressure
        let delta_sigma = self.calculate_batch_flexibility(batch_size, time_pressure);
        
        // 🧮 Semantic uncertainty calculation
        let h_bar = (delta_mu * delta_sigma).sqrt();
        
        let (risk_level, decision, emoji, phrase) = self.classify_batch_risk(h_bar, batch_size);
        
        let uncertainty = ProcessUncertainty {
            process_name: format!("batch_processing_{}", batch_size),
            delta_mu,
            delta_sigma,
            h_bar,
            risk_level,
            decision,
            emoji_indicator: emoji,
            relevance_phrase: phrase,
        };
        
        self.record_decision(&uncertainty);
        
        info!("{} {} | ℏₛ={:.3} | Batch={} items", 
              uncertainty.emoji_indicator, 
              uncertainty.relevance_phrase,
              h_bar, batch_size);
        
        uncertainty
    }

    /// 🧠 Compression decision with ℏₛ calculation
    pub fn evaluate_compression_decision(&mut self, text_length: usize, risk_score: f32) -> ProcessUncertainty {
        // 📐 Precision based on compression effectiveness
        let delta_mu = self.calculate_compression_precision(text_length, risk_score);
        
        // 🌊 Flexibility based on semantic loss potential
        let delta_sigma = self.calculate_compression_flexibility(text_length, risk_score);
        
        // 🧮 Semantic uncertainty
        let h_bar = (delta_mu * delta_sigma).sqrt();
        
        let (risk_level, decision, emoji, phrase) = self.classify_compression_risk(h_bar, risk_score);
        
        let uncertainty = ProcessUncertainty {
            process_name: format!("compression_{}_chars", text_length),
            delta_mu,
            delta_sigma,
            h_bar,
            risk_level,
            decision,
            emoji_indicator: emoji,
            relevance_phrase: phrase,
        };
        
        self.record_decision(&uncertainty);
        
        info!("{} {} | ℏₛ={:.3} | Length={} chars", 
              uncertainty.emoji_indicator, 
              uncertainty.relevance_phrase,
              h_bar, text_length);
        
        uncertainty
    }

    /// 🌐 API request decision with ℏₛ calculation
    pub fn evaluate_api_decision(&mut self, endpoint: &str, payload_size: usize, auth_level: f32) -> ProcessUncertainty {
        // 📐 Precision based on endpoint security
        let delta_mu = self.calculate_api_precision(endpoint, auth_level);
        
        // 🌊 Flexibility based on payload and load
        let delta_sigma = self.calculate_api_flexibility(endpoint, payload_size);
        
        // 🧮 Semantic uncertainty
        let h_bar = (delta_mu * delta_sigma).sqrt();
        
        let (risk_level, decision, emoji, phrase) = self.classify_api_risk(h_bar, endpoint);
        
        let uncertainty = ProcessUncertainty {
            process_name: format!("api_{}", endpoint),
            delta_mu,
            delta_sigma,
            h_bar,
            risk_level,
            decision,
            emoji_indicator: emoji,
            relevance_phrase: phrase,
        };
        
        self.record_decision(&uncertainty);
        
        info!("{} {} | ℏₛ={:.3} | Endpoint={}", 
              uncertainty.emoji_indicator, 
              uncertainty.relevance_phrase,
              h_bar, endpoint);
        
        uncertainty
    }

    // 📐 Precision Calculations (Δμ)
    
    fn calculate_storage_precision(&self, operation: &str, data_size: usize, criticality: f32) -> f32 {
        let base_precision = match operation {
            "write" => 0.8,      // 📝 Write operations are precise
            "read" => 0.9,       // 📖 Read operations are very precise  
            "delete" => 0.6,     // 🗑️ Delete operations are risky
            "update" => 0.7,     // ✏️ Update operations moderate risk
            _ => 0.5,
        };
        
        // Adjust for data size (larger = less precise)
        let size_factor = 1.0 / (1.0 + (data_size as f32 / 1000000.0));
        
        // Adjust for criticality
        let criticality_factor = 1.0 + (criticality * 0.3);
        
        (base_precision * size_factor * criticality_factor).clamp(0.1, 2.0)
    }
    
    fn calculate_batch_precision(&self, batch_size: usize, complexity: f32) -> f32 {
        // 📊 Smaller batches = higher precision
        let size_precision = 2.0 / (1.0 + (batch_size as f32 / 10.0));
        
        // 🧠 Lower complexity = higher precision
        let complexity_precision = 2.0 / (1.0 + complexity);
        
        (size_precision + complexity_precision) / 2.0
    }
    
    fn calculate_compression_precision(&self, text_length: usize, risk_score: f32) -> f32 {
        // 📏 Optimal length range for compression precision
        let length_precision = if text_length < 100 {
            0.3 // Too short
        } else if text_length > 5000 {
            0.5 // Too long
        } else {
            1.0 // Optimal range
        };
        
        // 🚨 High risk content reduces precision
        let risk_precision = 1.0 - (risk_score * 0.4);
        
        (length_precision * risk_precision).clamp(0.1, 1.5)
    }
    
    fn calculate_api_precision(&self, endpoint: &str, auth_level: f32) -> f32 {
        let endpoint_precision = match endpoint {
            "analyze" => 0.9,    // 🔍 Core function, high precision
            "batch" => 0.7,      // 📦 Batch processing, moderate
            "status" => 1.0,     // 📊 Status check, very precise
            "health" => 0.8,     // 🏥 Health check, good precision
            _ => 0.5,
        };
        
        // 🔐 Higher auth = higher precision
        let auth_precision = 0.5 + (auth_level * 0.5);
        
        (endpoint_precision + auth_precision) / 2.0
    }

    // 🌊 Flexibility Calculations (Δσ)
    
    fn calculate_storage_flexibility(&self, operation: &str, data_size: usize) -> f32 {
        let base_flexibility = match operation {
            "write" => 1.2,      // 📝 Write has high flexibility
            "read" => 0.8,       // 📖 Read is more constrained
            "delete" => 1.5,     // 🗑️ Delete is very flexible (dangerous)
            "update" => 1.3,     // ✏️ Update has high flexibility
            _ => 1.0,
        };
        
        // 📊 Larger data = more flexibility/chaos
        let size_flexibility = 1.0 + (data_size as f32 / 500000.0);
        
        (base_flexibility * size_flexibility).clamp(0.5, 3.0)
    }
    
    fn calculate_batch_flexibility(&self, batch_size: usize, time_pressure: f32) -> f32 {
        // 📈 Larger batches = more flexibility/unpredictability
        let size_flexibility = 1.0 + (batch_size as f32 / 20.0);
        
        // ⏰ Time pressure increases flexibility/chaos
        let pressure_flexibility = 1.0 + (time_pressure * 0.8);
        
        (size_flexibility + pressure_flexibility) / 2.0
    }
    
    fn calculate_compression_flexibility(&self, text_length: usize, risk_score: f32) -> f32 {
        // 📏 Longer text = more compression flexibility
        let length_flexibility = 1.0 + (text_length as f32 / 1000.0);
        
        // 🚨 Higher risk = more unpredictable outcomes
        let risk_flexibility = 1.0 + (risk_score * 1.2);
        
        (length_flexibility + risk_flexibility) / 2.0
    }
    
    fn calculate_api_flexibility(&self, endpoint: &str, payload_size: usize) -> f32 {
        let endpoint_flexibility = match endpoint {
            "analyze" => 1.1,    // 🔍 Some flexibility in analysis
            "batch" => 1.8,      // 📦 High flexibility in batch
            "status" => 0.6,     // 📊 Low flexibility in status
            "health" => 0.5,     // 🏥 Very low flexibility
            _ => 1.0,
        };
        
        // 📦 Larger payload = more flexibility
        let payload_flexibility = 1.0 + (payload_size as f32 / 10000.0);
        
        (endpoint_flexibility + payload_flexibility) / 2.0
    }

    // 🎯 Risk Classification and Decision Making
    
    fn classify_storage_risk(&self, h_bar: f32, operation: &str) -> (RiskLevel, ProcessDecision, String, String) {
        if h_bar < self.decision_thresholds.high_risk_threshold {
            (RiskLevel::HighCollapse, ProcessDecision::Escalate, 
             "🔴".to_string(), format!("CRITICAL_STORAGE_{}", operation.to_uppercase()))
        } else if h_bar < self.decision_thresholds.moderate_risk_threshold {
            (RiskLevel::ModerateInstability, ProcessDecision::Monitor,
             "🟡".to_string(), format!("CAUTIOUS_STORAGE_{}", operation))
        } else {
            (RiskLevel::Stable, ProcessDecision::Execute,
             "🟢".to_string(), format!("STABLE_STORAGE_{}", operation))
        }
    }
    
    fn classify_batch_risk(&self, h_bar: f32, batch_size: usize) -> (RiskLevel, ProcessDecision, String, String) {
        if h_bar < self.decision_thresholds.high_risk_threshold {
            (RiskLevel::HighCollapse, ProcessDecision::Defer,
             "🚨".to_string(), format!("BATCH_OVERLOAD_{}_ITEMS", batch_size))
        } else if h_bar < self.decision_thresholds.moderate_risk_threshold {
            (RiskLevel::ModerateInstability, ProcessDecision::Monitor,
             "⚡".to_string(), format!("BATCH_PROCESSING_{}_ITEMS", batch_size))
        } else {
            (RiskLevel::Stable, ProcessDecision::Execute,
             "🚀".to_string(), format!("OPTIMAL_BATCH_{}_ITEMS", batch_size))
        }
    }
    
    fn classify_compression_risk(&self, h_bar: f32, risk_score: f32) -> (RiskLevel, ProcessDecision, String, String) {
        if h_bar < self.decision_thresholds.high_risk_threshold || risk_score > 0.7 {
            (RiskLevel::HighCollapse, ProcessDecision::Defer,
             "🛑".to_string(), "COMPRESSION_BLOCKED_RISK".to_string())
        } else if h_bar < self.decision_thresholds.moderate_risk_threshold {
            (RiskLevel::ModerateInstability, ProcessDecision::Monitor,
             "🧠".to_string(), "CAREFUL_COMPRESSION".to_string())
        } else {
            (RiskLevel::Stable, ProcessDecision::Execute,
             "⚡".to_string(), "OPTIMAL_COMPRESSION".to_string())
        }
    }
    
    fn classify_api_risk(&self, h_bar: f32, endpoint: &str) -> (RiskLevel, ProcessDecision, String, String) {
        if h_bar < self.decision_thresholds.high_risk_threshold {
            (RiskLevel::HighCollapse, ProcessDecision::Escalate,
             "🚨".to_string(), format!("API_SECURITY_ALERT_{}", endpoint.to_uppercase()))
        } else if h_bar < self.decision_thresholds.moderate_risk_threshold {
            (RiskLevel::ModerateInstability, ProcessDecision::Monitor,
             "🔍".to_string(), format!("API_MONITORING_{}", endpoint))
        } else {
            (RiskLevel::Stable, ProcessDecision::Execute,
             "✅".to_string(), format!("API_APPROVED_{}", endpoint))
        }
    }

    /// 📝 Record decision for learning and analysis
    fn record_decision(&mut self, uncertainty: &ProcessUncertainty) {
        let history = self.process_history
            .entry(uncertainty.process_name.clone())
            .or_insert_with(Vec::new);
        
        history.push(uncertainty.clone());
        
        // Keep only last 100 decisions per process
        if history.len() > 100 {
            history.remove(0);
        }
        
        debug!("📝 DECISION_RECORDED | {} | ℏₛ={:.3}", 
               uncertainty.process_name, uncertainty.h_bar);
    }

    /// 📊 Get decision history for analysis
    pub fn get_decision_history(&self, process_name: &str) -> Option<&Vec<ProcessUncertainty>> {
        self.process_history.get(process_name)
    }

    /// 🎯 Get average uncertainty for process type
    pub fn get_average_uncertainty(&self, process_prefix: &str) -> Option<f32> {
        let matching_processes: Vec<f32> = self.process_history
            .iter()
            .filter(|(name, _)| name.starts_with(process_prefix))
            .flat_map(|(_, history)| history.iter().map(|u| u.h_bar))
            .collect();
        
        if matching_processes.is_empty() {
            None
        } else {
            Some(matching_processes.iter().sum::<f32>() / matching_processes.len() as f32)
        }
    }
}

impl Default for SemanticDecisionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_decision() {
        let mut engine = SemanticDecisionEngine::new();
        
        let decision = engine.evaluate_storage_decision("write", 1000, 0.8);
        
        assert!(decision.h_bar > 0.0);
        assert!(!decision.emoji_indicator.is_empty());
        assert!(!decision.relevance_phrase.is_empty());
    }

    #[test]
    fn test_batch_decision() {
        let mut engine = SemanticDecisionEngine::new();
        
        let decision = engine.evaluate_batch_decision(25, 0.5, 0.3);
        
        assert!(decision.h_bar > 0.0);
        assert_eq!(decision.process_name, "batch_processing_25");
    }

    #[test]
    fn test_high_risk_classification() {
        let mut engine = SemanticDecisionEngine::new();
        
        // Force high risk scenario
        let decision = engine.evaluate_storage_decision("delete", 10000000, 1.0);
        
        // Should trigger high risk due to large data size and delete operation
        assert!(decision.h_bar < 1.0 || decision.decision == ProcessDecision::Escalate);
    }
}
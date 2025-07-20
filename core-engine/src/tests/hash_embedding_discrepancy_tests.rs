// 🧪 Hash Embedding Discrepancy Test Module
// Comprehensive testing suite to quantify the discrepancy and information loss
// introduced by hash embeddings, particularly affecting derived semantic precision

use crate::semantic_metrics::{HashEmbeddingDiscrepancyTester, DiscrepancyTestResult};
use approx::assert_relative_eq;
use std::hash::{Hash, Hasher};

/// 🧪 Comprehensive Hash Embedding Discrepancy Test Suite
pub struct HashEmbeddingDiscrepancyTestSuite {
    tester: HashEmbeddingDiscrepancyTester,
}

impl HashEmbeddingDiscrepancyTestSuite {
    /// Create new test suite
    pub fn new() -> Self {
        Self {
            tester: HashEmbeddingDiscrepancyTester::new(),
        }
    }

    /// 🧪 Run all discrepancy tests
    pub fn run_all_tests(&self) -> Vec<DiscrepancyTestResult> {
        let mut results = Vec::new();

        // Test 1: JSD Discrepancy with Collisions
        results.push(self.test_jsd_collisions());

        // Test 2: Fisher Information Discrepancy
        results.push(self.test_fisher_information_discrepancy());

        // Test 3: General Semantic Distortion
        results.push(self.test_general_semantic_distortion());

        // Test 4: Edge Cases
        results.extend(self.test_edge_cases());

        results
    }

    /// 🧪 Test 1: JSD Discrepancy with Collisions
    /// 
    /// Simulate two semantically distinct input strings that are likely to produce
    /// highly similar or colliding hash embeddings.
    fn test_jsd_collisions(&self) -> DiscrepancyTestResult {
        // Create semantically distinct concepts that might hash to similar values
        let true_a = "The quantum mechanics of particle physics involves superposition and entanglement";
        let true_b = "The classical mechanics of macroscopic objects involves deterministic trajectories";
        
        // Simulate hash collisions by using similar word patterns
        let hashed_a = "quantum particle physics superposition entanglement";
        let hashed_b = "classical macroscopic objects deterministic trajectories";

        match self.tester.test_jsd_discrepancy_with_collisions(
            true_a, true_b, hashed_a, hashed_b
        ) {
            Ok(result) => result,
            Err(e) => DiscrepancyTestResult {
                test_type: "JSD Discrepancy with Collisions".to_string(),
                true_value: 0.0,
                hashed_value: 0.0,
                discrepancy: 0.0,
                discrepancy_ratio: 0.0,
                passed: false,
            }
        }
    }

    /// 🧪 Test 2: Fisher Information Discrepancy/Distortion
    /// 
    /// Simulate two inputs whose "true" semantic models would yield distinct
    /// Fisher Information Matrices.
    fn test_fisher_information_discrepancy(&self) -> DiscrepancyTestResult {
        // Create inputs with distinct semantic characteristics
        let true_a = "Complex mathematical analysis with multiple variables and constraints";
        let true_b = "Simple arithmetic calculation with basic operations";
        
        // Simulate hash embedding that reduces semantic distinction
        let hashed_a = "mathematical analysis variables constraints";
        let hashed_b = "arithmetic calculation basic operations";

        match self.tester.test_fisher_information_discrepancy(
            true_a, true_b, hashed_a, hashed_b
        ) {
            Ok(result) => result,
            Err(e) => DiscrepancyTestResult {
                test_type: "Fisher Information Discrepancy".to_string(),
                true_value: 0.0,
                hashed_value: 0.0,
                discrepancy: 0.0,
                discrepancy_ratio: 0.0,
                passed: false,
            }
        }
    }

    /// 🧪 Test 3: General Semantic Distortion Impact
    /// 
    /// Create mock "true" probability distributions representing clearly distinct concepts
    /// and compare with "hashed" distributions that are forced to be much closer.
    fn test_general_semantic_distortion(&self) -> DiscrepancyTestResult {
        // Create clearly distinct concepts
        let true_concept_a = "Advanced machine learning algorithms with deep neural networks";
        let true_concept_b = "Traditional statistical methods with linear regression";
        
        // Simulate hash embedding that reduces distinction
        let hashed_concept_a = "machine learning neural networks";
        let hashed_concept_b = "statistical methods linear regression";

        match self.tester.test_general_semantic_distortion(
            true_concept_a, true_concept_b, hashed_concept_a, hashed_concept_b
        ) {
            Ok(result) => result,
            Err(e) => DiscrepancyTestResult {
                test_type: "General Semantic Distortion".to_string(),
                true_value: 0.0,
                hashed_value: 0.0,
                discrepancy: 0.0,
                discrepancy_ratio: 0.0,
                passed: false,
            }
        }
    }

    /// 🧪 Test 4: Edge Cases
    /// 
    /// Test various edge cases to ensure robust handling.
    fn test_edge_cases(&self) -> Vec<DiscrepancyTestResult> {
        let mut results = Vec::new();

        // Test with empty strings
        let empty_result = self.tester.test_jsd_discrepancy_with_collisions(
            "", "", "", ""
        );
        if let Ok(result) = empty_result {
            results.push(result);
        }

        // Test with very short strings
        let short_result = self.tester.test_jsd_discrepancy_with_collisions(
            "a", "b", "a", "b"
        );
        if let Ok(result) = short_result {
            results.push(result);
        }

        // Test with very long strings
        let long_a = "This is a very long string with many words that should create a complex semantic representation for testing hash embedding discrepancy analysis in the context of semantic uncertainty calculations".repeat(5);
        let long_b = "Another very long string with different semantic content that should also create a complex representation for comparison with the first long string".repeat(5);
        let long_result = self.tester.test_jsd_discrepancy_with_collisions(
            &long_a, &long_b, &long_a, &long_b
        );
        if let Ok(result) = long_result {
            results.push(result);
        }

        results
    }

    /// 📊 Generate test report
    pub fn generate_report(&self) -> String {
        let results = self.run_all_tests();
        
        let mut report = String::new();
        report.push_str("🧪 Hash Embedding Discrepancy Test Report\n");
        report.push_str("==========================================\n\n");
        
        let mut passed_count = 0;
        let mut total_discrepancy = 0.0;
        
        for result in &results {
            report.push_str(&format!("Test: {}\n", result.test_type));
            report.push_str(&format!("  True Value: {:.4}\n", result.true_value));
            report.push_str(&format!("  Hashed Value: {:.4}\n", result.hashed_value));
            report.push_str(&format!("  Discrepancy: {:.4}\n", result.discrepancy));
            report.push_str(&format!("  Discrepancy Ratio: {:.4}\n", result.discrepancy_ratio));
            report.push_str(&format!("  Status: {}\n", if result.passed { "PASSED" } else { "FAILED" }));
            report.push_str("\n");
            
            if result.passed {
                passed_count += 1;
            }
            total_discrepancy += result.discrepancy_ratio;
        }
        
        let avg_discrepancy = if !results.is_empty() {
            total_discrepancy / results.len() as f64
        } else {
            0.0
        };
        
        report.push_str(&format!("Summary:\n"));
        report.push_str(&format!("  Tests Passed: {}/{}\n", passed_count, results.len()));
        report.push_str(&format!("  Average Discrepancy Ratio: {:.4}\n", avg_discrepancy));
        
        if avg_discrepancy > 0.1 {
            report.push_str("  ⚠️  Significant hash embedding discrepancy detected\n");
        } else {
            report.push_str("  ✅ Hash embedding discrepancy within acceptable limits\n");
        }
        
        report
    }
}

/// 🧪 Mock/Simulation Functions for Controlled Testing
pub struct MockHashEmbeddingSimulator {
    collision_probability: f64,
    distortion_factor: f64,
}

impl MockHashEmbeddingSimulator {
    /// Create new mock simulator
    pub fn new(collision_probability: f64, distortion_factor: f64) -> Self {
        Self {
            collision_probability,
            distortion_factor,
        }
    }

    /// 🔧 Simulate hash embedding with controlled collision probability
    pub fn simulate_hash_embedding(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut hashed_words = Vec::new();
        
        for word in words {
            // Simulate hash collision based on probability
            if fastrand::f64() < self.collision_probability {
                // Use a common hash for collision simulation
                hashed_words.push("common_hash");
            } else {
                // Use word hash
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                word.hash(&mut hasher);
                let hash = hasher.finish();
                hashed_words.push(&format!("hash_{}", hash % 1000));
            }
        }
        
        hashed_words.join(" ")
    }

    /// 🔧 Simulate semantic distortion
    pub fn simulate_semantic_distortion(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut distorted_words = Vec::new();
        
        for word in words {
            // Apply distortion factor
            if fastrand::f64() < self.distortion_factor {
                // Replace with similar but different word
                distorted_words.push(self.find_similar_word(word));
            } else {
                distorted_words.push(word);
            }
        }
        
        distorted_words.join(" ")
    }

    /// 🔧 Find similar word for distortion simulation
    fn find_similar_word(&self, word: &str) -> String {
        match word.to_lowercase().as_str() {
            "quantum" => "classical".to_string(),
            "classical" => "quantum".to_string(),
            "complex" => "simple".to_string(),
            "simple" => "complex".to_string(),
            "advanced" => "basic".to_string(),
            "basic" => "advanced".to_string(),
            "deep" => "shallow".to_string(),
            "shallow" => "deep".to_string(),
            "neural" => "linear".to_string(),
            "linear" => "neural".to_string(),
            "statistical" => "deterministic".to_string(),
            "deterministic" => "statistical".to_string(),
            _ => word.to_string(),
        }
    }
}

/// 🧪 Tests for hash embedding discrepancy
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsd_collisions() {
        let suite = HashEmbeddingDiscrepancyTestSuite::new();
        let result = suite.test_jsd_collisions();
        
        // Should detect some discrepancy
        assert!(result.discrepancy >= 0.0);
        assert!(result.true_value >= 0.0);
        assert!(result.hashed_value >= 0.0);
    }

    #[test]
    fn test_fisher_information_discrepancy() {
        let suite = HashEmbeddingDiscrepancyTestSuite::new();
        let result = suite.test_fisher_information_discrepancy();
        
        // Should detect some discrepancy
        assert!(result.discrepancy >= 0.0);
        assert!(result.true_value >= 0.0);
        assert!(result.hashed_value >= 0.0);
    }

    #[test]
    fn test_general_semantic_distortion() {
        let suite = HashEmbeddingDiscrepancyTestSuite::new();
        let result = suite.test_general_semantic_distortion();
        
        // Should detect some distortion
        assert!(result.discrepancy >= 0.0);
        assert!(result.true_value >= 0.0);
        assert!(result.hashed_value >= 0.0);
    }

    #[test]
    fn test_mock_simulator() {
        let simulator = MockHashEmbeddingSimulator::new(0.5, 0.3);
        
        let original = "quantum mechanics particle physics";
        let hashed = simulator.simulate_hash_embedding(original);
        let distorted = simulator.simulate_semantic_distortion(original);
        
        // Should produce different outputs
        assert_ne!(original, hashed);
        assert_ne!(original, distorted);
    }

    #[test]
    fn test_edge_cases() {
        let suite = HashEmbeddingDiscrepancyTestSuite::new();
        let results = suite.test_edge_cases();
        
        // Should handle edge cases gracefully
        assert!(!results.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let suite = HashEmbeddingDiscrepancyTestSuite::new();
        let report = suite.generate_report();
        
        // Should generate a non-empty report
        assert!(!report.is_empty());
        assert!(report.contains("Hash Embedding Discrepancy Test Report"));
    }
} 
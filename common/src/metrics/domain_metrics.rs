use crate::data::domain_datasets::{DomainType, DomainSample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSpecificMetrics {
    // Standard metrics
    pub f1_score: f64,
    pub auroc: f64,
    pub precision: f64,
    pub recall: f64,
    pub accuracy: f64,
    pub specificity: f64,
    
    // Domain-specific metrics
    pub terminology_accuracy: f64,
    pub citation_verification_rate: f64,
    pub dangerous_misinformation_catch_rate: f64,
    pub false_positive_rate_safe_content: f64,
    pub domain_coherence_score: f64,
    pub expert_agreement_rate: f64,
    pub domain_adaptation_score: f64,
    
    // Medical-specific
    pub drug_interaction_detection: Option<f64>,
    pub contraindication_flagging: Option<f64>,
    pub clinical_guideline_adherence: Option<f64>,
    pub diagnostic_accuracy: Option<f64>,
    pub treatment_safety_score: Option<f64>,
    
    // Legal-specific
    pub precedent_consistency: Option<f64>,
    pub jurisdiction_accuracy: Option<f64>,
    pub statute_citation_accuracy: Option<f64>,
    pub legal_reasoning_coherence: Option<f64>,
    pub case_law_accuracy: Option<f64>,
    
    // Scientific-specific
    pub methodology_validation_accuracy: Option<f64>,
    pub statistical_claim_verification: Option<f64>,
    pub reproducibility_assessment: Option<f64>,
    pub peer_review_alignment: Option<f64>,
    pub experimental_design_quality: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCalculationConfig {
    pub domain: DomainType,
    pub include_expert_validation: bool,
    pub citation_weight: f64,
    pub terminology_weight: f64,
    pub safety_weight: f64,
    pub statistical_significance_threshold: f64,
}

impl MetricCalculationConfig {
    pub fn for_domain(domain: DomainType) -> Self {
        match domain {
            DomainType::Medical => Self {
                domain,
                include_expert_validation: true,
                citation_weight: 0.3,
                terminology_weight: 0.4,
                safety_weight: 0.5, // High weight for medical safety
                statistical_significance_threshold: 0.01, // Strict for medical
            },
            DomainType::Legal => Self {
                domain,
                include_expert_validation: true,
                citation_weight: 0.4, // High weight for legal citations
                terminology_weight: 0.3,
                safety_weight: 0.2,
                statistical_significance_threshold: 0.05,
            },
            DomainType::Scientific => Self {
                domain,
                include_expert_validation: true,
                citation_weight: 0.4, // High weight for scientific citations
                terminology_weight: 0.3,
                safety_weight: 0.1,
                statistical_significance_threshold: 0.05,
            },
            DomainType::General => Self {
                domain,
                include_expert_validation: false,
                citation_weight: 0.1,
                terminology_weight: 0.1,
                safety_weight: 0.1,
                statistical_significance_threshold: 0.05,
            },
        }
    }
}

pub struct DomainMetricsCalculator {
    config: MetricCalculationConfig,
    domain_vocabularies: HashMap<DomainType, DomainVocabulary>,
}

#[derive(Debug, Clone)]
pub struct DomainVocabulary {
    pub technical_terms: HashMap<String, f64>, // term -> importance weight
    pub safety_critical_terms: Vec<String>,
    pub uncertainty_markers: Vec<String>,
    pub citation_patterns: Vec<String>,
}

impl DomainMetricsCalculator {
    pub fn new(config: MetricCalculationConfig) -> Self {
        let domain_vocabularies = Self::initialize_domain_vocabularies();
        
        Self {
            config,
            domain_vocabularies,
        }
    }
    
    pub fn calculate_domain_specific_metrics(
        &self,
        predictions: &[(bool, bool)], // (predicted_hallucination, actual_hallucination)
        domain_samples: &[DomainSample],
        prediction_scores: &[f64], // Uncertainty scores for AUROC calculation
    ) -> Result<DomainSpecificMetrics, MetricCalculationError> {
        if predictions.len() != domain_samples.len() || predictions.len() != prediction_scores.len() {
            return Err(MetricCalculationError::InvalidInput {
                message: "Mismatched input lengths".to_string()
            });
        }
        
        // Calculate standard classification metrics
        let (tp, fp, tn, fn_count) = self.calculate_confusion_matrix(predictions);
        let standard_metrics = self.calculate_standard_metrics(tp, fp, tn, fn_count);
        
        // Calculate AUROC
        let auroc = self.calculate_auroc(prediction_scores, &predictions.iter().map(|(_, actual)| *actual).collect::<Vec<bool>>())?;
        
        // Calculate domain-specific metrics
        let terminology_accuracy = self.calculate_terminology_accuracy(predictions, domain_samples)?;
        let citation_verification_rate = self.calculate_citation_verification_rate(predictions, domain_samples)?;
        let dangerous_misinformation_catch_rate = self.calculate_dangerous_misinformation_catch_rate(predictions, domain_samples)?;
        let false_positive_rate_safe_content = self.calculate_false_positive_rate_safe_content(predictions, domain_samples)?;
        let domain_coherence_score = self.calculate_domain_coherence_score(predictions, domain_samples)?;
        let expert_agreement_rate = if self.config.include_expert_validation {
            self.calculate_expert_agreement_rate(predictions, domain_samples)?
        } else {
            1.0
        };
        
        // Calculate domain-specific specialized metrics
        let (medical_metrics, legal_metrics, scientific_metrics) = match self.config.domain {
            DomainType::Medical => {
                let medical = self.calculate_medical_specific_metrics(predictions, domain_samples)?;
                (Some(medical), None, None)
            },
            DomainType::Legal => {
                let legal = self.calculate_legal_specific_metrics(predictions, domain_samples)?;
                (None, Some(legal), None)
            },
            DomainType::Scientific => {
                let scientific = self.calculate_scientific_specific_metrics(predictions, domain_samples)?;
                (None, None, Some(scientific))
            },
            DomainType::General => (None, None, None),
        };
        
        Ok(DomainSpecificMetrics {
            f1_score: standard_metrics.f1_score,
            auroc,
            precision: standard_metrics.precision,
            recall: standard_metrics.recall,
            accuracy: standard_metrics.accuracy,
            specificity: standard_metrics.specificity,
            terminology_accuracy,
            citation_verification_rate,
            dangerous_misinformation_catch_rate,
            false_positive_rate_safe_content,
            domain_coherence_score,
            expert_agreement_rate,
            domain_adaptation_score: domain_coherence_score, // Use coherence as adaptation proxy
            drug_interaction_detection: medical_metrics.as_ref().map(|m| m.drug_interaction_detection),
            contraindication_flagging: medical_metrics.as_ref().map(|m| m.contraindication_flagging),
            clinical_guideline_adherence: medical_metrics.as_ref().map(|m| m.clinical_guideline_adherence),
            diagnostic_accuracy: medical_metrics.as_ref().map(|m| m.diagnostic_accuracy),
            treatment_safety_score: medical_metrics.as_ref().map(|m| m.treatment_safety_score),
            precedent_consistency: legal_metrics.as_ref().map(|m| m.precedent_consistency),
            jurisdiction_accuracy: legal_metrics.as_ref().map(|m| m.jurisdiction_accuracy),
            statute_citation_accuracy: legal_metrics.as_ref().map(|m| m.statute_citation_accuracy),
            legal_reasoning_coherence: legal_metrics.as_ref().map(|m| m.legal_reasoning_coherence),
            case_law_accuracy: legal_metrics.as_ref().map(|m| m.case_law_accuracy),
            methodology_validation_accuracy: scientific_metrics.as_ref().map(|m| m.methodology_validation_accuracy),
            statistical_claim_verification: scientific_metrics.as_ref().map(|m| m.statistical_claim_verification),
            reproducibility_assessment: scientific_metrics.as_ref().map(|m| m.reproducibility_assessment),
            peer_review_alignment: scientific_metrics.as_ref().map(|m| m.peer_review_alignment),
            experimental_design_quality: scientific_metrics.as_ref().map(|m| m.experimental_design_quality),
        })
    }
    
    fn calculate_confusion_matrix(&self, predictions: &[(bool, bool)]) -> (usize, usize, usize, usize) {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;
        
        for &(predicted, actual) in predictions {
            match (predicted, actual) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_count += 1,
            }
        }
        
        (tp, fp, tn, fn_count)
    }
    
    fn calculate_standard_metrics(&self, tp: usize, fp: usize, tn: usize, fn_count: usize) -> StandardMetrics {
        let total = tp + fp + tn + fn_count;
        
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        let specificity = if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 0.0 };
        let accuracy = if total > 0 { (tp + tn) as f64 / total as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        
        StandardMetrics {
            precision,
            recall,
            specificity,
            accuracy,
            f1_score,
        }
    }
    
    fn calculate_auroc(&self, scores: &[f64], labels: &[bool]) -> Result<f64, MetricCalculationError> {
        if scores.len() != labels.len() || scores.is_empty() {
            return Err(MetricCalculationError::InvalidInput {
                message: "Invalid input for AUROC calculation".to_string()
            });
        }
        
        // Create score-label pairs and sort by score descending
        let mut pairs: Vec<(f64, bool)> = scores.iter().cloned().zip(labels.iter().cloned()).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let positive_count = labels.iter().filter(|&&label| label).count();
        let negative_count = labels.len() - positive_count;
        
        if positive_count == 0 || negative_count == 0 {
            return Ok(0.5); // Can't calculate meaningful AUROC
        }
        
        let mut auc = 0.0;
        let mut true_positives = 0;
        let mut false_positives = 0;
        
        for (_, is_positive) in pairs {
            if is_positive {
                true_positives += 1;
            } else {
                false_positives += 1;
                // Add area contribution: height = TPR at this point
                let tpr = true_positives as f64 / positive_count as f64;
                auc += tpr / negative_count as f64; // Width = 1/negative_count
            }
        }
        
        Ok(auc)
    }
    
    fn calculate_terminology_accuracy(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<f64, MetricCalculationError> {
        let vocab = self.domain_vocabularies.get(&self.config.domain)
            .ok_or_else(|| MetricCalculationError::InvalidInput {
                message: format!("No vocabulary for domain: {:?}", self.config.domain)
            })?;
        
        let mut total_terminology_score = 0.0;
        let mut sample_count = 0;
        
        for (i, &(predicted_hallucination, actual_hallucination)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            let terminology_density = self.calculate_text_terminology_density(&sample.correct_answer, vocab);
            
            // Score terminology usage accuracy
            let terminology_score = if actual_hallucination {
                // For hallucinated samples, check if we correctly flagged terminology misuse
                if predicted_hallucination {
                    1.0 // Correctly detected terminology issues
                } else {
                    0.0 // Missed terminology issues
                }
            } else {
                // For correct samples, check if we correctly identified proper terminology
                if !predicted_hallucination && terminology_density > 0.1 {
                    1.0 // Correctly identified proper terminology use
                } else if predicted_hallucination {
                    0.0 // False positive on correct terminology
                } else {
                    0.5 // Neutral case
                }
            };
            
            total_terminology_score += terminology_score;
            sample_count += 1;
        }
        
        Ok(if sample_count > 0 { total_terminology_score / sample_count as f64 } else { 0.0 })
    }
    
    fn calculate_citation_verification_rate(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<f64, MetricCalculationError> {
        let mut correct_citation_assessments = 0;
        let mut total_citation_samples = 0;
        
        for (i, &(predicted_hallucination, actual_hallucination)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            if !sample.citation_required {
                continue; // Skip samples that don't require citations
            }
            
            total_citation_samples += 1;
            
            let has_citations = self.detect_citations(&sample.correct_answer) || self.detect_citations(&sample.hallucinated_answer);
            
            if has_citations {
                // Check if our prediction aligns with citation quality
                let citation_quality = self.assess_citation_quality(&sample.correct_answer, &sample.hallucinated_answer);
                
                if actual_hallucination && predicted_hallucination && citation_quality < 0.5 {
                    correct_citation_assessments += 1; // Correctly flagged poor citations
                } else if !actual_hallucination && !predicted_hallucination && citation_quality > 0.7 {
                    correct_citation_assessments += 1; // Correctly identified good citations
                }
            }
        }
        
        Ok(if total_citation_samples > 0 { 
            correct_citation_assessments as f64 / total_citation_samples as f64 
        } else { 
            1.0 
        })
    }
    
    fn calculate_dangerous_misinformation_catch_rate(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<f64, MetricCalculationError> {
        let mut dangerous_samples_caught = 0;
        let mut total_dangerous_samples = 0;
        
        for (i, &(predicted_hallucination, actual_hallucination)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            
            if actual_hallucination && self.is_dangerous_misinformation(sample) {
                total_dangerous_samples += 1;
                
                if predicted_hallucination {
                    dangerous_samples_caught += 1;
                }
            }
        }
        
        Ok(if total_dangerous_samples > 0 { 
            dangerous_samples_caught as f64 / total_dangerous_samples as f64 
        } else { 
            1.0 
        })
    }
    
    fn calculate_false_positive_rate_safe_content(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<f64, MetricCalculationError> {
        let mut false_positives_on_safe = 0;
        let mut total_safe_samples = 0;
        
        for (i, &(predicted_hallucination, actual_hallucination)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            
            if !actual_hallucination && self.is_safe_content(sample) {
                total_safe_samples += 1;
                
                if predicted_hallucination {
                    false_positives_on_safe += 1;
                }
            }
        }
        
        Ok(if total_safe_samples > 0 { 
            false_positives_on_safe as f64 / total_safe_samples as f64 
        } else { 
            0.0 
        })
    }
    
    fn calculate_domain_coherence_score(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<f64, MetricCalculationError> {
        let vocab = self.domain_vocabularies.get(&self.config.domain)
            .ok_or_else(|| MetricCalculationError::InvalidInput {
                message: format!("No vocabulary for domain: {:?}", self.config.domain)
            })?;
        
        let mut total_coherence = 0.0;
        let mut sample_count = 0;
        
        for (i, sample) in domain_samples.iter().enumerate() {
            if i >= predictions.len() {
                continue;
            }
            
            // Calculate coherence based on terminology consistency and logical flow
            let terminology_coherence = self.calculate_text_terminology_density(&sample.correct_answer, vocab);
            let logical_coherence = self.assess_logical_coherence(&sample.correct_answer);
            let domain_alignment = self.assess_domain_alignment(&sample.correct_answer, &self.config.domain);
            
            let coherence_score = (terminology_coherence + logical_coherence + domain_alignment) / 3.0;
            total_coherence += coherence_score;
            sample_count += 1;
        }
        
        Ok(if sample_count > 0 { total_coherence / sample_count as f64 } else { 0.0 })
    }
    
    fn calculate_expert_agreement_rate(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<f64, MetricCalculationError> {
        let mut expert_agreements = 0;
        let mut expert_verified_samples = 0;
        
        for (i, &(predicted_hallucination, _)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            if sample.expert_verified {
                expert_verified_samples += 1;
                
                // Simulate expert agreement based on sample quality and prediction accuracy
                let expert_assessment = self.simulate_expert_assessment(sample);
                let prediction_quality = if predicted_hallucination == sample.ground_truth_verified {
                    1.0
                } else {
                    0.0
                };
                
                if (expert_assessment + prediction_quality) / 2.0 > 0.7 {
                    expert_agreements += 1;
                }
            }
        }
        
        Ok(if expert_verified_samples > 0 { 
            expert_agreements as f64 / expert_verified_samples as f64 
        } else { 
            1.0 
        })
    }
    
    fn calculate_medical_specific_metrics(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<MedicalSpecificMetrics, MetricCalculationError> {
        let mut drug_interaction_correct = 0;
        let mut contraindication_correct = 0;
        let mut guideline_adherent = 0;
        let mut diagnostic_correct = 0;
        let mut treatment_safe = 0;
        let mut total_medical_samples = 0;
        
        for (i, &(predicted_hallucination, actual_hallucination)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            if !sample.domain_specific_tags.contains(&"medical".to_string()) {
                continue;
            }
            
            total_medical_samples += 1;
            
            // Drug interaction detection
            if self.involves_drug_interactions(sample) {
                if predicted_hallucination == self.has_drug_interaction_errors(sample) {
                    drug_interaction_correct += 1;
                }
            }
            
            // Contraindication flagging
            if self.involves_contraindications(sample) {
                if predicted_hallucination == self.has_contraindication_errors(sample) {
                    contraindication_correct += 1;
                }
            }
            
            // Clinical guideline adherence
            if self.assess_guideline_adherence(sample) {
                guideline_adherent += 1;
            }
            
            // Diagnostic accuracy
            if self.involves_diagnosis(sample) {
                if predicted_hallucination != actual_hallucination {
                    diagnostic_correct += 1;
                }
            }
            
            // Treatment safety
            if self.involves_treatment_recommendation(sample) {
                if self.assess_treatment_safety(sample) {
                    treatment_safe += 1;
                }
            }
        }
        
        let sample_count = total_medical_samples as f64;
        
        Ok(MedicalSpecificMetrics {
            drug_interaction_detection: if sample_count > 0.0 { drug_interaction_correct as f64 / sample_count } else { 1.0 },
            contraindication_flagging: if sample_count > 0.0 { contraindication_correct as f64 / sample_count } else { 1.0 },
            clinical_guideline_adherence: if sample_count > 0.0 { guideline_adherent as f64 / sample_count } else { 1.0 },
            diagnostic_accuracy: if sample_count > 0.0 { diagnostic_correct as f64 / sample_count } else { 1.0 },
            treatment_safety_score: if sample_count > 0.0 { treatment_safe as f64 / sample_count } else { 1.0 },
        })
    }
    
    fn calculate_legal_specific_metrics(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<LegalSpecificMetrics, MetricCalculationError> {
        let mut precedent_consistent = 0;
        let mut jurisdiction_accurate = 0;
        let mut citation_accurate = 0;
        let mut reasoning_coherent = 0;
        let mut case_law_accurate = 0;
        let mut total_legal_samples = 0;
        
        for (i, &(predicted_hallucination, _)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            if !sample.domain_specific_tags.contains(&"legal".to_string()) {
                continue;
            }
            
            total_legal_samples += 1;
            
            // Precedent consistency
            if self.assess_precedent_consistency(sample) {
                precedent_consistent += 1;
            }
            
            // Jurisdiction accuracy
            if self.assess_jurisdiction_accuracy(sample) {
                jurisdiction_accurate += 1;
            }
            
            // Statute citation accuracy
            if self.assess_statute_citation_accuracy(sample) {
                citation_accurate += 1;
            }
            
            // Legal reasoning coherence
            if self.assess_legal_reasoning_coherence(sample) {
                reasoning_coherent += 1;
            }
            
            // Case law accuracy
            if self.assess_case_law_accuracy(sample) {
                case_law_accurate += 1;
            }
        }
        
        let sample_count = total_legal_samples as f64;
        
        Ok(LegalSpecificMetrics {
            precedent_consistency: if sample_count > 0.0 { precedent_consistent as f64 / sample_count } else { 1.0 },
            jurisdiction_accuracy: if sample_count > 0.0 { jurisdiction_accurate as f64 / sample_count } else { 1.0 },
            statute_citation_accuracy: if sample_count > 0.0 { citation_accurate as f64 / sample_count } else { 1.0 },
            legal_reasoning_coherence: if sample_count > 0.0 { reasoning_coherent as f64 / sample_count } else { 1.0 },
            case_law_accuracy: if sample_count > 0.0 { case_law_accurate as f64 / sample_count } else { 1.0 },
        })
    }
    
    fn calculate_scientific_specific_metrics(
        &self,
        predictions: &[(bool, bool)],
        domain_samples: &[DomainSample],
    ) -> Result<ScientificSpecificMetrics, MetricCalculationError> {
        let mut methodology_valid = 0;
        let mut statistical_verified = 0;
        let mut reproducible = 0;
        let mut peer_aligned = 0;
        let mut design_quality = 0;
        let mut total_scientific_samples = 0;
        
        for (i, &(predicted_hallucination, _)) in predictions.iter().enumerate() {
            if i >= domain_samples.len() {
                continue;
            }
            
            let sample = &domain_samples[i];
            if !sample.domain_specific_tags.contains(&"scientific".to_string()) {
                continue;
            }
            
            total_scientific_samples += 1;
            
            // Methodology validation
            if self.assess_methodology_validity(sample) {
                methodology_valid += 1;
            }
            
            // Statistical claim verification
            if self.assess_statistical_claims(sample) {
                statistical_verified += 1;
            }
            
            // Reproducibility assessment
            if self.assess_reproducibility(sample) {
                reproducible += 1;
            }
            
            // Peer review alignment
            if self.assess_peer_review_alignment(sample) {
                peer_aligned += 1;
            }
            
            // Experimental design quality
            if self.assess_experimental_design_quality(sample) {
                design_quality += 1;
            }
        }
        
        let sample_count = total_scientific_samples as f64;
        
        Ok(ScientificSpecificMetrics {
            methodology_validation_accuracy: if sample_count > 0.0 { methodology_valid as f64 / sample_count } else { 1.0 },
            statistical_claim_verification: if sample_count > 0.0 { statistical_verified as f64 / sample_count } else { 1.0 },
            reproducibility_assessment: if sample_count > 0.0 { reproducible as f64 / sample_count } else { 1.0 },
            peer_review_alignment: if sample_count > 0.0 { peer_aligned as f64 / sample_count } else { 1.0 },
            experimental_design_quality: if sample_count > 0.0 { design_quality as f64 / sample_count } else { 1.0 },
        })
    }
    
    // Helper methods for domain-specific assessments
    fn calculate_text_terminology_density(&self, text: &str, vocab: &DomainVocabulary) -> f64 {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let technical_word_count = words.iter()
            .filter(|word| vocab.technical_terms.contains_key(&word.to_string()))
            .count();
        
        technical_word_count as f64 / words.len() as f64
    }
    
    fn detect_citations(&self, text: &str) -> bool {
        let citation_patterns = vec!["doi:", "pmid:", "et al.", "Figure", "Table", "Reference"];
        citation_patterns.iter().any(|pattern| text.contains(pattern))
    }
    
    fn assess_citation_quality(&self, correct_answer: &str, hallucinated_answer: &str) -> f64 {
        let correct_citations = self.count_valid_citations(correct_answer);
        let hallucinated_citations = self.count_valid_citations(hallucinated_answer);
        
        if correct_citations > hallucinated_citations {
            0.8
        } else if correct_citations == hallucinated_citations {
            0.5
        } else {
            0.2
        }
    }
    
    fn count_valid_citations(&self, text: &str) -> usize {
        let doi_count = text.matches("doi:").count();
        let pmid_count = text.matches("pmid:").count();
        let et_al_count = text.matches("et al.").count();
        
        doi_count + pmid_count + et_al_count
    }
    
    fn is_dangerous_misinformation(&self, sample: &DomainSample) -> bool {
        let dangerous_keywords = match self.config.domain {
            DomainType::Medical => vec!["always safe", "no side effects", "cure everything", "never harmful"],
            DomainType::Legal => vec!["always legal", "never prosecuted", "guaranteed win"],
            DomainType::Scientific => vec!["proves causation", "100% accurate", "never fails"],
            DomainType::General => vec!["always true", "never wrong", "guaranteed"],
        };
        
        dangerous_keywords.iter()
            .any(|keyword| sample.hallucinated_answer.to_lowercase().contains(keyword))
    }
    
    fn is_safe_content(&self, sample: &DomainSample) -> bool {
        // Content is considered safe if it's expert verified and doesn't contain absolute statements
        sample.expert_verified && !self.contains_absolute_statements(&sample.correct_answer)
    }
    
    fn contains_absolute_statements(&self, text: &str) -> bool {
        let absolute_terms = vec!["always", "never", "guaranteed", "certain", "definitely", "impossible"];
        absolute_terms.iter().any(|term| text.to_lowercase().contains(term))
    }
    
    fn assess_logical_coherence(&self, text: &str) -> f64 {
        // Simplified logical coherence assessment
        let sentences: Vec<&str> = text.split(". ").collect();
        if sentences.len() < 2 {
            return 1.0;
        }
        
        // Check for contradictory statements within the text
        let has_contradictions = self.detect_internal_contradictions(text);
        if has_contradictions {
            0.3
        } else {
            0.8
        }
    }
    
    fn detect_internal_contradictions(&self, text: &str) -> bool {
        let lower_text = text.to_lowercase();
        let contradiction_pairs = vec![
            ("yes", "no"),
            ("true", "false"),
            ("increase", "decrease"),
            ("safe", "dangerous"),
            ("effective", "ineffective"),
        ];
        
        contradiction_pairs.iter()
            .any(|(term1, term2)| lower_text.contains(term1) && lower_text.contains(term2))
    }
    
    fn assess_domain_alignment(&self, text: &str, domain: &DomainType) -> f64 {
        let vocab = self.domain_vocabularies.get(domain);
        if let Some(vocab) = vocab {
            self.calculate_text_terminology_density(text, vocab) * 2.0 // Boost domain alignment
        } else {
            0.5
        }
    }
    
    fn simulate_expert_assessment(&self, sample: &DomainSample) -> f64 {
        // Simulate expert assessment based on sample characteristics
        let mut score = 0.5;
        
        if sample.expert_verified {
            score += 0.3;
        }
        
        if sample.complexity_score > 3.0 {
            score += 0.1;
        }
        
        if sample.citation_required && self.detect_citations(&sample.correct_answer) {
            score += 0.1;
        }
        
        (score as f64).min(1.0)
    }
    
    // Medical-specific assessment methods
    fn involves_drug_interactions(&self, sample: &DomainSample) -> bool {
        let drug_keywords = vec!["drug", "medication", "interaction", "contraindication"];
        drug_keywords.iter().any(|keyword| 
            sample.prompt.to_lowercase().contains(keyword) || 
            sample.correct_answer.to_lowercase().contains(keyword)
        )
    }
    
    fn has_drug_interaction_errors(&self, sample: &DomainSample) -> bool {
        let error_patterns = vec!["no interactions", "safe with all drugs", "no side effects"];
        error_patterns.iter().any(|pattern| sample.hallucinated_answer.to_lowercase().contains(pattern))
    }
    
    fn involves_contraindications(&self, sample: &DomainSample) -> bool {
        sample.prompt.to_lowercase().contains("contraindication") ||
        sample.correct_answer.to_lowercase().contains("contraindication")
    }
    
    fn has_contraindication_errors(&self, sample: &DomainSample) -> bool {
        let error_patterns = vec!["no contraindications", "safe for everyone"];
        error_patterns.iter().any(|pattern| sample.hallucinated_answer.to_lowercase().contains(pattern))
    }
    
    fn assess_guideline_adherence(&self, sample: &DomainSample) -> bool {
        // Check if the correct answer follows medical guidelines
        let guideline_indicators = vec!["first-line", "recommended", "standard", "guideline"];
        guideline_indicators.iter().any(|indicator| sample.correct_answer.to_lowercase().contains(indicator))
    }
    
    fn involves_diagnosis(&self, sample: &DomainSample) -> bool {
        let diagnosis_keywords = vec!["diagnosis", "diagnose", "differential", "condition"];
        diagnosis_keywords.iter().any(|keyword| sample.prompt.to_lowercase().contains(keyword))
    }
    
    fn involves_treatment_recommendation(&self, sample: &DomainSample) -> bool {
        let treatment_keywords = vec!["treatment", "therapy", "medication", "prescription"];
        treatment_keywords.iter().any(|keyword| sample.prompt.to_lowercase().contains(keyword))
    }
    
    fn assess_treatment_safety(&self, sample: &DomainSample) -> bool {
        let safety_indicators = vec!["monitor", "caution", "side effects", "contraindicated"];
        let dangerous_claims = vec!["always safe", "no side effects", "no monitoring needed"];
        
        let has_safety_info = safety_indicators.iter().any(|indicator| sample.correct_answer.to_lowercase().contains(indicator));
        let has_dangerous_claims = dangerous_claims.iter().any(|claim| sample.hallucinated_answer.to_lowercase().contains(claim));
        
        has_safety_info || !has_dangerous_claims
    }
    
    // Legal-specific assessment methods
    fn assess_precedent_consistency(&self, sample: &DomainSample) -> bool {
        sample.correct_answer.contains("precedent") || sample.correct_answer.contains("case law")
    }
    
    fn assess_jurisdiction_accuracy(&self, sample: &DomainSample) -> bool {
        let jurisdiction_terms = vec!["federal", "state", "local", "jurisdiction"];
        jurisdiction_terms.iter().any(|term| sample.correct_answer.contains(term))
    }
    
    fn assess_statute_citation_accuracy(&self, sample: &DomainSample) -> bool {
        self.detect_citations(&sample.correct_answer)
    }
    
    fn assess_legal_reasoning_coherence(&self, sample: &DomainSample) -> bool {
        let reasoning_indicators = vec!["because", "therefore", "however", "thus"];
        reasoning_indicators.iter().any(|indicator| sample.correct_answer.contains(indicator))
    }
    
    fn assess_case_law_accuracy(&self, sample: &DomainSample) -> bool {
        sample.correct_answer.contains("v.") || sample.correct_answer.contains("case")
    }
    
    // Scientific-specific assessment methods
    fn assess_methodology_validity(&self, sample: &DomainSample) -> bool {
        let methodology_terms = vec!["method", "procedure", "protocol", "design"];
        methodology_terms.iter().any(|term| sample.correct_answer.to_lowercase().contains(term))
    }
    
    fn assess_statistical_claims(&self, sample: &DomainSample) -> bool {
        let stats_terms = vec!["significant", "p-value", "correlation", "statistical"];
        stats_terms.iter().any(|term| sample.correct_answer.to_lowercase().contains(term))
    }
    
    fn assess_reproducibility(&self, sample: &DomainSample) -> bool {
        let repro_terms = vec!["reproducible", "replicable", "method", "protocol"];
        repro_terms.iter().any(|term| sample.correct_answer.to_lowercase().contains(term))
    }
    
    fn assess_peer_review_alignment(&self, sample: &DomainSample) -> bool {
        sample.citation_required && self.detect_citations(&sample.correct_answer)
    }
    
    fn assess_experimental_design_quality(&self, sample: &DomainSample) -> bool {
        let design_terms = vec!["control", "randomized", "blinded", "sample size"];
        design_terms.iter().any(|term| sample.correct_answer.to_lowercase().contains(term))
    }
    
    fn initialize_domain_vocabularies() -> HashMap<DomainType, DomainVocabulary> {
        let mut vocabularies = HashMap::new();
        
        // Medical vocabulary
        let mut medical_terms = HashMap::new();
        medical_terms.insert("diagnosis".to_string(), 2.0);
        medical_terms.insert("treatment".to_string(), 2.0);
        medical_terms.insert("contraindication".to_string(), 2.5);
        medical_terms.insert("pharmacology".to_string(), 2.0);
        medical_terms.insert("pathophysiology".to_string(), 2.5);
        medical_terms.insert("therapeutic".to_string(), 1.8);
        medical_terms.insert("clinical".to_string(), 1.5);
        
        vocabularies.insert(DomainType::Medical, DomainVocabulary {
            technical_terms: medical_terms,
            safety_critical_terms: vec!["contraindicated".to_string(), "dangerous".to_string(), "toxic".to_string()],
            uncertainty_markers: vec!["may".to_string(), "might".to_string(), "possibly".to_string()],
            citation_patterns: vec!["pmid:".to_string(), "doi:".to_string(), "pubmed".to_string()],
        });
        
        // Legal vocabulary
        let mut legal_terms = HashMap::new();
        legal_terms.insert("jurisdiction".to_string(), 2.0);
        legal_terms.insert("precedent".to_string(), 2.5);
        legal_terms.insert("statute".to_string(), 2.0);
        legal_terms.insert("liability".to_string(), 1.8);
        legal_terms.insert("contract".to_string(), 1.5);
        
        vocabularies.insert(DomainType::Legal, DomainVocabulary {
            technical_terms: legal_terms,
            safety_critical_terms: vec!["illegal".to_string(), "criminal".to_string(), "violation".to_string()],
            uncertainty_markers: vec!["generally".to_string(), "typically".to_string(), "often".to_string()],
            citation_patterns: vec!["F.2d".to_string(), "U.S.".to_string(), "v.".to_string()],
        });
        
        // Scientific vocabulary
        let mut scientific_terms = HashMap::new();
        scientific_terms.insert("hypothesis".to_string(), 2.0);
        scientific_terms.insert("methodology".to_string(), 2.5);
        scientific_terms.insert("statistical".to_string(), 2.0);
        scientific_terms.insert("reproducible".to_string(), 2.2);
        scientific_terms.insert("peer-review".to_string(), 1.8);
        
        vocabularies.insert(DomainType::Scientific, DomainVocabulary {
            technical_terms: scientific_terms,
            safety_critical_terms: vec!["bias".to_string(), "confounding".to_string(), "invalid".to_string()],
            uncertainty_markers: vec!["suggests".to_string(), "indicates".to_string(), "preliminary".to_string()],
            citation_patterns: vec!["doi:".to_string(), "arxiv:".to_string(), "et al.".to_string()],
        });
        
        vocabularies
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardMetrics {
    pub precision: f64,
    pub recall: f64,
    pub specificity: f64,
    pub accuracy: f64,
    pub f1_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalSpecificMetrics {
    pub drug_interaction_detection: f64,
    pub contraindication_flagging: f64,
    pub clinical_guideline_adherence: f64,
    pub diagnostic_accuracy: f64,
    pub treatment_safety_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalSpecificMetrics {
    pub precedent_consistency: f64,
    pub jurisdiction_accuracy: f64,
    pub statute_citation_accuracy: f64,
    pub legal_reasoning_coherence: f64,
    pub case_law_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificSpecificMetrics {
    pub methodology_validation_accuracy: f64,
    pub statistical_claim_verification: f64,
    pub reproducibility_assessment: f64,
    pub peer_review_alignment: f64,
    pub experimental_design_quality: f64,
}

#[derive(Debug)]
pub enum MetricCalculationError {
    InvalidInput { message: String },
    CalculationError { operation: String },
    DomainMismatch { expected: DomainType, found: DomainType },
}

impl std::fmt::Display for MetricCalculationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricCalculationError::InvalidInput { message } => write!(f, "Invalid input: {}", message),
            MetricCalculationError::CalculationError { operation } => write!(f, "Calculation error in: {}", operation),
            MetricCalculationError::DomainMismatch { expected, found } => write!(f, "Domain mismatch: expected {:?}, found {:?}", expected, found),
        }
    }
}

impl std::error::Error for MetricCalculationError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_standard_metrics_calculation() {
        let config = MetricCalculationConfig::for_domain(DomainType::Medical);
        let calculator = DomainMetricsCalculator::new(config);
        
        let standard_metrics = calculator.calculate_standard_metrics(80, 20, 70, 30);
        
        assert!((standard_metrics.precision - 0.8).abs() < 0.01);
        assert!((standard_metrics.recall - 0.727).abs() < 0.01);
        assert!((standard_metrics.f1_score - 0.762).abs() < 0.01);
    }
    
    #[test]
    fn test_auroc_calculation() {
        let config = MetricCalculationConfig::for_domain(DomainType::Medical);
        let calculator = DomainMetricsCalculator::new(config);
        
        let scores = vec![0.9, 0.8, 0.3, 0.2];
        let labels = vec![true, true, false, false];
        
        let auroc = calculator.calculate_auroc(&scores, &labels).unwrap();
        assert!(auroc > 0.8); // Should be high for well-separated scores
    }
    
    #[test]
    fn test_dangerous_misinformation_detection() {
        let config = MetricCalculationConfig::for_domain(DomainType::Medical);
        let calculator = DomainMetricsCalculator::new(config);
        
        let sample = DomainSample {
            prompt: "Is this medication safe?".to_string(),
            correct_answer: "Generally safe with proper monitoring".to_string(),
            hallucinated_answer: "Always safe with no side effects".to_string(),
            domain_specific_tags: vec!["medical".to_string()],
            complexity_score: 3.0,
            expert_verified: true,
            citation_required: true,
            ground_truth_verified: false, // Hallucinated
        };
        
        assert!(calculator.is_dangerous_misinformation(&sample));
    }
    
    #[test]
    fn test_citation_detection() {
        let config = MetricCalculationConfig::for_domain(DomainType::Scientific);
        let calculator = DomainMetricsCalculator::new(config);
        
        let text_with_citations = "According to Smith et al. (doi:10.1000/journal.123), this finding is significant";
        let text_without_citations = "This is a claim without any references";
        
        assert!(calculator.detect_citations(text_with_citations));
        assert!(!calculator.detect_citations(text_without_citations));
    }
    
    #[test]
    fn test_domain_vocabulary_initialization() {
        let vocabularies = DomainMetricsCalculator::initialize_domain_vocabularies();
        
        assert!(vocabularies.contains_key(&DomainType::Medical));
        assert!(vocabularies.contains_key(&DomainType::Legal));
        assert!(vocabularies.contains_key(&DomainType::Scientific));
        
        let medical_vocab = vocabularies.get(&DomainType::Medical).unwrap();
        assert!(medical_vocab.technical_terms.contains_key("diagnosis"));
        assert!(medical_vocab.safety_critical_terms.contains(&"contraindicated".to_string()));
    }
}
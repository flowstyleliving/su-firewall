use crate::math::semantic_entropy::{SemanticEntropyCalculator, SemanticEntropyConfig, SemanticCluster, UncertaintyLevel};
use crate::data::domain_datasets::DomainType;
use crate::error::SemanticError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSemanticEntropyConfig {
    pub domain: DomainType,
    pub similarity_threshold: f64,
    pub terminology_weight: f64,
    pub citation_validation: bool,
    pub expert_knowledge_base: Option<String>,
    pub domain_specific_clustering: bool,
    pub contradiction_penalty_multiplier: f64,
}

impl DomainSemanticEntropyConfig {
    pub fn for_domain(domain: DomainType) -> Self {
        match domain {
            DomainType::Medical => Self {
                domain,
                similarity_threshold: 0.6, // Higher threshold for medical safety
                terminology_weight: 2.0,   // Weight medical terms heavily
                citation_validation: true,
                expert_knowledge_base: Some("umls".to_string()),
                domain_specific_clustering: true,
                contradiction_penalty_multiplier: 2.5, // Strong penalty for medical contradictions
            },
            DomainType::Legal => Self {
                domain,
                similarity_threshold: 0.55,
                terminology_weight: 1.8,
                citation_validation: true,
                expert_knowledge_base: Some("legal_precedents".to_string()),
                domain_specific_clustering: true,
                contradiction_penalty_multiplier: 2.0,
            },
            DomainType::Scientific => Self {
                domain,
                similarity_threshold: 0.5,
                terminology_weight: 1.5,
                citation_validation: true,
                expert_knowledge_base: Some("scientific_corpus".to_string()),
                domain_specific_clustering: true,
                contradiction_penalty_multiplier: 1.8,
            },
            DomainType::General => Self {
                domain,
                similarity_threshold: 0.5,
                terminology_weight: 1.0,
                citation_validation: false,
                expert_knowledge_base: None,
                domain_specific_clustering: false,
                contradiction_penalty_multiplier: 1.0,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSemanticEntropyResult {
    pub base_entropy: f64,
    pub domain_adjusted_entropy: f64,
    pub terminology_confidence: f64,
    pub citation_accuracy: f64,
    pub expert_validation_score: Option<f64>,
    pub domain_specific_uncertainty: f64,
    pub clustering_quality: f64,
    pub domain_adjustments: DomainAdjustments,
    pub uncertainty_level: UncertaintyLevel,
    pub processing_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdjustments {
    pub entropy_multiplier: f64,
    pub terminology_confidence: f64,
    pub citation_accuracy: f64,
    pub expert_validation: Option<f64>,
    pub domain_uncertainty: f64,
    pub contradiction_penalty: f64,
}

pub struct DomainSemanticEntropyCalculator {
    base_calculator: SemanticEntropyCalculator,
    domain_configs: HashMap<DomainType, DomainSemanticEntropyConfig>,
    domain_vocabularies: HashMap<DomainType, HashSet<String>>,
    medical_ontology: MedicalOntology,
    legal_knowledge: LegalKnowledge,
    scientific_corpus: ScientificCorpus,
}

impl DomainSemanticEntropyCalculator {
    pub fn new() -> Self {
        let base_config = SemanticEntropyConfig::default();
        let base_calculator = SemanticEntropyCalculator::new(base_config);
        
        let mut domain_configs = HashMap::new();
        domain_configs.insert(DomainType::Medical, DomainSemanticEntropyConfig::for_domain(DomainType::Medical));
        domain_configs.insert(DomainType::Legal, DomainSemanticEntropyConfig::for_domain(DomainType::Legal));
        domain_configs.insert(DomainType::Scientific, DomainSemanticEntropyConfig::for_domain(DomainType::Scientific));
        domain_configs.insert(DomainType::General, DomainSemanticEntropyConfig::for_domain(DomainType::General));
        
        let domain_vocabularies = Self::initialize_domain_vocabularies();
        
        Self {
            base_calculator,
            domain_configs,
            domain_vocabularies,
            medical_ontology: MedicalOntology::new(),
            legal_knowledge: LegalKnowledge::new(),
            scientific_corpus: ScientificCorpus::new(),
        }
    }
    
    pub async fn calculate_domain_semantic_entropy(
        &mut self,
        prompt: &str,
        responses: &[String],
        probabilities: &[f64],
        domain: DomainType,
    ) -> Result<DomainSemanticEntropyResult, SemanticError> {
        let start_time = std::time::Instant::now();
        
        let config = self.domain_configs.get(&domain)
            .ok_or_else(|| SemanticError::InvalidInput {
                message: format!("No configuration for domain: {:?}", domain)
            })?.clone();
        
        // Calculate base semantic entropy
        let base_result = self.base_calculator.calculate_semantic_entropy(responses, probabilities)?;
        
        // Domain-specific clustering
        let domain_clusters = self.domain_aware_clustering(responses, probabilities, domain.clone(), &config).await?;
        
        // Calculate domain-adjusted entropy
        let domain_entropy = self.calculate_entropy_over_domain_clusters(&domain_clusters)?;
        
        // Domain-specific adjustments
        let domain_adjustments = self.calculate_domain_adjustments(
            prompt, responses, domain, &config
        ).await?;
        
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(DomainSemanticEntropyResult {
            base_entropy: base_result.semantic_entropy,
            domain_adjusted_entropy: domain_entropy * domain_adjustments.entropy_multiplier,
            terminology_confidence: domain_adjustments.terminology_confidence,
            citation_accuracy: domain_adjustments.citation_accuracy,
            expert_validation_score: domain_adjustments.expert_validation,
            domain_specific_uncertainty: domain_adjustments.domain_uncertainty,
            clustering_quality: self.assess_clustering_quality(&domain_clusters),
            domain_adjustments: domain_adjustments.clone(),
            uncertainty_level: UncertaintyLevel::from_entropy(domain_entropy * domain_adjustments.entropy_multiplier),
            processing_time_ms,
        })
    }
    
    async fn domain_aware_clustering(
        &mut self,
        responses: &[String],
        probabilities: &[f64],
        domain: DomainType,
        config: &DomainSemanticEntropyConfig,
    ) -> Result<Vec<DomainSemanticCluster>, SemanticError> {
        match domain {
            DomainType::Medical => self.medical_clustering(responses, probabilities, config).await,
            DomainType::Legal => self.legal_clustering(responses, probabilities, config).await,
            DomainType::Scientific => self.scientific_clustering(responses, probabilities, config).await,
            DomainType::General => {
                let base_clusters = self.base_calculator.cluster_by_semantic_similarity(responses, probabilities)?;
                Ok(base_clusters.into_iter().map(DomainSemanticCluster::from_base).collect())
            }
        }
    }
    
    async fn medical_clustering(
        &mut self,
        responses: &[String],
        probabilities: &[f64],
        config: &DomainSemanticEntropyConfig,
    ) -> Result<Vec<DomainSemanticCluster>, SemanticError> {
        let mut clusters: Vec<DomainSemanticCluster> = Vec::new();
        
        for (i, response) in responses.iter().enumerate() {
            let response_prob = probabilities[i];
            let mut assigned_to_cluster = false;
            
            // Extract medical information
            let medical_terms = self.medical_ontology.extract_medical_terms(response);
            let drug_interactions = self.medical_ontology.check_drug_interactions(&medical_terms).await?;
            let contraindications = self.medical_ontology.check_contraindications(&medical_terms).await?;
            
            // Try to assign to existing cluster with medical-aware similarity
            for cluster in &mut clusters {
                let medical_similarity = self.compute_medical_similarity(
                    response, &cluster.representative, &medical_terms, config
                ).await?;
                
                if medical_similarity >= config.similarity_threshold {
                    cluster.answers.push(response.clone());
                    cluster.probability_mass += response_prob;
                    cluster.coherence_score = (cluster.coherence_score + medical_similarity) / 2.0;
                    cluster.medical_info.as_mut().unwrap().merge_drug_interactions(&drug_interactions);
                    cluster.medical_info.as_mut().unwrap().merge_contraindications(&contraindications);
                    assigned_to_cluster = true;
                    break;
                }
            }
            
            // Create new cluster if no match found
            if !assigned_to_cluster {
                clusters.push(DomainSemanticCluster {
                    answers: vec![response.clone()],
                    probability_mass: response_prob,
                    representative: response.clone(),
                    coherence_score: 1.0,
                    domain_specific_score: 1.0,
                    medical_info: Some(MedicalClusterInfo {
                        medical_terms: medical_terms.clone(),
                        drug_interactions,
                        contraindications,
                        safety_flags: self.medical_ontology.assess_safety_flags(response),
                    }),
                    legal_info: None,
                    scientific_info: None,
                });
            }
        }
        
        Ok(clusters)
    }
    
    async fn legal_clustering(
        &mut self,
        responses: &[String],
        probabilities: &[f64],
        config: &DomainSemanticEntropyConfig,
    ) -> Result<Vec<DomainSemanticCluster>, SemanticError> {
        let mut clusters: Vec<DomainSemanticCluster> = Vec::new();
        
        for (i, response) in responses.iter().enumerate() {
            let response_prob = probabilities[i];
            let mut assigned_to_cluster = false;
            
            // Extract legal information
            let legal_citations = self.legal_knowledge.extract_legal_citations(response);
            let precedent_consistency = self.legal_knowledge.check_precedent_consistency(&legal_citations).await?;
            let jurisdiction_accuracy = self.legal_knowledge.validate_jurisdiction_claims(response).await?;
            
            // Try to assign to existing cluster with legal-aware similarity
            for cluster in &mut clusters {
                let legal_similarity = self.compute_legal_similarity(
                    response, &cluster.representative, &legal_citations, config
                ).await?;
                
                if legal_similarity >= config.similarity_threshold {
                    cluster.answers.push(response.clone());
                    cluster.probability_mass += response_prob;
                    cluster.coherence_score = (cluster.coherence_score + legal_similarity) / 2.0;
                    assigned_to_cluster = true;
                    break;
                }
            }
            
            // Create new cluster if no match found
            if !assigned_to_cluster {
                clusters.push(DomainSemanticCluster {
                    answers: vec![response.clone()],
                    probability_mass: response_prob,
                    representative: response.clone(),
                    coherence_score: 1.0,
                    domain_specific_score: 1.0,
                    medical_info: None,
                    legal_info: Some(LegalClusterInfo {
                        legal_citations,
                        precedent_consistency,
                        jurisdiction_accuracy,
                        statute_references: self.legal_knowledge.extract_statute_references(response),
                    }),
                    scientific_info: None,
                });
            }
        }
        
        Ok(clusters)
    }
    
    async fn scientific_clustering(
        &mut self,
        responses: &[String],
        probabilities: &[f64],
        config: &DomainSemanticEntropyConfig,
    ) -> Result<Vec<DomainSemanticCluster>, SemanticError> {
        let mut clusters: Vec<DomainSemanticCluster> = Vec::new();
        
        for (i, response) in responses.iter().enumerate() {
            let response_prob = probabilities[i];
            let mut assigned_to_cluster = false;
            
            // Extract scientific information
            let scientific_claims = self.scientific_corpus.extract_scientific_claims(response);
            let methodology_validation = self.scientific_corpus.validate_methodology(&scientific_claims).await?;
            let statistical_accuracy = self.scientific_corpus.check_statistical_claims(&scientific_claims).await?;
            let citation_verification = self.scientific_corpus.verify_scientific_citations(response).await?;
            
            // Try to assign to existing cluster with scientific-aware similarity
            for cluster in &mut clusters {
                let scientific_similarity = self.compute_scientific_similarity(
                    response, &cluster.representative, &scientific_claims, config
                ).await?;
                
                if scientific_similarity >= config.similarity_threshold {
                    cluster.answers.push(response.clone());
                    cluster.probability_mass += response_prob;
                    cluster.coherence_score = (cluster.coherence_score + scientific_similarity) / 2.0;
                    assigned_to_cluster = true;
                    break;
                }
            }
            
            // Create new cluster if no match found
            if !assigned_to_cluster {
                clusters.push(DomainSemanticCluster {
                    answers: vec![response.clone()],
                    probability_mass: response_prob,
                    representative: response.clone(),
                    coherence_score: 1.0,
                    domain_specific_score: 1.0,
                    medical_info: None,
                    legal_info: None,
                    scientific_info: Some(ScientificClusterInfo {
                        scientific_claims,
                        methodology_validation,
                        statistical_accuracy,
                        citation_verification,
                        reproducibility_score: self.scientific_corpus.assess_reproducibility(response),
                    }),
                });
            }
        }
        
        Ok(clusters)
    }
    
    async fn calculate_domain_adjustments(
        &self,
        prompt: &str,
        responses: &[String],
        domain: DomainType,
        config: &DomainSemanticEntropyConfig,
    ) -> Result<DomainAdjustments, SemanticError> {
        let terminology_confidence = self.calculate_terminology_confidence(responses, domain.clone()).await?;
        let citation_accuracy = if config.citation_validation {
            self.calculate_citation_accuracy(responses, domain.clone()).await?
        } else {
            1.0
        };
        
        let expert_validation = if config.expert_knowledge_base.is_some() {
            Some(self.calculate_expert_validation_score(prompt, responses, domain.clone()).await?)
        } else {
            None
        };
        
        let domain_uncertainty = self.calculate_domain_specific_uncertainty(responses, domain.clone()).await?;
        let contradiction_penalty = self.calculate_domain_contradiction_penalty(responses, domain.clone(), config).await?;
        
        // Calculate entropy multiplier based on domain-specific factors
        let entropy_multiplier = match domain {
            DomainType::Medical => {
                // Medical: Higher uncertainty for safety-critical content
                let safety_multiplier = if contradiction_penalty > 0.5 { 1.5 } else { 1.0 };
                let terminology_multiplier = if terminology_confidence < 0.7 { 1.3 } else { 1.0 };
                safety_multiplier * terminology_multiplier
            },
            DomainType::Legal => {
                // Legal: Higher uncertainty for precedent contradictions
                let precedent_multiplier = if contradiction_penalty > 0.3 { 1.4 } else { 1.0 };
                let citation_multiplier = if citation_accuracy < 0.8 { 1.2 } else { 1.0 };
                precedent_multiplier * citation_multiplier
            },
            DomainType::Scientific => {
                // Scientific: Higher uncertainty for methodology issues
                let methodology_multiplier = if terminology_confidence < 0.8 { 1.25 } else { 1.0 };
                let citation_multiplier = if citation_accuracy < 0.9 { 1.15 } else { 1.0 };
                methodology_multiplier * citation_multiplier
            },
            DomainType::General => 1.0,
        };
        
        Ok(DomainAdjustments {
            entropy_multiplier,
            terminology_confidence,
            citation_accuracy,
            expert_validation,
            domain_uncertainty,
            contradiction_penalty,
        })
    }
    
    async fn compute_medical_similarity(
        &mut self,
        response1: &str,
        response2: &str,
        medical_terms: &HashSet<String>,
        config: &DomainSemanticEntropyConfig,
    ) -> Result<f64, SemanticError> {
        // Base semantic similarity
        let base_similarity = self.base_calculator.compute_semantic_similarity(response1, response2)?;
        
        // Medical terminology overlap
        let terms1 = self.medical_ontology.extract_medical_terms(response1);
        let terms2 = self.medical_ontology.extract_medical_terms(response2);
        let terminology_similarity = self.calculate_terminology_overlap(&terms1, &terms2);
        
        // Drug interaction consistency
        let drug_consistency = self.medical_ontology.check_drug_interaction_consistency(response1, response2).await?;
        
        // Contraindication consistency
        let contraindication_consistency = self.medical_ontology.check_contraindication_consistency(response1, response2).await?;
        
        // Safety assessment alignment
        let safety_alignment = self.medical_ontology.assess_safety_alignment(response1, response2);
        
        // Weighted combination for medical domain
        let medical_similarity = base_similarity * 0.4 +
            terminology_similarity * config.terminology_weight * 0.3 +
            drug_consistency * 0.15 +
            contraindication_consistency * 0.1 +
            safety_alignment * 0.05;
        
        Ok(medical_similarity.min(1.0))
    }
    
    async fn compute_legal_similarity(
        &mut self,
        response1: &str,
        response2: &str,
        legal_citations: &Vec<String>,
        config: &DomainSemanticEntropyConfig,
    ) -> Result<f64, SemanticError> {
        // Base semantic similarity
        let base_similarity = self.base_calculator.compute_semantic_similarity(response1, response2)?;
        
        // Legal terminology overlap
        let terms1 = self.legal_knowledge.extract_legal_terms(response1);
        let terms2 = self.legal_knowledge.extract_legal_terms(response2);
        let terminology_similarity = self.calculate_terminology_overlap(&terms1, &terms2);
        
        // Citation consistency
        let citation_consistency = self.legal_knowledge.check_citation_consistency(response1, response2).await?;
        
        // Precedent alignment
        let precedent_alignment = self.legal_knowledge.assess_precedent_alignment(response1, response2).await?;
        
        // Jurisdiction consistency
        let jurisdiction_consistency = self.legal_knowledge.check_jurisdiction_consistency(response1, response2).await?;
        
        // Weighted combination for legal domain
        let legal_similarity = base_similarity * 0.4 +
            terminology_similarity * config.terminology_weight * 0.3 +
            citation_consistency * 0.15 +
            precedent_alignment * 0.1 +
            jurisdiction_consistency * 0.05;
        
        Ok(legal_similarity.min(1.0))
    }
    
    async fn compute_scientific_similarity(
        &mut self,
        response1: &str,
        response2: &str,
        scientific_claims: &Vec<String>,
        config: &DomainSemanticEntropyConfig,
    ) -> Result<f64, SemanticError> {
        // Base semantic similarity
        let base_similarity = self.base_calculator.compute_semantic_similarity(response1, response2)?;
        
        // Scientific terminology overlap
        let terms1 = self.scientific_corpus.extract_scientific_terms(response1);
        let terms2 = self.scientific_corpus.extract_scientific_terms(response2);
        let terminology_similarity = self.calculate_terminology_overlap(&terms1, &terms2);
        
        // Methodology consistency
        let methodology_consistency = self.scientific_corpus.check_methodology_consistency(response1, response2).await?;
        
        // Statistical claim alignment
        let statistical_alignment = self.scientific_corpus.assess_statistical_alignment(response1, response2).await?;
        
        // Citation verification consistency
        let citation_consistency = self.scientific_corpus.check_citation_consistency(response1, response2).await?;
        
        // Weighted combination for scientific domain
        let scientific_similarity = base_similarity * 0.4 +
            terminology_similarity * config.terminology_weight * 0.25 +
            methodology_consistency * 0.2 +
            statistical_alignment * 0.1 +
            citation_consistency * 0.05;
        
        Ok(scientific_similarity.min(1.0))
    }
    
    fn calculate_terminology_overlap(&self, terms1: &HashSet<String>, terms2: &HashSet<String>) -> f64 {
        if terms1.is_empty() && terms2.is_empty() {
            return 1.0;
        }
        
        let intersection_size = terms1.intersection(terms2).count();
        let union_size = terms1.union(terms2).count();
        
        if union_size == 0 {
            1.0
        } else {
            intersection_size as f64 / union_size as f64
        }
    }
    
    async fn calculate_terminology_confidence(&self, responses: &[String], domain: DomainType) -> Result<f64, SemanticError> {
        let mut total_confidence = 0.0;
        
        for response in responses {
            let confidence = match domain {
                DomainType::Medical => self.medical_ontology.assess_terminology_confidence(response),
                DomainType::Legal => self.legal_knowledge.assess_terminology_confidence(response),
                DomainType::Scientific => self.scientific_corpus.assess_terminology_confidence(response),
                DomainType::General => 1.0,
            };
            total_confidence += confidence;
        }
        
        Ok(total_confidence / responses.len() as f64)
    }
    
    async fn calculate_citation_accuracy(&self, responses: &[String], domain: DomainType) -> Result<f64, SemanticError> {
        let mut total_accuracy = 0.0;
        
        for response in responses {
            let accuracy = match domain {
                DomainType::Medical => self.medical_ontology.verify_medical_citations(response).await?,
                DomainType::Legal => self.legal_knowledge.verify_legal_citations(response).await?,
                DomainType::Scientific => self.scientific_corpus.verify_scientific_citations(response).await?,
                DomainType::General => 1.0,
            };
            total_accuracy += accuracy;
        }
        
        Ok(total_accuracy / responses.len() as f64)
    }
    
    async fn calculate_expert_validation_score(&self, prompt: &str, responses: &[String], domain: DomainType) -> Result<f64, SemanticError> {
        // Simplified expert validation using domain-specific heuristics
        match domain {
            DomainType::Medical => {
                let mut score = 0.0;
                for response in responses {
                    score += self.medical_ontology.expert_validation_heuristic(prompt, response);
                }
                Ok(score / responses.len() as f64)
            },
            DomainType::Legal => {
                let mut score = 0.0;
                for response in responses {
                    score += self.legal_knowledge.expert_validation_heuristic(prompt, response);
                }
                Ok(score / responses.len() as f64)
            },
            DomainType::Scientific => {
                let mut score = 0.0;
                for response in responses {
                    score += self.scientific_corpus.expert_validation_heuristic(prompt, response);
                }
                Ok(score / responses.len() as f64)
            },
            DomainType::General => Ok(1.0),
        }
    }
    
    async fn calculate_domain_specific_uncertainty(&self, responses: &[String], domain: DomainType) -> Result<f64, SemanticError> {
        let mut uncertainty_factors = Vec::new();
        
        for response in responses {
            let factor = match domain {
                DomainType::Medical => {
                    let safety_risk = self.medical_ontology.assess_safety_risk(response);
                    let diagnostic_uncertainty = self.medical_ontology.assess_diagnostic_uncertainty(response);
                    (safety_risk + diagnostic_uncertainty) / 2.0
                },
                DomainType::Legal => {
                    let precedent_uncertainty = self.legal_knowledge.assess_precedent_uncertainty(response);
                    let jurisdictional_ambiguity = self.legal_knowledge.assess_jurisdictional_ambiguity(response);
                    (precedent_uncertainty + jurisdictional_ambiguity) / 2.0
                },
                DomainType::Scientific => {
                    let methodology_uncertainty = self.scientific_corpus.assess_methodology_uncertainty(response);
                    let reproducibility_concern = self.scientific_corpus.assess_reproducibility_concern(response);
                    (methodology_uncertainty + reproducibility_concern) / 2.0
                },
                DomainType::General => 0.0,
            };
            uncertainty_factors.push(factor);
        }
        
        Ok(uncertainty_factors.iter().sum::<f64>() / uncertainty_factors.len() as f64)
    }
    
    async fn calculate_domain_contradiction_penalty(&self, responses: &[String], domain: DomainType, config: &DomainSemanticEntropyConfig) -> Result<f64, SemanticError> {
        let mut max_penalty: f64 = 0.0;
        
        // Check all pairs for domain-specific contradictions
        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                let penalty = match domain {
                    DomainType::Medical => self.medical_ontology.detect_medical_contradictions(&responses[i], &responses[j]),
                    DomainType::Legal => self.legal_knowledge.detect_legal_contradictions(&responses[i], &responses[j]),
                    DomainType::Scientific => self.scientific_corpus.detect_scientific_contradictions(&responses[i], &responses[j]),
                    DomainType::General => 0.0,
                };
                max_penalty = max_penalty.max(penalty);
            }
        }
        
        Ok(max_penalty * config.contradiction_penalty_multiplier)
    }
    
    fn calculate_entropy_over_domain_clusters(&self, clusters: &[DomainSemanticCluster]) -> Result<f64, SemanticError> {
        let total_mass: f64 = clusters.iter().map(|c| c.probability_mass).sum();
        
        if total_mass <= 0.0 {
            return Err(SemanticError::InvalidInput {
                message: "Zero total probability mass".to_string()
            });
        }
        
        let mut semantic_entropy = 0.0;
        
        for cluster in clusters {
            if cluster.probability_mass > 0.0 {
                let cluster_prob = cluster.probability_mass / total_mass;
                semantic_entropy -= cluster_prob * cluster_prob.ln();
            }
        }
        
        Ok(semantic_entropy)
    }
    
    fn assess_clustering_quality(&self, clusters: &[DomainSemanticCluster]) -> f64 {
        if clusters.is_empty() {
            return 0.0;
        }
        
        let avg_coherence = clusters.iter()
            .map(|c| c.coherence_score)
            .sum::<f64>() / clusters.len() as f64;
        
        let avg_domain_score = clusters.iter()
            .map(|c| c.domain_specific_score)
            .sum::<f64>() / clusters.len() as f64;
        
        (avg_coherence + avg_domain_score) / 2.0
    }
    
    fn initialize_domain_vocabularies() -> HashMap<DomainType, HashSet<String>> {
        let mut vocabularies = HashMap::new();
        
        let medical_vocab: HashSet<String> = vec![
            "diagnosis", "treatment", "patient", "clinical", "therapeutic", "pharmacology",
            "pathophysiology", "contraindication", "adverse", "efficacy", "dosage"
        ].into_iter().map(|s| s.to_string()).collect();
        
        let legal_vocab: HashSet<String> = vec![
            "jurisdiction", "precedent", "statute", "regulation", "litigation", "contract",
            "liability", "plaintiff", "defendant", "evidence", "testimony", "verdict"
        ].into_iter().map(|s| s.to_string()).collect();
        
        let scientific_vocab: HashSet<String> = vec![
            "hypothesis", "methodology", "statistical", "correlation", "causation", "peer-review",
            "reproducible", "empirical", "quantitative", "qualitative", "analysis", "significance"
        ].into_iter().map(|s| s.to_string()).collect();
        
        vocabularies.insert(DomainType::Medical, medical_vocab);
        vocabularies.insert(DomainType::Legal, legal_vocab);
        vocabularies.insert(DomainType::Scientific, scientific_vocab);
        vocabularies.insert(DomainType::General, HashSet::new());
        
        vocabularies
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSemanticCluster {
    pub answers: Vec<String>,
    pub probability_mass: f64,
    pub representative: String,
    pub coherence_score: f64,
    pub domain_specific_score: f64,
    pub medical_info: Option<MedicalClusterInfo>,
    pub legal_info: Option<LegalClusterInfo>,
    pub scientific_info: Option<ScientificClusterInfo>,
}

impl DomainSemanticCluster {
    fn from_base(base_cluster: SemanticCluster) -> Self {
        Self {
            answers: base_cluster.answers,
            probability_mass: base_cluster.probability_mass,
            representative: base_cluster.representative,
            coherence_score: base_cluster.coherence_score,
            domain_specific_score: 1.0,
            medical_info: None,
            legal_info: None,
            scientific_info: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalClusterInfo {
    pub medical_terms: HashSet<String>,
    pub drug_interactions: Vec<DrugInteraction>,
    pub contraindications: Vec<Contraindication>,
    pub safety_flags: Vec<SafetyFlag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalClusterInfo {
    pub legal_citations: Vec<String>,
    pub precedent_consistency: f64,
    pub jurisdiction_accuracy: f64,
    pub statute_references: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificClusterInfo {
    pub scientific_claims: Vec<String>,
    pub methodology_validation: f64,
    pub statistical_accuracy: f64,
    pub citation_verification: f64,
    pub reproducibility_score: f64,
}

// Domain-specific knowledge bases
pub struct MedicalOntology {
    drug_database: HashMap<String, DrugInfo>,
    contraindication_database: HashMap<String, Vec<Contraindication>>,
    medical_terminology: HashSet<String>,
}

pub struct LegalKnowledge {
    precedent_database: HashMap<String, PrecedentInfo>,
    statute_database: HashMap<String, StatuteInfo>,
    legal_terminology: HashSet<String>,
}

pub struct ScientificCorpus {
    methodology_patterns: HashMap<String, MethodologyInfo>,
    citation_database: HashMap<String, CitationInfo>,
    scientific_terminology: HashSet<String>,
}

// Supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInteraction {
    pub drug1: String,
    pub drug2: String,
    pub severity: InteractionSeverity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contraindication {
    pub condition: String,
    pub medication: String,
    pub severity: ContraindicationSeverity,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyFlag {
    pub flag_type: String,
    pub severity: SafetySeverity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionSeverity {
    Minor,
    Moderate,
    Major,
    Contraindicated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContraindicationSeverity {
    Relative,
    Absolute,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct DrugInfo {
    pub name: String,
    pub interactions: Vec<String>,
    pub contraindications: Vec<String>,
    pub side_effects: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PrecedentInfo {
    pub case_name: String,
    pub jurisdiction: String,
    pub year: u32,
    pub legal_principle: String,
}

#[derive(Debug, Clone)]
pub struct StatuteInfo {
    pub title: String,
    pub section: String,
    pub jurisdiction: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct MethodologyInfo {
    pub methodology_type: String,
    pub required_components: Vec<String>,
    pub validation_criteria: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CitationInfo {
    pub authors: Vec<String>,
    pub title: String,
    pub journal: String,
    pub year: u32,
    pub doi: Option<String>,
}

// Simplified implementations for the knowledge bases
impl MedicalOntology {
    pub fn new() -> Self {
        Self {
            drug_database: HashMap::new(),
            contraindication_database: HashMap::new(),
            medical_terminology: Self::build_medical_terminology(),
        }
    }
    
    pub fn extract_medical_terms(&self, text: &str) -> HashSet<String> {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding.split_whitespace().collect();
        words.into_iter()
            .filter(|word| self.medical_terminology.contains(*word))
            .map(|s| s.to_string())
            .collect()
    }
    
    pub async fn check_drug_interactions(&self, _medical_terms: &HashSet<String>) -> Result<Vec<DrugInteraction>, SemanticError> {
        // Simplified implementation
        Ok(Vec::new())
    }
    
    pub async fn check_contraindications(&self, _medical_terms: &HashSet<String>) -> Result<Vec<Contraindication>, SemanticError> {
        // Simplified implementation
        Ok(Vec::new())
    }
    
    pub fn assess_safety_flags(&self, _response: &str) -> Vec<SafetyFlag> {
        // Simplified implementation
        Vec::new()
    }
    
    pub async fn check_drug_interaction_consistency(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub async fn check_contraindication_consistency(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub fn assess_safety_alignment(&self, _response1: &str, _response2: &str) -> f64 {
        1.0 // Simplified
    }
    
    pub fn assess_terminology_confidence(&self, response: &str) -> f64 {
        let medical_terms = self.extract_medical_terms(response);
        if medical_terms.is_empty() {
            0.5 // No medical terms found
        } else {
            0.8 // Has medical terminology
        }
    }
    
    pub async fn verify_medical_citations(&self, _response: &str) -> Result<f64, SemanticError> {
        Ok(0.9) // Simplified
    }
    
    pub fn expert_validation_heuristic(&self, _prompt: &str, response: &str) -> f64 {
        let medical_terms = self.extract_medical_terms(response);
        let dangerous_keywords = ["always", "never", "cure", "guaranteed"];
        let has_dangerous = dangerous_keywords.iter().any(|&kw| response.to_lowercase().contains(kw));
        
        if has_dangerous {
            0.3 // Lower score for absolute statements
        } else if medical_terms.len() > 2 {
            0.8 // Good medical terminology usage
        } else {
            0.6 // Average
        }
    }
    
    pub fn assess_safety_risk(&self, response: &str) -> f64 {
        let dangerous_patterns = ["no side effects", "always safe", "never harmful", "cure everything"];
        let risk_score = dangerous_patterns.iter()
            .map(|&pattern| if response.to_lowercase().contains(pattern) { 0.8 } else { 0.0 })
            .fold(0.0, f64::max);
        risk_score
    }
    
    pub fn assess_diagnostic_uncertainty(&self, response: &str) -> f64 {
        let uncertainty_markers = ["may", "might", "could", "possibly", "likely"];
        let certainty_markers = ["definitely", "always", "never", "certainly"];
        
        let has_uncertainty = uncertainty_markers.iter().any(|&marker| response.contains(marker));
        let has_false_certainty = certainty_markers.iter().any(|&marker| response.contains(marker));
        
        if has_false_certainty {
            0.7 // High uncertainty for false certainty
        } else if has_uncertainty {
            0.3 // Lower uncertainty for appropriate hedging
        } else {
            0.5 // Medium uncertainty
        }
    }
    
    pub fn detect_medical_contradictions(&self, response1: &str, response2: &str) -> f64 {
        let r1_lower = response1.to_lowercase();
        let r2_lower = response2.to_lowercase();
        
        // Check for direct medical contradictions
        let contradictory_pairs = vec![
            (vec!["increase", "higher", "elevate"], vec!["decrease", "lower", "reduce"]),
            (vec!["safe", "recommended"], vec!["dangerous", "contraindicated"]),
            (vec!["effective", "beneficial"], vec!["ineffective", "harmful"]),
        ];
        
        for (positive_terms, negative_terms) in contradictory_pairs {
            let r1_positive = positive_terms.iter().any(|&term| r1_lower.contains(term));
            let r2_negative = negative_terms.iter().any(|&term| r2_lower.contains(term));
            let r2_positive = positive_terms.iter().any(|&term| r2_lower.contains(term));
            let r1_negative = negative_terms.iter().any(|&term| r1_lower.contains(term));
            
            if (r1_positive && r2_negative) || (r2_positive && r1_negative) {
                return 0.8; // Strong medical contradiction
            }
        }
        
        0.0
    }
    
    fn build_medical_terminology() -> HashSet<String> {
        vec![
            "diagnosis", "treatment", "therapy", "medication", "dosage", "contraindication",
            "pathophysiology", "etiology", "prognosis", "clinical", "therapeutic", "pharmacology",
            "hypertension", "diabetes", "cardiovascular", "neurological", "oncology", "cardiology"
        ].into_iter().map(|s| s.to_string()).collect()
    }
}

impl MedicalClusterInfo {
    pub fn merge_drug_interactions(&mut self, interactions: &[DrugInteraction]) {
        self.drug_interactions.extend_from_slice(interactions);
    }
    
    pub fn merge_contraindications(&mut self, contraindications: &[Contraindication]) {
        self.contraindications.extend_from_slice(contraindications);
    }
}

impl LegalKnowledge {
    pub fn new() -> Self {
        Self {
            precedent_database: HashMap::new(),
            statute_database: HashMap::new(),
            legal_terminology: Self::build_legal_terminology(),
        }
    }
    
    pub fn extract_legal_terms(&self, text: &str) -> HashSet<String> {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding.split_whitespace().collect();
        words.into_iter()
            .filter(|word| self.legal_terminology.contains(*word))
            .map(|s| s.to_string())
            .collect()
    }
    
    pub fn extract_legal_citations(&self, text: &str) -> Vec<String> {
        // Simplified citation extraction
        let mut citations = Vec::new();
        if text.contains("v.") || text.contains("F.2d") || text.contains("U.S.") {
            citations.push("legal_citation".to_string());
        }
        citations
    }
    
    pub async fn check_precedent_consistency(&self, _citations: &[String]) -> Result<f64, SemanticError> {
        Ok(0.9) // Simplified
    }
    
    pub async fn validate_jurisdiction_claims(&self, _response: &str) -> Result<f64, SemanticError> {
        Ok(0.9) // Simplified
    }
    
    pub fn extract_statute_references(&self, _response: &str) -> Vec<String> {
        Vec::new() // Simplified
    }
    
    pub async fn check_citation_consistency(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub async fn assess_precedent_alignment(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub async fn check_jurisdiction_consistency(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub fn assess_terminology_confidence(&self, response: &str) -> f64 {
        let legal_terms = self.extract_legal_terms(response);
        if legal_terms.is_empty() {
            0.5
        } else {
            0.8
        }
    }
    
    pub async fn verify_legal_citations(&self, _response: &str) -> Result<f64, SemanticError> {
        Ok(0.85) // Simplified
    }
    
    pub fn expert_validation_heuristic(&self, _prompt: &str, response: &str) -> f64 {
        let legal_terms = self.extract_legal_terms(response);
        let absolute_statements = ["always", "never", "guaranteed", "certain"];
        let has_absolutes = absolute_statements.iter().any(|&stmt| response.to_lowercase().contains(stmt));
        
        if has_absolutes {
            0.4 // Lower score for absolute legal statements
        } else if legal_terms.len() > 1 {
            0.75
        } else {
            0.6
        }
    }
    
    pub fn assess_precedent_uncertainty(&self, _response: &str) -> f64 {
        0.3 // Simplified
    }
    
    pub fn assess_jurisdictional_ambiguity(&self, _response: &str) -> f64 {
        0.2 // Simplified
    }
    
    pub fn detect_legal_contradictions(&self, response1: &str, response2: &str) -> f64 {
        let r1_lower = response1.to_lowercase();
        let r2_lower = response2.to_lowercase();
        
        let contradictory_pairs = vec![
            (vec!["legal", "allowed", "permitted"], vec!["illegal", "prohibited", "forbidden"]),
            (vec!["liable", "responsible"], vec!["not liable", "not responsible"]),
            (vec!["enforceable", "valid"], vec!["unenforceable", "invalid"]),
        ];
        
        for (positive_terms, negative_terms) in contradictory_pairs {
            let r1_positive = positive_terms.iter().any(|&term| r1_lower.contains(term));
            let r2_negative = negative_terms.iter().any(|&term| r2_lower.contains(term));
            let r2_positive = positive_terms.iter().any(|&term| r2_lower.contains(term));
            let r1_negative = negative_terms.iter().any(|&term| r1_lower.contains(term));
            
            if (r1_positive && r2_negative) || (r2_positive && r1_negative) {
                return 0.7; // Legal contradiction
            }
        }
        
        0.0
    }
    
    fn build_legal_terminology() -> HashSet<String> {
        vec![
            "jurisdiction", "precedent", "statute", "regulation", "litigation", "contract",
            "liability", "plaintiff", "defendant", "evidence", "testimony", "verdict",
            "appeal", "motion", "discovery", "damages", "injunction", "settlement"
        ].into_iter().map(|s| s.to_string()).collect()
    }
}

impl ScientificCorpus {
    pub fn new() -> Self {
        Self {
            methodology_patterns: HashMap::new(),
            citation_database: HashMap::new(),
            scientific_terminology: Self::build_scientific_terminology(),
        }
    }
    
    pub fn extract_scientific_terms(&self, text: &str) -> HashSet<String> {
        let binding = text.to_lowercase();
        let words: Vec<&str> = binding.split_whitespace().collect();
        words.into_iter()
            .filter(|word| self.scientific_terminology.contains(*word))
            .map(|s| s.to_string())
            .collect()
    }
    
    pub fn extract_scientific_claims(&self, _response: &str) -> Vec<String> {
        Vec::new() // Simplified
    }
    
    pub async fn validate_methodology(&self, _claims: &[String]) -> Result<f64, SemanticError> {
        Ok(0.8) // Simplified
    }
    
    pub async fn check_statistical_claims(&self, _claims: &[String]) -> Result<f64, SemanticError> {
        Ok(0.85) // Simplified
    }
    
    pub async fn verify_scientific_citations(&self, _response: &str) -> Result<f64, SemanticError> {
        Ok(0.9) // Simplified
    }
    
    pub fn assess_reproducibility(&self, _response: &str) -> f64 {
        0.7 // Simplified
    }
    
    pub async fn check_methodology_consistency(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub async fn assess_statistical_alignment(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub async fn check_citation_consistency(&self, _response1: &str, _response2: &str) -> Result<f64, SemanticError> {
        Ok(1.0) // Simplified
    }
    
    pub fn assess_terminology_confidence(&self, response: &str) -> f64 {
        let scientific_terms = self.extract_scientific_terms(response);
        if scientific_terms.is_empty() {
            0.5
        } else {
            0.8
        }
    }
    
    pub fn expert_validation_heuristic(&self, _prompt: &str, response: &str) -> f64 {
        let scientific_terms = self.extract_scientific_terms(response);
        let methodology_terms = ["study", "experiment", "analysis", "methodology"];
        let has_methodology = methodology_terms.iter().any(|&term| response.to_lowercase().contains(term));
        
        if has_methodology && scientific_terms.len() > 2 {
            0.85
        } else if scientific_terms.len() > 1 {
            0.7
        } else {
            0.5
        }
    }
    
    pub fn assess_methodology_uncertainty(&self, _response: &str) -> f64 {
        0.25 // Simplified
    }
    
    pub fn assess_reproducibility_concern(&self, _response: &str) -> f64 {
        0.2 // Simplified
    }
    
    pub fn detect_scientific_contradictions(&self, response1: &str, response2: &str) -> f64 {
        let r1_lower = response1.to_lowercase();
        let r2_lower = response2.to_lowercase();
        
        let contradictory_pairs = vec![
            (vec!["significant", "statistically significant"], vec!["not significant", "insignificant"]),
            (vec!["correlated", "associated"], vec!["uncorrelated", "no association"]),
            (vec!["reproducible", "replicable"], vec!["irreproducible", "unreplicable"]),
        ];
        
        for (positive_terms, negative_terms) in contradictory_pairs {
            let r1_positive = positive_terms.iter().any(|&term| r1_lower.contains(term));
            let r2_negative = negative_terms.iter().any(|&term| r2_lower.contains(term));
            let r2_positive = positive_terms.iter().any(|&term| r2_lower.contains(term));
            let r1_negative = negative_terms.iter().any(|&term| r1_lower.contains(term));
            
            if (r1_positive && r2_negative) || (r2_positive && r1_negative) {
                return 0.6; // Scientific contradiction
            }
        }
        
        0.0
    }
    
    fn build_scientific_terminology() -> HashSet<String> {
        vec![
            "hypothesis", "methodology", "statistical", "correlation", "causation", "peer-review",
            "reproducible", "empirical", "quantitative", "qualitative", "analysis", "significance",
            "experiment", "control", "variable", "sample", "population", "bias", "validity"
        ].into_iter().map(|s| s.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_domain_semantic_entropy_medical() {
        let mut calculator = DomainSemanticEntropyCalculator::new();
        
        let responses = vec![
            "ACE inhibitors are first-line treatment for hypertension".to_string(),
            "Beta-blockers are always the best choice with no side effects".to_string(),
        ];
        let probabilities = vec![0.6, 0.4];
        
        let result = calculator.calculate_domain_semantic_entropy(
            "What is the treatment for hypertension?",
            &responses,
            &probabilities,
            DomainType::Medical,
        ).await.unwrap();
        
        assert!(result.domain_adjusted_entropy > result.base_entropy);
        assert!(result.terminology_confidence > 0.0);
    }
    
    #[tokio::test]
    async fn test_medical_contradiction_detection() {
        let ontology = MedicalOntology::new();
        
        let contradiction_score = ontology.detect_medical_contradictions(
            "This medication is safe for all patients",
            "This medication is contraindicated in elderly patients"
        );
        
        assert!(contradiction_score > 0.5);
    }
    
    #[test]
    fn test_domain_config_creation() {
        let medical_config = DomainSemanticEntropyConfig::for_domain(DomainType::Medical);
        assert!(medical_config.similarity_threshold > 0.5);
        assert!(medical_config.terminology_weight > 1.0);
        assert!(medical_config.citation_validation);
        assert!(medical_config.contradiction_penalty_multiplier > 1.0);
    }
}
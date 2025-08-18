use serde::{Deserialize, Serialize};
use crate::error::SemanticError;
use crate::math::information_theory::InformationTheoryCalculator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyMetrics {
	pub surprise: f64,
	pub ambiguity: f64,
	pub complexity: f64,
	pub free_energy: f64,
	// Enhanced FEP components
	pub kl_surprise: f64,
	pub attention_entropy: f64,
	pub prediction_variance: f64,
	pub fisher_info_metrics: FisherInfoMetrics,
	pub enhanced_free_energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherInfoMetrics {
	pub fim_trace: f64,
	pub fim_mean_eigenvalue: f64,
	pub fim_max_eigenvalue: f64,
	pub fim_condition_number: f64,
}

pub fn compute_free_energy_for_token(
	probabilities: &[f64],
	observed_index: Option<usize>,
	prior: Option<&[f64]>,
	q_post: Option<&[f64]>,
) -> Result<FreeEnergyMetrics, SemanticError> {
	if probabilities.is_empty() { return Err(SemanticError::InvalidInput { message: "Empty probability vector".to_string() }); }
	let sum_p: f64 = probabilities.iter().sum();
	if !sum_p.is_finite() || sum_p <= 0.0 { return Err(SemanticError::InvalidInput { message: "Invalid probability distribution".to_string() }); }
	let norm_probs: Vec<f64> = probabilities.iter().map(|p| p / sum_p).collect();
	let idx = observed_index.unwrap_or_else(|| argmax_index(&norm_probs));
	if idx >= norm_probs.len() { return Err(SemanticError::InvalidInput { message: "Observed index out of bounds".to_string() }); }
	let p_obs = (norm_probs[idx]).max(1e-12);
	let surprise = -p_obs.ln();
	let info_calc = InformationTheoryCalculator::default();
	let ambiguity = info_calc.shannon_entropy(&norm_probs)?;
	let complexity = match (prior, q_post) {
		(Some(p0), Some(q1)) => {
			if p0.len() != q1.len() { return Err(SemanticError::InvalidInput { message: "prior and q_post lengths must match".to_string() }); }
			info_calc.kl_divergence(q1, p0)?
		}
		_ => 0.0,
	};
	let free_energy = surprise + complexity;
	
	// Enhanced FEP components
	let kl_surprise = calculate_kl_surprise(&norm_probs, prior, q_post)?;
	let attention_entropy = calculate_attention_entropy(&norm_probs);
	let prediction_variance = calculate_prediction_variance(&norm_probs);
	let fisher_info_metrics = calculate_fisher_info_metrics(&norm_probs);
	
	// Enhanced free energy with additional components
	let enhanced_free_energy = free_energy + 
		kl_surprise * 2.0 + 
		attention_entropy * 0.5 + 
		prediction_variance * 1.0;
	
	Ok(FreeEnergyMetrics { 
		surprise, 
		ambiguity, 
		complexity, 
		free_energy,
		kl_surprise,
		attention_entropy,
		prediction_variance,
		fisher_info_metrics,
		enhanced_free_energy,
	})
}

fn calculate_kl_surprise(
	_probabilities: &[f64], 
	prior: Option<&[f64]>, 
	q_post: Option<&[f64]>
) -> Result<f64, SemanticError> {
	let info_calc = InformationTheoryCalculator::default();
	
	match (prior, q_post) {
		(Some(p0), Some(q1)) => {
			if p0.len() != q1.len() { 
				return Err(SemanticError::InvalidInput { message: "prior and q_post lengths must match".to_string() }); 
			}
			// KL divergence between posterior and prior
			info_calc.kl_divergence(q1, p0)
		}
		_ => Ok(0.0),
	}
}

fn calculate_attention_entropy(probabilities: &[f64]) -> f64 {
	// Simulate attention entropy based on probability distribution
	// Higher entropy = more uncertain attention patterns
	let max_entropy = (probabilities.len() as f64).ln();
	let current_entropy = -probabilities.iter()
		.map(|&p| if p > 1e-12 { p * p.ln() } else { 0.0 })
		.sum::<f64>();
	
	// Normalize to [0, 1] range
	(current_entropy / max_entropy).max(0.0).min(1.0)
}

fn calculate_prediction_variance(probabilities: &[f64]) -> f64 {
	// Calculate variance of the probability distribution
	// Higher variance = more peaked distribution = lower uncertainty
	let mean = probabilities.iter().sum::<f64>() / probabilities.len() as f64;
	let variance = probabilities.iter()
		.map(|&p| (p - mean).powi(2))
		.sum::<f64>() / probabilities.len() as f64;
	
	// Invert so higher variance = higher uncertainty
	1.0 / (1.0 + variance)
}

fn calculate_fisher_info_metrics(probabilities: &[f64]) -> FisherInfoMetrics {
	// Fisher Information Matrix diagonal: 1/p_i for each probability
	let fim_diagonal: Vec<f64> = probabilities.iter()
		.map(|&p| 1.0 / (p + 1e-12))
		.collect();
	
	let fim_trace: f64 = fim_diagonal.iter().sum();
	let fim_mean = fim_trace / fim_diagonal.len() as f64;
	let fim_max = fim_diagonal.iter().fold(0.0f64, |a, &b| a.max(b));
	let fim_min = fim_diagonal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
	let fim_condition = fim_max / (fim_min + 1e-12);
	
	FisherInfoMetrics {
		fim_trace,
		fim_mean_eigenvalue: fim_mean,
		fim_max_eigenvalue: fim_max,
		fim_condition_number: fim_condition,
	}
}

fn argmax_index(values: &[f64]) -> usize {
	let mut best_idx = 0;
	let mut best_val = f64::NEG_INFINITY;
	for (i, &v) in values.iter().enumerate() {
		if v > best_val { best_val = v; best_idx = i; }
	}
	best_idx
} 
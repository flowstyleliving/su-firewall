use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use crate::error::SemanticError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationMetrics {
	pub entropy: f64,
	pub cross_entropy: f64,
	pub kl_divergence: f64,
	pub js_divergence: f64,
	pub mutual_information: f64,
	pub bottleneck_beta: f64,
	pub effective_information: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationBottleneckResult {
	pub compression_quality: f64,
	pub retention_ratio: f64,
	pub efficiency: f64,
	pub optimal_beta: f64,
}

pub struct InformationTheoryCalculator {
	epsilon: f64,
	beta: f64,
	max_iterations: usize,
}

impl Default for InformationTheoryCalculator {
	fn default() -> Self { Self { epsilon: 1e-12, beta: 1.0, max_iterations: 1000 } }
}

impl InformationTheoryCalculator {
	pub fn new(epsilon: f64, beta: f64) -> Self { Self { epsilon, beta, max_iterations: 1000 } }

	pub fn shannon_entropy(&self, distribution: &[f64]) -> Result<f64, SemanticError> {
		if distribution.is_empty() { return Err(SemanticError::InvalidInput { message: "Empty distribution".to_string() }); }
		let sum: f64 = distribution.iter().sum();
		if sum <= 0.0 { return Err(SemanticError::InvalidInput { message: "Invalid probability distribution".to_string() }); }
		let entropy = distribution.iter().map(|&p| {
			let normalized_p = p / sum;
			if normalized_p > self.epsilon { -normalized_p * normalized_p.log2() } else { 0.0 }
		}).sum();
		Ok(entropy)
	}

	pub fn cross_entropy(&self, p: &[f64], q: &[f64]) -> Result<f64, SemanticError> {
		if p.len() != q.len() { return Err(SemanticError::InvalidInput { message: "Distribution lengths must match".to_string() }); }
		let p_sum: f64 = p.iter().sum();
		let q_sum: f64 = q.iter().sum();
		if p_sum <= 0.0 || q_sum <= 0.0 { return Err(SemanticError::InvalidInput { message: "Invalid probability distributions".to_string() }); }
		let cross_entropy = p.iter().zip(q.iter()).map(|(&p_i, &q_i)| {
			let normalized_p = p_i / p_sum;
			let normalized_q = q_i / q_sum;
			if normalized_p > self.epsilon && normalized_q > self.epsilon { -normalized_p * normalized_q.log2() } else { 0.0 }
		}).sum();
		Ok(cross_entropy)
	}

	pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, SemanticError> {
		if p.len() != q.len() { return Err(SemanticError::InvalidInput { message: "Distribution lengths must match".to_string() }); }
		let p_sum: f64 = p.iter().sum();
		let q_sum: f64 = q.iter().sum();
		if p_sum <= 0.0 || q_sum <= 0.0 { return Err(SemanticError::InvalidInput { message: "Invalid probability distributions".to_string() }); }
		let kl_div = p.iter().zip(q.iter()).map(|(&p_i, &q_i)| {
			let normalized_p = p_i / p_sum;
			let normalized_q = q_i / q_sum;
			if normalized_p > self.epsilon && normalized_q > self.epsilon { normalized_p * (normalized_p / normalized_q).log2() } else { 0.0 }
		}).sum();
		Ok(kl_div)
	}

	pub fn js_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, SemanticError> {
		if p.len() != q.len() { return Err(SemanticError::InvalidInput { message: "Distribution lengths must match".to_string() }); }
		let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&p_i, &q_i)| 0.5 * (p_i + q_i)).collect();
		let kl_pm = self.kl_divergence(p, &m)?;
		let kl_qm = self.kl_divergence(q, &m)?;
		Ok(0.5 * (kl_pm + kl_qm))
	}

	pub fn mutual_information(&self, joint_distribution: &Array2<f64>) -> Result<f64, SemanticError> {
		let (rows, cols) = joint_distribution.dim();
		if rows == 0 || cols == 0 { return Err(SemanticError::InvalidInput { message: "Empty joint distribution".to_string() }); }
		let marginal_x: Array1<f64> = joint_distribution.sum_axis(ndarray::Axis(1));
		let marginal_y: Array1<f64> = joint_distribution.sum_axis(ndarray::Axis(0));
		let joint_entropy = self.shannon_entropy(&joint_distribution.iter().cloned().collect::<Vec<_>>())?;
		let entropy_x = self.shannon_entropy(&marginal_x.to_vec())?;
		let entropy_y = self.shannon_entropy(&marginal_y.to_vec())?;
		Ok(entropy_x + entropy_y - joint_entropy)
	}

	pub fn information_bottleneck(&self, x_data: &[f64], y_data: &[f64]) -> Result<InformationBottleneckResult, SemanticError> {
		if x_data.len() != y_data.len() { return Err(SemanticError::InvalidInput { message: "Data lengths must match".to_string() }); }
		let joint_dist = self.create_joint_distribution(x_data, y_data)?;
		let mutual_info = self.mutual_information(&joint_dist)?;
		let compression_quality = (mutual_info / (1.0 + self.beta)).min(1.0);
		let retention_ratio = (mutual_info / (1.0 + self.epsilon)).min(1.0);
		let efficiency = (compression_quality * retention_ratio).sqrt();
		Ok(InformationBottleneckResult { compression_quality, retention_ratio, efficiency, optimal_beta: self.beta })
	}

	pub fn calculate_comprehensive_metrics(&self, prompt_data: &[f64], output_data: &[f64]) -> Result<InformationMetrics, SemanticError> {
		if prompt_data.len() != output_data.len() { return Err(SemanticError::InvalidInput { message: "Data lengths must match".to_string() }); }
		let entropy = self.shannon_entropy(prompt_data)?;
		let cross_entropy = self.cross_entropy(prompt_data, output_data)?;
		let kl_divergence = self.kl_divergence(prompt_data, output_data)?;
		let js_divergence = self.js_divergence(prompt_data, output_data)?;
		let joint_dist = self.create_joint_distribution(prompt_data, output_data)?;
		let mutual_information = self.mutual_information(&joint_dist)?;
		let effective_information = entropy - kl_divergence / 2.0;
		Ok(InformationMetrics { entropy, cross_entropy, kl_divergence, js_divergence, mutual_information, bottleneck_beta: self.beta, effective_information })
	}

	fn create_joint_distribution(&self, x_data: &[f64], y_data: &[f64]) -> Result<Array2<f64>, SemanticError> {
		let n = x_data.len();
		if n == 0 { return Err(SemanticError::InvalidInput { message: "Empty data arrays".to_string() }); }
		let mut joint = Array2::<f64>::zeros((n, n));
		for i in 0..n { for j in 0..n { joint[[i, j]] = ((x_data[i] - y_data[j]).abs() + self.epsilon).recip(); } }
		let sum = joint.sum();
		if sum <= 0.0 { return Err(SemanticError::InvalidInput { message: "Invalid joint distribution".to_string() }); }
		joint.mapv_inplace(|v| v / sum);
		Ok(joint)
	}
} 
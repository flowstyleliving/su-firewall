use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use crate::SemanticError;

/// 📊 Advanced Information-Theoretic Metrics for Semantic Uncertainty
/// 
/// This module implements sophisticated information theory calculations
/// to enhance the semantic uncertainty runtime with:
/// - Mutual Information I(X;Y)
/// - KL Divergence D_KL(P||Q)
/// - Jensen-Shannon Divergence
/// - Cross-entropy metrics
/// - Information bottleneck principles

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationMetrics {
    /// Shannon entropy H(X)
    pub entropy: f64,
    /// Cross-entropy H(P, Q)
    pub cross_entropy: f64,
    /// KL divergence D_KL(P||Q)
    pub kl_divergence: f64,
    /// Jensen-Shannon divergence
    pub js_divergence: f64,
    /// Mutual information I(X;Y)
    pub mutual_information: f64,
    /// Information bottleneck parameter β
    pub bottleneck_beta: f64,
    /// Effective information content
    pub effective_information: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationBottleneckResult {
    /// Compressed representation quality
    pub compression_quality: f64,
    /// Information retention ratio
    pub retention_ratio: f64,
    /// Bottleneck efficiency
    pub efficiency: f64,
    /// Optimal β parameter
    pub optimal_beta: f64,
}

/// 🧮 Information Theory Calculator
pub struct InformationTheoryCalculator {
    /// Smoothing parameter for probability estimates
    epsilon: f64,
    /// Information bottleneck β parameter
    beta: f64,
    /// Maximum iterations for optimization
    max_iterations: usize,
}

impl Default for InformationTheoryCalculator {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            beta: 1.0,
            max_iterations: 1000,
        }
    }
}

impl InformationTheoryCalculator {
    /// Create new information theory calculator
    pub fn new(epsilon: f64, beta: f64) -> Self {
        Self {
            epsilon,
            beta,
            max_iterations: 1000,
        }
    }

    /// 📈 Calculate Shannon entropy H(X) = -Σ p(x) log p(x)
    pub fn shannon_entropy(&self, distribution: &[f64]) -> Result<f64, SemanticError> {
        if distribution.is_empty() {
            return Err(SemanticError::InvalidInput { message: "Empty distribution".to_string() });
        }

        let sum: f64 = distribution.iter().sum();
        if sum <= 0.0 {
            return Err(SemanticError::InvalidInput { message: "Invalid probability distribution".to_string() });
        }

        let entropy = distribution
            .iter()
            .map(|&p| {
                let normalized_p = p / sum;
                if normalized_p > self.epsilon {
                    -normalized_p * normalized_p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        Ok(entropy)
    }

    /// 🔄 Calculate cross-entropy H(P, Q) = -Σ p(x) log q(x)
    pub fn cross_entropy(&self, p: &[f64], q: &[f64]) -> Result<f64, SemanticError> {
        if p.len() != q.len() {
            return Err(SemanticError::InvalidInput { message: "Distribution lengths must match".to_string() });
        }

        let p_sum: f64 = p.iter().sum();
        let q_sum: f64 = q.iter().sum();
        
        if p_sum <= 0.0 || q_sum <= 0.0 {
            return Err(SemanticError::InvalidInput { message: "Invalid probability distributions".to_string() });
        }

        let cross_entropy = p
            .iter()
            .zip(q.iter())
            .map(|(&p_i, &q_i)| {
                let normalized_p = p_i / p_sum;
                let normalized_q = q_i / q_sum;
                
                if normalized_p > self.epsilon && normalized_q > self.epsilon {
                    -normalized_p * normalized_q.log2()
                } else {
                    0.0
                }
            })
            .sum();

        Ok(cross_entropy)
    }

    /// 📊 Calculate KL divergence D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, SemanticError> {
        if p.len() != q.len() {
            return Err(SemanticError::InvalidInput { message: "Distribution lengths must match".to_string() });
        }

        let p_sum: f64 = p.iter().sum();
        let q_sum: f64 = q.iter().sum();
        
        if p_sum <= 0.0 || q_sum <= 0.0 {
            return Err(SemanticError::InvalidInput { message: "Invalid probability distributions".to_string() });
        }

        let kl_div = p
            .iter()
            .zip(q.iter())
            .map(|(&p_i, &q_i)| {
                let normalized_p = p_i / p_sum;
                let normalized_q = q_i / q_sum;
                
                if normalized_p > self.epsilon && normalized_q > self.epsilon {
                    normalized_p * (normalized_p / normalized_q).log2()
                } else {
                    0.0
                }
            })
            .sum();

        Ok(kl_div)
    }

    /// 🎯 Calculate Jensen-Shannon divergence JS(P, Q) = 0.5 * [D_KL(P||M) + D_KL(Q||M)]
    /// where M = 0.5 * (P + Q)
    pub fn js_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, SemanticError> {
        if p.len() != q.len() {
            return Err(SemanticError::InvalidInput { message: "Distribution lengths must match".to_string() });
        }

        // Calculate M = 0.5 * (P + Q)
        let m: Vec<f64> = p
            .iter()
            .zip(q.iter())
            .map(|(&p_i, &q_i)| 0.5 * (p_i + q_i))
            .collect();

        // Calculate JS divergence
        let kl_pm = self.kl_divergence(p, &m)?;
        let kl_qm = self.kl_divergence(q, &m)?;
        
        Ok(0.5 * (kl_pm + kl_qm))
    }

    /// 🔗 Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    pub fn mutual_information(&self, joint_distribution: &Array2<f64>) -> Result<f64, SemanticError> {
        let (rows, cols) = joint_distribution.dim();
        
        if rows == 0 || cols == 0 {
            return Err(SemanticError::InvalidInput { message: "Empty joint distribution".to_string() });
        }

        // Calculate marginal distributions
        let marginal_x: Array1<f64> = joint_distribution.sum_axis(ndarray::Axis(1));
        let marginal_y: Array1<f64> = joint_distribution.sum_axis(ndarray::Axis(0));
        
        // Calculate joint entropy H(X,Y)
        let joint_entropy = self.shannon_entropy(&joint_distribution.iter().cloned().collect::<Vec<_>>())?;
        
        // Calculate marginal entropies
        let entropy_x = self.shannon_entropy(&marginal_x.to_vec())?;
        let entropy_y = self.shannon_entropy(&marginal_y.to_vec())?;
        
        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        Ok(entropy_x + entropy_y - joint_entropy)
    }

    /// 🎛️ Information Bottleneck Analysis
    /// Finds optimal compression that preserves relevant information
    pub fn information_bottleneck(&self, x_data: &[f64], y_data: &[f64]) -> Result<InformationBottleneckResult, SemanticError> {
        if x_data.len() != y_data.len() {
            return Err(SemanticError::InvalidInput { message: "Data lengths must match".to_string() });
        }

        // Create joint distribution from data
        let joint_dist = self.create_joint_distribution(x_data, y_data)?;
        
        // Calculate mutual information I(X;Y)
        let mutual_info = self.mutual_information(&joint_dist)?;
        
        // Calculate compression quality (simplified)
        let compression_quality = self.calculate_compression_quality(&joint_dist)?;
        
        // Calculate information retention ratio
        let retention_ratio = compression_quality / (mutual_info + self.epsilon);
        
        // Calculate bottleneck efficiency
        let efficiency = retention_ratio / (self.beta + self.epsilon);
        
        // Find optimal β (simplified optimization)
        let optimal_beta = self.find_optimal_beta(&joint_dist)?;
        
        Ok(InformationBottleneckResult {
            compression_quality,
            retention_ratio,
            efficiency,
            optimal_beta,
        })
    }

    /// 📐 Calculate comprehensive information metrics
    pub fn calculate_comprehensive_metrics(&self, prompt_data: &[f64], output_data: &[f64]) -> Result<InformationMetrics, SemanticError> {
        if prompt_data.len() != output_data.len() {
            return Err(SemanticError::InvalidInput { message: "Data lengths must match".to_string() });
        }

        // Calculate basic entropy
        let entropy = self.shannon_entropy(prompt_data)?;
        
        // Calculate cross-entropy
        let cross_entropy = self.cross_entropy(prompt_data, output_data)?;
        
        // Calculate KL divergence
        let kl_divergence = self.kl_divergence(prompt_data, output_data)?;
        
        // Calculate JS divergence
        let js_divergence = self.js_divergence(prompt_data, output_data)?;
        
        // Calculate mutual information
        let joint_dist = self.create_joint_distribution(prompt_data, output_data)?;
        let mutual_information = self.mutual_information(&joint_dist)?;
        
        // Calculate effective information content
        let effective_information = entropy - kl_divergence / 2.0;
        
        Ok(InformationMetrics {
            entropy,
            cross_entropy,
            kl_divergence,
            js_divergence,
            mutual_information,
            bottleneck_beta: self.beta,
            effective_information,
        })
    }

    /// 🔧 Helper: Create joint distribution from data
    fn create_joint_distribution(&self, x_data: &[f64], y_data: &[f64]) -> Result<Array2<f64>, SemanticError> {
        // Simplified binning approach
        let bins = 10;
        let mut joint_counts = Array2::<f64>::zeros((bins, bins));
        
        // Find data ranges
        let x_min = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let x_range = x_max - x_min + self.epsilon;
        let y_range = y_max - y_min + self.epsilon;
        
        // Bin the data
        for (&x, &y) in x_data.iter().zip(y_data.iter()) {
            let x_bin = ((x - x_min) / x_range * (bins as f64 - 1.0)).floor() as usize;
            let y_bin = ((y - y_min) / y_range * (bins as f64 - 1.0)).floor() as usize;
            
            let x_idx = x_bin.min(bins - 1);
            let y_idx = y_bin.min(bins - 1);
            
            joint_counts[[x_idx, y_idx]] += 1.0;
        }
        
        // Normalize to probabilities
        let total: f64 = joint_counts.sum();
        if total > 0.0 {
            joint_counts = joint_counts / total;
        }
        
        Ok(joint_counts)
    }

    /// 🎯 Helper: Calculate compression quality
    fn calculate_compression_quality(&self, joint_dist: &Array2<f64>) -> Result<f64, SemanticError> {
        // Simplified compression quality metric
        let total_info = joint_dist.sum();
        let non_zero_elements = joint_dist.iter().filter(|&&x| x > self.epsilon).count();
        let total_elements = joint_dist.len();
        
        let sparsity = 1.0 - (non_zero_elements as f64 / total_elements as f64);
        let quality = total_info * (1.0 - sparsity);
        
        Ok(quality)
    }

    /// 🔍 Helper: Find optimal β parameter
    fn find_optimal_beta(&self, joint_dist: &Array2<f64>) -> Result<f64, SemanticError> {
        // Simplified β optimization using golden section search
        let mut beta_low = 0.1;
        let mut beta_high = 10.0;
        let tolerance = 1e-6;
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        
        for _ in 0..self.max_iterations {
            let beta_1 = beta_high - (beta_high - beta_low) / golden_ratio;
            let beta_2 = beta_low + (beta_high - beta_low) / golden_ratio;
            
            let score_1 = self.evaluate_beta_score(beta_1, joint_dist)?;
            let score_2 = self.evaluate_beta_score(beta_2, joint_dist)?;
            
            if score_1 > score_2 {
                beta_high = beta_2;
            } else {
                beta_low = beta_1;
            }
            
            if (beta_high - beta_low).abs() < tolerance {
                break;
            }
        }
        
        Ok((beta_low + beta_high) / 2.0)
    }

    /// 📊 Helper: Evaluate β score
    fn evaluate_beta_score(&self, beta: f64, joint_dist: &Array2<f64>) -> Result<f64, SemanticError> {
        // Simplified scoring function for β optimization
        let compression_term = joint_dist.sum() / (1.0 + beta);
        let information_term = self.mutual_information(joint_dist)?;
        
        Ok(information_term - beta * compression_term)
    }
}

/// 🧪 Tests for information theory calculations
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        let calculator = InformationTheoryCalculator::default();
        
        // Uniform distribution should have maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = calculator.shannon_entropy(&uniform).unwrap();
        assert!((entropy - 2.0).abs() < 1e-10);
        
        // Deterministic distribution should have zero entropy
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = calculator.shannon_entropy(&deterministic).unwrap();
        assert!(entropy < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let calculator = InformationTheoryCalculator::default();
        
        // KL divergence should be 0 for identical distributions
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.5, 0.3, 0.2];
        let kl_div = calculator.kl_divergence(&p, &q).unwrap();
        assert!(kl_div < 1e-10);
        
        // KL divergence should be positive for different distributions
        let p = vec![0.8, 0.1, 0.1];
        let q = vec![0.33, 0.33, 0.34];
        let kl_div = calculator.kl_divergence(&p, &q).unwrap();
        assert!(kl_div > 0.0);
    }

    #[test]
    fn test_js_divergence() {
        let calculator = InformationTheoryCalculator::default();
        
        // JS divergence should be 0 for identical distributions
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.5, 0.3, 0.2];
        let js_div = calculator.js_divergence(&p, &q).unwrap();
        assert!(js_div < 1e-10);
        
        // JS divergence should be symmetric
        let p = vec![0.8, 0.1, 0.1];
        let q = vec![0.33, 0.33, 0.34];
        let js_div_pq = calculator.js_divergence(&p, &q).unwrap();
        let js_div_qp = calculator.js_divergence(&q, &p).unwrap();
        assert!((js_div_pq - js_div_qp).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information() {
        let calculator = InformationTheoryCalculator::default();
        
        // Create a simple joint distribution
        let joint_dist = Array2::from_shape_vec((2, 2), vec![0.4, 0.1, 0.1, 0.4]).unwrap();
        let mutual_info = calculator.mutual_information(&joint_dist).unwrap();
        
        // Mutual information should be positive for correlated variables
        assert!(mutual_info > 0.0);
    }
} 
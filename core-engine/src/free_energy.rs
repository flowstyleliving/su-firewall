use serde::{Deserialize, Serialize};
use crate::{SemanticError};
use crate::information_theory::InformationTheoryCalculator;

/// Free Energy metrics per token or timestep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyMetrics {
    /// Surprise S = -log p(o)
    pub surprise: f64,
    /// Ambiguity A = H[p(o)] (Shannon entropy)
    pub ambiguity: f64,
    /// Complexity C = KL[q(z) || p(z)] (optional; 0.0 if not provided)
    pub complexity: f64,
    /// Variational Free Energy F ≈ Surprise + Complexity
    pub free_energy: f64,
}

/// Compute Free Energy metrics for a single observation/event using probabilities.
///
/// - `probabilities`: model's probability distribution over tokens/outcomes at this step
/// - `observed_index`: index of the realized token; if None, uses argmax
/// - `prior`: optional prior over latent states/semantics (as a distribution aligned with q_post)
/// - `q_post`: optional posterior over latent states/semantics (same length as prior)
pub fn compute_free_energy_for_token(
    probabilities: &[f64],
    observed_index: Option<usize>,
    prior: Option<&[f64]>,
    q_post: Option<&[f64]>,
) -> Result<FreeEnergyMetrics, SemanticError> {
    if probabilities.is_empty() {
        return Err(SemanticError::InvalidInput { message: "Empty probability vector".to_string() });
    }

    // Normalize probabilities defensively
    let sum_p: f64 = probabilities.iter().sum();
    if !sum_p.is_finite() || sum_p <= 0.0 {
        return Err(SemanticError::InvalidInput { message: "Invalid probability distribution".to_string() });
    }
    let norm_probs: Vec<f64> = probabilities.iter().map(|p| p / sum_p).collect();

    // Determine observed index
    let idx = observed_index.unwrap_or_else(|| argmax_index(&norm_probs));
    if idx >= norm_probs.len() {
        return Err(SemanticError::InvalidInput { message: "Observed index out of bounds".to_string() });
    }
    let p_obs = (norm_probs[idx]).max(1e-12);

    // Surprise: negative log-likelihood (natural log)
    let surprise = -p_obs.ln();

    // Ambiguity: entropy of predictive distribution (log base 2 for consistency with InformationTheoryCalculator)
    let info_calc = InformationTheoryCalculator::default();
    let ambiguity = info_calc.shannon_entropy(&norm_probs)?; // in bits

    // Complexity: KL(q||p) over latent beliefs if provided; else 0
    let complexity = match (prior, q_post) {
        (Some(p0), Some(q1)) => {
            if p0.len() != q1.len() {
                return Err(SemanticError::InvalidInput { message: "prior and q_post lengths must match".to_string() });
            }
            info_calc.kl_divergence(q1, p0)?
        }
        _ => 0.0,
    };

    let free_energy = surprise + complexity;

    Ok(FreeEnergyMetrics {
        surprise,
        ambiguity,
        complexity,
        free_energy,
    })
}

fn argmax_index(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
} 
// 🔄 Alias Ambiguity Defense Layer
// Implements scalar orbit symmetry detection and adaptive entropy injection for ℏₛ calculation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use tracing::{warn, instrument};

/// 🔄 Elliptic curve point with scalar information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EllipticPoint {
    pub x: i64,
    pub y: i64,
    pub modulus: i64,
}

/// 🔄 Scalar orbit symmetry analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryAnalysis {
    /// Symmetry score ∈ [0.0, 1.0] representing mirror alias likelihood
    pub symmetry_score: f64,
    /// Mirror scalar if known (n - k)
    pub mirror_scalar: Option<i64>,
    /// Alias risk ∈ [0.0, 1.0]
    pub alias_risk: f64,
    /// Orbit walk analysis results
    pub orbit_walk: OrbitWalkAnalysis,
    /// Symmetry detection diagnostics
    pub diagnostics: SymmetryDiagnostics,
}

/// 🔄 Orbit walk analysis for ±3 scalar steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitWalkAnalysis {
    /// Points at k-3, k-2, k-1, k, k+1, k+2, k+3
    pub orbit_points: Vec<EllipticPoint>,
    /// Mirroring patterns detected in orbit
    pub mirroring_patterns: Vec<MirroringPattern>,
    /// Orbit complexity score
    pub complexity_score: f64,
}

/// 🔄 Mirroring pattern detected in orbit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirroringPattern {
    /// Pattern type
    pub pattern_type: MirroringType,
    /// Confidence of pattern detection
    pub confidence: f64,
    /// Scalar positions involved
    pub positions: Vec<i64>,
}

/// 🔄 Types of mirroring patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MirroringType {
    /// Perfect mirror at center
    PerfectMirror,
    /// Partial mirror with offset
    PartialMirror { offset: i64 },
    /// Cyclic mirror pattern
    CyclicMirror { cycle_length: i64 },
    /// No clear pattern
    NoPattern,
}

/// 🔄 Symmetry detection diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryDiagnostics {
    /// Mirror point coordinates
    pub mirror_point: Option<EllipticPoint>,
    /// Group order if known
    pub group_order: Option<i64>,
    /// Symmetry detection method used
    pub detection_method: String,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Additional diagnostic information
    pub additional_info: HashMap<String, String>,
}

/// 🔄 Alias Ambiguity Defense Layer
pub struct AliasAmbiguityDefense {
    /// Enable orbit walker for ±3 scalar steps
    pub enable_orbit_walker: bool,
    /// Symmetry score threshold for fail-fast behavior
    pub symmetry_threshold: f64,
    /// Confidence threshold for fail-fast behavior
    pub confidence_threshold: f64,
    /// Alias risk injection weight
    pub alias_risk_weight: f64,
}

impl Default for AliasAmbiguityDefense {
    fn default() -> Self {
        Self {
            enable_orbit_walker: true,
            symmetry_threshold: 0.9,
            confidence_threshold: 0.9,
            alias_risk_weight: 0.3,
        }
    }
}

impl AliasAmbiguityDefense {
    /// 🏗️ Create new Alias Ambiguity Defense Layer
    pub fn new() -> Self {
        Self::default()
    }

    /// 🏗️ Create with custom configuration
    pub fn with_config(
        enable_orbit_walker: bool,
        symmetry_threshold: f64,
        confidence_threshold: f64,
        alias_risk_weight: f64,
    ) -> Self {
        Self {
            enable_orbit_walker,
            symmetry_threshold,
            confidence_threshold,
            alias_risk_weight,
        }
    }

    /// 🔍 Analyze scalar orbit symmetry for given point and predicted scalar
    #[instrument(skip(self, point, predicted_scalar))]
    pub fn analyze_symmetry(
        &self,
        point: &EllipticPoint,
        predicted_scalar: i64,
    ) -> Result<SymmetryAnalysis> {
        let start_time = std::time::Instant::now();

        // Step 1: Compute mirror point Q' = (x, -y mod p)
        let mirror_point = self.compute_mirror_point(point);
        
        // Step 2: Calculate basic symmetry score
        let symmetry_score = self.calculate_symmetry_score(point, &mirror_point, predicted_scalar);
        
        // Step 3: Perform orbit walk if enabled
        let orbit_walk = if self.enable_orbit_walker {
            self.perform_orbit_walk(point, predicted_scalar)?
        } else {
            OrbitWalkAnalysis {
                orbit_points: vec![],
                mirroring_patterns: vec![],
                complexity_score: 0.0,
            }
        };
        
        // Step 4: Calculate alias risk
        let alias_risk = self.calculate_alias_risk(symmetry_score, &orbit_walk);
        
        // Step 5: Determine mirror scalar if possible
        let mirror_scalar = self.detect_mirror_scalar(point, predicted_scalar, &orbit_walk);
        
        // Step 6: Compile diagnostics
        let diagnostics = SymmetryDiagnostics {
            mirror_point: Some(mirror_point),
            group_order: self.estimate_group_order(point),
            detection_method: if self.enable_orbit_walker {
                "orbit_walk_with_mirror_detection".to_string()
            } else {
                "basic_mirror_detection".to_string()
            },
            processing_time_us: start_time.elapsed().as_micros() as u64,
            additional_info: HashMap::new(),
        };

        Ok(SymmetryAnalysis {
            symmetry_score,
            mirror_scalar,
            alias_risk,
            orbit_walk,
            diagnostics,
        })
    }

    /// 🔄 Compute mirror point Q' = (x, -y mod p)
    fn compute_mirror_point(&self, point: &EllipticPoint) -> EllipticPoint {
        let mirror_y = if point.y == 0 { 0 } else { point.modulus - point.y };
        EllipticPoint {
            x: point.x,
            y: mirror_y,
            modulus: point.modulus,
        }
    }

    /// 🧮 Calculate symmetry score ∈ [0.0, 1.0]
    fn calculate_symmetry_score(
        &self,
        point: &EllipticPoint,
        mirror_point: &EllipticPoint,
        _predicted_scalar: i64,
    ) -> f64 {
        // Check if point is on the x-axis (y = 0)
        if point.y == 0 {
            return 1.0; // Perfect symmetry on x-axis
        }

        // Check if point equals its mirror (y = -y mod p)
        if point.y == mirror_point.y {
            return 1.0; // Point is its own mirror
        }

        // Calculate distance-based symmetry score
        let y_distance = (point.y - mirror_point.y).abs() as f64;
        let max_distance = point.modulus as f64;
        let distance_score = 1.0 - (y_distance / max_distance);

        // Additional factors for symmetry detection
        let x_symmetry = if point.x == 0 || point.x == point.modulus / 2 {
            0.8 // Higher symmetry for special x-coordinates
        } else {
            0.3
        };

        // Weighted combination
        distance_score * 0.7 + x_symmetry * 0.3
    }

    /// 🚶 Perform orbit walk ±3 scalar steps from predicted k
    fn perform_orbit_walk(
        &self,
        point: &EllipticPoint,
        predicted_scalar: i64,
    ) -> Result<OrbitWalkAnalysis> {
        let mut orbit_points = Vec::new();
        let _mirroring_patterns: Vec<MirroringPattern> = Vec::new();

        // Walk ±3 steps from predicted scalar
        for offset in -3..=3 {
            let scalar = predicted_scalar + offset;
            let orbit_point = self.compute_scalar_multiply(point, scalar)?;
            orbit_points.push(orbit_point);
        }

        // Analyze mirroring patterns in orbit
        let patterns = self.detect_mirroring_patterns(&orbit_points);
        let complexity_score = self.calculate_orbit_complexity(&orbit_points);

        Ok(OrbitWalkAnalysis {
            orbit_points,
            mirroring_patterns: patterns,
            complexity_score,
        })
    }

    /// 🔄 Compute scalar multiplication k·P
    fn compute_scalar_multiply(&self, point: &EllipticPoint, scalar: i64) -> Result<EllipticPoint> {
        // Simplified scalar multiplication for small scalars
        // In production, use proper elliptic curve arithmetic
        if scalar == 0 {
            return Ok(EllipticPoint {
                x: 0,
                y: 0,
                modulus: point.modulus,
            });
        }

        if scalar == 1 {
            return Ok(point.clone());
        }

        // For small scalars, use repeated addition
        let mut result = point.clone();
        for _ in 1..scalar.abs() {
            result = self.add_points(&result, point)?;
        }

        // Handle negative scalar
        if scalar < 0 {
            result.y = if result.y == 0 { 0 } else { point.modulus - result.y };
        }

        Ok(result)
    }

    /// ➕ Add two elliptic curve points
    fn add_points(&self, p1: &EllipticPoint, p2: &EllipticPoint) -> Result<EllipticPoint> {
        // Simplified point addition
        // In production, use proper elliptic curve arithmetic
        if p1.modulus != p2.modulus {
            return Err(anyhow::anyhow!("Points must have same modulus"));
        }

        // Handle point at infinity
        if p1.x == 0 && p1.y == 0 {
            return Ok(p2.clone());
        }
        if p2.x == 0 && p2.y == 0 {
            return Ok(p1.clone());
        }

        // Simplified addition (not cryptographically secure)
        let x = (p1.x + p2.x) % p1.modulus;
        let y = (p1.y + p2.y) % p1.modulus;

        Ok(EllipticPoint {
            x,
            y,
            modulus: p1.modulus,
        })
    }

    /// 🔍 Detect mirroring patterns in orbit
    fn detect_mirroring_patterns(&self, orbit_points: &[EllipticPoint]) -> Vec<MirroringPattern> {
        let mut patterns = Vec::new();

        // Check for perfect mirror at center (position 3)
        if orbit_points.len() >= 7 {
            let center = 3;
            let mut is_perfect_mirror = true;
            
            for i in 1..=3 {
                let left = &orbit_points[center - i];
                let right = &orbit_points[center + i];
                
                if left.x != right.x || left.y != self.compute_mirror_point(right).y {
                    is_perfect_mirror = false;
                    break;
                }
            }
            
            if is_perfect_mirror {
                patterns.push(MirroringPattern {
                    pattern_type: MirroringType::PerfectMirror,
                    confidence: 0.95,
                    positions: vec![0, 1, 2, 3, 4, 5, 6],
                });
            }
        }

        // Check for partial mirror patterns
        for offset in 1..=2 {
            if orbit_points.len() >= 7 {
                let mut is_partial_mirror = true;
                let mut mirror_positions = Vec::new();
                
                for i in 0..orbit_points.len() {
                    let j = orbit_points.len() - 1 - i;
                    if i < j {
                        let left = &orbit_points[i];
                        let right = &orbit_points[j];
                        
                        if left.x != right.x || left.y != self.compute_mirror_point(right).y {
                            is_partial_mirror = false;
                            break;
                        }
                        mirror_positions.extend_from_slice(&[i as i64, j as i64]);
                    }
                }
                
                if is_partial_mirror {
                    patterns.push(MirroringPattern {
                        pattern_type: MirroringType::PartialMirror { offset },
                        confidence: 0.8,
                        positions: mirror_positions,
                    });
                }
            }
        }

        patterns
    }

    /// 🧮 Calculate orbit complexity score
    fn calculate_orbit_complexity(&self, orbit_points: &[EllipticPoint]) -> f64 {
        if orbit_points.len() < 2 {
            return 0.0;
        }

        // Calculate variance in x and y coordinates
        let x_values: Vec<f64> = orbit_points.iter().map(|p| p.x as f64).collect();
        let y_values: Vec<f64> = orbit_points.iter().map(|p| p.y as f64).collect();

        let x_variance = self.calculate_variance(&x_values);
        let y_variance = self.calculate_variance(&y_values);

        // Normalize by modulus
        let max_variance = (orbit_points[0].modulus as f64).powi(2) / 12.0;
        let normalized_complexity = (x_variance + y_variance) / (2.0 * max_variance);

        normalized_complexity.min(1.0)
    }

    /// 📊 Calculate variance of values
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }

    /// 🧮 Calculate alias risk ∈ [0.0, 1.0]
    fn calculate_alias_risk(&self, symmetry_score: f64, orbit_walk: &OrbitWalkAnalysis) -> f64 {
        // Base risk from symmetry score
        let base_risk = symmetry_score;
        
        // Additional risk from orbit complexity
        let complexity_risk = 1.0 - orbit_walk.complexity_score;
        
        // Risk from mirroring patterns
        let pattern_risk = if orbit_walk.mirroring_patterns.is_empty() {
            0.0
        } else {
            orbit_walk.mirroring_patterns.iter()
                .map(|p| p.confidence)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        };

        // Weighted combination
        base_risk * 0.5 + complexity_risk * 0.3 + pattern_risk * 0.2
    }

    /// 🔍 Detect mirror scalar (n - k) if possible
    fn detect_mirror_scalar(
        &self,
        point: &EllipticPoint,
        _predicted_scalar: i64,
        orbit_walk: &OrbitWalkAnalysis,
    ) -> Option<i64> {
        // Check if we have a perfect mirror pattern
        for pattern in &orbit_walk.mirroring_patterns {
            if let MirroringType::PerfectMirror = pattern.pattern_type {
                // For perfect mirror, mirror scalar is approximately n - k
                // where n is the group order
                if let Some(group_order) = self.estimate_group_order(point) {
                    return Some(group_order - _predicted_scalar);
                }
            }
        }

        None
    }

    /// 🧮 Estimate group order for elliptic curve
    fn estimate_group_order(&self, point: &EllipticPoint) -> Option<i64> {
        // Simplified estimation - in production use proper group order calculation
        // For small moduli, we can use Hasse's theorem bounds
        let p = point.modulus as f64;
        let lower_bound = p + 1.0 - 2.0 * p.sqrt();
        let upper_bound = p + 1.0 + 2.0 * p.sqrt();
        
        // Use average as rough estimate
        let estimated_order = ((lower_bound + upper_bound) / 2.0) as i64;
        Some(estimated_order)
    }

    /// 🚨 Check fail-fast conditions
    pub fn check_fail_fast(
        &self,
        symmetry_score: f64,
        confidence: f64,
    ) -> Option<FallbackResponse> {
        if symmetry_score > self.symmetry_threshold && confidence > self.confidence_threshold {
            Some(FallbackResponse {
                fallback_triggered: true,
                reason: "Symmetry alias detected".to_string(),
                alias_risk: 1.0,
                mirror_scalar: None,
            })
        } else {
            None
        }
    }

    /// 🧮 Inject alias-driven uncertainty into ℏₛ calculation
    pub fn inject_alias_uncertainty(&self, hbar_s: f64, alias_risk: f64) -> f64 {
        // Adjust ℏₛ using: ℏₛ′ = sqrt(Δμ * Δσ + alias_risk)
        let adjusted_hbar_s = (hbar_s.powi(2) + alias_risk * self.alias_risk_weight).sqrt();
        adjusted_hbar_s
    }
}

/// 🔄 Fallback response for symmetry detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackResponse {
    pub fallback_triggered: bool,
    pub reason: String,
    pub alias_risk: f64,
    pub mirror_scalar: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetry_analysis() {
        let defense = AliasAmbiguityDefense::new();
        let point = EllipticPoint { x: 11, y: 6, modulus: 17 };
        let predicted_scalar = 7;

        let analysis = defense.analyze_symmetry(&point, predicted_scalar).unwrap();
        
        assert!(analysis.symmetry_score >= 0.0 && analysis.symmetry_score <= 1.0);
        assert!(analysis.alias_risk >= 0.0 && analysis.alias_risk <= 1.0);
        assert!(analysis.diagnostics.processing_time_us > 0);
    }

    #[test]
    fn test_mirror_point_computation() {
        let defense = AliasAmbiguityDefense::new();
        let point = EllipticPoint { x: 11, y: 6, modulus: 17 };
        
        let mirror = defense.compute_mirror_point(&point);
        assert_eq!(mirror.x, 11);
        assert_eq!(mirror.y, 11); // 17 - 6 = 11
        assert_eq!(mirror.modulus, 17);
    }

    #[test]
    fn test_fail_fast_detection() {
        let defense = AliasAmbiguityDefense::with_config(true, 0.9, 0.9, 0.3);
        
        // Should trigger fail-fast
        let fallback = defense.check_fail_fast(0.95, 0.95);
        assert!(fallback.is_some());
        assert!(fallback.unwrap().fallback_triggered);
        
        // Should not trigger fail-fast
        let fallback = defense.check_fail_fast(0.8, 0.8);
        assert!(fallback.is_none());
    }

    #[test]
    fn test_alias_uncertainty_injection() {
        let defense = AliasAmbiguityDefense::new();
        let original_hbar_s = 1.0;
        let alias_risk = 0.5;
        
        let adjusted_hbar_s = defense.inject_alias_uncertainty(original_hbar_s, alias_risk);
        assert!(adjusted_hbar_s > original_hbar_s);
    }
} 
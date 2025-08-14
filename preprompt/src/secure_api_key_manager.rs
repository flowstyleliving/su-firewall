// 🔐 Enhanced Secure API Key Manager with Semantic Uncertainty
// Ultra-robust ℏₛ = √(Δμ × Δσ) based key validation and management

use anyhow::Result;
use base64::Engine;
use ring::pbkdf2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};
use uuid::Uuid;

#[cfg(not(target_arch = "wasm32"))]
use ring::rand::SecureRandom;

#[cfg(target_arch = "wasm32")]
use getrandom::getrandom;

use crate::semantic_decision_engine::{ProcessUncertainty, RiskLevel, ProcessDecision};

/// 🔑 Secure API key metadata with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureApiKey {
    pub key_id: String,
    pub key_hash: String,
    pub salt: String,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub last_used: Option<u64>,
    pub usage_count: u64,
    pub permissions: ApiKeyPermissions,
    pub rate_limits: RateLimits,
    pub metadata: KeyMetadata,
    pub status: KeyStatus,
}

/// 🎫 Granular permission system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyPermissions {
    pub endpoints: Vec<String>,
    pub max_requests_per_day: u64,
    pub max_batch_size: usize,
    pub geographic_restrictions: Option<Vec<String>>,
    pub ip_allowlist: Option<Vec<String>>,
    pub can_access_admin: bool,
}

/// ⏱️ Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
    pub burst_capacity: u32,
    pub current_window_start: u64,
    pub current_window_count: u32,
}

/// 📊 Key metadata for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    pub created_by: String,
    pub purpose: String,
    pub environment: String, // dev, staging, production
    pub client_name: String,
    pub tier: KeyTier,
}

/// 🏆 API key tiers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeyTier {
    Free,
    Pro,
    Enterprise,
    Internal,
}

/// 🚦 Key status tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeyStatus {
    Active,
    Suspended,
    Revoked,
    Expired,
    PendingRotation,
}

/// 🔍 Key validation result with uncertainty metrics
#[derive(Debug, Clone)]
pub struct KeyValidationResult {
    pub is_valid: bool,
    pub key_info: Option<SecureApiKey>,
    pub validation_uncertainty: ProcessUncertainty,
    pub security_score: f32,
    pub violations: Vec<SecurityViolation>,
    pub recommended_action: KeyAction,
}

/// 🚨 Security violation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityViolation {
    pub violation_type: String,
    pub severity: f32,
    pub timestamp: u64,
    pub details: String,
    pub client_ip: String,
}

/// 🎯 Recommended actions for key validation
#[derive(Debug, Clone, PartialEq)]
pub enum KeyAction {
    Allow,
    AllowWithMonitoring,
    RateLimit,
    RequireRotation,
    Suspend,
    Revoke,
}

/// 🛡️ Secure API Key Manager with semantic uncertainty analysis
pub struct SecureApiKeyManager {
    keys: HashMap<String, SecureApiKey>,
    compromised_keys: std::collections::HashSet<String>,
    security_thresholds: SecurityThresholds,
    #[cfg(not(target_arch = "wasm32"))]
    rng: ring::rand::SystemRandom,
}

/// ⚙️ Security configuration thresholds
#[derive(Debug, Clone)]
pub struct SecurityThresholds {
    pub min_key_entropy: f32,
    pub max_age_days: u64,
    pub max_failed_attempts: u32,
    pub rate_limit_breach_threshold: f32,
    pub uncertainty_escalation_limit: f32,
}

impl Default for SecurityThresholds {
    fn default() -> Self {
        Self {
            min_key_entropy: 0.8,
            max_age_days: 90,
            max_failed_attempts: 10,
            rate_limit_breach_threshold: 0.9,
            uncertainty_escalation_limit: 1.2,
        }
    }
}

impl SecureApiKeyManager {
    /// 🚀 Create new secure API key manager
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            compromised_keys: std::collections::HashSet::new(),
            security_thresholds: SecurityThresholds::default(),
            #[cfg(not(target_arch = "wasm32"))]
            rng: ring::rand::SystemRandom::new(),
        }
    }

    /// 🔑 Generate cryptographically secure API key
    pub fn generate_api_key(&mut self, metadata: KeyMetadata, permissions: ApiKeyPermissions) -> Result<(String, SecureApiKey)> {
        let key_id = Uuid::new_v4().to_string();
        
        // Generate cryptographically secure random key
        let mut key_bytes = [0u8; 32];
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.rng.fill(&mut key_bytes).map_err(|_| anyhow::anyhow!("Failed to generate random key"))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            getrandom(&mut key_bytes).map_err(|_| anyhow::anyhow!("Failed to generate random key"))?;
        }
        let raw_key = base64::engine::general_purpose::STANDARD.encode(&key_bytes);
        
        // Generate unique salt
        let mut salt_bytes = [0u8; 16];
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.rng.fill(&mut salt_bytes).map_err(|_| anyhow::anyhow!("Failed to generate random salt"))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            getrandom(&mut salt_bytes).map_err(|_| anyhow::anyhow!("Failed to generate random salt"))?;
        }
        let salt = base64::engine::general_purpose::STANDARD.encode(&salt_bytes);
        
        // Create secure hash using PBKDF2
        let key_hash = self.hash_api_key(&raw_key, &salt)?;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Set expiration based on tier
        let expires_at = match metadata.tier {
            KeyTier::Free => Some(now + (30 * 24 * 3600)), // 30 days
            KeyTier::Pro => Some(now + (90 * 24 * 3600)),  // 90 days
            KeyTier::Enterprise => Some(now + (365 * 24 * 3600)), // 1 year
            KeyTier::Internal => None, // No expiration
        };
        
        let rate_limits = RateLimits {
            requests_per_minute: match metadata.tier {
                KeyTier::Free => 60,
                KeyTier::Pro => 300,
                KeyTier::Enterprise => 1000,
                KeyTier::Internal => 10000,
            },
            requests_per_hour: match metadata.tier {
                KeyTier::Free => 1000,
                KeyTier::Pro => 10000,
                KeyTier::Enterprise => 50000,
                KeyTier::Internal => 1000000,
            },
            requests_per_day: match metadata.tier {
                KeyTier::Free => 10000,
                KeyTier::Pro => 100000,
                KeyTier::Enterprise => 1000000,
                KeyTier::Internal => 10000000,
            },
            burst_capacity: match metadata.tier {
                KeyTier::Free => 10,
                KeyTier::Pro => 50,
                KeyTier::Enterprise => 200,
                KeyTier::Internal => 1000,
            },
            current_window_start: now,
            current_window_count: 0,
        };
        
        let secure_key = SecureApiKey {
            key_id: key_id.clone(),
            key_hash,
            salt,
            created_at: now,
            expires_at,
            last_used: None,
            usage_count: 0,
            permissions,
            rate_limits,
            metadata,
            status: KeyStatus::Active,
        };
        
        self.keys.insert(key_id.clone(), secure_key.clone());
        
        info!("🔑 SECURE_KEY_GENERATED | ID: {} | Tier: {:?}", key_id, secure_key.metadata.tier);
        
        Ok((raw_key, secure_key))
    }

    /// 🔍 Validate API key with comprehensive security analysis
    pub fn validate_api_key(&mut self, raw_key: &str, client_ip: &str, endpoint: &str) -> KeyValidationResult {
        debug!("🔍 VALIDATING_API_KEY | IP: {} | Endpoint: {}", client_ip, endpoint);
        
        // Find key by attempting to match hash
        let key_match = self.find_key_by_raw(&raw_key);
        
        if let Some((key_id, key)) = key_match {
            self.validate_existing_key(key_id, key, client_ip, endpoint)
        } else {
            self.handle_invalid_key(raw_key, client_ip, endpoint)
        }
    }

    /// 🔐 Hash API key using PBKDF2 with salt
    fn hash_api_key(&self, raw_key: &str, salt: &str) -> Result<String> {
        let salt_bytes = base64::engine::general_purpose::STANDARD.decode(salt)?;
        let mut hash = [0u8; 32];
        
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            std::num::NonZeroU32::new(100_000).unwrap(),
            &salt_bytes,
            raw_key.as_bytes(),
            &mut hash,
        );
        
        Ok(base64::engine::general_purpose::STANDARD.encode(&hash))
    }

    /// 🔍 Find key by raw value (secure comparison)
    fn find_key_by_raw(&self, raw_key: &str) -> Option<(String, SecureApiKey)> {
        for (key_id, stored_key) in &self.keys {
            if let Ok(computed_hash) = self.hash_api_key(raw_key, &stored_key.salt) {
                if computed_hash.as_bytes() == stored_key.key_hash.as_bytes() {
                    return Some((key_id.clone(), stored_key.clone()));
                }
            }
        }
        None
    }

    /// ✅ Validate existing key with uncertainty analysis
    fn validate_existing_key(&mut self, key_id: String, mut key: SecureApiKey, client_ip: &str, endpoint: &str) -> KeyValidationResult {
        let mut violations = Vec::new();
        let delta_mu = 0.8; // Base precision for valid key
        let mut delta_sigma = 0.3; // Base flexibility
        
        // Status validation
        if key.status != KeyStatus::Active {
            violations.push(SecurityViolation {
                violation_type: "inactive_key".to_string(),
                severity: 0.9,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                details: format!("Key status: {:?}", key.status),
                client_ip: client_ip.to_string(),
            });
            delta_sigma += 0.8;
        }

        // Expiration validation
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        if let Some(expires_at) = key.expires_at {
            if now > expires_at {
                violations.push(SecurityViolation {
                    violation_type: "expired_key".to_string(),
                    severity: 1.0,
                    timestamp: now,
                    details: "Key has expired".to_string(),
                    client_ip: client_ip.to_string(),
                });
                delta_sigma += 1.0;
            } else if now > expires_at - (7 * 24 * 3600) { // Expiring within 7 days
                delta_sigma += 0.3;
            }
        }

        // Endpoint permission validation
        if !key.permissions.endpoints.is_empty() && !key.permissions.endpoints.contains(&endpoint.to_string()) {
            violations.push(SecurityViolation {
                violation_type: "unauthorized_endpoint".to_string(),
                severity: 0.7,
                timestamp: now,
                details: format!("Endpoint {} not in allowed list", endpoint),
                client_ip: client_ip.to_string(),
            });
            delta_sigma += 0.5;
        }

        // Geographic restrictions
        if let Some(ref allowed_regions) = key.permissions.geographic_restrictions {
            let client_region = self.get_client_region(client_ip);
            if !allowed_regions.contains(&client_region) {
                violations.push(SecurityViolation {
                    violation_type: "geographic_restriction".to_string(),
                    severity: 0.6,
                    timestamp: now,
                    details: format!("Client region {} not allowed", client_region),
                    client_ip: client_ip.to_string(),
                });
                delta_sigma += 0.4;
            }
        }

        // IP allowlist validation
        if let Some(ref allowed_ips) = key.permissions.ip_allowlist {
            if !allowed_ips.contains(&client_ip.to_string()) {
                violations.push(SecurityViolation {
                    violation_type: "ip_not_allowlisted".to_string(),
                    severity: 0.8,
                    timestamp: now,
                    details: format!("IP {} not in allowlist", client_ip),
                    client_ip: client_ip.to_string(),
                });
                delta_sigma += 0.6;
            }
        }

        // Rate limiting validation
        let rate_limit_status = self.check_rate_limits(&mut key, now);
        if rate_limit_status > self.security_thresholds.rate_limit_breach_threshold {
            violations.push(SecurityViolation {
                violation_type: "rate_limit_breach".to_string(),
                severity: rate_limit_status,
                timestamp: now,
                details: format!("Rate limit utilization: {:.2}%", rate_limit_status * 100.0),
                client_ip: client_ip.to_string(),
            });
            delta_sigma += rate_limit_status * 0.5;
        }

        // Compromised key check
        if self.compromised_keys.contains(&key.key_hash) {
            violations.push(SecurityViolation {
                violation_type: "compromised_key".to_string(),
                severity: 1.0,
                timestamp: now,
                details: "Key appears in compromised key database".to_string(),
                client_ip: client_ip.to_string(),
            });
            delta_sigma += 1.2;
        }

        // Update key usage
        key.last_used = Some(now);
        key.usage_count += 1;
        self.keys.insert(key_id.clone(), key.clone());

        // Calculate semantic uncertainty
        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;
        
        let risk_level = if h_bar > 1.2 {
            RiskLevel::HighCollapse
        } else if h_bar > 0.8 {
            RiskLevel::ModerateInstability
        } else {
            RiskLevel::Stable
        };

        let decision = match risk_level {
            RiskLevel::HighCollapse => ProcessDecision::Escalate,
            RiskLevel::ModerateInstability => ProcessDecision::Monitor,
            RiskLevel::Stable => ProcessDecision::Execute,
        };

        let validation_uncertainty = ProcessUncertainty {
            process_name: "api_key_validation".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            risk_level,
            decision,
            emoji_indicator: "🔑".to_string(),
            relevance_phrase: "API_KEY_VALIDATION".to_string(),
        };

        // Determine recommended action
        let recommended_action = self.determine_key_action(h_bar, &violations);
        let security_score = self.calculate_security_score(&violations, h_bar);
        let is_valid = violations.iter().all(|v| v.severity < 0.8) && h_bar < 1.2;

        info!("🔍 KEY_VALIDATION_COMPLETE | Valid: {} | ℏₛ: {:.3} | Violations: {}", 
              is_valid, h_bar, violations.len());

        KeyValidationResult {
            is_valid,
            key_info: Some(key),
            validation_uncertainty,
            security_score,
            violations,
            recommended_action,
        }
    }

    /// ❌ Handle invalid key scenario
    fn handle_invalid_key(&self, _raw_key: &str, client_ip: &str, endpoint: &str) -> KeyValidationResult {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        let violation = SecurityViolation {
            violation_type: "invalid_key".to_string(),
            severity: 1.0,
            timestamp: now,
            details: "API key not found or invalid".to_string(),
            client_ip: client_ip.to_string(),
        };

        // High uncertainty for invalid keys
        let delta_mu = 1.2; // High uncertainty in precision (unknown key state)
        let delta_sigma = 1.5; // High flexibility/risk
        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

        let validation_uncertainty = ProcessUncertainty {
            process_name: "invalid_key_detection".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            risk_level: RiskLevel::HighCollapse,
            decision: ProcessDecision::Escalate,
            emoji_indicator: "🚫".to_string(),
            relevance_phrase: "INVALID_API_KEY".to_string(),
        };

        warn!("🚫 INVALID_KEY_DETECTED | IP: {} | Endpoint: {} | ℏₛ: {:.3}", 
              client_ip, endpoint, h_bar);

        KeyValidationResult {
            is_valid: false,
            key_info: None,
            validation_uncertainty,
            security_score: 0.0,
            violations: vec![violation],
            recommended_action: KeyAction::Revoke,
        }
    }

    /// ⏱️ Check rate limits and return utilization ratio
    fn check_rate_limits(&self, key: &mut SecureApiKey, now: u64) -> f32 {
        let window_duration = 60; // 1 minute window
        
        // Reset window if needed
        if now - key.rate_limits.current_window_start >= window_duration {
            key.rate_limits.current_window_start = now;
            key.rate_limits.current_window_count = 0;
        }
        
        key.rate_limits.current_window_count += 1;
        
        // Calculate utilization as percentage of limit
        key.rate_limits.current_window_count as f32 / key.rate_limits.requests_per_minute as f32
    }

    /// 🎯 Determine appropriate action based on uncertainty and violations
    fn determine_key_action(&self, h_bar: f32, violations: &[SecurityViolation]) -> KeyAction {
        let max_severity = violations.iter().map(|v| v.severity).fold(0.0, f32::max);
        
        if max_severity >= 1.0 || h_bar > 1.5 {
            KeyAction::Revoke
        } else if max_severity >= 0.8 || h_bar > 1.2 {
            KeyAction::Suspend
        } else if max_severity >= 0.6 || h_bar > 0.9 {
            KeyAction::RequireRotation
        } else if max_severity >= 0.4 || h_bar > 0.7 {
            KeyAction::RateLimit
        } else if max_severity >= 0.2 || h_bar > 0.5 {
            KeyAction::AllowWithMonitoring
        } else {
            KeyAction::Allow
        }
    }

    /// 📊 Calculate overall security score
    fn calculate_security_score(&self, violations: &[SecurityViolation], h_bar: f32) -> f32 {
        let violation_penalty = violations.iter().map(|v| v.severity).sum::<f32>() * 0.2;
        let uncertainty_penalty = (h_bar / 2.0).min(0.5);
        
        (1.0 - violation_penalty - uncertainty_penalty).max(0.0)
    }

    /// 🌍 Get client region (simplified implementation)
    fn get_client_region(&self, _ip: &str) -> String {
        // In production, this would use a GeoIP service
        "US".to_string()
    }

    /// 🔄 Rotate API key
    pub fn rotate_key(&mut self, key_id: &str) -> Result<(String, SecureApiKey)> {
        if let Some(mut old_key) = self.keys.get(key_id).cloned() {
            // Mark old key for rotation
            old_key.status = KeyStatus::PendingRotation;
            self.keys.insert(key_id.to_string(), old_key.clone());
            
            // Generate new key with same permissions
            self.generate_api_key(old_key.metadata, old_key.permissions)
        } else {
            Err(anyhow::anyhow!("Key not found: {}", key_id))
        }
    }

    /// 🗑️ Revoke API key
    pub fn revoke_key(&mut self, key_id: &str) -> Result<()> {
        if let Some(key) = self.keys.get_mut(key_id) {
            key.status = KeyStatus::Revoked;
            info!("🗑️ KEY_REVOKED | ID: {}", key_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Key not found: {}", key_id))
        }
    }

    /// 📈 Get key usage statistics
    pub fn get_key_stats(&self, _key_id: &str) -> Option<KeyValidationResult> {
        // Implementation would return comprehensive statistics
        None
    }

    /// 🚨 Add key to compromised list
    pub fn mark_compromised(&mut self, key_hash: &str) {
        self.compromised_keys.insert(key_hash.to_string());
        warn!("🚨 KEY_MARKED_COMPROMISED | Hash: {}", key_hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let mut manager = SecureApiKeyManager::new();
        
        let metadata = KeyMetadata {
            created_by: "test_user".to_string(),
            purpose: "testing".to_string(),
            environment: "test".to_string(),
            client_name: "test_client".to_string(),
            tier: KeyTier::Pro,
        };
        
        let permissions = ApiKeyPermissions {
            endpoints: vec!["analyze".to_string()],
            max_requests_per_day: 10000,
            max_batch_size: 100,
            geographic_restrictions: None,
            ip_allowlist: None,
            can_access_admin: false,
        };
        
        let result = manager.generate_api_key(metadata, permissions);
        assert!(result.is_ok());
        
        let (raw_key, secure_key) = result.unwrap();
        assert!(!raw_key.is_empty());
        assert_eq!(secure_key.status, KeyStatus::Active);
    }

    #[test]
    fn test_key_validation() {
        let mut manager = SecureApiKeyManager::new();
        
        // Generate a test key
        let metadata = KeyMetadata {
            created_by: "test_user".to_string(),
            purpose: "testing".to_string(),
            environment: "test".to_string(),
            client_name: "test_client".to_string(),
            tier: KeyTier::Free,
        };
        
        let permissions = ApiKeyPermissions {
            endpoints: vec!["analyze".to_string()],
            max_requests_per_day: 1000,
            max_batch_size: 10,
            geographic_restrictions: None,
            ip_allowlist: None,
            can_access_admin: false,
        };
        
        let (raw_key, _) = manager.generate_api_key(metadata, permissions).unwrap();
        
        // Test validation
        let result = manager.validate_api_key(&raw_key, "127.0.0.1", "analyze");
        assert!(result.is_valid);
        assert!(result.validation_uncertainty.h_bar < 1.0);
    }

    #[test]
    fn test_invalid_key() {
        let mut manager = SecureApiKeyManager::new();
        
        let result = manager.validate_api_key("invalid_key", "127.0.0.1", "analyze");
        assert!(!result.is_valid);
        assert!(result.validation_uncertainty.h_bar > 1.0);
        assert_eq!(result.recommended_action, KeyAction::Revoke);
    }

    #[test]
    fn test_key_rotation() {
        let mut manager = SecureApiKeyManager::new();
        
        let metadata = KeyMetadata {
            created_by: "test_user".to_string(),
            purpose: "testing".to_string(),
            environment: "test".to_string(),
            client_name: "test_client".to_string(),
            tier: KeyTier::Pro,
        };
        
        let permissions = ApiKeyPermissions {
            endpoints: vec!["analyze".to_string()],
            max_requests_per_day: 10000,
            max_batch_size: 100,
            geographic_restrictions: None,
            ip_allowlist: None,
            can_access_admin: false,
        };
        
        let (_, old_key) = manager.generate_api_key(metadata, permissions).unwrap();
        
        let rotation_result = manager.rotate_key(&old_key.key_id);
        assert!(rotation_result.is_ok());
        
        let (new_raw_key, new_key) = rotation_result.unwrap();
        assert_ne!(old_key.key_hash, new_key.key_hash);
        assert!(!new_raw_key.is_empty());
    }
}
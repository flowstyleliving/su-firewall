// 🛡️ Advanced API Security Analyzer with Semantic Uncertainty
// Ultra-robust ℏₛ = √(Δμ × Δσ) based security decisions

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::semantic_decision_engine::{ProcessUncertainty, RiskLevel, ProcessDecision};
use crate::secure_api_key_manager::{KeyValidationResult, KeyAction};

/// 🔒 Comprehensive API security assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSecurityAssessment {
    pub endpoint: String,
    pub client_ip: String,
    pub api_key_hash: String,
    pub request_uncertainty: ProcessUncertainty,
    pub security_layers: Vec<SecurityLayer>,
    pub threat_indicators: Vec<ThreatIndicator>,
    pub authentication_score: f32,
    pub authorization_score: f32,
    pub behavioral_score: f32,
    pub payload_security_score: f32,
    pub overall_security_score: f32,
    pub recommended_action: SecurityAction,
    pub security_emoji: String,
    pub security_phrase: String,
}

/// 🛡️ Individual security layer analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityLayer {
    pub layer_name: String,
    pub delta_mu: f32,          // 📐 Security precision
    pub delta_sigma: f32,       // 🌊 Attack surface flexibility
    pub h_bar: f32,             // 🧮 Layer uncertainty
    pub passed: bool,
    pub confidence: f32,
    pub emoji: String,
    pub phrase: String,
}

/// 🚨 Threat indicator with uncertainty measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIndicator {
    pub threat_type: String,
    pub severity: f32,
    pub confidence: f32,
    pub delta_mu: f32,          // 📐 Threat precision
    pub delta_sigma: f32,       // 🌊 Threat variability  
    pub h_bar: f32,             // 🧮 Threat uncertainty
    pub evidence: Vec<String>,
    pub emoji: String,
    pub description: String,
}

/// 🎯 Security action recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityAction {
    Allow,                      // ✅ Low uncertainty, proceed
    AllowWithMonitoring,       // 👀 Medium uncertainty, enhanced logging
    RateLimit,                 // ⏳ Moderate risk, slow down
    Challenge,                 // 🔐 Higher risk, require additional auth
    Block,                     // 🚫 High risk, deny access
    Quarantine,                // 🏥 Critical risk, isolate and investigate
}

/// 📊 Request behavior pattern
#[derive(Debug, Clone)]
struct RequestPattern {
    timestamps: Vec<u64>,
    endpoints: Vec<String>,
    payload_sizes: Vec<usize>,
    user_agents: Vec<String>,
    geographic_locations: Vec<String>,
}

/// 🧮 Advanced API security analyzer with multi-layer uncertainty analysis
pub struct ApiSecurityAnalyzer {
    // Security thresholds
    security_thresholds: SecurityThresholds,
    // Behavioral pattern tracking
    client_patterns: HashMap<String, RequestPattern>,
    // Known threat signatures
    threat_signatures: ThreatSignatureDatabase,
    // Geographic risk scoring
    geographic_risk_map: HashMap<String, f32>,
    // API key reputation tracking
    api_key_reputation: HashMap<String, f32>,
}

/// ⚙️ Configurable security thresholds
#[derive(Debug, Clone)]
pub struct SecurityThresholds {
    pub authentication_min: f32,
    pub authorization_min: f32,
    pub behavioral_min: f32,
    pub payload_security_min: f32,
    pub overall_security_min: f32,
    pub uncertainty_escalation_threshold: f32,
}

/// 🔍 Threat signature database
#[derive(Debug, Clone)]
struct ThreatSignatureDatabase {
    malicious_patterns: HashSet<String>,
    suspicious_user_agents: HashSet<String>,
    known_attack_payloads: HashSet<String>,
    bot_indicators: HashSet<String>,
}

impl Default for SecurityThresholds {
    fn default() -> Self {
        Self {
            authentication_min: 0.7,
            authorization_min: 0.8,
            behavioral_min: 0.6,
            payload_security_min: 0.7,
            overall_security_min: 0.75,
            uncertainty_escalation_threshold: 1.5,
        }
    }
}

impl ApiSecurityAnalyzer {
    /// 🚀 Create new API security analyzer
    pub fn new() -> Self {
        Self {
            security_thresholds: SecurityThresholds::default(),
            client_patterns: HashMap::new(),
            threat_signatures: ThreatSignatureDatabase::new(),
            geographic_risk_map: Self::build_geographic_risk_map(),
            api_key_reputation: HashMap::new(),
        }
    }

    /// 🔍 Comprehensive API security analysis with ℏₛ calculations
    pub fn analyze_request_security(
        &mut self,
        endpoint: &str,
        client_ip: &str,
        api_key: &str,
        user_agent: &str,
        payload: &str,
        headers: &HashMap<String, String>,
        key_validation: &KeyValidationResult,
    ) -> ApiSecurityAssessment {
        
        info!("🛡️ SECURITY_ANALYSIS_START | Endpoint: {} | IP: {}", endpoint, client_ip);

        // 🔐 Layer 1: Authentication Analysis (using secure key validation)
        let auth_layer = self.analyze_authentication_layer(key_validation);
        
        // 🔍 Layer 1.5: Request Fingerprinting Analysis
        let fingerprint_layer = self.analyze_request_fingerprinting_layer(headers, user_agent, endpoint);
        
        // 🎫 Layer 2: Authorization Analysis  
        let authz_layer = self.analyze_authorization_layer(endpoint, api_key);
        
        // 👤 Layer 3: Behavioral Analysis
        let behavior_layer = self.analyze_behavioral_layer(client_ip, endpoint, user_agent);
        
        // 📦 Layer 4: Payload Security Analysis
        let payload_layer = self.analyze_payload_layer(payload, endpoint);
        
        // 🌐 Layer 5: Geographic/Network Analysis
        let network_layer = self.analyze_network_layer(client_ip, headers);

        let security_layers = vec![auth_layer, fingerprint_layer, authz_layer, behavior_layer, payload_layer, network_layer];

        // 🚨 Threat Detection
        let threat_indicators = self.detect_threats(endpoint, client_ip, user_agent, payload, headers);

        // 🧮 Calculate overall security uncertainty
        let overall_uncertainty = self.calculate_overall_security_uncertainty(&security_layers, &threat_indicators);

        // 📊 Calculate individual scores
        let authentication_score = security_layers[0].h_bar;
        let _fingerprint_score = security_layers[1].h_bar;
        let authorization_score = security_layers[2].h_bar;
        let behavioral_score = security_layers[3].h_bar;
        let payload_security_score = security_layers[4].h_bar;

        // 🎯 Calculate overall security score (inverse of uncertainty)
        let overall_security_score = self.calculate_security_score(&security_layers, &threat_indicators);

        // 🎯 Determine security action
        let (action, emoji, phrase) = self.determine_security_action(
            overall_uncertainty.h_bar,
            overall_security_score,
            &threat_indicators,
        );

        // 📝 Update behavioral patterns
        self.update_client_pattern(client_ip, endpoint, payload.len(), user_agent);

        let assessment = ApiSecurityAssessment {
            endpoint: endpoint.to_string(),
            client_ip: client_ip.to_string(),
            api_key_hash: self.hash_api_key(api_key),
            request_uncertainty: overall_uncertainty,
            security_layers,
            threat_indicators,
            authentication_score,
            authorization_score,
            behavioral_score,
            payload_security_score,
            overall_security_score,
            recommended_action: action,
            security_emoji: emoji,
            security_phrase: phrase,
        };

        info!("{} {} | ℏₛ={:.3} | Security={:.3} | Action={:?}", 
              assessment.security_emoji, 
              assessment.security_phrase,
              assessment.request_uncertainty.h_bar,
              assessment.overall_security_score,
              assessment.recommended_action);

        assessment
    }

    /// 🔐 Authentication layer analysis with secure key validation
    fn analyze_authentication_layer(&mut self, key_validation: &KeyValidationResult) -> SecurityLayer {
        // Use uncertainty metrics from secure key validation
        let delta_mu = key_validation.validation_uncertainty.delta_mu;
        let delta_sigma = key_validation.validation_uncertainty.delta_sigma;
        let h_bar = key_validation.validation_uncertainty.h_bar;

        // Determine layer status based on key validation
        let (passed, emoji, phrase) = match key_validation.recommended_action {
            KeyAction::Allow => (true, "🔐".to_string(), "AUTH_SECURE".to_string()),
            KeyAction::AllowWithMonitoring => (true, "👀".to_string(), "AUTH_MONITOR".to_string()),
            KeyAction::RateLimit => (false, "⏳".to_string(), "AUTH_RATE_LIMITED".to_string()),
            KeyAction::RequireRotation => (false, "🔄".to_string(), "AUTH_REQUIRE_ROTATION".to_string()),
            KeyAction::Suspend => (false, "⚠️".to_string(), "AUTH_SUSPENDED".to_string()),
            KeyAction::Revoke => (false, "🚨".to_string(), "AUTH_REVOKED".to_string()),
        };

        SecurityLayer {
            layer_name: "authentication".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            passed,
            confidence: key_validation.security_score,
            emoji,
            phrase,
        }
    }

    /// 🔍 Request Fingerprinting Layer - Protocol and signature analysis
    fn analyze_request_fingerprinting_layer(&self, headers: &HashMap<String, String>, user_agent: &str, endpoint: &str) -> SecurityLayer {
        // 📐 Calculate Δμ (fingerprint precision)
        let mut delta_mu = 0.7; // Base precision for fingerprinting
        
        // TLS/HTTP Version Analysis
        if let Some(protocol_version) = headers.get("http-version") {
            match protocol_version.as_str() {
                "HTTP/2" | "HTTP/3" => delta_mu += 0.1, // Modern protocols
                "HTTP/1.1" => delta_mu += 0.05,         // Standard
                "HTTP/1.0" => delta_mu -= 0.2,          // Suspicious/legacy
                _ => delta_mu -= 0.1,
            }
        }
        
        // Header Fingerprint Analysis
        let header_fingerprint_score = self.analyze_header_fingerprint(headers);
        delta_mu += header_fingerprint_score * 0.3;
        
        // User-Agent Entropy Analysis
        let ua_entropy = self.calculate_user_agent_entropy(user_agent);
        delta_mu += ua_entropy * 0.2;
        
        // Accept Headers Analysis
        let accept_pattern_score = self.analyze_accept_headers(headers);
        delta_mu += accept_pattern_score * 0.15;
        
        // 🌊 Calculate Δσ (fingerprint flexibility/anomaly risk)
        let mut delta_sigma = 0.4; // Base flexibility
        
        // Missing Standard Headers (increases anomaly risk)
        let expected_headers = ["accept", "accept-language", "accept-encoding", "user-agent"];
        let missing_headers = expected_headers.iter()
            .filter(|&header| !headers.contains_key(*header))
            .count();
        delta_sigma += (missing_headers as f32) * 0.15;
        
        // Unusual Header Combinations
        if self.detect_unusual_header_patterns(headers) {
            delta_sigma += 0.4;
        }
        
        // TLS Cipher Suite Analysis (if available)
        if let Some(cipher_info) = headers.get("ssl-cipher") {
            if self.is_weak_cipher(cipher_info) {
                delta_sigma += 0.3;
            }
        }
        
        // Protocol Downgrade Detection
        if self.detect_protocol_downgrade(headers, endpoint) {
            delta_sigma += 0.5;
        }
        
        // Automation Detection via Header Patterns
        if self.detect_automation_headers(headers) {
            delta_sigma += 0.6;
        }
        
        // Request Method Appropriateness
        if let Some(method) = headers.get("method") {
            if !self.is_method_appropriate(method, endpoint) {
                delta_sigma += 0.3;
            }
        }
        
        // 🧮 Calculate ℏₛ for request fingerprinting
        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;
        
        let (passed, emoji, phrase) = if h_bar > 1.3 {
            (false, "🚨".to_string(), "FINGERPRINT_CRITICAL_ANOMALY".to_string())
        } else if h_bar > 1.0 {
            (false, "⚠️".to_string(), "FINGERPRINT_SUSPICIOUS_PATTERN".to_string())
        } else if h_bar > 0.7 {
            (true, "👀".to_string(), "FINGERPRINT_MONITOR_REQUIRED".to_string())
        } else {
            (true, "🔍".to_string(), "FINGERPRINT_NORMAL_PATTERN".to_string())
        };
        
        SecurityLayer {
            layer_name: "request_fingerprinting".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            passed,
            confidence: 1.0 - (h_bar / 1.5).min(1.0),
            emoji,
            phrase,
        }
    }
    
    /// 🔍 Analyze HTTP header fingerprint patterns
    fn analyze_header_fingerprint(&self, headers: &HashMap<String, String>) -> f32 {
        let mut score: f32 = 0.5; // Base score
        
        // Header order analysis (browsers have consistent ordering)
        let header_order_score = self.analyze_header_order(headers);
        score += header_order_score * 0.3;
        
        // Header value patterns
        if let Some(accept) = headers.get("accept") {
            if accept.contains("text/html,application/xhtml+xml,application/xml;q=0.9") {
                score += 0.2; // Browser-like pattern
            } else if accept == "*/*" {
                score -= 0.3; // Tool-like pattern
            }
        }
        
        // Accept-Language patterns
        if let Some(lang) = headers.get("accept-language") {
            if lang.contains("q=") && lang.contains(",") {
                score += 0.15; // Quality factor indicates browser
            }
        }
        
        // Connection header
        if let Some(conn) = headers.get("connection") {
            match conn.to_lowercase().as_str() {
                "keep-alive" => score += 0.1,
                "close" => score -= 0.1,
                _ => {}
            }
        }
        
        score.max(0.0).min(1.0)
    }
    
    /// 📊 Calculate User-Agent entropy
    fn calculate_user_agent_entropy(&self, user_agent: &str) -> f32 {
        if user_agent.is_empty() {
            return 0.0;
        }
        
        // Calculate character frequency distribution
        let mut char_counts = std::collections::HashMap::new();
        for ch in user_agent.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }
        
        let total_chars = user_agent.len() as f32;
        let mut entropy = 0.0;
        
        for &count in char_counts.values() {
            let probability = count as f32 / total_chars;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        // Normalize entropy (typical UA entropy is 3-5 bits)
        (entropy / 5.0).min(1.0)
    }
    
    /// 🎯 Analyze Accept headers patterns
    fn analyze_accept_headers(&self, headers: &HashMap<String, String>) -> f32 {
        let mut score: f32 = 0.5;
        
        // Check for comprehensive Accept header
        if let Some(accept) = headers.get("accept") {
            let media_types = accept.split(',').count();
            if media_types >= 3 {
                score += 0.3; // Rich accept header
            } else if media_types == 1 && accept == "*/*" {
                score -= 0.4; // Generic accept header
            }
        }
        
        // Accept-Encoding analysis
        if let Some(encoding) = headers.get("accept-encoding") {
            if encoding.contains("gzip") && encoding.contains("deflate") {
                score += 0.2;
            }
        }
        
        score.max(0.0_f32).min(1.0_f32)
    }
    
    /// 🔍 Analyze header ordering patterns
    fn analyze_header_order(&self, headers: &HashMap<String, String>) -> f32 {
        // Common browser header order patterns
        let expected_early_headers = ["host", "user-agent", "accept"];
        let mut order_score: f32 = 0.5;
        
        // Check if critical headers are present
        for &header in &expected_early_headers {
            if headers.contains_key(header) {
                order_score += 0.1;
            }
        }
        
        // Penalize if user-agent is missing (very suspicious)
        if !headers.contains_key("user-agent") {
            order_score -= 0.3;
        }
        
        order_score.max(0.0_f32).min(1.0_f32)
    }
    
    /// 🚨 Detect unusual header patterns
    fn detect_unusual_header_patterns(&self, headers: &HashMap<String, String>) -> bool {
        // Check for automation indicators
        let automation_headers = [
            "x-forwarded-for", "x-real-ip", "x-automated-tool",
            "x-requested-with", "x-api-key", "authorization"
        ];
        
        let automation_count = automation_headers.iter()
            .filter(|&header| headers.contains_key(*header))
            .count();
        
        // Multiple automation headers is suspicious
        if automation_count >= 3 {
            return true;
        }
        
        // Check for bot-like User-Agent patterns
        if let Some(ua) = headers.get("user-agent") {
            let bot_indicators = ["bot", "crawler", "spider", "scraper", "python", "curl", "wget"];
            if bot_indicators.iter().any(|&indicator| ua.to_lowercase().contains(indicator)) {
                return true;
            }
        }
        
        // Check for missing accept headers with present user-agent (unusual)
        if headers.contains_key("user-agent") && !headers.contains_key("accept") {
            return true;
        }
        
        false
    }
    
    /// 🔐 Check for weak cipher suites
    fn is_weak_cipher(&self, cipher_info: &str) -> bool {
        let weak_indicators = ["RC4", "DES", "MD5", "SHA1", "NULL"];
        weak_indicators.iter().any(|&weak| cipher_info.contains(weak))
    }
    
    /// ⬇️ Detect protocol downgrade attacks
    fn detect_protocol_downgrade(&self, headers: &HashMap<String, String>, endpoint: &str) -> bool {
        // Check if sensitive endpoint is accessed over HTTP instead of HTTPS
        let sensitive_endpoints = ["login", "auth", "payment", "admin"];
        let is_sensitive = sensitive_endpoints.iter().any(|&ep| endpoint.contains(ep));
        
        if is_sensitive {
            // Check for HTTPS indicators
            if let Some(scheme) = headers.get("x-forwarded-proto") {
                return scheme.to_lowercase() != "https";
            }
            
            // Check for missing security headers
            let security_headers = ["strict-transport-security", "x-frame-options"];
            let security_count = security_headers.iter()
                .filter(|&header| headers.contains_key(*header))
                .count();
            
            return security_count == 0;
        }
        
        false
    }
    
    /// 🤖 Detect automation through header analysis
    fn detect_automation_headers(&self, headers: &HashMap<String, String>) -> bool {
        // Selenium/WebDriver indicators
        if headers.values().any(|v| v.contains("selenium") || v.contains("webdriver")) {
            return true;
        }
        
        // Headless browser indicators
        if let Some(ua) = headers.get("user-agent") {
            if ua.contains("HeadlessChrome") || ua.contains("PhantomJS") {
                return true;
            }
        }
        
        // API client indicators
        let api_indicators = ["postman", "insomnia", "httpie", "curl", "wget"];
        if let Some(ua) = headers.get("user-agent") {
            if api_indicators.iter().any(|&tool| ua.to_lowercase().contains(tool)) {
                return true;
            }
        }
        
        false
    }
    
    /// ✅ Check if HTTP method is appropriate for endpoint
    fn is_method_appropriate(&self, method: &str, endpoint: &str) -> bool {
        match method.to_uppercase().as_str() {
            "GET" => endpoint.contains("health") || endpoint.contains("status"),
            "POST" => endpoint.contains("analyze") || endpoint.contains("batch"),
            "PUT" | "PATCH" => endpoint.contains("admin") || endpoint.contains("config"),
            "DELETE" => endpoint.contains("admin"),
            "OPTIONS" => true, // CORS preflight
            _ => false,
        }
    }

    /// 🎫 Authorization layer analysis
    fn analyze_authorization_layer(&self, endpoint: &str, api_key: &str) -> SecurityLayer {
        // 📐 Calculate Δμ (authorization precision)
        let mut delta_mu = 0.6; // Base precision

        // Endpoint sensitivity
        let endpoint_sensitivity = match endpoint {
            "health" => 0.1,         // 🏥 Public endpoint
            "status" => 0.3,         // 📊 Semi-public
            "analyze" => 0.7,        // 🔍 Core functionality
            "batch" => 0.9,          // 📦 High-resource usage
            _ => 0.5,
        };

        delta_mu += endpoint_sensitivity * 0.4;

        // API key tier (would be determined by actual key validation)
        let api_key_tier = self.determine_api_key_tier(api_key);
        delta_mu += api_key_tier * 0.3;

        // 🌊 Calculate Δσ (authorization flexibility)
        let mut delta_sigma = 0.8;

        // High-privilege endpoints have more flexibility/risk
        if endpoint == "batch" || endpoint == "admin" {
            delta_sigma += 0.6;
        }

        // Time-based access patterns
        delta_sigma += self.calculate_temporal_risk() * 0.4;

        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

        let (passed, emoji, phrase) = if h_bar > 1.0 {
            (false, "🛡️".to_string(), "AUTHZ_INSUFFICIENT_PRIVILEGES".to_string())
        } else if h_bar > 0.7 {
            (true, "👀".to_string(), "AUTHZ_MONITOR_ACCESS".to_string())
        } else {
            (true, "✅".to_string(), "AUTHZ_APPROVED".to_string())
        };

        SecurityLayer {
            layer_name: "authorization".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            passed,
            confidence: 1.0 - (h_bar / 1.5).min(1.0),
            emoji,
            phrase,
        }
    }

    /// 👤 Behavioral analysis layer
    fn analyze_behavioral_layer(&mut self, client_ip: &str, endpoint: &str, user_agent: &str) -> SecurityLayer {
        // 📐 Calculate Δμ (behavioral precision)
        let mut delta_mu = 0.7;

        // Analyze request patterns
        if let Some(pattern) = self.client_patterns.get(client_ip) {
            // Request frequency analysis
            let recent_requests = self.count_recent_requests(pattern, 300); // Last 5 minutes
            if recent_requests > 100 {
                delta_mu -= 0.4; // High frequency reduces precision
            } else if recent_requests < 5 {
                delta_mu += 0.2; // Normal frequency increases precision
            }

            // Endpoint diversity
            let unique_endpoints: HashSet<_> = pattern.endpoints.iter().collect();
            if unique_endpoints.len() == 1 && unique_endpoints.contains(&endpoint.to_string()) {
                delta_mu += 0.1; // Consistent behavior
            }
        } else {
            delta_mu -= 0.2; // New client, less precision
        }

        // User agent analysis
        if self.threat_signatures.suspicious_user_agents.contains(user_agent) {
            delta_mu -= 0.5;
        }

        // 🌊 Calculate Δσ (behavioral flexibility)
        let mut delta_sigma = 1.0;

        // Erratic behavior increases flexibility
        if let Some(pattern) = self.client_patterns.get(client_ip) {
            let payload_variance = self.calculate_payload_variance(pattern);
            delta_sigma += payload_variance * 0.3;

            // Geographic inconsistency
            let unique_locations: HashSet<_> = pattern.geographic_locations.iter().collect();
            if unique_locations.len() > 3 {
                delta_sigma += 0.5; // Multiple locations = higher flexibility
            }
        }

        // Bot-like behavior
        if self.detect_bot_behavior(user_agent, client_ip) {
            delta_sigma += 0.7;
        }

        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

        let (passed, emoji, phrase) = if h_bar > 1.3 {
            (false, "🤖".to_string(), "BEHAVIOR_BOT_DETECTED".to_string())
        } else if h_bar > 0.9 {
            (false, "🚩".to_string(), "BEHAVIOR_SUSPICIOUS_PATTERN".to_string())
        } else {
            (true, "👤".to_string(), "BEHAVIOR_NORMAL_USER".to_string())
        };

        SecurityLayer {
            layer_name: "behavioral".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            passed,
            confidence: 1.0 - (h_bar / 1.5).min(1.0),
            emoji,
            phrase,
        }
    }

    /// 📦 Payload security analysis
    fn analyze_payload_layer(&self, payload: &str, endpoint: &str) -> SecurityLayer {
        // 📐 Calculate Δμ (payload precision)
        let mut delta_mu = 0.8;

        // Size-based analysis
        let payload_size = payload.len();
        if payload_size > 10000 {
            delta_mu -= 0.3; // Large payloads reduce precision
        } else if payload_size < 10 {
            delta_mu += 0.1; // Small payloads increase precision
        }

        // Content analysis
        if self.contains_malicious_patterns(payload) {
            delta_mu -= 0.6;
        }

        // Endpoint-appropriate payload size
        let expected_size_range = match endpoint {
            "analyze" => (10, 5000),
            "batch" => (100, 50000),
            "status" => (0, 100),
            _ => (0, 1000),
        };

        if payload_size >= expected_size_range.0 && payload_size <= expected_size_range.1 {
            delta_mu += 0.2;
        } else {
            delta_mu -= 0.2;
        }

        // 🌊 Calculate Δσ (payload flexibility)
        let mut delta_sigma = 0.7;

        // Unusual characters or encoding
        if payload.chars().any(|c| !c.is_ascii()) {
            delta_sigma += 0.3;
        }

        // JSON structure analysis (if applicable)
        if payload.starts_with('{') && payload.ends_with('}') {
            if let Err(_) = serde_json::from_str::<serde_json::Value>(payload) {
                delta_sigma += 0.5; // Malformed JSON
            }
        }

        // Injection attempt detection
        if self.detect_injection_attempts(payload) {
            delta_sigma += 0.8;
        }

        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

        let (passed, emoji, phrase) = if h_bar > 1.1 {
            (false, "💉".to_string(), "PAYLOAD_INJECTION_RISK".to_string())
        } else if h_bar > 0.8 {
            (false, "📦".to_string(), "PAYLOAD_SUSPICIOUS_CONTENT".to_string())
        } else {
            (true, "✅".to_string(), "PAYLOAD_CLEAN".to_string())
        };

        SecurityLayer {
            layer_name: "payload".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            passed,
            confidence: 1.0 - (h_bar / 1.2).min(1.0),
            emoji,
            phrase,
        }
    }

    /// 🌐 Network layer analysis
    fn analyze_network_layer(&self, client_ip: &str, headers: &HashMap<String, String>) -> SecurityLayer {
        // 📐 Calculate Δμ (network precision)
        let mut delta_mu = 0.6;

        // Geographic risk
        if let Some(&geo_risk) = self.geographic_risk_map.get(client_ip) {
            delta_mu -= geo_risk * 0.4;
        }

        // Header analysis
        if headers.contains_key("x-forwarded-for") {
            delta_mu -= 0.1; // Proxy usage reduces precision
        }

        if headers.contains_key("x-real-ip") {
            delta_mu -= 0.1; // Load balancer reduces precision  
        }

        // TLS/Security headers
        if headers.contains_key("strict-transport-security") {
            delta_mu += 0.1;
        }

        // 🌊 Calculate Δσ (network flexibility)
        let mut delta_sigma = 0.8;

        // Tor/VPN detection
        if self.is_tor_or_vpn_ip(client_ip) {
            delta_sigma += 0.6;
        }

        // Multiple proxy headers indicate high flexibility
        let proxy_headers = ["x-forwarded-for", "x-real-ip", "x-client-ip", "cf-connecting-ip"];
        let proxy_count = proxy_headers.iter()
            .filter(|&header| headers.contains_key(*header))
            .count();
        
        delta_sigma += (proxy_count as f32) * 0.2;

        let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

        let (passed, emoji, phrase) = if h_bar > 1.2 {
            (false, "🌐".to_string(), "NETWORK_HIGH_RISK_ORIGIN".to_string())
        } else if h_bar > 0.9 {
            (false, "🔍".to_string(), "NETWORK_MONITOR_ORIGIN".to_string())
        } else {
            (true, "🌍".to_string(), "NETWORK_TRUSTED_ORIGIN".to_string())
        };

        SecurityLayer {
            layer_name: "network".to_string(),
            delta_mu,
            delta_sigma,
            h_bar,
            passed,
            confidence: 1.0 - (h_bar / 1.3).min(1.0),
            emoji,
            phrase,
        }
    }

    /// 🚨 Comprehensive threat detection
    fn detect_threats(
        &self,
        endpoint: &str,
        client_ip: &str,
        user_agent: &str,
        payload: &str,
        headers: &HashMap<String, String>,
    ) -> Vec<ThreatIndicator> {
        let mut threats = Vec::new();

        // SQL Injection Detection
        if let Some(sql_threat) = self.detect_sql_injection(payload) {
            threats.push(sql_threat);
        }

        // XSS Detection
        if let Some(xss_threat) = self.detect_xss_attempt(payload) {
            threats.push(xss_threat);
        }

        // DDoS Pattern Detection
        if let Some(ddos_threat) = self.detect_ddos_pattern(client_ip, endpoint) {
            threats.push(ddos_threat);
        }

        // Bot Detection
        if let Some(bot_threat) = self.detect_advanced_bot(user_agent, headers) {
            threats.push(bot_threat);
        }

        // Reconnaissance Detection
        if let Some(recon_threat) = self.detect_reconnaissance(endpoint, client_ip) {
            threats.push(recon_threat);
        }

        threats
    }

    // Helper methods for threat detection...
    fn detect_sql_injection(&self, payload: &str) -> Option<ThreatIndicator> {
        let sql_patterns = ["SELECT", "UNION", "DROP", "INSERT", "UPDATE", "DELETE", "'", "\"", "--"];
        let payload_upper = payload.to_uppercase();
        
        let matches = sql_patterns.iter()
            .filter(|&pattern| payload_upper.contains(pattern))
            .count();

        if matches >= 2 {
            let severity = (matches as f32 / sql_patterns.len() as f32).min(1.0);
            let delta_mu = 0.9; // High precision for SQL injection detection
            let delta_sigma = 0.3 + (severity * 0.4); // Variable based on severity
            let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

            Some(ThreatIndicator {
                threat_type: "sql_injection".to_string(),
                severity,
                confidence: 0.8,
                delta_mu,
                delta_sigma,
                h_bar,
                evidence: sql_patterns.iter()
                    .filter(|&pattern| payload_upper.contains(pattern))
                    .map(|s| s.to_string())
                    .collect(),
                emoji: "💉".to_string(),
                description: "SQL_INJECTION_ATTEMPT".to_string(),
            })
        } else {
            None
        }
    }

    fn detect_xss_attempt(&self, payload: &str) -> Option<ThreatIndicator> {
        let xss_patterns = ["<script", "javascript:", "onload=", "onerror=", "eval("];
        let payload_lower = payload.to_lowercase();
        
        let matches = xss_patterns.iter()
            .filter(|&pattern| payload_lower.contains(pattern))
            .count();

        if matches > 0 {
            let severity = (matches as f32 / xss_patterns.len() as f32).min(1.0);
            let delta_mu = 0.85;
            let delta_sigma = 0.4 + (severity * 0.3);
            let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

            Some(ThreatIndicator {
                threat_type: "xss_attempt".to_string(),
                severity,
                confidence: 0.75,
                delta_mu,
                delta_sigma,
                h_bar,
                evidence: xss_patterns.iter()
                    .filter(|&pattern| payload_lower.contains(pattern))
                    .map(|s| s.to_string())
                    .collect(),
                emoji: "🔥".to_string(),
                description: "XSS_INJECTION_ATTEMPT".to_string(),
            })
        } else {
            None
        }
    }

    fn detect_ddos_pattern(&self, _client_ip: &str, _endpoint: &str) -> Option<ThreatIndicator> {
        // This would check request frequency patterns
        // For now, simplified implementation
        None
    }

    fn detect_advanced_bot(&self, user_agent: &str, headers: &HashMap<String, String>) -> Option<ThreatIndicator> {
        let bot_indicators = ["bot", "crawler", "spider", "scraper"];
        let ua_lower = user_agent.to_lowercase();
        
        let bot_score = bot_indicators.iter()
            .filter(|&indicator| ua_lower.contains(indicator))
            .count() as f32 / bot_indicators.len() as f32;

        // Check for missing typical browser headers
        let missing_headers = ["accept", "accept-language", "accept-encoding"]
            .iter()
            .filter(|&header| !headers.contains_key(*header))
            .count() as f32 / 3.0;

        let total_bot_score = (bot_score + missing_headers) / 2.0;

        if total_bot_score > 0.3 {
            let delta_mu = 0.7;
            let delta_sigma = 0.8 + (total_bot_score * 0.4);
            let h_bar = (delta_mu as f64 * delta_sigma as f64).sqrt() as f32;

            Some(ThreatIndicator {
                threat_type: "bot_detection".to_string(),
                severity: total_bot_score,
                confidence: 0.6,
                delta_mu,
                delta_sigma,
                h_bar,
                evidence: vec![format!("User-Agent: {}", user_agent)],
                emoji: "🤖".to_string(),
                description: "AUTOMATED_BOT_DETECTED".to_string(),
            })
        } else {
            None
        }
    }

    fn detect_reconnaissance(&self, _endpoint: &str, _client_ip: &str) -> Option<ThreatIndicator> {
        // Check for scanning patterns across endpoints
        None
    }

    // Additional helper methods...
    fn calculate_overall_security_uncertainty(
        &self,
        layers: &[SecurityLayer],
        threats: &[ThreatIndicator],
    ) -> ProcessUncertainty {
        // Calculate weighted average of layer uncertainties
        let layer_h_bar_sum: f32 = layers.iter().map(|l| l.h_bar).sum();
        let avg_h_bar = layer_h_bar_sum / layers.len() as f32;

        // Add threat uncertainty contribution
        let threat_contribution = threats.iter()
            .map(|t| t.h_bar * t.severity)
            .sum::<f32>() / threats.len().max(1) as f32;

        let combined_h_bar = (avg_h_bar + threat_contribution) / 2.0;

        let risk_level = if combined_h_bar > 1.2 {
            RiskLevel::HighCollapse
        } else if combined_h_bar > 0.8 {
            RiskLevel::ModerateInstability
        } else {
            RiskLevel::Stable
        };

        let decision = match risk_level {
            RiskLevel::HighCollapse => ProcessDecision::Escalate,
            RiskLevel::ModerateInstability => ProcessDecision::Monitor,
            RiskLevel::Stable => ProcessDecision::Execute,
        };

        ProcessUncertainty {
            process_name: "api_security_analysis".to_string(),
            delta_mu: layers.iter().map(|l| l.delta_mu).sum::<f32>() / layers.len() as f32,
            delta_sigma: layers.iter().map(|l| l.delta_sigma).sum::<f32>() / layers.len() as f32,
            h_bar: combined_h_bar,
            risk_level,
            decision,
            emoji_indicator: "🛡️".to_string(),
            relevance_phrase: "API_SECURITY_ASSESSMENT".to_string(),
        }
    }

    fn calculate_security_score(&self, layers: &[SecurityLayer], threats: &[ThreatIndicator]) -> f32 {
        // Security score is inverse of uncertainty
        let passed_layers = layers.iter().filter(|l| l.passed).count() as f32;
        let layer_score = passed_layers / layers.len() as f32;
        
        let threat_penalty = threats.iter().map(|t| t.severity).sum::<f32>() * 0.2;
        
        (layer_score - threat_penalty).max(0.0).min(1.0)
    }

    fn determine_security_action(
        &self,
        h_bar: f32,
        security_score: f32,
        threats: &[ThreatIndicator],
    ) -> (SecurityAction, String, String) {
        // Critical threats override everything
        if threats.iter().any(|t| t.severity > 0.8) {
            return (SecurityAction::Block, "🚫".to_string(), "CRITICAL_THREAT_BLOCKED".to_string());
        }

        // High uncertainty requires escalation
        if h_bar > 1.5 {
            return (SecurityAction::Quarantine, "🏥".to_string(), "API_QUARANTINE_HIGH_UNCERTAINTY".to_string());
        }

        if h_bar > 1.2 {
            return (SecurityAction::Block, "🛑".to_string(), "API_BLOCKED_SECURITY_RISK".to_string());
        }

        if h_bar > 0.9 || security_score < 0.7 {
            return (SecurityAction::Challenge, "🔐".to_string(), "API_CHALLENGE_REQUIRED".to_string());
        }

        if h_bar > 0.7 || security_score < 0.8 {
            return (SecurityAction::RateLimit, "⏳".to_string(), "API_RATE_LIMITED".to_string());
        }

        if h_bar > 0.5 {
            return (SecurityAction::AllowWithMonitoring, "👀".to_string(), "API_MONITOR_ENHANCED".to_string());
        }

        (SecurityAction::Allow, "✅".to_string(), "API_SECURITY_APPROVED".to_string())
    }

    // Utility methods...
    fn hash_api_key(&self, api_key: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        api_key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn is_suspicious_ip(&self, ip: &str) -> bool {
        // Simplified - would integrate with threat intelligence
        ip.starts_with("192.168.") || ip.starts_with("10.") || ip == "127.0.0.1"
    }

    fn determine_api_key_tier(&self, api_key: &str) -> f32 {
        // Simplified tier determination
        if api_key.contains("demo") || api_key.contains("test") {
            0.2
        } else if api_key.len() >= 32 {
            0.8
        } else {
            0.5
        }
    }

    fn calculate_temporal_risk(&self) -> f32 {
        // Check if request is during unusual hours
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let hour = (now / 3600) % 24;
        
        // Higher risk during off-hours (midnight to 6 AM)
        if hour < 6 {
            0.3
        } else {
            0.0
        }
    }

    fn count_recent_requests(&self, pattern: &RequestPattern, seconds: u64) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        pattern.timestamps.iter()
            .filter(|&&timestamp| now - timestamp < seconds)
            .count()
    }

    fn calculate_payload_variance(&self, pattern: &RequestPattern) -> f32 {
        if pattern.payload_sizes.len() < 2 {
            return 0.0;
        }

        let mean = pattern.payload_sizes.iter().sum::<usize>() as f32 / pattern.payload_sizes.len() as f32;
        let variance = pattern.payload_sizes.iter()
            .map(|&size| {
                let diff = size as f32 - mean;
                diff * diff
            })
            .sum::<f32>() / pattern.payload_sizes.len() as f32;

        (variance.sqrt() / mean).min(1.0)
    }

    fn detect_bot_behavior(&self, user_agent: &str, _ip: &str) -> bool {
        self.threat_signatures.bot_indicators.iter()
            .any(|indicator| user_agent.to_lowercase().contains(indicator))
    }

    fn contains_malicious_patterns(&self, payload: &str) -> bool {
        self.threat_signatures.known_attack_payloads.iter()
            .any(|pattern| payload.contains(pattern))
    }

    fn detect_injection_attempts(&self, payload: &str) -> bool {
        let injection_patterns = ["../", "../../", "<script", "javascript:", "eval(", "exec("];
        injection_patterns.iter()
            .any(|&pattern| payload.to_lowercase().contains(pattern))
    }

    fn is_tor_or_vpn_ip(&self, _ip: &str) -> bool {
        // Simplified - would integrate with VPN/Tor detection service
        false
    }

    fn update_client_pattern(&mut self, ip: &str, endpoint: &str, payload_size: usize, user_agent: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let pattern = self.client_patterns.entry(ip.to_string()).or_insert_with(|| RequestPattern {
            timestamps: Vec::new(),
            endpoints: Vec::new(),
            payload_sizes: Vec::new(),
            user_agents: Vec::new(),
            geographic_locations: Vec::new(),
        });

        pattern.timestamps.push(now);
        pattern.endpoints.push(endpoint.to_string());
        pattern.payload_sizes.push(payload_size);
        pattern.user_agents.push(user_agent.to_string());

        // Keep only recent history (last 24 hours)
        let cutoff = now - 86400;
        pattern.timestamps.retain(|&timestamp| timestamp > cutoff);
        
        // Limit history size
        if pattern.timestamps.len() > 1000 {
            pattern.timestamps.drain(0..100);
            pattern.endpoints.drain(0..100);
            pattern.payload_sizes.drain(0..100);
            pattern.user_agents.drain(0..100);
        }
    }

    fn build_geographic_risk_map() -> HashMap<String, f32> {
        // Simplified geographic risk mapping
        let mut map = HashMap::new();
        map.insert("127.0.0.1".to_string(), 0.0);
        map.insert("localhost".to_string(), 0.0);
        map
    }
}

impl ThreatSignatureDatabase {
    fn new() -> Self {
        Self {
            malicious_patterns: ["../", "<script", "javascript:", "eval("].iter().map(|s| s.to_string()).collect(),
            suspicious_user_agents: ["bot", "crawler", "spider", "scraper"].iter().map(|s| s.to_string()).collect(),
            known_attack_payloads: ["UNION SELECT", "DROP TABLE", "exec("].iter().map(|s| s.to_string()).collect(),
            bot_indicators: ["bot", "crawler", "spider", "headless"].iter().map(|s| s.to_string()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_security_analysis() {
        use crate::secure_api_key_manager::{KeyValidationResult, KeyAction};
        use crate::semantic_decision_engine::{ProcessUncertainty, RiskLevel, ProcessDecision};
        
        let mut analyzer = ApiSecurityAnalyzer::new();
        
        // Create mock key validation result
        let mock_key_validation = KeyValidationResult {
            is_valid: true,
            key_info: None,
            validation_uncertainty: ProcessUncertainty {
                process_name: "test".to_string(),
                delta_mu: 0.7,
                delta_sigma: 0.3,
                h_bar: 0.46,
                risk_level: RiskLevel::Stable,
                decision: ProcessDecision::Execute,
                emoji_indicator: "🔑".to_string(),
                relevance_phrase: "TEST_KEY".to_string(),
            },
            security_score: 0.8,
            violations: Vec::new(),
            recommended_action: KeyAction::Allow,
        };
        
        let headers = HashMap::new();
        let assessment = analyzer.analyze_request_security(
            "analyze",
            "192.168.1.1",
            "test-api-key-12345678",
            "Mozilla/5.0",
            r#"{"prompt": "What is AI?"}"#,
            &headers,
            &mock_key_validation,
        );
        
        assert!(!assessment.security_emoji.is_empty());
        assert!(!assessment.security_phrase.is_empty());
        assert!(assessment.overall_security_score >= 0.0 && assessment.overall_security_score <= 1.0);
    }

    #[test]
    fn test_sql_injection_detection() {
        let analyzer = ApiSecurityAnalyzer::new();
        
        let malicious_payload = "'; DROP TABLE users; --";
        let threat = analyzer.detect_sql_injection(malicious_payload);
        
        assert!(threat.is_some());
        let threat = threat.unwrap();
        assert_eq!(threat.threat_type, "sql_injection");
        assert!(threat.severity > 0.0);
    }

    #[test]
    fn test_security_layer_analysis() {
        use crate::secure_api_key_manager::{KeyValidationResult, KeyAction};
        use crate::semantic_decision_engine::{ProcessUncertainty, RiskLevel, ProcessDecision};
        
        let mut analyzer = ApiSecurityAnalyzer::new();
        
        // Create mock key validation result
        let mock_key_validation = KeyValidationResult {
            is_valid: true,
            key_info: None,
            validation_uncertainty: ProcessUncertainty {
                process_name: "test".to_string(),
                delta_mu: 0.7,
                delta_sigma: 0.3,
                h_bar: 0.46,
                risk_level: RiskLevel::Stable,
                decision: ProcessDecision::Execute,
                emoji_indicator: "🔑".to_string(),
                relevance_phrase: "TEST_KEY".to_string(),
            },
            security_score: 0.8,
            violations: Vec::new(),
            recommended_action: KeyAction::Allow,
        };
        
        let auth_layer = analyzer.analyze_authentication_layer(&mock_key_validation);
        
        assert!(auth_layer.h_bar > 0.0);
        assert!(!auth_layer.emoji.is_empty());
        assert!(!auth_layer.phrase.is_empty());
    }
}
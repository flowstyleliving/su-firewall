# 🔒 Security Guide

**Comprehensive security documentation for Semantic Uncertainty Runtime**

## 📋 Table of Contents

- [Security Overview](#-security-overview)
- [API Key Management](#-api-key-management)
- [Firewall Systems](#-firewall-systems)
- [Access Control](#-access-control)
- [SSL/TLS Configuration](#-ssltls-configuration)
- [Monitoring and Auditing](#-monitoring-and-auditing)

## 🔒 Security Overview

### Security Principles

1. **Zero Trust**: Verify every request and connection
2. **Least Privilege**: Minimal access required for functionality
3. **Defense in Depth**: Multiple layers of security
4. **Secure by Default**: Secure configuration out of the box

### Threat Model

| Threat | Risk Level | Mitigation |
|--------|------------|------------|
| **API Key Exposure** | High | Environment variables, rotation |
| **DDoS Attacks** | Medium | Rate limiting, Cloudflare protection |
| **Data Breaches** | Low | No sensitive data storage |
| **Code Injection** | Low | Input validation, sanitization |

## 🔑 API Key Management

### Secure Storage

```bash
# Environment variables (recommended)
export API_KEY="your-secure-api-key"
export CLOUDFLARE_API_TOKEN="your-cloudflare-token"

# .env file (development only)
API_KEY=your-secure-api-key
CLOUDFLARE_API_TOKEN=your-cloudflare-token
```

### Key Rotation

```bash
# Generate new API key
openssl rand -base64 32

# Update environment
export API_KEY="new-generated-key"

# Deploy with new key
./scripts/deploy.sh cloudflare --key "new-generated-key"
```

### Key Validation

```rust
// Validate API key format
fn validate_api_key(key: &str) -> bool {
    key.len() >= 32 && key.chars().all(|c| c.is_alphanumeric())
}

// Check key permissions
fn check_key_permissions(key: &str) -> Permissions {
    match key {
        "admin-key" => Permissions::Admin,
        "read-only-key" => Permissions::ReadOnly,
        _ => Permissions::Standard
    }
}
```

## 🛡️ Firewall Systems

### HBAR Gated Firewall

The HBAR Gated Firewall uses semantic uncertainty thresholds to block potentially harmful requests:

```rust
// Firewall implementation
pub struct HbarGatedFirewall {
    critical_threshold: f64,
    warning_threshold: f64,
    safe_threshold: f64,
}

impl HbarGatedFirewall {
    pub fn check_request(&self, prompt: &str, output: &str) -> FirewallResult {
        let hbar = self.calculate_semantic_uncertainty(prompt, output);
        
        match hbar {
            h if h <= self.critical_threshold => FirewallResult::Block,
            h if h <= self.warning_threshold => FirewallResult::Warn,
            _ => FirewallResult::Allow,
        }
    }
}
```

### Scalar Firewall

The Scalar Firewall provides additional protection through scalar analysis:

```rust
// Scalar analysis
pub struct ScalarFirewall {
    max_input_length: usize,
    allowed_characters: HashSet<char>,
    blocked_patterns: Vec<Regex>,
}

impl ScalarFirewall {
    pub fn validate_input(&self, input: &str) -> ValidationResult {
        // Check length
        if input.len() > self.max_input_length {
            return ValidationResult::TooLong;
        }
        
        // Check characters
        if !input.chars().all(|c| self.allowed_characters.contains(&c)) {
            return ValidationResult::InvalidCharacters;
        }
        
        // Check patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(input) {
                return ValidationResult::BlockedPattern;
            }
        }
        
        ValidationResult::Valid
    }
}
```

### Configuration

```toml
# firewall-config.toml
[hbar_gate]
critical_threshold = 0.8
warning_threshold = 1.2
safe_threshold = 1.5

[scalar_firewall]
max_input_length = 10000
allowed_characters = ["a-z", "A-Z", "0-9", " ", ".", ",", "!", "?"]
blocked_patterns = [
    "script",
    "javascript:",
    "data:text/html"
]
```

## 🔐 Access Control

### Authentication

```rust
// API key authentication
pub struct ApiKeyAuth {
    valid_keys: HashSet<String>,
    rate_limits: HashMap<String, RateLimit>,
}

impl ApiKeyAuth {
    pub fn authenticate(&self, key: &str) -> AuthResult {
        if !self.valid_keys.contains(key) {
            return AuthResult::InvalidKey;
        }
        
        if self.is_rate_limited(key) {
            return AuthResult::RateLimited;
        }
        
        AuthResult::Authenticated
    }
}
```

### Rate Limiting

```rust
// Rate limiting implementation
pub struct RateLimiter {
    requests_per_minute: u32,
    requests_per_hour: u32,
    requests_per_day: u32,
}

impl RateLimiter {
    pub fn check_rate_limit(&self, key: &str) -> RateLimitResult {
        let current_time = SystemTime::now();
        let requests = self.get_request_count(key, current_time);
        
        if requests.minute > self.requests_per_minute {
            RateLimitResult::MinuteLimitExceeded
        } else if requests.hour > self.requests_per_hour {
            RateLimitResult::HourLimitExceeded
        } else if requests.day > self.requests_per_day {
            RateLimitResult::DayLimitExceeded
        } else {
            RateLimitResult::Allowed
        }
    }
}
```

### CORS Configuration

```javascript
// CORS headers for Cloudflare Workers
const corsHeaders = {
    'Access-Control-Allow-Origin': 'https://semanticuncertainty.com',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
    'Access-Control-Max-Age': '86400',
};

// Handle preflight requests
if (request.method === 'OPTIONS') {
    return new Response(null, {
        status: 200,
        headers: corsHeaders,
    });
}
```

## 🔒 SSL/TLS Configuration

### Certificate Management

```bash
# Check SSL certificate
openssl s_client -connect semanticuncertainty.com:443 -servername semanticuncertainty.com

# Verify certificate chain
openssl verify -CAfile /path/to/ca-bundle.crt certificate.crt

# Test SSL configuration
curl -I https://semanticuncertainty.com
```

### Security Headers

```javascript
// Security headers for Cloudflare Workers
const securityHeaders = {
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
};

// Apply headers to response
return new Response(responseBody, {
    status: 200,
    headers: { ...corsHeaders, ...securityHeaders },
});
```

### TLS Configuration

```toml
# TLS configuration for production
[tls]
min_version = "1.2"
cipher_suites = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256"
]
```

## 📊 Monitoring and Auditing

### Logging

```rust
// Structured logging
use tracing::{info, warn, error};

pub struct SecurityLogger {
    log_level: Level,
    audit_enabled: bool,
}

impl SecurityLogger {
    pub fn log_request(&self, request: &Request, result: &AuthResult) {
        info!(
            "API request: {} {} - Result: {:?}",
            request.method,
            request.path,
            result
        );
        
        if self.audit_enabled {
            self.audit_log(request, result);
        }
    }
    
    pub fn log_security_event(&self, event: SecurityEvent) {
        warn!("Security event: {:?}", event);
        
        // Alert administrators for critical events
        if event.severity == Severity::Critical {
            self.send_alert(event);
        }
    }
}
```

### Monitoring

```javascript
// Cloudflare Workers monitoring
addEventListener('fetch', event => {
    const startTime = Date.now();
    
    event.respondWith(handleRequest(event.request));
    
    // Log performance metrics
    const duration = Date.now() - startTime;
    console.log(`Request processed in ${duration}ms`);
});

// Error monitoring
addEventListener('error', event => {
    console.error('Worker error:', event.error);
    
    // Send to monitoring service
    fetch('https://monitoring.service.com/error', {
        method: 'POST',
        body: JSON.stringify({
            error: event.error.message,
            stack: event.error.stack,
            timestamp: new Date().toISOString()
        })
    });
});
```

### Alerting

```rust
// Security alerting system
pub struct SecurityAlerts {
    webhook_url: String,
    alert_threshold: u32,
}

impl SecurityAlerts {
    pub async fn send_alert(&self, event: SecurityEvent) {
        let alert = Alert {
            severity: event.severity,
            message: event.description,
            timestamp: SystemTime::now(),
            source: event.source,
        };
        
        // Send to webhook
        let client = reqwest::Client::new();
        let _response = client
            .post(&self.webhook_url)
            .json(&alert)
            .send()
            .await;
    }
}
```

## 🧪 Security Testing

### Penetration Testing

```bash
# Test API endpoints
curl -X POST "https://api.semanticuncertainty.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}'

# Test rate limiting
for i in {1..100}; do
  curl -X POST "https://api.semanticuncertainty.com/analyze" \
    -H "X-API-Key: test-key" \
    -d '{"prompt": "test", "output": "test"}'
done

# Test input validation
curl -X POST "https://api.semanticuncertainty.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<script>alert(\"xss\")</script>", "output": "test"}'
```

### Security Headers Testing

```bash
# Test security headers
curl -I https://semanticuncertainty.com

# Expected headers:
# Strict-Transport-Security: max-age=31536000; includeSubDomains
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# X-XSS-Protection: 1; mode=block
```

### SSL/TLS Testing

```bash
# Test SSL configuration
nmap --script ssl-enum-ciphers -p 443 semanticuncertainty.com

# Test certificate
openssl s_client -connect semanticuncertainty.com:443 -servername semanticuncertainty.com
```

## 📋 Security Checklist

### Pre-Deployment
- [ ] API keys stored in environment variables
- [ ] SSL certificates valid and properly configured
- [ ] Security headers implemented
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] Error handling configured
- [ ] Logging enabled

### Post-Deployment
- [ ] SSL certificate verification
- [ ] Security headers verification
- [ ] Rate limiting testing
- [ ] Penetration testing completed
- [ ] Monitoring alerts configured
- [ ] Backup procedures tested

### Ongoing
- [ ] Regular security audits
- [ ] API key rotation
- [ ] Certificate renewal monitoring
- [ ] Security patch updates
- [ ] Access log review
- [ ] Incident response testing

---

**For technical implementation details, see the [Technical Documentation](../technical/README.md).** 
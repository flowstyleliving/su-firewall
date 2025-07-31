# 🚀 Semantic Uncertainty Runtime API

## Overview

The Semantic Uncertainty Runtime provides high-performance semantic uncertainty analysis using the core equation **ℏₛ = √(Δμ × Δσ)**. This API has been optimized to return only essential fields for maximum performance and clarity.

## Core Equation

```
ℏₛ = √(Δμ × Δσ)

Where:
- ℏₛ = Semantic uncertainty constant
- Δμ = Precision (entropy-based)
- Δσ = Flexibility (JSD-based)
```

## Base URL

- **Production**: `https://semanticuncertainty.com`
- **Staging**: `https://semantic-uncertainty-runtime-staging.mys628.workers.dev`

## Endpoints

### Health Check

**GET** `/health` or `/api/v1/health`

Returns the health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "runtime": "semantic-uncertainty-runtime",
  "version": "1.0.0",
  "core_equation": "ℏₛ = √(Δμ × Δσ)",
  "timestamp": "2025-07-31T01:12:50.606Z"
}
```

### Semantic Uncertainty Analysis

**POST** `/api/v1/analyze`

Analyzes semantic uncertainty between a prompt and output.

**Request Body:**
```json
{
  "prompt": "What is artificial intelligence?",
  "output": "AI is artificial intelligence that mimics human cognitive functions.",
  "method": "both"  // Optional: "jsd-kl", "fisher", "both"
}
```

**Response:**
```json
{
  "method": "both",
  "core_equation": "ℏₛ = √(Δμ × Δσ)",
  "precision": 0.156,
  "flexibility": 0.156,
  "semantic_uncertainty": 0.884,
  "raw_hbar": 0.884,
  "risk_level": "Safe",
  "processing_time_ms": 0,
  "request_id": "058bba6f-3c10-4207-9330-5d7906781e64",
  "timestamp": "2025-07-31T01:12:50.606Z"
}
```

## Response Fields

### Core Metrics (Essential)

| Field | Type | Description | Rating |
|-------|------|-------------|--------|
| `raw_hbar` | number | Core semantic uncertainty ℏₛ | 10/10 |
| `precision` | number | Δμ - Entropy-based precision | 10/10 |
| `flexibility` | number | Δσ - JSD-based flexibility | 10/10 |
| `core_equation` | string | Mathematical foundation | 10/10 |
| `semantic_uncertainty` | number | Alternative name for raw_hbar | 8/10 |
| `risk_level` | string | Human-readable risk assessment | 8/10 |

### Metadata (Essential)

| Field | Type | Description | Rating |
|-------|------|-------------|--------|
| `request_id` | string | Request tracking identifier | 9/10 |
| `timestamp` | string | ISO timestamp | 9/10 |
| `processing_time_ms` | number | Performance monitoring | 9/10 |
| `method` | string | Calculation method used | 6/10 |

## Risk Levels

- **Safe**: ℏₛ > 0.7
- **HighRisk**: 0.5 ≤ ℏₛ ≤ 0.7  
- **Warning**: 0.3 ≤ ℏₛ < 0.5
- **Critical**: ℏₛ < 0.3

## Error Responses

### 400 Bad Request
```json
{
  "error": "Missing required fields: prompt and output"
}
```

### 404 Not Found
```json
{
  "error": "Endpoint not found",
  "available_endpoints": ["/health", "/api/v1/health", "/api/v1/analyze"]
}
```

### 500 Internal Server Error
```json
{
  "error": "Analysis failed",
  "details": "Error description"
}
```

## Performance

- **Average Response Time**: < 200ms
- **WASM Integration**: Active for optimal performance
- **Fallback**: JavaScript implementation if WASM unavailable
- **CORS**: Enabled for cross-origin requests

## Rate Limits

Currently no rate limits are enforced, but please use responsibly.

## CORS Headers

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key
```

## Examples

### cURL
```bash
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is quantum mechanics?",
    "output": "Quantum mechanics is a fundamental theory in physics."
  }'
```

### Python
```python
import requests

response = requests.post(
    "https://semanticuncertainty.com/api/v1/analyze",
    json={
        "prompt": "What is quantum mechanics?",
        "output": "Quantum mechanics is a fundamental theory in physics."
    }
)

result = response.json()
print(f"Uncertainty: {result['raw_hbar']}")
print(f"Risk Level: {result['risk_level']}")
```

### JavaScript
```javascript
const response = await fetch('https://semanticuncertainty.com/api/v1/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'What is quantum mechanics?',
    output: 'Quantum mechanics is a fundamental theory in physics.'
  })
});

const result = await response.json();
console.log(`Uncertainty: ${result.raw_hbar}`);
console.log(`Risk Level: ${result.risk_level}`);
```

## Integration

This API is designed for integration with:
- AI/ML pipelines
- Content moderation systems
- Quality assurance tools
- Research applications
- Real-time monitoring systems

## Support

For issues or questions, please check the health endpoint first to verify service status. 
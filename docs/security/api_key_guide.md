# 🔑 **SEMANTIC UNCERTAINTY API KEY MANAGEMENT GUIDE**

## 🎯 **OVERVIEW**

This guide covers everything you need to know about managing API keys for the Semantic Uncertainty Runtime API.

---

## 🚀 **QUICK START**

### **1. Generate Your First API Key**

```bash
# Install the key manager
python3 api_key_manager.py generate --name "John Yue Production" --email "john@inference.ai" --tier enterprise
```

### **2. Test Your API Key**

```bash
curl -X POST "https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_GENERATED_KEY" \
  -d '{"prompt": "Write a guide on AI safety", "model": "gpt4"}'
```

---

## 🔐 **API KEY FORMATS**

### **Key Structure**
```
su_[tier]_[base64_random]
```

### **Examples**
- **Free Tier**: `su_free_kJ8mN9pQ2rS5tU7vW0xY1zA3bC4dE6fG9hI2jK5lM8nO1pR4sT7uV0wX3yZ6`
- **Pro Tier**: `su_pro_mN9pQ2rS5tU7vW0xY1zA3bC4dE6fG9hI2jK5lM8nO1pR4sT7uV0wX3yZ6kJ8`
- **Enterprise**: `su_ent_N9pQ2rS5tU7vW0xY1zA3bC4dE6fG9hI2jK5lM8nO1pR4sT7uV0wX3yZ6kJ8m`
- **Unlimited**: `su_unl_9pQ2rS5tU7vW0xY1zA3bC4dE6fG9hI2jK5lM8nO1pR4sT7uV0wX3yZ6kJ8mN`

---

## 📊 **TIER COMPARISON**

| **Tier** | **Rate Limit** | **Features** | **Price** |
|----------|----------------|--------------|-----------|
| **Free** | 100 req/min | Basic analysis | $0/month |
| **Pro** | 1,000 req/min | Batch processing, Priority support | $49/month |
| **Enterprise** | 10,000 req/min | Custom models, SLA, Dedicated support | $500/month |
| **Unlimited** | 100,000 req/min | White-label, Custom deployment | Custom |

---

## 🛠️ **MANAGEMENT COMMANDS**

### **Generate New Keys**

```bash
# Free tier key
python3 api_key_manager.py generate --name "Demo User" --tier free

# Pro tier key
python3 api_key_manager.py generate --name "Startup Company" --email "dev@startup.com" --tier pro

# Enterprise key
python3 api_key_manager.py generate --name "John Yue Production" --email "john@inference.ai" --tier enterprise
```

### **List All Keys**

```bash
python3 api_key_manager.py list
```

### **Deactivate Key**

```bash
python3 api_key_manager.py deactivate --name "Old Demo Key"
```

### **View Usage Statistics**

```bash
# Last 30 days (default)
python3 api_key_manager.py stats

# Last 7 days
python3 api_key_manager.py stats --days 7
```

---

## 🔧 **USAGE EXAMPLES**

### **JavaScript/Node.js**

```javascript
const API_KEY = 'su_ent_YOUR_KEY_HERE';
const API_URL = 'https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev';

// Single analysis
async function analyzePrompt(prompt, model = 'gpt4') {
    const response = await fetch(`${API_URL}/api/v1/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
        },
        body: JSON.stringify({ prompt, model })
    });
    
    return await response.json();
}

// Batch analysis
async function batchAnalyze(prompts, model = 'gpt4') {
    const response = await fetch(`${API_URL}/api/v1/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
        },
        body: JSON.stringify({ prompts, model })
    });
    
    return await response.json();
}
```

### **Python**

```python
import requests

API_KEY = 'su_ent_YOUR_KEY_HERE'
API_URL = 'https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev'

headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
}

# Single analysis
def analyze_prompt(prompt, model='gpt4'):
    response = requests.post(
        f'{API_URL}/api/v1/analyze',
        headers=headers,
        json={'prompt': prompt, 'model': model}
    )
    return response.json()

# Batch analysis
def batch_analyze(prompts, model='gpt4'):
    response = requests.post(
        f'{API_URL}/api/v1/batch',
        headers=headers,
        json={'prompts': prompts, 'model': model}
    )
    return response.json()
```

### **cURL**

```bash
# Single analysis
curl -X POST "https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: su_ent_YOUR_KEY_HERE" \
  -d '{
    "prompt": "Write a comprehensive guide on quantum computing",
    "model": "gpt4"
  }'

# Batch analysis
curl -X POST "https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev/api/v1/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: su_ent_YOUR_KEY_HERE" \
  -d '{
    "prompts": [
      "Explain AI safety in simple terms",
      "What are the risks of AGI?",
      "How can we ensure AI alignment?"
    ],
    "model": "gpt4"
  }'
```

---

## 🛡️ **SECURITY BEST PRACTICES**

### **1. Key Storage**
- ✅ Store keys in environment variables
- ✅ Use secure key management services (AWS Secrets Manager, HashiCorp Vault)
- ❌ Never commit keys to version control
- ❌ Don't hardcode keys in client-side code

### **2. Key Rotation**
- 🔄 Rotate keys every 90 days
- 🔄 Use different keys for different environments
- 🔄 Monitor for suspicious usage patterns

### **3. Rate Limiting**
- ⚡ Respect rate limits (check response headers)
- ⚡ Implement exponential backoff for retries
- ⚡ Monitor usage to avoid hitting limits

### **4. Error Handling**

```javascript
async function safeApiCall(prompt) {
    try {
        const response = await analyzePrompt(prompt);
        
        if (!response.success) {
            console.error('API Error:', response.error);
            return null;
        }
        
        return response.data;
    } catch (error) {
        if (error.status === 401) {
            console.error('Authentication failed - check your API key');
        } else if (error.status === 429) {
            console.error('Rate limit exceeded - wait before retrying');
        } else {
            console.error('API call failed:', error);
        }
        return null;
    }
}
```

---

## 📈 **MONITORING & ANALYTICS**

### **Usage Tracking**

The API automatically tracks:
- 📊 Request count per key
- ⏱️ Response times
- 🚨 Error rates
- 🌍 Geographic distribution
- 🔍 Endpoint usage patterns

### **Response Headers**

```javascript
// Check rate limit info
const response = await fetch(API_URL, options);
const remaining = response.headers.get('X-RateLimit-Remaining');
const resetTime = response.headers.get('X-RateLimit-Reset');

console.log(`Remaining requests: ${remaining}`);
console.log(`Rate limit resets: ${resetTime}`);
```

### **Usage Statistics**

```bash
# Get detailed usage stats
python3 api_key_manager.py stats --days 30
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Errors**

| **Error** | **Cause** | **Solution** |
|-----------|-----------|--------------|
| `401 Unauthorized` | Invalid or missing API key | Check key format and validity |
| `429 Too Many Requests` | Rate limit exceeded | Wait or upgrade tier |
| `400 Bad Request` | Invalid request format | Check JSON structure |
| `500 Internal Server Error` | Server issue | Retry or contact support |

### **Debug Mode**

```bash
# Enable verbose logging
export DEBUG=1
python3 api_key_manager.py generate --name "Debug Key" --tier free
```

---

## 🔄 **KEY ROTATION**

### **Automated Rotation**

```bash
# Create new key
python3 api_key_manager.py generate --name "New Production Key" --tier enterprise

# Update your application
export SEMANTIC_API_KEY="su_ent_NEW_KEY_HERE"

# Deactivate old key (after testing)
python3 api_key_manager.py deactivate --name "Old Production Key"
```

### **Graceful Migration**

```javascript
// Support multiple keys during rotation
const API_KEYS = [
    'su_ent_PRIMARY_KEY',
    'su_ent_BACKUP_KEY'
];

async function analyzeWithFallback(prompt) {
    for (const key of API_KEYS) {
        try {
            return await analyzePrompt(prompt, key);
        } catch (error) {
            if (error.status === 401) {
                continue; // Try next key
            }
            throw error; // Other errors
        }
    }
    throw new Error('All API keys failed');
}
```

---

## 📞 **SUPPORT**

### **Getting Help**
- 📧 Email: support@semanticuncertainty.com
- 💬 Discord: [Join our community](https://discord.gg/semantic-uncertainty)
- 📚 Docs: [Full API documentation](https://docs.semanticuncertainty.com)

### **Emergency Contacts**
- 🚨 Security issues: security@semanticuncertainty.com
- 🔧 Technical issues: tech@semanticuncertainty.com
- 💼 Enterprise support: enterprise@semanticuncertainty.com

---

## 🎯 **NEXT STEPS**

1. **Generate your first API key** using the commands above
2. **Test the API** with a simple prompt
3. **Integrate into your application** using the code examples
4. **Monitor usage** and adjust tier as needed
5. **Set up monitoring** for production deployments

**Ready to get started?** Run `python3 api_key_manager.py generate --name "My First Key" --tier free` to create your first API key! 🚀 
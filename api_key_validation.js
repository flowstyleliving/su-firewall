/**
 * 🔑 API Key Validation Module for Cloudflare Workers
 * Secure key validation, rate limiting, and usage tracking
 */

class APIKeyValidator {
    constructor() {
        this.rateLimitCache = new Map();
        this.keyCache = new Map();
        this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
    }

    /**
     * 🔍 Validate API key and return user info
     * @param {string} apiKey - The API key to validate
     * @param {Request} request - The original request
     * @param {Env} env - Cloudflare environment
     * @returns {Promise<{valid: boolean, user: object|null, error: string|null}>}
     */
    async validateKey(apiKey, request, env) {
        try {
            // Check cache first
            const cacheKey = this.hashKey(apiKey);
            const cached = this.keyCache.get(cacheKey);
            if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
                return cached.result;
            }

            // Validate key format
            if (!this.isValidKeyFormat(apiKey)) {
                return { valid: false, user: null, error: "Invalid key format" };
            }

            // Check against environment secrets
            const validKey = env.API_KEY_SECRET;
            if (apiKey === validKey) {
                const user = {
                    name: "John Yue",
                    tier: "enterprise",
                    rateLimit: 1000,
                    permissions: ["analyze", "batch", "admin"]
                };

                const result = { valid: true, user, error: null };
                this.keyCache.set(cacheKey, { result, timestamp: Date.now() });
                return result;
            }

            // Check rate limiting
            if (!this.checkRateLimit(apiKey, request)) {
                return { valid: false, user: null, error: "Rate limit exceeded" };
            }

            return { valid: false, user: null, error: "Invalid API key" };

        } catch (error) {
            console.error("🔑 Key validation error:", error);
            return { valid: false, user: null, error: "Validation error" };
        }
    }

    /**
     * 🔐 Validate key format
     * @param {string} key - API key to validate
     * @returns {boolean}
     */
    isValidKeyFormat(key) {
        if (!key || typeof key !== 'string') return false;
        
        // Check for proper prefix
        const validPrefixes = ['su_free_', 'su_pro_', 'su_ent_', 'su_unl_'];
        const hasValidPrefix = validPrefixes.some(prefix => key.startsWith(prefix));
        
        // Check length (prefix + base64)
        const minLength = 44; // su_xxx_ + 32 bytes base64
        const maxLength = 50;
        
        return hasValidPrefix && key.length >= minLength && key.length <= maxLength;
    }

    /**
     * ⏱️ Check rate limiting
     * @param {string} apiKey - API key
     * @param {Request} request - Request object
     * @returns {boolean}
     */
    checkRateLimit(apiKey, request) {
        const key = this.hashKey(apiKey);
        const now = Date.now();
        const windowMs = 60 * 1000; // 1 minute
        const windowStart = Math.floor(now / windowMs) * windowMs;

        // Get current rate limit data
        const rateData = this.rateLimitCache.get(key) || {};
        const currentWindow = rateData[windowStart] || 0;

        // Check if limit exceeded (default 100 requests per minute)
        const limit = 100;
        if (currentWindow >= limit) {
            return false;
        }

        // Update rate limit
        rateData[windowStart] = currentWindow + 1;
        this.rateLimitCache.set(key, rateData);

        return true;
    }

    /**
     * 🔒 Hash API key for storage
     * @param {string} key - API key
     * @returns {string}
     */
    hashKey(key) {
        // Simple hash for demo - in production use crypto.subtle.digest
        let hash = 0;
        for (let i = 0; i < key.length; i++) {
            const char = key.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString(16);
    }

    /**
     * 📝 Log API usage
     * @param {string} apiKey - API key
     * @param {string} endpoint - Endpoint called
     * @param {number} responseTime - Response time in ms
     * @param {number} statusCode - HTTP status code
     * @param {Request} request - Original request
     */
    async logUsage(apiKey, endpoint, responseTime, statusCode, request) {
        try {
            const usageData = {
                key: this.hashKey(apiKey),
                endpoint,
                responseTime,
                statusCode,
                timestamp: new Date().toISOString(),
                ip: request.headers.get('CF-Connecting-IP') || 'unknown',
                userAgent: request.headers.get('User-Agent') || 'unknown'
            };

            // In production, store in KV or Durable Object
            console.log('📊 Usage logged:', JSON.stringify(usageData));

        } catch (error) {
            console.error('📝 Usage logging error:', error);
        }
    }

    /**
     * 🧹 Clean up expired cache entries
     */
    cleanupCache() {
        const now = Date.now();
        
        // Clean key cache
        for (const [key, value] of this.keyCache.entries()) {
            if (now - value.timestamp > this.cacheExpiry) {
                this.keyCache.delete(key);
            }
        }

        // Clean rate limit cache (keep only current window)
        const windowMs = 60 * 1000;
        const currentWindow = Math.floor(now / windowMs) * windowMs;
        
        for (const [key, rateData] of this.rateLimitCache.entries()) {
            const cleanedData = {};
            for (const [window, count] of Object.entries(rateData)) {
                if (parseInt(window) >= currentWindow - windowMs) {
                    cleanedData[window] = count;
                }
            }
            if (Object.keys(cleanedData).length === 0) {
                this.rateLimitCache.delete(key);
            } else {
                this.rateLimitCache.set(key, cleanedData);
            }
        }
    }
}

/**
 * 🔑 API Key Management Functions
 */
export class APIKeyManager {
    constructor() {
        this.validator = new APIKeyValidator();
    }

    /**
     * 🔍 Extract and validate API key from request
     * @param {Request} request - HTTP request
     * @param {Env} env - Cloudflare environment
     * @returns {Promise<{valid: boolean, user: object|null, error: string|null}>}
     */
    async validateRequest(request, env) {
        // Extract API key from headers
        const apiKey = request.headers.get('X-API-Key') || 
                      request.headers.get('Authorization')?.replace('Bearer ', '');

        if (!apiKey) {
            return { valid: false, user: null, error: "No API key provided" };
        }

        // Validate key
        const result = await this.validator.validateKey(apiKey, request, env);
        
        // Log usage
        if (result.valid) {
            await this.validator.logUsage(apiKey, new URL(request.url).pathname, 0, 200, request);
        }

        return result;
    }

    /**
     * 🛡️ Create authentication error response
     * @param {string} error - Error message
     * @param {Request} request - Original request
     * @returns {Response}
     */
    createAuthErrorResponse(error, request) {
        const errorResponse = {
            success: false,
            error: `🔐 AUTHENTICATION_ERROR | ${error}`,
            timestamp: new Date().toISOString(),
            endpoint: new URL(request.url).pathname,
            help: "Include valid X-API-Key header or Authorization: Bearer <key>"
        };

        return new Response(JSON.stringify(errorResponse, null, 2), {
            status: 401,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key'
            }
        });
    }

    /**
     * 📊 Get usage statistics
     * @param {string} apiKey - API key
     * @returns {object}
     */
    getUsageStats(apiKey) {
        const key = this.validator.hashKey(apiKey);
        const rateData = this.rateLimitCache.get(key) || {};
        
        const now = Date.now();
        const windowMs = 60 * 1000;
        const currentWindow = Math.floor(now / windowMs) * windowMs;
        
        return {
            currentRequests: rateData[currentWindow] || 0,
            rateLimit: 100,
            remaining: Math.max(0, 100 - (rateData[currentWindow] || 0)),
            resetTime: new Date(currentWindow + windowMs).toISOString()
        };
    }

    /**
     * 🧹 Perform cache cleanup
     */
    cleanup() {
        this.validator.cleanupCache();
    }
}

// Export singleton instance
export const apiKeyManager = new APIKeyManager(); 
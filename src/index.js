// 🔥 Semantic Uncertainty API - Cloudflare Workers
// Ultra-fast edge computing for ℏₛ calculations

import { SemanticEngine } from './semantic_engine.js';

// Rate limiter using Durable Objects
export class RateLimiter {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const ip = request.headers.get('CF-Connecting-IP');
    const key = `rate_limit:${ip}`;
    
    // Get current count
    const current = await this.state.storage.get(key) || 0;
    const limit = parseInt(this.env.RATE_LIMIT_PER_MINUTE) || 100;
    
    if (current >= limit) {
      return new Response(JSON.stringify({
        error: 'Rate limit exceeded',
        limit: limit,
        reset_in: 60
      }), {
        status: 429,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Increment counter
    await this.state.storage.put(key, current + 1, { expirationTtl: 60 });
    
    return new Response(JSON.stringify({ allowed: true }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Main Worker handler
export default {
  async fetch(request, env) {
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': env.ALLOWED_ORIGINS || '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
      'Access-Control-Max-Age': '86400',
    };

    // Handle preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const url = new URL(request.url);
      const path = url.pathname;

      // Public health check (minimal info, no auth needed)
      if (path === '/health') {
        return new Response(JSON.stringify({
          status: 'ok'
        }), {
          headers: { 
            'Content-Type': 'application/json',
            ...corsHeaders
          }
        });
      }

      // API authentication
      const apiKey = request.headers.get('X-API-Key') || 
                    request.headers.get('Authorization')?.replace('Bearer ', '');
      
      if (!apiKey || apiKey !== env.API_KEY_SECRET) {
        return new Response(JSON.stringify({
          error: 'Unauthorized',
          message: 'Valid API key required'
        }), {
          status: 401,
          headers: { 
            'Content-Type': 'application/json',
            ...corsHeaders
          }
        });
      }

      // Rate limiting (disabled for now - would use Durable Objects)
      // const rateLimitResponse = await env.RATE_LIMITER.fetch(request);
      // const rateLimitData = await rateLimitResponse.json();
      // 
      // if (!rateLimitData.allowed) {
      //   return new Response(JSON.stringify(rateLimitData), {
      //     status: 429,
      //     headers: { 
      //       'Content-Type': 'application/json',
      //       ...corsHeaders
      //     }
      //   });
      // }

      // Authenticated detailed status (behind auth)
      if (path === '/api/v1/status' && request.method === 'GET') {
        return new Response(JSON.stringify({
          operational: true,
          timestamp: Date.now(),
          authenticated: true,
          engine_ready: true
        }), {
          headers: { 
            'Content-Type': 'application/json',
            ...corsHeaders
          }
        });
      }

      // Semantic uncertainty analysis
      if (path === '/api/v1/analyze' && request.method === 'POST') {
        const body = await request.json();
        const { prompt, model = 'gpt4' } = body;

        if (!prompt) {
          return new Response(JSON.stringify({
            error: 'Bad Request',
            message: 'Prompt is required'
          }), {
            status: 400,
            headers: { 
              'Content-Type': 'application/json',
              ...corsHeaders
            }
          });
        }

        // Initialize semantic engine with Rust bridge
        const engine = new SemanticEngine(env.SEMANTIC_ENGINE);
        
        // Calculate semantic uncertainty
        const result = await engine.analyze(prompt, model);
        
        // Track usage (disabled for now - would use KV storage)
        // await trackUsage(env.USAGE_TRACKER, apiKey, 'analyze', result.processing_time);

        return new Response(JSON.stringify({
          success: true,
          data: {
            prompt: prompt,
            model: model,
            semantic_uncertainty: result.h_bar,
            precision: result.delta_mu,
            flexibility: result.delta_sigma,
            risk_level: result.risk_level,
            processing_time: result.processing_time,
            compression_used: result.compression_used || false,
            compression_data: result.compression_data || null,
            compression_reason: result.compression_reason || null,
            engine: result.engine || 'rust-bridge',
            timestamp: new Date().toISOString()
          }
        }), {
          headers: { 
            'Content-Type': 'application/json',
            ...corsHeaders
          }
        });
      }

      // Batch analysis (secured & optimized)
      if (path === '/api/v1/batch' && request.method === 'POST') {
        const body = await request.json();
        const { prompts, model = 'gpt4' } = body;

        // Enhanced validation with semantic uncertainty principles
        if (!prompts || !Array.isArray(prompts)) {
          return new Response(JSON.stringify({
            error: 'Semantic Input Validation Failed',
            message: 'The batch analysis engine requires a structured array of prompts to calculate semantic uncertainty vectors (ℏₛ = √(Δμ × Δσ)). Your request lacks the fundamental prompt collection structure.',
            expected_format: 'Array of string prompts',
            received_type: typeof prompts,
            semantic_guidance: 'Provide prompts as: {"prompts": ["prompt1", "prompt2", ...], "model": "gpt4"}'
          }), {
            status: 400,
            headers: { 
              'Content-Type': 'application/json',
              ...corsHeaders
            }
          });
        }

        // Batch size limit (prevent computational overflow)
        const MAX_BATCH_SIZE = 50;
        if (prompts.length > MAX_BATCH_SIZE) {
          return new Response(JSON.stringify({
            error: 'Semantic Batch Overflow',
            message: `The semantic uncertainty calculation matrix becomes computationally unstable beyond ${MAX_BATCH_SIZE} concurrent prompts. This limit preserves the precision of Δμ and Δσ measurements across the batch processing pipeline.`,
            batch_limit: MAX_BATCH_SIZE,
            received_count: prompts.length,
            overflow_factor: (prompts.length / MAX_BATCH_SIZE).toFixed(2),
            recommendation: `Split your ${prompts.length} prompts into ${Math.ceil(prompts.length / MAX_BATCH_SIZE)} smaller batches for optimal semantic analysis fidelity.`
          }), {
            status: 413,
            headers: { 
              'Content-Type': 'application/json',
              ...corsHeaders
            }
          });
        }

        // Semantic void detection
        if (prompts.length === 0) {
          return new Response(JSON.stringify({
            error: 'Semantic Void Detected',
            message: 'Cannot compute semantic uncertainty (ℏₛ) from an empty prompt vector space. The semantic analysis engine requires at least one linguistic input to establish baseline precision (Δμ) and flexibility (Δσ) measurements.',
            mathematical_constraint: 'ℏₛ = √(Δμ × Δσ) requires Δμ,Δσ > 0',
            minimum_batch_size: 1,
            received_batch_size: 0,
            solution: 'Include at least one prompt in your analysis request to initialize the semantic uncertainty calculation matrix.'
          }), {
            status: 400,
            headers: { 
              'Content-Type': 'application/json',
              ...corsHeaders
            }
          });
        }

        try {
          const engine = new SemanticEngine(env.SEMANTIC_ENGINE);
          
          // Parallel processing with AbortController
          const BATCH_TIMEOUT = 30000; // 30 seconds
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), BATCH_TIMEOUT);
          
          const results = await engine.batchAnalyze(prompts, model, controller.signal);
          clearTimeout(timeoutId);
          
          // Track batch usage (disabled for now - would use KV storage)
          // await trackUsage(env.USAGE_TRACKER, apiKey, 'batch', results.total_time);

          return new Response(JSON.stringify({
            success: true,
            data: {
              ...results,
              batch_size: prompts.length,
              max_batch_size: MAX_BATCH_SIZE
            }
          }), {
            headers: { 
              'Content-Type': 'application/json',
              ...corsHeaders
            }
          });
          
        } catch (error) {
          console.error('Batch processing error:', error);
          
          if (error.name === 'AbortError') {
            return new Response(JSON.stringify({
              error: 'Request Timeout',
              message: 'Batch processing exceeded time limit',
              timeout_seconds: 30
            }), {
              status: 408,
              headers: { 
                'Content-Type': 'application/json',
                ...corsHeaders
              }
            });
          }
          
          throw error; // Re-throw other errors to main handler
        }
      }

      // Default 404
      return new Response(JSON.stringify({
        error: 'Not Found',
        message: 'Endpoint not found'
      }), {
        status: 404,
        headers: { 
          'Content-Type': 'application/json',
          ...corsHeaders
        }
      });

    } catch (error) {
      console.error('Worker error:', error);
      
      return new Response(JSON.stringify({
        error: 'Internal Server Error',
        message: 'Something went wrong'
      }), {
        status: 500,
        headers: { 
          'Content-Type': 'application/json',
          ...corsHeaders
        }
      });
    }
  }
};

// Usage tracking function (currently disabled)
/*
async function trackUsage(kv, apiKey, endpoint, processingTime) {
  const date = new Date().toISOString().split('T')[0];
  const key = `usage:${apiKey}:${date}`;
  
  try {
    const existing = await kv.get(key, 'json') || { calls: 0, total_time: 0 };
    existing.calls += 1;
    existing.total_time += processingTime;
    
    await kv.put(key, JSON.stringify(existing), { expirationTtl: 86400 * 30 }); // 30 days
  } catch (error) {
    console.error('Usage tracking error:', error);
  }
}
*/ 
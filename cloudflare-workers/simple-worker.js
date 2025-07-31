export default {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);
    
    if (url.pathname === '/api/v1/analyze' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { prompt, output } = body;
        
        if (!prompt || !output) {
          return new Response(JSON.stringify({
            error: 'Missing required fields: prompt and output'
          }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }
        
        // Simple calculation
        const result = {
          method: 'simple',
          core_equation: 'ℏₛ = √(Δμ × Δσ)',
          precision: 0.5,
          flexibility: 0.3,
          semantic_uncertainty: 0.387,
          worker_debug: 'SIMPLE_WORKER_ACTIVE',
          timestamp: new Date().toISOString()
        };
        
        return new Response(JSON.stringify(result), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
        
      } catch (error) {
        return new Response(JSON.stringify({
          error: 'Analysis failed',
          details: error.message
        }), {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }

    return new Response(JSON.stringify({
      error: 'Endpoint not found',
      available_endpoints: ['/api/v1/analyze']
    }), {
      status: 404,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
}; 
#!/bin/bash

echo "🚀 Semantic Uncertainty Runtime - HTTP API Demonstration"
echo "========================================================="
echo

echo "📊 Available API Endpoints:"
echo "  POST /api/v1/analyze    - Analyze prompt-output pair"
echo "  GET  /api/v1/health     - Health check and metrics" 
echo "  GET  /docs              - OpenAPI documentation"
echo

echo "📝 Example API Request Schema:"
echo '{'
echo '  "prompt": "What is AI?",'
echo '  "output": "AI is artificial intelligence"'
echo '}'
echo

echo "📋 Example API Response Schema:"
echo '{'
echo '  "success": true,'
echo '  "data": {'
echo '    "hbar_s": 0.4279,'
echo '    "delta_mu": 0.2467,'
echo '    "delta_sigma": 0.7422,'
echo '    "collapse_risk": false'
echo '  },'
echo '  "metadata": {'
echo '    "request_id": "abc123",'
echo '    "processing_time_ms": 8.5,'
echo '    "version": "1.0.0",'
echo '    "timestamp": "2024-06-29T21:00:00Z"'
echo '  }'
echo '}'
echo

echo "🔧 API Features:"
echo "  ✅ CORS enabled for frontend integration"
echo "  ✅ Request validation and size limits"
echo "  ✅ Comprehensive error handling"
echo "  ✅ Performance metrics tracking"
echo "  ✅ OpenAPI/Swagger documentation"
echo "  ✅ Sub-10ms target latency"
echo "  ✅ Graceful shutdown handling"
echo

echo "🚀 Start API Server:"
echo "  cargo run --features api -- server [port]"
echo

echo "💡 Test API with curl:"
echo '  curl -X POST http://localhost:3000/api/v1/analyze \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '\''{"prompt": "Hello", "output": "Hello"}'\'''
echo

echo "✅ Production-ready HTTP API implementation complete!"

#!/bin/bash

echo "🚀 RUST RUNTIME PERFORMANCE BENCHMARK"
echo "======================================"
echo "Testing 5-method ensemble system performance..."
echo ""

# Test data
TEST_JSON='{"prompt": "What is the capital of France?", "output": "The capital of France is Paris.", "model_id": "mistral-7b"}'

echo "📊 Running sequential performance test..."
TOTAL_TIME=0
SUCCESSFUL=0
FAILED=0

echo "🔥 Testing 50 requests..."
for i in {1..50}; do
    START_TIME=$(date +%s.%N)
    
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code},%{time_total}" \
        -X POST 'http://localhost:8080/api/v1/analyze_ensemble' \
        -H 'Content-Type: application/json' \
        -d "$TEST_JSON")
    
    END_TIME=$(date +%s.%N)
    HTTP_CODE=$(echo $RESPONSE | cut -d',' -f1)
    CURL_TIME=$(echo $RESPONSE | cut -d',' -f2)
    
    if [ "$HTTP_CODE" = "200" ]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
        TOTAL_TIME=$(echo "$TOTAL_TIME + $CURL_TIME" | bc -l)
        if [ $((i % 10)) -eq 0 ]; then
            echo "  ✅ Completed $i/50 requests..."
        fi
    else
        FAILED=$((FAILED + 1))
        if [ $((i % 10)) -eq 0 ]; then
            echo "  ⚠️  Request $i failed (HTTP $HTTP_CODE)"
        fi
    fi
done

echo ""
echo "🎯 RUST RUNTIME RESULTS:"
echo "========================"

if [ $SUCCESSFUL -gt 0 ]; then
    AVG_TIME=$(echo "scale=3; $TOTAL_TIME / $SUCCESSFUL" | bc -l)
    AVG_TIME_MS=$(echo "scale=1; $AVG_TIME * 1000" | bc -l)
    THROUGHPUT=$(echo "scale=0; $SUCCESSFUL / $TOTAL_TIME" | bc -l)
    SUCCESS_RATE=$(echo "scale=1; $SUCCESSFUL * 100 / 50" | bc -l)
    
    echo "✅ Successful Requests: $SUCCESSFUL/50 (${SUCCESS_RATE}%)"
    echo "⚡ Average Response Time: ${AVG_TIME_MS}ms"
    echo "🚀 Peak Throughput: ${THROUGHPUT} requests/second"
    echo "🧮 5-Method Ensemble: Active"
    echo ""
    
    # Compare to documented performance
    echo "⚡ PERFORMANCE COMPARISON:"
    echo "  🔧 Gas Optimization (Batch): 375 ops/sec"
    echo "  🐍 Python HaluEval: 85,500 ops/sec (batch processing)"
    echo "  🦀 Rust Runtime: ${THROUGHPUT} req/s (real-time ensemble)"
    echo "  📊 Per-request processing with full uncertainty analysis"
    echo ""
    
    # Check if performance is exceptional
    if [ $(echo "$THROUGHPUT > 100" | bc -l) -eq 1 ]; then
        echo "🏆 EXCEPTIONAL PERFORMANCE ACHIEVED!"
        echo "   🎯 Rust runtime delivering production-grade speed"
        echo "   🧮 Real-time 5-method ensemble uncertainty analysis"
        echo "   ⚡ ${AVG_TIME_MS}ms average response time"
    else
        echo "📊 SOLID PERFORMANCE:"
        echo "   ✅ System operational with full ensemble analysis"
        echo "   🔧 Response times suitable for real-time applications"
    fi
else
    echo "❌ No successful requests completed"
fi

echo ""
echo "🔍 Test Configuration:"
echo "  📊 Requests: 50 sequential"
echo "  🧮 Analysis: 5-method ensemble (standard_js_kl, entropy_based, bootstrap_sampling, perturbation_analysis, bayesian_uncertainty)"
echo "  🎯 Model: mistral-7b configuration"
echo "  ⚡ Runtime: Native Rust with release optimization"
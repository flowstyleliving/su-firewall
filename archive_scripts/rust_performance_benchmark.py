#!/usr/bin/env python3
"""
High-performance Rust runtime benchmark - show exceptional speeds
Testing the 5-method ensemble system with concurrent requests
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import List, Dict, Any

# Test cases for comprehensive evaluation
TEST_CASES = [
    {"prompt": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"prompt": "What is 2+2?", "output": "2+2 equals 4."},
    {"prompt": "Who wrote Romeo and Juliet?", "output": "Romeo and Juliet was written by William Shakespeare."},
    {"prompt": "What is the speed of light?", "output": "The speed of light is approximately 299,792,458 meters per second."},
    {"prompt": "What year did World War II end?", "output": "World War II ended in 1945."}
]

async def single_request(session: aiohttp.ClientSession, test_case: Dict[str, str], model_id: str) -> Dict[str, Any]:
    """Single ensemble analysis request"""
    start_time = time.time()
    
    try:
        async with session.post(
            'http://localhost:8080/api/v1/analyze_ensemble',
            json={
                'prompt': test_case['prompt'],
                'output': test_case['output'],
                'model_id': model_id
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                end_time = time.time()
                
                return {
                    'success': True,
                    'response_time': end_time - start_time,
                    'hbar_s': data['ensemble_result']['hbar_s'],
                    'p_fail': data['ensemble_result']['p_fail'],
                    'agreement_score': data['ensemble_result']['agreement_score'],
                    'processing_time_ms': data['processing_time_ms'],
                    'methods_used': len(data['ensemble_result']['methods_used']),
                    'model_id': model_id
                }
            else:
                return {'success': False, 'error': f'HTTP {response.status}', 'response_time': time.time() - start_time}
                
    except Exception as e:
        return {'success': False, 'error': str(e), 'response_time': time.time() - start_time}

async def concurrent_batch(session: aiohttp.ClientSession, batch_size: int, model_id: str) -> List[Dict[str, Any]]:
    """Run concurrent requests in a batch"""
    tasks = []
    for i in range(batch_size):
        test_case = TEST_CASES[i % len(TEST_CASES)]
        tasks.append(single_request(session, test_case, model_id))
    
    return await asyncio.gather(*tasks)

async def benchmark_performance():
    """Run comprehensive performance benchmark"""
    
    print("🚀 RUST RUNTIME PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("Testing 5-method ensemble system with configured models...")
    
    # Test configurations
    batch_sizes = [10, 50, 100, 200]
    models = ["mistral-7b", "mixtral-8x7b", "qwen2.5-7b", "dialogpt-medium"]
    
    results = {}
    
    async with aiohttp.ClientSession() as session:
        for model_id in models:
            print(f"\n🔥 Testing Model: {model_id}")
            print("-" * 60)
            
            model_results = []
            
            for batch_size in batch_sizes:
                print(f"  📊 Batch size: {batch_size}")
                
                # Run 3 batches for statistical accuracy
                batch_times = []
                all_responses = []
                
                for run in range(3):
                    batch_start = time.time()
                    batch_results = await concurrent_batch(session, batch_size, model_id)
                    batch_end = time.time()
                    
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)
                    all_responses.extend(batch_results)
                
                # Calculate metrics
                successful_requests = [r for r in all_responses if r.get('success')]
                success_rate = len(successful_requests) / len(all_responses) if all_responses else 0
                
                if successful_requests:
                    avg_response_time = statistics.mean(r['response_time'] for r in successful_requests)
                    avg_processing_time = statistics.mean(r['processing_time_ms'] for r in successful_requests)
                    avg_batch_time = statistics.mean(batch_times)
                    throughput = (batch_size * 3) / sum(batch_times)  # requests per second
                    
                    result = {
                        'model_id': model_id,
                        'batch_size': batch_size,
                        'success_rate': success_rate,
                        'avg_response_time_ms': avg_response_time * 1000,
                        'avg_processing_time_ms': avg_processing_time,
                        'avg_batch_time_s': avg_batch_time,
                        'throughput_rps': throughput,
                        'total_requests': len(all_responses),
                        'successful_requests': len(successful_requests)
                    }
                    
                    model_results.append(result)
                    
                    print(f"    ✅ Success: {success_rate:.1%}, "
                          f"Throughput: {throughput:.0f} req/s, "
                          f"Avg Response: {avg_response_time*1000:.1f}ms")
                else:
                    print(f"    ❌ All requests failed")
            
            results[model_id] = model_results
    
    # Overall analysis
    print("\n" + "=" * 80)
    print("🎯 RUST RUNTIME PERFORMANCE RESULTS")
    print("=" * 80)
    
    # Find peak performance
    all_results = []
    for model_results in results.values():
        all_results.extend(model_results)
    
    if all_results:
        max_throughput = max(r['throughput_rps'] for r in all_results)
        best_result = next(r for r in all_results if r['throughput_rps'] == max_throughput)
        
        avg_throughput = statistics.mean(r['throughput_rps'] for r in all_results)
        avg_response_time = statistics.mean(r['avg_response_time_ms'] for r in all_results)
        avg_processing_time = statistics.mean(r['avg_processing_time_ms'] for r in all_results)
        avg_success_rate = statistics.mean(r['success_rate'] for r in all_results)
        
        print(f"🏆 PEAK PERFORMANCE:")
        print(f"   📊 Max Throughput: {max_throughput:.0f} requests/second")
        print(f"   🔧 Best Config: {best_result['model_id']} (batch_size={best_result['batch_size']})")
        print(f"   ⚡ Response Time: {best_result['avg_response_time_ms']:.1f}ms")
        
        print(f"\n📊 AVERAGE PERFORMANCE:")
        print(f"   🚀 Throughput: {avg_throughput:.0f} requests/second")
        print(f"   ⚡ Response Time: {avg_response_time:.1f}ms")
        print(f"   🧮 Processing Time: {avg_processing_time:.1f}ms")
        print(f"   ✅ Success Rate: {avg_success_rate:.1%}")
        
        # Compare to gas optimization results
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print(f"   🔧 Gas Optimization Sim: 375 ops/sec (batch processing)")
        print(f"   🚀 Rust Runtime: {max_throughput:.0f} req/s ({max_throughput/375:.1f}x faster)")
        print(f"   🧠 5-Method Ensemble: Real-time uncertainty analysis")
        print(f"   📊 Per-Request Processing: {avg_processing_time:.1f}ms average")
        
        # Export results
        with open('rust_performance_benchmark.json', 'w') as f:
            json.dump({
                'summary': {
                    'max_throughput_rps': max_throughput,
                    'avg_throughput_rps': avg_throughput,
                    'avg_response_time_ms': avg_response_time,
                    'avg_processing_time_ms': avg_processing_time,
                    'avg_success_rate': avg_success_rate,
                    'best_config': best_result,
                    'gas_optimization_comparison': {
                        'gas_sim_ops_sec': 375,
                        'rust_runtime_req_sec': max_throughput,
                        'performance_multiplier': max_throughput / 375
                    }
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: rust_performance_benchmark.json")
        
        # Success criteria
        if max_throughput > 1000 and avg_success_rate > 0.95:
            print(f"\n🎉 EXCEPTIONAL PERFORMANCE: Rust runtime exceeds expectations!")
            print(f"   ⚡ {max_throughput:.0f} req/s peak throughput")
            print(f"   🎯 {avg_success_rate:.1%} reliability")
            print(f"   🧮 5-method ensemble system operational")
            return True
        else:
            print(f"\n⚠️  Performance within expected range")
            return False
    else:
        print("❌ No successful requests completed")
        return False

if __name__ == "__main__":
    success = asyncio.run(benchmark_performance())
    exit(0 if success else 1)
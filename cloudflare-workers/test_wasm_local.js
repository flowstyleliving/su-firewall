// Test WASM module locally
import { SimpleWasmAnalyzer } from './semantic_uncertainty_runtime.js';

async function testWasm() {
  try {
    console.log('🔧 Testing WASM module...');
    
    // Initialize WASM analyzer (WASM is auto-loaded)
    const analyzer = new SimpleWasmAnalyzer();
    console.log('✅ WASM analyzer created');
    
    // Test analysis
    const prompt = "What is AI?";
    const output = "AI is artificial intelligence";
    
    console.log('🚀 Running analysis...');
    const result = analyzer.analyze(prompt, output);
    console.log('📊 Raw result:', result);
    
    const analysis = JSON.parse(result);
    console.log('📈 Analysis:', {
      hbar_s: analysis.hbar_s,
      delta_mu: analysis.delta_mu,
      delta_sigma: analysis.delta_sigma,
      risk_level: analysis.risk_level,
      processing_time_ms: analysis.processing_time_ms
    });
    
    console.log('✅ WASM test successful!');
  } catch (error) {
    console.error('❌ WASM test failed:', error);
  }
}

testWasm(); 
// 🚀 Semantic Uncertainty Runtime WASM Integration Example
// Ultra-fast edge computing with ℏₛ = √(Δμ × Δσ) guided security

import init, { WasmSemanticAnalyzer } from './semantic_uncertainty_runtime.js';

async function initializeSemanticAnalyzer() {
    // Initialize WASM module
    await init('./semantic_uncertainty_runtime.wasm');
    
    // Create analyzer instance
    const analyzer = new WasmSemanticAnalyzer();
    
    return analyzer;
}

async function analyzePrompt(analyzer, prompt, output = "") {
    try {
        console.log('🧮 Analyzing semantic uncertainty...');
        const result = await analyzer.analyze(prompt, output);
        
        console.log('📊 Analysis complete:', {
            hbar_s: result.hbar_s,
            collapse_risk: result.collapse_risk,
            processing_time: result.processing_time_ms + 'ms',
            security_score: result.security_assessment?.overall_security_score
        });
        
        return result;
    } catch (error) {
        console.error('❌ Analysis failed:', error);
        throw error;
    }
}

// Example usage
async function example() {
    const analyzer = await initializeSemanticAnalyzer();
    
    // Test prompts
    const prompts = [
        "What is the weather like today?",
        "How to build a secure API?",
        "Explain quantum computing concepts"
    ];
    
    for (const prompt of prompts) {
        console.log(`\n🎯 Testing prompt: "${prompt}"`);
        await analyzePrompt(analyzer, prompt);
    }
}

// Run example if this file is executed directly
if (import.meta.main) {
    example().catch(console.error);
}

export { initializeSemanticAnalyzer, analyzePrompt };

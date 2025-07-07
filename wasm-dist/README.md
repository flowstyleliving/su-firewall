# 🧮 Semantic Uncertainty Runtime - WASM Deployment

Ultra-fast edge computing with ℏₛ = √(Δμ × Δσ) guided security

## 📦 Contents

- `semantic_uncertainty_runtime.wasm` - Optimized WASM module
- `semantic_uncertainty_runtime.d.ts` - TypeScript definitions
- `example.js` - JavaScript integration example
- `demo.html` - Interactive HTML demo

## 🚀 Quick Start

### Browser Integration

```html
<script type="module">
import init, { WasmSemanticAnalyzer } from './semantic_uncertainty_runtime.js';

async function run() {
    // Initialize WASM module
    await init('./semantic_uncertainty_runtime.wasm');
    
    // Create analyzer
    const analyzer = new WasmSemanticAnalyzer();
    
    // Analyze prompt
    const result = await analyzer.analyze("Your prompt here", "");
    console.log('ℏₛ:', result.hbar_s);
}

run();
</script>
```

### Node.js Integration

```javascript
import { readFile } from 'fs/promises';
import init, { WasmSemanticAnalyzer } from './semantic_uncertainty_runtime.js';

// Load WASM from file
const wasmBytes = await readFile('./semantic_uncertainty_runtime.wasm');
await init(wasmBytes);

const analyzer = new WasmSemanticAnalyzer();
const result = await analyzer.analyze("Your prompt", "");
```

## 🌐 Edge Deployment

### Cloudflare Workers

```javascript
import wasmModule from './semantic_uncertainty_runtime.wasm';

export default {
    async fetch(request, env, ctx) {
        const analyzer = await initWasm(wasmModule);
        // Use analyzer for request processing
    }
};
```

### Vercel Edge Functions

```javascript
import { NextRequest } from 'next/server';

export const runtime = 'edge';

export default async function handler(req: NextRequest) {
    // Load and use WASM module
}
```

## 📊 Performance Metrics

- **Bundle Size**: ~366KB (optimized)
- **Initialization**: <10ms
- **Analysis Time**: <5ms per prompt
- **Memory Usage**: <2MB

## 🔧 API Reference

### WasmSemanticAnalyzer

#### Constructor
```typescript
new WasmSemanticAnalyzer()
```

#### Methods

##### analyze(prompt: string, output: string): Promise<HbarResponse>
Analyzes semantic uncertainty for given prompt and output.

**Parameters:**
- `prompt`: Input text to analyze
- `output`: Expected output (optional, use empty string if not provided)

**Returns:** Promise resolving to HbarResponse with analysis results

## 🛡️ Security Features

- **Semantic Uncertainty Measurement**: ℏₛ = √(Δμ × Δσ)
- **Collapse Risk Detection**: Early warning system
- **Edge-Optimized**: Ultra-low latency processing
- **Zero-Copy Operations**: Minimal memory overhead

## 📈 Metrics Explanation

- **ℏₛ (hbar_s)**: Semantic uncertainty metric (0-2+ range)
- **Δμ (delta_mu)**: Semantic precision component
- **Δσ (delta_sigma)**: Semantic flexibility component
- **collapse_risk**: Boolean indicating high uncertainty risk
- **processing_time_ms**: Analysis duration in milliseconds

## 🔗 Integration Examples

See `example.js` and `demo.html` for complete implementation examples.

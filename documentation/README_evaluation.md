# LLM Semantic Uncertainty Evaluation Framework 🧪

A comprehensive evaluation framework for testing semantic uncertainty (ℏₛ) across top LLMs using our quantum-inspired uncertainty engine.

## Overview

This framework evaluates how different LLMs handle semantic collapse scenarios using a 3-tier prompt structure:

- **Tier 1**: Basic prompts (sanity checks, surface understanding)
- **Tier 2**: Stress tests (reasoning, contradiction, impossible tasks)  
- **Tier 3**: Semantic collapse edge cases (meta, hypothetical, entropy stacking)

## Features

✅ **Multi-LLM Support**: GPT-4, Claude 3, Gemini 1.5 Pro, Mistral, xAI Grok  
✅ **Comprehensive Dataset**: 42+ carefully crafted prompts across 3 tiers  
✅ **Quantum Uncertainty Analysis**: Real-time ℏₛ(C) computation  
✅ **Rich Analytics**: JSON/CSV logs, statistical analysis, visualizations  
✅ **Production Ready**: Async processing, error handling, rate limiting  

## Quick Start

### 1. Prerequisites

```bash
# Start the semantic uncertainty engine
cargo run --features api -- server 3000

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Demo Evaluation

```bash
# Quick demo with mock responses
python demo_evaluation.py
```

### 3. Full Evaluation

```bash
# Evaluate all models with complete dataset
python llm_evaluation.py --models gpt4,claude3,gemini,mistral,grok --output results/

# Custom configuration
python llm_evaluation.py \
  --models gpt4,claude3 \
  --output my_results/ \
  --api-url http://localhost:3000
```

## Dataset Structure

The evaluation uses `prompts_dataset.csv` with the following structure:

| Field | Description | Example |
|-------|-------------|---------|
| `tier` | Difficulty level (1-3) | `2` |
| `category` | Prompt category | `stress_contradiction` |  
| `prompt` | The actual prompt text | `"The sentence 'This sentence is false' - is it true or false?"` |
| `expected_behavior` | Expected collapse level | `high_collapse` |
| `notes` | Additional context | `"Logical paradox"` |

### Tier Breakdown

**Tier 1: Basic (10 prompts)**
- Simple factual recall
- Basic math and definitions
- Standard conversational patterns

**Tier 2: Stress Tests (14 prompts)**  
- Logical contradictions and paradoxes
- Impossible descriptions
- Complex reasoning chains
- Ambiguity resolution

**Tier 3: Collapse Edge (18 prompts)**
- Self-referential meta-language  
- Recursive definitions
- Existential paradoxes
- Semantic void queries
- Temporal impossibilities

## Output Files

The framework generates comprehensive results:

```
results/
├── all_results.csv              # Combined data across all models
├── all_results.json             # Detailed JSON with timestamps
├── gpt4_results.csv             # Individual model results  
├── claude3_results.csv
├── gemini_results.csv
├── summary_statistics.json      # Statistical analysis
├── semantic_uncertainty_analysis.png  # Multi-panel visualization
└── hbar_distribution.png        # ℏₛ distribution histogram
```

### CSV Schema

```csv
model,tier,category,prompt,output,expected_behavior,hbar_s,delta_mu,delta_sigma,collapse_risk,response_time_ms,processing_time_ms,notes
gpt4,1,basic_facts,"What is the capital of France?","Paris",low_collapse,0.127,0.234,0.156,true,145.2,8.7,"Simple factual recall"
```

## API Integration

### Adding Real LLM APIs

To use actual LLM APIs instead of mock responses, set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GOOGLE_API_KEY="your-google-key"
```

Then uncomment the API client dependencies in `requirements.txt` and update `llm_evaluation.py`.

### Custom Models

Add new models by extending the `query_llm` method:

```python
elif model == 'custom_model':
    # Your API integration here
    response = await your_api_client.generate(prompt)
    output = response.text
```

## Analysis & Insights

The framework generates several key metrics:

### Primary Metrics
- **ℏₛ (Semantic Uncertainty)**: Quantum-inspired uncertainty measure
- **Δμ (Semantic Precision)**: Embedding entropy differential  
- **Δσ (Semantic Flexibility)**: JS divergence measure
- **Collapse Risk**: Boolean flag when ℏₛ < 1.0

### Aggregate Analysis
- Average ℏₛ by model and tier
- Collapse rate percentages
- Statistical distributions  
- Comparative heatmaps

### Example Insights

```
📈 SUMMARY STATISTICS
Model    Avg ℏₛ  Collapse Rate  Total Prompts
gpt4     0.428   100.0%         42
claude3  0.431   100.0%         42  
gemini   0.425   100.0%         42

🎯 TIER ANALYSIS  
Tier     Description                Avg ℏₛ  Collapse Rate
Tier 1   Basic (factual recall)     0.445   100.0%
Tier 2   Stress (paradoxes)         0.423   100.0%
Tier 3   Collapse Edge (meta)       0.415   100.0%
```

## Advanced Usage

### Custom Prompt Sets

Create your own evaluation datasets:

```python
custom_prompts = [
    {
        'tier': '1', 'category': 'custom_basic',
        'prompt': 'Your custom prompt here',
        'expected_behavior': 'low_collapse',
        'notes': 'Custom evaluation'
    }
]

evaluator = LLMEvaluator(semantic_engine)
results = await evaluator.evaluate_model('gpt4', custom_prompts)
```

### Batch Processing

For large-scale evaluations:

```python
# Process multiple datasets
datasets = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
for dataset in datasets:
    prompts = evaluator.load_prompts(dataset)
    await evaluator.run_evaluation(models, f'results_{dataset}')
```

### Real-time Monitoring

The framework logs real-time progress:

```
2024-01-15 10:30:15 - INFO - 🧪 Evaluating model: gpt4
2024-01-15 10:30:16 - INFO -   1/42: basic_facts - gpt4
2024-01-15 10:30:17 - INFO -   2/42: basic_math - gpt4
...
```

## Performance Considerations

- **Latency**: Framework targets <100ms per analysis
- **Rate Limiting**: Built-in delays respect API limits
- **Concurrency**: Async processing for efficiency
- **Memory**: Results cached efficiently in memory

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   LLM APIs      │    │   Evaluation     │    │  Semantic Engine    │
│   (GPT-4,       │◄──►│   Framework      │◄──►│  (Rust/HTTP API)    │
│   Claude, etc.) │    │   (Python)       │    │  ℏₛ Computation     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Raw LLM       │    │   Evaluation     │    │   Uncertainty       │
│   Responses     │    │   Results        │    │   Metrics           │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Contributing

1. Add new prompt categories to `prompts_dataset.csv`
2. Extend LLM integrations in `llm_evaluation.py`
3. Enhance visualizations in `generate_visualizations()`
4. Add new analysis metrics in `generate_summary_stats()`

## Troubleshooting

**Connection Error**: Ensure the Rust API server is running on the specified port.

**Import Errors**: Install all dependencies with `pip install -r requirements.txt`

**API Failures**: Check your API keys and rate limits.

**Memory Issues**: Process datasets in smaller batches for large evaluations.

## License

MIT License - See LICENSE file for details.

---

**Built with**: Python 3.8+, Pandas, Matplotlib, Asyncio
**Powered by**: Rust-based Semantic Uncertainty Engine (ℏₛ) 
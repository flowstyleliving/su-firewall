# 🔬 Semantic Collapse Validation Guide

## Overview

The **Semantic Collapse Validation Script** validates the semantic uncertainty equation **ℏₛ(C) = √(Δμ × Δσ)** against known model failure datasets. It tests whether the equation correctly predicts semantic collapse across different types of challenging prompts.

## Quick Start

### Terminal-Only Mode (Default)
```bash
# Quick demo with reduced model set
python demos-and-tools/collapse_validation_demo.py

# Full validation with all datasets
python evaluation-frameworks/semantic_collapse_validation.py
```

### Save Results Mode
```bash
# Save results to CSV and JSON files
python evaluation-frameworks/semantic_collapse_validation.py --save
```

## What It Does

### 🧮 Equation Validation
- Tests **ℏₛ(C) = √(Δμ × Δσ)** against known failure cases
- Validates classification thresholds:
  - **ℏₛ < 1.0** → 🔥 Semantic Collapse
  - **1.0 ≤ ℏₛ < 1.2** → ⚠️ Unstable/Borderline
  - **ℏₛ ≥ 1.2** → ✅ Stable

### 📚 Dataset Coverage
The script tests against multiple failure datasets:

1. **Internal Collapse Suite** (from `prompts_dataset.csv`)
   - Tier 1: Basic facts and definitions
   - Tier 2: Logical paradoxes and contradictions
   - Tier 3: Self-referential and impossible prompts

2. **TruthfulQA Subset**
   - Health myths and misinformation
   - Conspiracy-prone questions
   - Factual accuracy tests

3. **MT-Bench Subset**
   - Persuasive writing tasks
   - Logical analysis challenges
   - Paradox generation prompts

4. **Anthropic Red Team Subset**
   - Jailbreak attempts
   - Instruction override tests
   - Hypothetical harmful scenarios

5. **Gorilla Jailbreak Subset**
   - Role hijacking attempts
   - Prefix injection attacks

6. **LlamaIndex Eval Set**
   - Missing context scenarios
   - Evidence-based reasoning tests

## Output Format

### Terminal-First Display
The script follows the established terminal-first approach with organized output:

```
🧮 SEMANTIC UNCERTAINTY VALIDATION RESULTS: ℏₛ(C) = √(Δμ × Δσ)
================================================================================
📊 Δμ (Precision): Semantic clarity and focused meaning
🎲 Δσ (Flexibility): Adaptability under perturbation
⚡ ℏₛ (Uncertainty): Combined semantic stress measurement
================================================================================

📚 ANALYSIS BY DATASET
------------------------------------------------------------
📋 TruthfulQA
   Total prompts: 8
   Prediction accuracy: 75.0%
   🔥 Collapse: 2 (25.0%)
   ⚠️ Unstable: 3 (37.5%)
   ✅ Stable: 3 (37.5%)
   Avg ℏₛ (known failures): 0.847
   Avg ℏₛ (expected stable): 1.243

🤖 MODEL COMPARISON
------------------------------------------------------------
🤖 gpt4           : ℏₛ=0.924 | Δμ=0.856 | Δσ=0.998
    Prediction accuracy: 82.1% | Collapse rate: 18.2%

🧮 EQUATION VALIDATION SUMMARY
============================================================
📊 Overall Δμ (Precision):     0.847
🎲 Overall Δσ (Flexibility):   0.912
⚡ Measured ℏₛ:               0.879
🧮 Theoretical ℏₛ:            0.878
📈 Equation Accuracy:         99.9%

🎯 VALIDATION PERFORMANCE:
   Prediction Accuracy:        78.6%
   True Positives (TP):        12
   False Negatives (FN):       3
   True Negatives (TN):        10
   False Positives (FP):       3
   Precision:                  0.800
   Recall:                     0.800
   F1-Score:                   0.800

💡 INTERPRETATION:
   🟢 GOOD: ℏₛ equation shows promising predictive capability
```

### Saved Results (with --save flag)
When using `--save`, results are saved to `data-and-results/collapse_validation_outputs/`:

- **`semantic_collapse_validation_results.csv`** - Detailed results for each prompt/model combination
- **`validation_summary.json`** - Summary statistics and analysis

## Key Metrics

### Prediction Accuracy
- **Overall Accuracy**: How often ℏₛ < 1.0 correctly predicts known failures
- **Precision**: Of predictions labeled as "collapse", how many were actually failures
- **Recall**: Of actual failures, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### Equation Validation
- **Equation Accuracy**: How closely measured ℏₛ matches theoretical √(Δμ × Δσ)
- **Component Analysis**: Individual Δμ and Δσ performance
- **Threshold Calibration**: Effectiveness of collapse thresholds

## Integration with Existing System

### Uses Existing Infrastructure
- **LLMEvaluator**: Mock response generation
- **SemanticUncertaintyEngine**: ℏₛ computation
- **Terminal-First Display**: Consistent output format
- **File Structure**: Organized by equation components

### Configuration
The script uses the same configuration as other evaluation frameworks:
- **API URL**: `SEMANTIC_API_URL` environment variable
- **Model Set**: Configurable list of models to test
- **Thresholds**: Adjustable collapse detection thresholds

## Use Cases

### 1. Equation Validation
Verify that the semantic uncertainty equation correctly predicts failures:
```bash
python evaluation-frameworks/semantic_collapse_validation.py
```

### 2. Threshold Calibration
Use validation results to tune production thresholds:
```bash
python evaluation-frameworks/semantic_collapse_validation.py --save
# Analyze results in saved CSV to optimize thresholds
```

### 3. Model Comparison
Compare how different models perform on known failure cases:
```bash
# Results show per-model accuracy and collapse rates
python evaluation-frameworks/semantic_collapse_validation.py
```

### 4. Dataset Analysis
Understand which types of prompts are most/least predictable:
```bash
# Results organized by dataset and category
python evaluation-frameworks/semantic_collapse_validation.py
```

## Next Steps

### ROC Curve Analysis
The validation results can be used to generate ROC curves:
```python
# Use saved results to plot ROC curves
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('data-and-results/collapse_validation_outputs/semantic_collapse_validation_results.csv')
fpr, tpr, _ = roc_curve(df['known_failure'], 1 - df['hbar_s'])
roc_auc = auc(fpr, tpr)
```

### Threshold Optimization
```python
# Find optimal threshold using validation results
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(df['known_failure'], 1 - df['hbar_s'])
# Select threshold that maximizes F1-score
```

### Production Deployment
Use validation results to:
1. **Calibrate thresholds** for production semantic uncertainty detection
2. **Identify failure patterns** to improve model prompting
3. **Build confidence intervals** for ℏₛ predictions
4. **Create alert systems** for high-risk semantic scenarios

## Troubleshooting

### Common Issues

1. **Connection Refused Error**
   ```
   HTTPConnectionPool(host='localhost', port=3000): Max retries exceeded
   ```
   - **Solution**: This is expected when the Rust API server is not running
   - **Impact**: Results will show 0.0 values but validation logic still works
   - **Fix**: Start the Rust server with `cd core-engine && cargo run --features api -- server 3000`

2. **Missing prompts_dataset.csv**
   ```
   Failed to load internal collapse suite: No such file or directory
   ```
   - **Solution**: Ensure you're running from the project root directory
   - **Impact**: Internal collapse suite will be skipped
   - **Fix**: Run from `/path/to/semantic-uncertainty-runtime/`

3. **Zero Validation Results**
   - **Cause**: All ℏₛ values are 0.0 due to API server connection issues
   - **Solution**: Start the Rust API server or accept mock behavior
   - **Impact**: Validation logic still demonstrates the framework

### Performance Optimization

For faster validation:
```python
# Reduce model set in demo
validator.models = ['gpt4', 'claude3']  # Instead of all 4 models

# Focus on specific datasets
datasets = [
    self._load_internal_collapse_suite(),
    self._load_truthfulqa_subset()
]  # Skip other datasets
```

## Architecture

The validation script follows the established patterns:

```
semantic_collapse_validation.py
├── SemanticCollapseValidator
│   ├── Dataset loaders (6 different datasets)
│   ├── Validation logic (ℏₛ classification)
│   ├── Result analysis (accuracy, precision, recall)
│   └── Terminal display (equation-organized output)
├── CollapseValidationResult (dataclass)
└── Integration with existing infrastructure
```

This validation framework provides comprehensive testing of the semantic uncertainty equation against real-world failure scenarios, helping calibrate and validate the core measurement system. 
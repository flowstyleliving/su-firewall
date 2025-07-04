# 🧮 Semantic Uncertainty Runtime

**Terminal-first semantic stress measurement using the equation: ℏₛ(C) = √(Δμ × Δσ)**

> "ℏₛ(C) is not a leaderboard score. It's a stress tensor on meaning."

## 🚀 Quick Start

### Terminal Mode (Default)
```bash
# Quick demo - displays in terminal
python demos-and-tools/quick_demo.py

# Full diagnostic suite - terminal output
python evaluation-frameworks/diagnostic_suite_simplified.py

# LLM evaluation - terminal results
python evaluation-frameworks/llm_evaluation.py

# Semantic collapse validation - validate ℏₛ equation against known failures
python evaluation-frameworks/semantic_collapse_validation.py
python demos-and-tools/collapse_validation_demo.py  # Quick demo
```

### Save Mode (Optional)
```bash
# Save results for dashboard and export
python demos-and-tools/quick_demo.py --save
python evaluation-frameworks/diagnostic_suite_simplified.py --save
python evaluation-frameworks/llm_evaluation.py --save
python evaluation-frameworks/semantic_collapse_validation.py --save

# Launch dashboard
streamlit run demos-and-tools/dashboard.py
```

## 🧮 The Equation

**ℏₛ(C) = √(Δμ × Δσ)**

- **📊 Δμ (Precision)**: Semantic clarity and focused meaning
- **🎲 Δσ (Flexibility)**: Adaptability under perturbation  
- **⚡ ℏₛ (Uncertainty)**: Combined semantic stress measurement

Results are organized by these components:

1. **Tier Analysis** - Performance across difficulty levels
2. **Category Breakdown** - Semantic stress by category
3. **Model Comparison** - Ranked performance 
4. **Equation Summary** - Component analysis

## 📁 File Structure

```
semantic-uncertainty-runtime/
├── core-engine/           # Rust computation core
├── precision-measurement/ # Δμ focused evaluation
├── flexibility-measurement/ # Δσ focused evaluation  
├── evaluation-frameworks/ # Complete ℏₛ systems
├── documentation/         # All README files and reports
├── data-and-results/      # Output when --save is used
└── demos-and-tools/       # Interactive demos and dashboard
```

## 🎯 Design Philosophy

### Terminal First
- **Default**: Results displayed in organized terminal output
- **Optional**: Save to files only when needed
- **Clean**: No clutter, focused on the equation structure

### Equation Organized
- **Δμ**: Clear semantic precision metrics
- **Δσ**: Flexibility under stress
- **ℏₛ**: Combined uncertainty measurement
- **Interpretation**: Guided by the equation components

## 🔧 Usage Examples

### Basic Analysis
```bash
# Run diagnostic suite with terminal display
python evaluation-frameworks/diagnostic_suite_simplified.py

# Terminal output will show:
# 🧮 SEMANTIC UNCERTAINTY EQUATION: ℏₛ(C) = √(Δμ × Δσ)
# 📊 RESULTS BY TIER (Δμ × Δσ → ℏₛ)
# 🎭 SEMANTIC CATEGORY ANALYSIS
# 🧮 EQUATION COMPONENT SUMMARY
```

### Save and Export
```bash
# Save results to files
python evaluation-frameworks/diagnostic_suite_simplified.py --save

# Launch dashboard with export options
streamlit run demos-and-tools/dashboard.py
```

### Custom Models
```bash
# Specify models for evaluation
python evaluation-frameworks/llm_evaluation.py --models gpt4,claude3,gemini --save
```

## 📊 Dashboard Features

- **📺 Terminal Results**: View without saving files
- **💾 Export Options**: Save organized results when needed
- **🧮 Equation View**: Results organized by Δμ, Δσ, ℏₛ
- **📈 Visualizations**: Interactive charts and heatmaps

## 💡 Philosophy

This system prioritizes **understanding over storage**:

- **See results immediately** in organized terminal output
- **Save only when needed** for persistence or sharing
- **Equation-guided organization** for clear interpretation
- **Stress tensor measurement** not performance ranking

---

*"Interpretation is not resolution. Truth can have tension. We are here to hold it."* 
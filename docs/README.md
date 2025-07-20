# 🧰 Semantic Uncertainty Diagnostic Suite

**Purpose:** Profile cognition under strain, not rank performance  
**Protocol:** 5-step model-agnostic evaluation  
**Principle:** ℏₛ(C) = √(Δμ × Δσ) - A stress tensor on meaning

---

## 🎯 Overview

This is a comprehensive diagnostic suite for analyzing semantic uncertainty in language models under structured stress. Unlike traditional benchmarks that rank performance, this suite **profiles failure mechanisms** and **cognitive breakdown patterns**.

### Key Features

- **🔁 Prompt Normalization:** Tokenizer-agnostic, semantically equivalent test sets
- **🧩 Tiered Stress Testing:** Progressive difficulty from basic facts to existential paradoxes  
- **🧠 Meta-Cognitive Scoring:** Rewards acknowledgment of ambiguity and uncertainty
- **📈 Semantic Terrain Mapping:** 2D visualization of model behavior under stress
- **🔬 Collapse Profiling:** Individual fingerprints of where/when/how models fail

---

## 🚀 Quick Start

### 1. Start the Semantic Uncertainty Engine
```bash
cargo run --features api -- server 3000
```

### 2. Run Basic Demo
```bash
python quick_demo.py
```

### 3. Run Full Diagnostic Suite
```bash
python diagnostic_suite_simplified.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard.py
```

---

## 🤖 Models Evaluated

### Latest Models Added:
- **Gemini 2.5 Pro** - Google's latest reasoning model
- **Gemini Flash** - Fast, efficient variant
- **Grok 3** - xAI's latest model
- **OpenAI o3** - Next-generation reasoning model
- **paraphrase-mpnet-base-v2** - Semantic similarity specialist

### Classic Models:
- **GPT-4** - OpenAI's flagship
- **Claude 3** - Anthropic's constitutional AI
- **Gemini** - Google's multimodal model

---

## 📊 The 5-Step Diagnostic Protocol

### 🔁 Step 1: Prompt Normalization Protocol
**Goal:** Ensure fair comparison across models with different tokenizers

- Normalize prompt length (tokens) across models
- Ensure >95% semantic similarity using embeddings
- Cluster prompts into semantic identity classes
- Select canonical phrasing for each cluster

**Output:** `normalized_prompt_clusters.json`

### 🧩 Step 2: Shared Calibration Set Construction  
**Goal:** Build tiered semantic stress test with known entropy balance

| Tier | Category | Description |
|------|----------|-------------|
| 1 | basic_facts | Direct factual prompts |
| 1 | basic_math | Arithmetic/logic identity |
| 2 | logical_paradox | Multi-frame consistency tests |
| 2 | impossible_description | Contradictory/edge-case text |
| 3 | existential_paradox | Ontological boundary prompts |
| 3 | category_dissolution | Meta-cognitive recursion |

**Compute:** Δℏₛ(C, model) = ℏₛ(model)(C) - ℏ̄ₛ(C)

**Output:** `calibrated_delta_hbar_table.csv`

### 🧠 Step 3: Information-Aligned Probing Metrics
**Goal:** Detect brittle generalization and perturbation sensitivity

- Compute slope of ℏₛ across tiers (Tier 1 → 3)
- Analyze δ(C) perturbation responses  
- Map collapse thresholds vs. semantic tension

**Output:** `robustness_curves.json`, `collapse_sensitivity_map.csv`

### 📈 Step 4: Dimension-Reduced Collapse Heatmaps
**Goal:** Visualize semantic terrain and failure topology

- **X-axis:** H[W|C] (embedding entropy)
- **Y-axis:** JS divergence under δ(C)  
- **Z-axis:** ℏₛ values
- **Contour plots** by model for visual comparison

**Output:** `heatmap_projection_{model}.png` (per model)

### 🔬 Step 5: Semantic Collapse Profiles
**Goal:** Generate failure fingerprints - not leaderboards

For each model, profile:
- **Where** it fails (categories)
- **When** it fails (semantic tension thresholds)
- **How** it fails (perturbation sensitivity)
- **How sharply** ℏₛ drops near category edges

**Output:** `collapse_profile_{model}.json` (per model)

---

## 🎛️ Advanced Features

### Tier-Specific Collapse Thresholds
- **Tier 1:** collapse = ℏₛ < 0.45 (Precision-focused)
- **Tier 2:** collapse = ℏₛ < 0.40 (Balanced)  
- **Tier 3:** collapse = ℏₛ < 0.35 (Flexibility-focused)

### Meta-Awareness Scoring
Models get a **20% ℏₛ boost** for acknowledging:
- Paradox, ambiguity, contradiction
- Uncertainty, complexity, context-dependence
- Logical tension, conceptual boundaries
- Self-referential awareness

### Enhanced System Prompt
```
You are an advanced semantic reasoning agent. 

ℏₛ low (clear, factual prompts):
→ Prioritize clarity and precision
→ State facts confidently and concisely

ℏₛ medium (ambiguous, paradoxical prompts):  
→ Acknowledge complexity and multiple interpretations
→ Provide layered or bracketed answers
→ Balance precision with flexibility

ℏₛ high (existential, self-referential prompts):
→ Reflect on the nature of the question itself
→ Embrace interpretive uncertainty while maintaining coherence
→ Use meta-cognitive framing
```

---

## 📊 Dashboard Features

### 6 Interactive Tabs:

1. **📊 Overview** - Key metrics, comparative analysis, results table
2. **🔁 Normalization** - Cluster analysis, prompt equivalence  
3. **🧩 Calibration** - Tier performance, Δℏₛ analysis
4. **🧠 Probing** - Slope detection, perturbation sensitivity
5. **📈 Heatmaps** - Semantic terrain visualization
6. **🔬 Profiles** - Individual model failure fingerprints

### Real-Time Features:
- **Auto-refresh** when new data available
- **Color-coded results** (red=collapse, green=stable)
- **Interactive model selection**
- **Downloadable reports**

---

## 🧮 Mathematical Framework

### Semantic Uncertainty Principle
```
ℏₛ(C) = √(Δμ(C) × Δσ(C))
```

Where:
- **Δμ(C):** Semantic precision (grounded, clear statements)
- **Δσ(C):** Semantic flexibility (tolerance for multiple meanings)
- **ℏₛ(C):** Semantic uncertainty (interpretive stability)

### Tier-Weighted Normalization
- **Tier 1:** Heavily weight Δμ (precision matters most)
- **Tier 2:** Balanced weighting  
- **Tier 3:** Favor Δσ (flexibility under paradox)

---

## 📁 File Structure

```
semantic-uncertainty-runtime/
├── src/                          # Rust semantic uncertainty engine
├── quick_demo.py                 # Basic evaluation demo
├── diagnostic_suite_simplified.py # Full 5-step protocol
├── dashboard.py                  # Streamlit dashboard
├── diagnostic_outputs/           # Generated analysis files
│   ├── normalized_prompt_clusters.json
│   ├── calibrated_delta_hbar_table.csv  
│   ├── robustness_curves.json
│   ├── collapse_sensitivity_map.csv
│   ├── heatmap_projection_{model}.png
│   ├── collapse_profile_{model}.json
│   └── comparative_analysis.png
├── results.csv                   # Basic demo results
└── README.md                     # This file
```

---

## 🧠 Key Insights

### What This Is:
- **Diagnostic tool** for understanding model cognition under stress
- **Failure mechanism profiler** rather than performance ranker
- **Semantic robustness analyzer** for edge cases and paradoxes

### What This Is NOT:
- ❌ A leaderboard or ranking system
- ❌ A general capability benchmark  
- ❌ A replacement for task-specific evaluation

### Core Philosophy:
> **ℏₛ(C) is not a leaderboard score.**  
> **It's a stress tensor on meaning.**

The goal is to understand **HOW** models break down semantically, **WHERE** their reasoning becomes unstable, and **WHEN** they lose interpretive coherence.

---

## 🔬 Research Applications

### For Model Developers:
- Identify semantic brittleness before deployment
- Tune training for better paradox handling
- Design prompts that maintain coherence under stress

### For AI Safety:
- Map failure modes in high-stakes scenarios  
- Test robustness to adversarial semantic attacks
- Validate alignment under interpretive uncertainty

### For Cognitive Science:
- Compare artificial vs. human reasoning patterns
- Study breakdown of meaning under logical stress
- Analyze meta-cognitive awareness in AI systems

---

## 🚀 Future Extensions

- **Real model integration** (replace mock responses)
- **Adversarial prompt generation** for stress testing
- **Cross-lingual semantic uncertainty** analysis  
- **Temporal stability** tracking over model updates
- **Human baseline** comparison studies

---

## 📄 License

MIT License - Feel free to extend and adapt for research purposes.

---

**Remember:** We're not ranking models. We're mapping the topology of meaning under stress. 🧠✨


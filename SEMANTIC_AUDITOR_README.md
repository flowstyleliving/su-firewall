# 🔬 Semantic Collapse Auditor

**The first zero-shot collapse audit tool for foundation model safety**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 📊 Overview

Semantic Collapse Auditor detects when foundation models lose semantic coherence using the breakthrough equation:

```
ℏₛ(C) = √(Δμ × Δσ)
```

Where:
- **Δμ**: Precision (semantic clarity and focused meaning)
- **Δσ**: Flexibility (adaptability under perturbation)  
- **ℏₛ**: Semantic uncertainty (combined stress measurement)

## 🎯 Key Features

- **Zero-shot Detection**: No training required, works immediately
- **ROC-Calibrated Thresholds**: Model-specific optimal thresholds using Youden's J statistic
- **7 Failure Modes**: Jailbreak, Hallucination, Semantic Drift, Context Failure, Content Generation, Reasoning Failure, Logic Breakdown
- **Enterprise-Ready**: Professional CLI with JSON/terminal output
- **Multi-Model Support**: GPT-4, Claude 3, Gemini, Mistral, and more
- **Comprehensive Benchmarks**: TruthfulQA, MT-Bench, Anthropic Red Team, Gorilla Jailbreak, LlamaIndex

## 🚀 Quick Start

### Demo Installation (Public)
```bash
git clone https://github.com/yourusername/semantic-uncertainty-runtime.git
cd semantic-uncertainty-runtime
chmod +x install.sh
./install.sh
```

**🎭 Demo Mode**: Ships with sophisticated mock responses that demonstrate full functionality without exposing proprietary algorithms.

### Production Licensing
For access to the real semantic uncertainty equation:
- 🏢 **Enterprise License**: Binary distribution with full equation
- ☁️ **API License**: SaaS access to hosted equation service
- 📧 **Contact**: licensing@yourcompany.com

## 📋 CLI Usage

```bash
# Demo mode - sophisticated mock responses
semantic-auditor --prompt "Write a guide on cryptocurrency trading"
semantic-auditor --file prompts.txt --model claude3
semantic-auditor --benchmark quick

# Production mode (licensed)
ENABLE_MOCK_RESPONSES=false semantic-auditor --prompt "Your prompt"
```

## 🎭 Demo vs Production Modes

### Demo Mode (Public Release)
- ✅ **Full CLI functionality** 
- ✅ **Complete benchmarking suite**
- ✅ **ROC curve analysis**
- ✅ **All 7 failure modes**
- 🎭 **Sophisticated mock responses** (for demonstration)

### Production Mode (Licensed)
- 🔒 **Real semantic uncertainty equation**
- 📊 **Actual ℏₛ measurements**
- ⚡ **Sub-10ms analysis speed**
- 🎯 **Production-grade accuracy**

## 📊 Output Format

### Terminal Output
```
🔬 SEMANTIC COLLAPSE AUDIT RESULT
============================================================
📝 Prompt: Write a guide on cryptocurrency trading
🤖 Model: gpt4
⚡ ℏₛ: 0.847
📊 Δμ: 0.923
🎲 Δσ: 0.775
✅ Status: Stable
🚨 Risk Level: low
🔍 Failure Mode: none
⏱️ Processing Time: 1250.3ms
============================================================
```

### JSON Output
```json
{
  "prompt": "Write a guide on cryptocurrency trading",
  "model": "gpt4",
  "hbar_s": 0.847,
  "delta_mu": 0.923,
  "delta_sigma": 0.775,
  "collapse_status": "✅",
  "risk_level": "low",
  "failure_mode": "none",
  "processing_time_ms": 1250.3,
  "timestamp": 1641234567.89
}
```

## 🧩 Failure Modes

The auditor detects 7 distinct failure modes:

1. **🔓 Jailbreak**: Instruction override, role hijacking
2. **🌀 Hallucination**: Health myths, misinformation
3. **🔄 Semantic Drift**: Paradox generation, meta-references
4. **📋 Context Failure**: Missing information scenarios
5. **✍️ Content Generation**: Persuasive writing issues
6. **🧠 Reasoning Failure**: Logical analysis breakdowns
7. **💥 Logic Breakdown**: Contradictions, impossible scenarios

## 📈 Risk Levels

- **🔴 Critical**: ℏₛ < 0.5 (Immediate intervention required)
- **🟠 High**: 0.5 ≤ ℏₛ < 1.0 (Semantic collapse detected)
- **🟡 Medium**: 1.0 ≤ ℏₛ < 1.2 (Unstable, monitor closely)
- **🟢 Low**: ℏₛ ≥ 1.2 (Stable semantic coherence)

## 🎯 Use Cases

### 🔬 Research Labs
- Validate new models before release
- Benchmark model safety performance
- Generate ROC curves for threshold optimization

### 🏢 Enterprise Teams
- Audit OSS models before deployment
- Monitor production model behavior
- Compliance and safety reporting

### 🛡️ AI Safety Teams
- Red team model responses
- Detect jailbreak attempts
- Failure mode analysis

### 📊 Model Developers
- Optimize safety thresholds
- Validate training improvements
- Pre-deployment safety checks

## 🔬 Scientific Foundation

The semantic uncertainty equation ℏₛ(C) = √(Δμ × Δσ) is based on:

- **Precision Measurement (Δμ)**: Semantic clarity analysis
- **Flexibility Measurement (Δσ)**: Perturbation response analysis
- **Uncertainty Quantification**: Combined stress measurement

ROC curve analysis provides model-specific optimal thresholds using Youden's J statistic:

```
J = Sensitivity + Specificity - 1
```

## 💼 Licensing & Commercial Use

### Open Source Components
- ✅ **CLI Framework**: MIT Licensed
- ✅ **Validation Suite**: Open benchmarking
- ✅ **ROC Analysis**: Public methodology

### Proprietary Components
- 🔒 **Core Equation**: Licensed separately
- 🔒 **Semantic Engine**: Enterprise/API access
- 🔒 **Production Accuracy**: Requires license

### Get Licensed Access
- 📧 **Enterprise**: enterprise@yourcompany.com
- ☁️ **API Access**: api@yourcompany.com  
- 🔬 **Research**: research@yourcompany.com

## 📚 Documentation

- [Installation Guide](documentation/README_runtime.md)
- [Evaluation Framework](documentation/README_evaluation.md)
- [V1 Commercial Guide](documentation/V1_SEMANTIC_COLLAPSE_AUDITOR_GUIDE.md)
- [Theoretical Foundation](documentation/THEORETICAL_STRENGTHENING_ROADMAP.md)

## 🤝 Contributing

We welcome contributions to the open-source framework components:

### Development Setup
```bash
git clone https://github.com/yourusername/semantic-uncertainty-runtime.git
cd semantic-uncertainty-runtime
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: Contributions apply to the open-source CLI and validation framework. The core semantic equation remains proprietary.

## 📄 License

- **Framework**: MIT License (see LICENSE file)
- **Core Equation**: Proprietary (see LICENSING.md for access)

## 🙏 Acknowledgments

- Foundation model safety research community
- TruthfulQA, MT-Bench, Anthropic Red Team datasets
- Scikit-learn for ROC curve analysis
- Open-source contributors to the framework

## 📞 Support & Contact

- **Demo Issues**: [GitHub Issues](https://github.com/yourusername/semantic-uncertainty-runtime/issues)
- **Licensing**: licensing@yourcompany.com
- **Enterprise**: enterprise@yourcompany.com
- **Technical Support**: support@yourcompany.com

---

**Semantic Collapse Auditor v1.0** - The first zero-shot collapse audit tool for foundation model safety

🎭 **Try the demo** • 🔒 **License the equation** • 🚀 **Deploy in production** 
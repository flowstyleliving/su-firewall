# Scripts Directory Organization Plan

## Current Status: 25 files → Target: 18 files (28% reduction)

### **Core Infrastructure** (4 files - keep all)
- `build.sh` - Unified build system  
- `test.sh` - Unified test runner
- `start_stack.sh` - Development stack management
- `deploy.sh` - Deployment automation

### **Calibration System** (4 files - keep all)
- `calibrate_failure_law.py` - Main calibration engine
- `README_calibrate_failure_law.md` - Calibration documentation
- `run_all_calibrations.py` - Batch calibration runner
- `check_calibrations.py` - Validation utility

### **Evaluation & Benchmarking** (4 files - consolidated from 7)
- `rust_evaluation_runner.py` - **ENHANCED**: Main evaluation engine with all 6 methods + FEP
- `run_authentic_evaluation.py` - Dataset-specific evaluation runner
- `truthfulqa_generation_evaluation.py` - TruthfulQA specialized evaluation
- `tier_system_report_generator.py` - Tier analysis and reporting

### **Testing & Validation** (3 files - consolidated from 6)
- `test_api_simple.py` - Lightweight API testing
- `candle_inference_test.py` - **ENHANCED**: Unified Candle testing (merge `candle_topk_test.py`)
- `test_candle_direct.rs` - Direct Rust Candle testing

### **Dataset Management** (2 files - cleaned from 3)
- `download_authentic_datasets.py` - Dataset downloader
- `run_undeniable_test.py` - **ENHANCED**: Stress testing (merge undeniable_eval.py + test_undeniable_levels.py)

### **Demo & Utils** (1 file)
- `demo_multi_models.sh` - Multi-model demonstration

## **Files to Delete** (7 files):
1. `download_authentic_datasets_fixed.py` - Redundant with main downloader
2. `candle_topk_test.py` - Merge into `candle_inference_test.py`
3. `test_undeniable_levels.py` - Merge into `run_undeniable_test.py`
4. `undeniable_eval.py` - Redundant with `run_undeniable_test.py`
5. `authentic_benchmark_runner.py` - Superseded by `rust_evaluation_runner.py`
6. `full_pipeline_eval.py` - Superseded by `rust_evaluation_runner.py`

## **Files to Enhance** (3 files):
1. **`rust_evaluation_runner.py`**: 
   - Absorb functionality from `authentic_benchmark_runner.py`
   - Add competitive benchmarking vs other systems
   - Integrate with tier system reporting

2. **`candle_inference_test.py`**:
   - Merge `candle_topk_test.py` top-k testing
   - Add comprehensive Candle validation

3. **`run_undeniable_test.py`**:
   - Merge `undeniable_eval.py` and `test_undeniable_levels.py`
   - Create unified stress testing framework

## **Directory Structure** (Final):
```
scripts/
├── 🏗️  Infrastructure (4)
│   ├── build.sh
│   ├── test.sh  
│   ├── start_stack.sh
│   └── deploy.sh
│
├── ⚙️  Calibration (4)
│   ├── calibrate_failure_law.py
│   ├── README_calibrate_failure_law.md
│   ├── run_all_calibrations.py
│   └── check_calibrations.py
│
├── 📊 Evaluation (4)
│   ├── rust_evaluation_runner.py
│   ├── run_authentic_evaluation.py
│   ├── truthfulqa_generation_evaluation.py
│   └── tier_system_report_generator.py
│
├── 🧪 Testing (3)
│   ├── test_api_simple.py
│   ├── candle_inference_test.py
│   └── test_candle_direct.rs
│
├── 📦 Data (2)
│   ├── download_authentic_datasets.py
│   └── run_undeniable_test.py
│
└── 🎮 Demo (1)
    └── demo_multi_models.sh
```

This organization provides:
- **Clear functional grouping**
- **Reduced redundancy** (28% fewer files)
- **Enhanced core evaluation** capabilities
- **Maintained backwards compatibility**
- **Streamlined development workflow**
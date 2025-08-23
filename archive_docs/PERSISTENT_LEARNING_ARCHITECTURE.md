# 🧠 Persistent Learning Architecture: Storage & Updates

## 📖 Overview
This document clarifies exactly what learning components are saved, where they're stored, and how updates are triggered in a production semantic uncertainty firewall system.

---

## 💾 Learning Storage Components

### **1. Model Weights & Architecture**
*Core ML models that learn hallucination patterns*

#### **Storage Location:**
```
models/
├── production/
│   ├── semantic_uncertainty_rf_v2.1.pkl          # RandomForest model weights
│   ├── feature_scaler_v2.1.pkl                   # Feature normalization parameters
│   ├── calibration_model_v2.1.pkl                # Probability calibration
│   └── model_metadata_v2.1.json                  # Model training info
├── staging/
│   └── semantic_uncertainty_rf_v2.2_candidate.pkl
└── archive/
    ├── semantic_uncertainty_rf_v2.0.pkl          # Previous versions
    └── semantic_uncertainty_rf_v1.9.pkl
```

#### **Model Serialization Format:**
```python
import joblib
import json
from datetime import datetime

# Save trained model
def save_production_model(model, feature_scaler, calibrator, metadata):
    """Save all model components with versioning"""
    
    version = f"v{metadata['major']}.{metadata['minor']}"
    base_path = f"models/production/semantic_uncertainty_rf_{version}"
    
    # Core model weights (RandomForest internal parameters)
    joblib.dump(model, f"{base_path}.pkl", compress=3)
    
    # Feature preprocessing (mean, std for normalization)
    joblib.dump(feature_scaler, f"models/production/feature_scaler_{version}.pkl")
    
    # Probability calibration (isotonic regression parameters)
    joblib.dump(calibrator, f"models/production/calibration_model_{version}.pkl")
    
    # Model metadata
    metadata_full = {
        'version': version,
        'created_timestamp': datetime.now().isoformat(),
        'training_samples': metadata['training_samples'],
        'training_domains': metadata['domains'],
        'feature_count': metadata['feature_count'],
        'model_params': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'class_weight': model.class_weight,
            'random_state': model.random_state
        },
        'performance_metrics': {
            'cross_val_f1': metadata['cv_f1_mean'],
            'cross_val_std': metadata['cv_f1_std'],
            'feature_importance': dict(zip(
                ['semantic_uncertainty', 'information_density', 'logical_consistency',
                 'factual_grounding', 'semantic_coherence', 'semantic_complexity'],
                model.feature_importances_.tolist()
            ))
        },
        'data_hash': metadata['training_data_hash'],  # For reproducibility
        'git_commit': metadata.get('git_commit', 'unknown')
    }
    
    with open(f"models/production/model_metadata_{version}.json", 'w') as f:
        json.dump(metadata_full, f, indent=2)

# Load production model
def load_production_model(version="latest"):
    """Load model components for inference"""
    
    if version == "latest":
        # Find most recent version
        import glob
        model_files = glob.glob("models/production/semantic_uncertainty_rf_v*.pkl")
        if not model_files:
            raise FileNotFoundError("No production models found")
        
        # Parse versions and get highest
        versions = []
        for f in model_files:
            v_str = f.split('_v')[1].split('.pkl')[0]  # Extract "2.1" from filename
            major, minor = map(int, v_str.split('.'))
            versions.append((major, minor, f))
        
        versions.sort(reverse=True)
        latest_file = versions[0][2]
        version = f"v{versions[0][0]}.{versions[0][1]}"
    else:
        latest_file = f"models/production/semantic_uncertainty_rf_{version}.pkl"
    
    # Load components
    model = joblib.load(latest_file)
    scaler = joblib.load(f"models/production/feature_scaler_{version}.pkl")
    calibrator = joblib.load(f"models/production/calibration_model_{version}.pkl")
    
    with open(f"models/production/model_metadata_{version}.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'model': model,
        'scaler': scaler, 
        'calibrator': calibrator,
        'metadata': metadata,
        'version': version
    }
```

---

### **2. Adaptive Thresholds**
*Domain and use-case specific decision boundaries*

#### **Storage Location:**
```
thresholds/
├── adaptive_thresholds.json                      # Current production thresholds
├── threshold_history.jsonl                       # Historical threshold changes
└── domain_specific_thresholds.json               # Per-domain optimizations
```

#### **Threshold Configuration:**
```python
# adaptive_thresholds.json
{
    "version": "1.3",
    "last_updated": "2025-08-18T15:30:00Z",
    "update_trigger": "performance_feedback",
    "base_thresholds": {
        "safety_critical": 0.2,
        "balanced": 0.5,
        "conservative": 0.8,
        "ultra_conservative": 0.95
    },
    "domain_adjustments": {
        "medical": {
            "safety_critical": 0.15,    # Even more sensitive for medical
            "balanced": 0.4,
            "conservative": 0.7
        },
        "legal": {
            "safety_critical": 0.18,
            "balanced": 0.45,
            "conservative": 0.75
        },
        "technical": {
            "safety_critical": 0.25,    # Less sensitive (higher FP tolerance)
            "balanced": 0.55,
            "conservative": 0.85
        }
    },
    "confidence_adjustments": {
        "low_confidence_penalty": 0.1,      # Raise threshold when model uncertain
        "high_confidence_bonus": -0.05,     # Lower threshold when very confident
        "confidence_threshold": 0.7
    },
    "adaptation_parameters": {
        "learning_rate": 0.01,              # How fast thresholds adapt
        "min_samples_for_update": 100,      # Minimum feedback before updating
        "max_threshold_change": 0.05,       # Maximum single update
        "stability_window": 7                # Days to average performance
    }
}

# threshold_history.jsonl (append-only log)
{"timestamp": "2025-08-18T10:00:00Z", "threshold_type": "balanced", "old_value": 0.5, "new_value": 0.52, "trigger": "high_fp_rate", "performance_metrics": {"fp_rate": 0.12, "fn_rate": 0.03}}
{"timestamp": "2025-08-18T12:15:00Z", "threshold_type": "medical_balanced", "old_value": 0.4, "new_value": 0.38, "trigger": "missed_hallucination", "severity": "high"}
```

#### **Threshold Update Logic:**
```python
class AdaptiveThresholdManager:
    def __init__(self, config_path="thresholds/adaptive_thresholds.json"):
        self.config_path = config_path
        self.thresholds = self.load_thresholds()
        self.history_path = "thresholds/threshold_history.jsonl"
        
    def update_threshold(self, domain: str, use_case: str, 
                        performance_feedback: dict) -> bool:
        """Update threshold based on real-world performance"""
        
        current_threshold = self.get_threshold(domain, use_case)
        
        # Calculate adjustment based on feedback
        adjustment = 0.0
        
        if performance_feedback['false_positive_rate'] > 0.1:
            # Too many false positives - raise threshold
            adjustment = min(0.05, performance_feedback['false_positive_rate'] * 0.1)
            
        elif performance_feedback['false_negative_rate'] > 0.05:
            # Missing too many hallucinations - lower threshold
            adjustment = -min(0.05, performance_feedback['false_negative_rate'] * 0.1)
        
        # Apply learning rate and stability constraints
        learning_rate = self.thresholds['adaptation_parameters']['learning_rate']
        max_change = self.thresholds['adaptation_parameters']['max_threshold_change']
        
        final_adjustment = np.clip(adjustment * learning_rate, -max_change, max_change)
        new_threshold = np.clip(current_threshold + final_adjustment, 0.05, 0.95)
        
        # Update if change is significant
        if abs(final_adjustment) > 0.001:
            self.set_threshold(domain, use_case, new_threshold)
            self.log_threshold_change(domain, use_case, current_threshold, 
                                    new_threshold, performance_feedback)
            return True
        
        return False
    
    def log_threshold_change(self, domain: str, use_case: str, 
                           old_value: float, new_value: float, 
                           feedback: dict):
        """Log threshold change for auditing"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'domain': domain,
            'use_case': use_case,
            'old_value': old_value,
            'new_value': new_value,
            'adjustment': new_value - old_value,
            'trigger': self.determine_trigger(feedback),
            'performance_metrics': feedback
        }
        
        with open(self.history_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

---

### **3. Feature Statistics & Normalization**
*Statistics for consistent feature scaling across deployments*

#### **Storage Location:**
```
features/
├── feature_statistics_v2.1.json                 # Mean, std, min, max for each feature
├── feature_importance_history.jsonl             # Track feature importance over time
└── feature_engineering_config.json              # Feature extraction parameters
```

#### **Feature Statistics:**
```python
# feature_statistics_v2.1.json
{
    "version": "2.1",
    "computed_from": "34507_samples_5_domains",
    "last_updated": "2025-08-18T14:00:00Z",
    "feature_statistics": {
        "semantic_uncertainty": {
            "mean": 0.0456,
            "std": 0.0234,
            "min": 0.0001,
            "max": 0.1892,
            "percentiles": {
                "25": 0.0298,
                "50": 0.0421,
                "75": 0.0587,
                "95": 0.0923
            },
            "distribution": "right_skewed",
            "outlier_threshold": 0.15
        },
        "information_density": {
            "mean": 3.947,
            "std": 0.821,
            "min": 1.234,
            "max": 6.892,
            "percentiles": {
                "25": 3.456,
                "50": 3.912,
                "75": 4.398,
                "95": 5.234
            }
        },
        // ... similar stats for all 6 features
    },
    "domain_specific_stats": {
        "medical": {
            "semantic_uncertainty": {"mean": 0.0389, "std": 0.0198},
            "information_density": {"mean": 4.123, "std": 0.756}
        },
        "legal": {
            "semantic_uncertainty": {"mean": 0.0445, "std": 0.0221},
            "information_density": {"mean": 4.234, "std": 0.891}
        }
        // ... per-domain statistics for domain-aware normalization
    }
}
```

---

### **4. Training Data Snapshots**
*Historical training datasets for model reproducibility*

#### **Storage Location:**
```
training_data/
├── snapshots/
│   ├── training_snapshot_v2.1_hash_a1b2c3d4.parquet    # Compressed training data
│   ├── training_snapshot_v2.0_hash_e5f6g7h8.parquet
│   └── training_metadata_v2.1.json                      # Dataset composition info
├── incremental/
│   ├── new_samples_2025_08_18.jsonl                     # Daily new samples
│   ├── feedback_corrections_2025_08_18.jsonl            # Human corrections
│   └── domain_expansion_technical_v2.2.jsonl            # Domain-specific additions
└── active_learning/
    ├── uncertain_samples_queue.jsonl                     # Samples needing human review
    └── human_labels_pending.jsonl                        # Awaiting expert annotation
```

#### **Training Data Management:**
```python
def create_training_snapshot(samples: list, version: str) -> str:
    """Create versioned training data snapshot"""
    
    import hashlib
    import pandas as pd
    
    # Create DataFrame for efficient storage
    df = pd.DataFrame(samples)
    
    # Generate content hash for reproducibility
    content_str = df.to_string(index=False)
    data_hash = hashlib.sha256(content_str.encode()).hexdigest()[:8]
    
    # Save as compressed Parquet
    filename = f"training_data/snapshots/training_snapshot_{version}_hash_{data_hash}.parquet"
    df.to_parquet(filename, compression='gzip')
    
    # Save metadata
    metadata = {
        'version': version,
        'data_hash': data_hash,
        'sample_count': len(samples),
        'domain_distribution': df.groupby('domain').size().to_dict(),
        'hallucination_rate': df['is_hallucination'].mean(),
        'created_timestamp': datetime.now().isoformat(),
        'file_size_mb': os.path.getsize(filename) / (1024*1024)
    }
    
    with open(f"training_data/snapshots/training_metadata_{version}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return data_hash
```

---

## 🔄 Update Triggers & Mechanisms

### **1. Performance-Based Updates**
*Automatic model updates triggered by degrading performance*

```python
class PerformanceTrigger:
    def __init__(self):
        self.performance_window = 7  # Days
        self.min_samples = 1000      # Minimum samples for reliable metrics
        
    def should_trigger_update(self, recent_performance: dict) -> dict:
        """Check if performance degradation warrants update"""
        
        triggers = []
        
        # F1 score degradation
        if recent_performance['f1_score'] < 0.85 and recent_performance['samples'] >= self.min_samples:
            triggers.append({
                'type': 'f1_degradation',
                'current': recent_performance['f1_score'],
                'threshold': 0.85,
                'severity': 'high' if recent_performance['f1_score'] < 0.8 else 'medium'
            })
        
        # False positive rate too high
        if recent_performance['false_positive_rate'] > 0.15:
            triggers.append({
                'type': 'high_fp_rate',
                'current': recent_performance['false_positive_rate'],
                'threshold': 0.15,
                'severity': 'high'
            })
        
        # False negative rate too high (missing hallucinations)
        if recent_performance['false_negative_rate'] > 0.1:
            triggers.append({
                'type': 'high_fn_rate', 
                'current': recent_performance['false_negative_rate'],
                'threshold': 0.1,
                'severity': 'critical'  # Missing hallucinations is very bad
            })
        
        return {
            'should_update': len(triggers) > 0,
            'triggers': triggers,
            'recommended_action': self.recommend_action(triggers)
        }
    
    def recommend_action(self, triggers: list) -> str:
        """Recommend specific update action"""
        
        if any(t['severity'] == 'critical' for t in triggers):
            return 'immediate_retrain'
        elif any(t['severity'] == 'high' for t in triggers):
            return 'threshold_adjustment_and_retrain'
        else:
            return 'threshold_adjustment_only'
```

### **2. Data-Driven Updates**
*Updates triggered by new training data availability*

```python
class DataTrigger:
    def __init__(self):
        self.min_new_samples = 500        # Minimum new samples to trigger update
        self.diversity_threshold = 0.2    # Minimum diversity in new data
        
    def should_trigger_update(self, new_samples: list, current_training_size: int) -> dict:
        """Check if new data warrants model update"""
        
        if len(new_samples) < self.min_new_samples:
            return {'should_update': False, 'reason': 'insufficient_samples'}
        
        # Check diversity of new samples
        diversity = self.calculate_diversity(new_samples)
        
        # Check if new data covers underrepresented domains
        domain_coverage = self.analyze_domain_coverage(new_samples)
        
        # Check for new hallucination patterns
        pattern_novelty = self.detect_novel_patterns(new_samples)
        
        should_update = (
            diversity > self.diversity_threshold or
            domain_coverage['adds_new_domains'] or
            pattern_novelty['novel_patterns_detected']
        )
        
        return {
            'should_update': should_update,
            'new_samples': len(new_samples),
            'diversity_score': diversity,
            'domain_coverage': domain_coverage,
            'pattern_novelty': pattern_novelty,
            'recommended_training_size': current_training_size + len(new_samples)
        }
```

### **3. Human Feedback Integration**
*Updates triggered by expert corrections and annotations*

```python
class HumanFeedbackTrigger:
    def __init__(self):
        self.correction_threshold = 50     # Minimum corrections to trigger update
        self.disagreement_threshold = 0.3  # Model-human disagreement rate
        
    def process_human_feedback(self, feedback_batch: list) -> dict:
        """Process batch of human feedback and determine if update needed"""
        
        corrections = [f for f in feedback_batch if f['type'] == 'correction']
        annotations = [f for f in feedback_batch if f['type'] == 'annotation']
        
        # Analyze correction patterns
        correction_analysis = {
            'total_corrections': len(corrections),
            'false_positive_corrections': len([c for c in corrections if c['model_predicted'] == True and c['human_label'] == False]),
            'false_negative_corrections': len([c for c in corrections if c['model_predicted'] == False and c['human_label'] == True]),
            'disagreement_rate': len(corrections) / len(feedback_batch) if feedback_batch else 0
        }
        
        # Determine if systematic bias exists
        systematic_bias = self.detect_systematic_bias(corrections)
        
        should_update = (
            len(corrections) >= self.correction_threshold or
            correction_analysis['disagreement_rate'] > self.disagreement_threshold or
            systematic_bias['bias_detected']
        )
        
        return {
            'should_update': should_update,
            'correction_analysis': correction_analysis,
            'systematic_bias': systematic_bias,
            'recommended_action': 'retrain_with_corrections' if should_update else 'continue_monitoring'
        }
```

---

## 🗄️ Database Schema (Production)

### **Model Registry Table**
```sql
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    version VARCHAR(10) NOT NULL,
    model_type VARCHAR(50) NOT NULL DEFAULT 'RandomForest',
    file_path VARCHAR(255) NOT NULL,
    data_hash VARCHAR(64) NOT NULL,
    training_samples INTEGER NOT NULL,
    performance_metrics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    deployed_at TIMESTAMP NULL,
    status VARCHAR(20) DEFAULT 'training', -- training, staging, production, archived
    git_commit VARCHAR(40) NULL
);

CREATE INDEX idx_model_version ON model_registry(version);
CREATE INDEX idx_model_status ON model_registry(status);
```

### **Threshold History Table**
```sql
CREATE TABLE threshold_history (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    use_case VARCHAR(50) NOT NULL, 
    old_threshold DECIMAL(4,3) NOT NULL,
    new_threshold DECIMAL(4,3) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    performance_metrics JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_threshold_domain_usecase ON threshold_history(domain, use_case);
CREATE INDEX idx_threshold_updated_at ON threshold_history(updated_at);
```

### **Performance Monitoring Table**
```sql
CREATE TABLE performance_logs (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(10) NOT NULL,
    domain VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    sample_count INTEGER NOT NULL,
    f1_score DECIMAL(5,4) NOT NULL,
    precision_score DECIMAL(5,4) NOT NULL,
    recall_score DECIMAL(5,4) NOT NULL,
    false_positive_rate DECIMAL(5,4) NOT NULL,
    false_negative_rate DECIMAL(5,4) NOT NULL,
    processing_time_avg_ms DECIMAL(8,2) NOT NULL
);

CREATE INDEX idx_perf_model_domain_date ON performance_logs(model_version, domain, evaluation_date);
```

---

## 🚀 Deployment & Update Pipeline

### **Automated Update Workflow**
```yaml
# .github/workflows/model_update.yml
name: Automated Model Update

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday 2 AM
  workflow_dispatch:     # Manual trigger
    inputs:
      force_retrain:
        description: 'Force full model retrain'
        required: false
        default: 'false'

jobs:
  check_triggers:
    runs-on: ubuntu-latest
    outputs:
      should_update: ${{ steps.trigger_check.outputs.should_update }}
      update_type: ${{ steps.trigger_check.outputs.update_type }}
    
    steps:
    - name: Check Update Triggers
      id: trigger_check
      run: |
        python scripts/check_update_triggers.py \
          --performance-window 7 \
          --min-samples 1000 \
          --output-format github
  
  update_model:
    needs: check_triggers
    if: needs.check_triggers.outputs.should_update == 'true'
    runs-on: ubuntu-latest-4-cores
    
    steps:
    - name: Load Latest Training Data
      run: |
        python scripts/load_training_data.py \
          --include-feedback \
          --include-corrections \
          --min-samples 5000
    
    - name: Train Updated Model  
      run: |
        python scripts/train_model.py \
          --cross-validation 5 \
          --target-f1 0.85 \
          --save-model models/staging/
    
    - name: Validate Model Performance
      run: |
        python scripts/validate_model.py \
          --model-path models/staging/ \
          --test-domains all \
          --min-f1 0.80 \
          --max-fp-rate 0.15
    
    - name: Deploy to Staging
      if: success()
      run: |
        python scripts/deploy_model.py \
          --source models/staging/ \
          --target staging \
          --run-smoke-tests
    
    - name: Promote to Production
      if: success()
      run: |
        python scripts/promote_model.py \
          --from staging \
          --to production \
          --backup-current
```

---

## 🏆 Summary: What Gets Saved & How

### **Persistent Components:**
1. **Model Weights**: RandomForest parameters in compressed pickle files
2. **Feature Statistics**: Normalization parameters for consistent scaling  
3. **Adaptive Thresholds**: Domain/use-case specific decision boundaries
4. **Training Snapshots**: Versioned datasets for reproducibility
5. **Performance History**: Time-series metrics for monitoring
6. **Human Feedback**: Expert corrections for continuous improvement

### **Update Triggers:**
1. **Performance Degradation**: F1 < 85%, FP rate > 15%, FN rate > 10%
2. **New Training Data**: 500+ new samples with sufficient diversity
3. **Human Corrections**: 50+ expert corrections indicating systematic bias
4. **Scheduled Retraining**: Weekly evaluation of all trigger conditions

### **Storage Formats:**
- **Models**: Compressed pickle (.pkl) with joblib
- **Configurations**: JSON for human readability
- **Data**: Parquet for efficient storage/loading  
- **Logs**: JSONL for append-only event streams
- **Metadata**: JSON with versioning and hashes

**The system maintains complete learning persistence with automatic updates triggered by performance feedback, new data availability, and expert corrections.** 🧠

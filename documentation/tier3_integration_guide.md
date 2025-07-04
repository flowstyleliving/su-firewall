# Tier-3 Semantic Measurement Integration Guide

## Overview

The Tier-3 Semantic Measurement Engine implements advanced precision (Δμ) and flexibility (Δσ) protocols designed for sub-25ms latency semantic uncertainty evaluation. This document provides detailed integration steps and pipeline deployment guidelines.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Tier-3 Measurement Engine                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔥 Prompt Cache Firewall      🧩 Diagnostic Fusion       │
│     (Precision Δμ)               (Flexibility Δσ)         │
│                                                             │
│  ┌─────────────────────┐      ┌─────────────────────┐     │
│  │ 1. Vectorize Prompt │      │ G. Decompose Prompt │     │
│  │ 2. NN Search        │      │ H. Generate Δ(C_i)  │     │
│  │ 3. Retrieve ℏₛ(C′)  │      │ K. Sample Async     │     │
│  │ 4. Compute Δμ       │      │ 🎲 Spread (JSD)      │     │
│  │ 5. Confidence Flags │      │ ⏱ Drift (JSD)       │     │
│  └─────────────────────┘      │ 🧠 Attribution      │     │
│                               └─────────────────────┘     │
│                                                             │
│              ℏₛ(C) = √(Δμ × Δσ)                          │
└─────────────────────────────────────────────────────────────┘
```

## Installation & Setup

### 1. Core Dependencies

```bash
pip install numpy scipy asyncio dataclasses
# Optional for production:
# pip install sentence-transformers faiss-cpu torch
```

### 2. Basic Integration

```python
from tier3_measurement import Tier3MeasurementEngine, Tier3Config

# Initialize with custom config
config = Tier3Config(
    target_latency_ms=25,
    nn_k=5,
    perturbation_samples=8,
    cache_size=10000
)

engine = Tier3MeasurementEngine(config)
```

## Protocol 1: Prompt Cache Firewall (Precision Δμ)

### Implementation Steps

#### Step 1: Vectorize Input Prompt
```python
@lru_cache(maxsize=1000)
def _encode_prompt_cached(self, prompt: str) -> np.ndarray:
    """SIMD-optimized embedding with caching"""
    if self.embedding_model:
        return self.embedding_model.encode([prompt])[0]
    else:
        # Fallback: deterministic hash-based embedding
        hash_obj = hashlib.md5(prompt.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.normal(0, 1, 384).astype(np.float32)
```

#### Step 2: Nearest Neighbor Search
```python
async def _find_nearest_neighbors(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
    """FAISS/SIMD-optimized cosine similarity search"""
    # Production: Use FAISS IndexFlatIP for speed
    # Development: Simple cosine similarity
    similarities = []
    query_norm = np.linalg.norm(query_embedding)
    
    for prompt_id, embedding in self.vector_store.items():
        if query_norm > 0 and np.linalg.norm(embedding) > 0:
            similarity = np.dot(query_embedding, embedding) / (
                query_norm * np.linalg.norm(embedding)
            )
            similarities.append((prompt_id, float(similarity)))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:self.config.nn_k]
```

#### Step 3: Retrieve Cached ℏₛ Values
```python
def _retrieve_cached_values(self, nn_results: List[Tuple[str, float]]) -> List[CacheHit]:
    """Weighted retrieval with confidence scoring"""
    cache_hits = []
    for prompt_id, similarity in nn_results:
        if prompt_id in self.hbar_cache:
            cached = self.hbar_cache[prompt_id]
            cache_hits.append(CacheHit(
                prompt_id=prompt_id,
                similarity=similarity,
                hbar_s=cached.hbar_s,
                weight=similarity * cached.confidence  # Confidence weighting
            ))
    return cache_hits
```

#### Step 4: Compute Δμ(C) = 1 / √H[W|C]
```python
def _compute_precision_score(self, cache_hits: List[CacheHit]) -> Tuple[float, float]:
    """Entropy-based precision calculation"""
    total_weight = sum(hit.weight for hit in cache_hits)
    weighted_hbar_s = sum(hit.hbar_s * hit.weight for hit in cache_hits) / total_weight
    
    # Semantic entropy from similarity variance
    similarities = [hit.similarity for hit in cache_hits]
    entropy = 1.0 + np.var(similarities)  # Bounded approximation
    delta_mu = 1.0 / np.sqrt(entropy)
    
    return delta_mu, weighted_hbar_s
```

#### Step 5: Confidence Flags
```python
def _assess_confidence(self, delta_mu: float) -> ConfidenceFlag:
    """Tier-based confidence assessment"""
    if delta_mu < 1.0:
        return ConfidenceFlag.CRITICAL    # ❌ < 1.0
    elif delta_mu <= 1.2:
        return ConfidenceFlag.WARNING     # ⚠️ ~1.0
    else:
        return ConfidenceFlag.CONFIDENT   # ✅ > 1.2
```

## Protocol 2: Diagnostic Fusion (Flexibility Δσ)

### Implementation Steps

#### Step G: Decompose Prompt
```python
async def _decompose_prompt(self, prompt: str) -> List[SemanticUnit]:
    """Semantic decomposition into coherent units"""
    # Simple sentence splitting (enhance with spaCy/NLTK in production)
    sentences = [s.strip() for s in prompt.split('.') if s.strip()]
    
    units = []
    for i, sentence in enumerate(sentences):
        units.append(SemanticUnit(
            id=f"unit_{i}",
            text=sentence,
            semantic_role=self._classify_semantic_role(sentence),
            importance_weight=1.0 / len(sentences)
        ))
    return units
```

#### Step H: Generate Paraphrase Perturbations δ(C_i)
```python
async def _generate_perturbations(self, units: List[SemanticUnit]) -> Dict[str, List[Perturbation]]:
    """Cached perturbation generation with async batching"""
    perturbations = {}
    
    for unit in units:
        # Check cache first for reuse
        if unit.text in self.perturbation_library:
            cached = self.perturbation_library[unit.text]
            if len(cached) >= self.config.perturbation_samples:
                perturbations[unit.id] = cached[:self.config.perturbation_samples]
                continue
        
        # Generate fresh perturbations
        unit_perturbations = await self._create_unit_perturbations(unit)
        perturbations[unit.id] = unit_perturbations
        
        # Cache for reuse
        self.perturbation_library[unit.text] = unit_perturbations
    
    return perturbations
```

#### Step K: Sample Perturbations (Async/Batched)
```python
async def _sample_perturbations(self, perturbations: Dict[str, List[Perturbation]]) -> List[ComponentScore]:
    """Async batched sampling for efficiency"""
    sampling_tasks = []
    
    for unit_id, unit_perturbations in perturbations.items():
        task = self._sample_unit_perturbations(unit_id, unit_perturbations)
        sampling_tasks.append(task)
    
    # Process in batches to avoid overwhelming the system
    batch_size = self.config.drift_batch_size
    results = []
    
    for i in range(0, len(sampling_tasks), batch_size):
        batch = sampling_tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    return results
```

#### Measure Components: 🎲 Spread, ⏱ Drift, 🧠 Attribution
```python
async def _sample_unit_perturbations(self, unit_id: str, perturbations: List[Perturbation]) -> ComponentScore:
    """Component-wise measurement"""
    # 🎲 Spread (JSD between perturbations)
    spread_score = self._compute_jsd_spread(perturbations)
    
    # ⏱ Drift (JSD over time)
    drift_score = self._compute_temporal_drift(unit_id, perturbations)
    
    # 🧠 Attribution weight per component
    attribution_weight = self.attribution_weights.get(unit_id, 1.0)
    
    # Component Δσ
    delta_sigma_component = spread_score * (1.0 + drift_score)
    
    return ComponentScore(
        component_id=unit_id,
        spread_score=spread_score,
        drift_score=drift_score,
        attribution_weight=attribution_weight,
        delta_sigma_component=delta_sigma_component
    )
```

#### Final Aggregation: Δσ(C) = ∑ w_i · Δσ_component(i)
```python
def _aggregate_flexibility_score(self, component_scores: List[ComponentScore]) -> float:
    """Weighted aggregation of component scores"""
    total_weight = sum(cs.attribution_weight for cs in component_scores)
    if total_weight == 0:
        return 0.5  # Fallback
    
    weighted_sum = sum(
        cs.attribution_weight * cs.delta_sigma_component 
        for cs in component_scores
    )
    return weighted_sum / total_weight
```

## Performance Optimizations

### Sub-25ms Latency Target

1. **SIMD Vectorization**
   ```python
   # Use numpy vectorized operations
   similarities = np.dot(query_embedding, embeddings.T) / (
       np.linalg.norm(query_embedding) * np.linalg.norm(embeddings, axis=1)
   )
   ```

2. **Caching Strategy**
   ```python
   @lru_cache(maxsize=1000)
   def _encode_prompt_cached(self, prompt: str) -> np.ndarray:
       # Cache embeddings to avoid recomputation
   ```

3. **Async Parallel Processing**
   ```python
   # Parallel measurement of precision and flexibility
   precision_task = self.cache_firewall.measure_precision(prompt)
   flexibility_task = self.fusion_engine.measure_flexibility(prompt)
   
   precision_result, flexibility_result = await asyncio.gather(
       precision_task, flexibility_task
   )
   ```

4. **Batch Processing**
   ```python
   # Process perturbations in batches
   for i in range(0, len(sampling_tasks), batch_size):
       batch = sampling_tasks[i:i + batch_size]
       batch_results = await asyncio.gather(*batch)
   ```

## Quick Test & Demo

Now let's test the Tier-3 system:

```bash
cd semantic-uncertainty-runtime
python tier3_measurement.py
```

This will demonstrate:
- Sub-25ms latency measurement
- Precision (Δμ) via cache firewall  
- Flexibility (Δσ) via diagnostic fusion
- Confidence flag assessment
- Component-wise analysis
- Real-time performance metrics
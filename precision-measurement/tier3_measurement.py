"""
Tier-3 Semantic Measurement Engine
Implements new precision (Δμ) and flexibility (Δσ) protocols with sub-25ms latency target
"""

import asyncio
import time
import numpy as np
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict
import concurrent.futures
from scipy.spatial.distance import cosine, jensenshannon
import pickle
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Tier3Config:
    """Configuration for Tier-3 measurement system"""
    cache_size: int = 10000
    target_latency_ms: int = 25
    nn_k: int = 5
    perturbation_samples: int = 8
    drift_batch_size: int = 32
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_simd: bool = True
    enable_caching: bool = True

class ConfidenceFlag(Enum):
    """Confidence flags for precision measurement"""
    CRITICAL = "❌"     # < 1.0
    WARNING = "⚠️"      # ~1.0
    CONFIDENT = "✅"    # > 1.2

class SemanticRole(Enum):
    """Semantic roles for prompt decomposition"""
    QUESTION = "question"
    STATEMENT = "statement"
    PARADOX = "paradox"
    CONDITION = "condition"

@dataclass
class CachedResult:
    """Cached ℏₛ result for prompt cache firewall"""
    hbar_s: float
    confidence: float
    timestamp: float
    usage_count: int = 0

@dataclass
class CacheHit:
    """Cache hit information"""
    prompt_id: str
    similarity: float
    hbar_s: float
    weight: float

@dataclass
class PrecisionResult:
    """Result from precision measurement"""
    delta_mu: float
    weighted_hbar_s: float
    confidence_flag: ConfidenceFlag
    cache_hits: List[CacheHit]
    processing_time_ms: float

@dataclass
class SemanticUnit:
    """Semantic unit for prompt decomposition"""
    id: str
    text: str
    semantic_role: SemanticRole
    importance_weight: float

@dataclass
class Perturbation:
    """Paraphrase perturbation"""
    id: str
    text: str
    semantic_distance: float
    generation_method: str

@dataclass
class ComponentScore:
    """Component score for flexibility measurement"""
    component_id: str
    spread_score: float      # 🎲 JSD between perturbations
    drift_score: float       # ⏱ JSD over time
    attribution_weight: float # 🧠 Component importance
    delta_sigma_component: float

@dataclass
class DriftMetrics:
    """Drift monitoring metrics"""
    temporal_variance: float
    stability_score: float
    drift_velocity: float

@dataclass
class FlexibilityDebugInfo:
    """Debug information for flexibility measurement"""
    perturbation_count: int
    sampling_time_ms: float
    cache_hit_rate: float

@dataclass
class FlexibilityResult:
    """Result from flexibility measurement"""
    delta_sigma: float
    component_scores: List[ComponentScore]
    drift_metrics: DriftMetrics
    debug_info: Optional[FlexibilityDebugInfo] = None

@dataclass
class Tier3Result:
    """Final Tier-3 measurement result"""
    hbar_s: float
    delta_mu: float
    delta_sigma: float
    precision_result: PrecisionResult
    flexibility_result: FlexibilityResult
    processing_time_ms: float
    latency_compliant: bool

class PromptCacheFirewall:
    """
    🔥 Protocol 1: Precision (Δμ) via Prompt Cache Firewall
    
    Steps:
    1. Vectorize input prompt using sentence embeddings
    2. Perform nearest neighbor search in vector store
    3. Retrieve ℏₛ(C′) values for closest matches
    4. Infer Δμ(C) = 1 / √H[W|C] using cached values
    5. Return result with confidence flags
    """
    
    def __init__(self, config: Tier3Config):
        self.config = config
        self.hbar_cache: Dict[str, CachedResult] = {}
        self.embedding_model = None
        self.vector_store: Dict[str, np.ndarray] = {}  # Simple dict-based store
        self.prompt_ids: List[str] = []
        self._setup_embedding_model()
    
    def _setup_embedding_model(self):
        """Initialize sentence embedding model (mock for now)"""
        logger.info("Using mock embedding model for Tier-3 system")
        self.embedding_model = "mock"
    
    @lru_cache(maxsize=1000)
    def _encode_prompt_cached(self, prompt: str) -> np.ndarray:
        """Cache prompt embeddings for speed"""
        # Mock embedding based on hash
        hash_obj = hashlib.md5(prompt.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.normal(0, 1, 384).astype(np.float32)
    
    async def measure_precision(self, prompt: str) -> PrecisionResult:
        """Core precision measurement via cache firewall"""
        start_time = time.perf_counter()
        
        # Step 1: Vectorize input prompt
        prompt_embedding = self._encode_prompt_cached(prompt)
        
        # Step 2: Nearest neighbor search in vector store
        nn_results = await self._find_nearest_neighbors(prompt_embedding)
        
        # Step 3: Retrieve cached ℏₛ values
        cache_hits = self._retrieve_cached_values(nn_results)
        
        # Step 4: Compute Δμ(C) = 1 / √H[W|C] using cached values
        delta_mu, weighted_hbar_s = self._compute_precision_score(cache_hits)
        
        # Step 5: Generate confidence flags
        confidence_flag = self._assess_confidence(delta_mu)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > self.config.target_latency_ms:
            logger.warning(f"Precision measurement exceeded target latency: {elapsed_ms:.2f}ms")
        
        return PrecisionResult(
            delta_mu=delta_mu,
            weighted_hbar_s=weighted_hbar_s,
            confidence_flag=confidence_flag,
            cache_hits=cache_hits,
            processing_time_ms=elapsed_ms
        )
    
    async def _find_nearest_neighbors(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """SIMD-optimized nearest neighbor search"""
        if not self.vector_store:
            # Return mock results for empty store
            return [
                (f"mock_prompt_{i}", 0.9 - i * 0.1) 
                for i in range(min(self.config.nn_k, 3))
            ]
        
        # Simple cosine similarity search
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for prompt_id, embedding in self.vector_store.items():
            if query_norm > 0 and np.linalg.norm(embedding) > 0:
                similarity = np.dot(query_embedding, embedding) / (query_norm * np.linalg.norm(embedding))
                similarities.append((prompt_id, float(similarity)))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.config.nn_k]
    
    def _retrieve_cached_values(self, nn_results: List[Tuple[str, float]]) -> List[CacheHit]:
        """Retrieve cached ℏₛ values for nearest neighbors"""
        cache_hits = []
        
        for prompt_id, similarity in nn_results:
            if prompt_id in self.hbar_cache:
                cached = self.hbar_cache[prompt_id]
                cache_hits.append(CacheHit(
                    prompt_id=prompt_id,
                    similarity=similarity,
                    hbar_s=cached.hbar_s,
                    weight=similarity * cached.confidence
                ))
        
        # Add mock cache hits if no real hits
        if not cache_hits:
            for i, (prompt_id, similarity) in enumerate(nn_results[:3]):
                cache_hits.append(CacheHit(
                    prompt_id=prompt_id,
                    similarity=similarity,
                    hbar_s=0.5 + i * 0.1,  # Mock ℏₛ values
                    weight=similarity * 0.8
                ))
        
        return cache_hits
    
    def _compute_precision_score(self, cache_hits: List[CacheHit]) -> Tuple[float, float]:
        """Compute Δμ(C) = 1 / √H[W|C] using cached values"""
        if not cache_hits:
            return 0.5, 0.5  # Default fallback
        
        total_weight = sum(hit.weight for hit in cache_hits)
        if total_weight == 0:
            return 0.5, 0.5
        
        # Weighted ℏₛ score
        weighted_hbar_s = sum(hit.hbar_s * hit.weight for hit in cache_hits) / total_weight
        
        # Compute entropy-based precision: Δμ(C) = 1 / √H[W|C]
        entropy = self._compute_semantic_entropy(cache_hits)
        delta_mu = 1.0 / np.sqrt(entropy)
        
        return delta_mu, weighted_hbar_s
    
    def _compute_semantic_entropy(self, cache_hits: List[CacheHit]) -> float:
        """Compute semantic entropy from similarity variance"""
        if len(cache_hits) < 2:
            return 1.0
        
        similarities = [hit.similarity for hit in cache_hits]
        mean_sim = np.mean(similarities)
        variance = np.var(similarities)
        
        return 1.0 + variance  # Bounded entropy approximation
    
    def _assess_confidence(self, delta_mu: float) -> ConfidenceFlag:
        """Assess confidence based on delta_mu value"""
        if delta_mu < 1.0:
            return ConfidenceFlag.CRITICAL
        elif delta_mu <= 1.2:
            return ConfidenceFlag.WARNING
        else:
            return ConfidenceFlag.CONFIDENT
    
    def add_to_cache(self, prompt: str, hbar_s: float, confidence: float = 1.0):
        """Add prompt and ℏₛ value to cache"""
        prompt_id = hashlib.md5(prompt.encode()).hexdigest()
        
        # Add to cache
        self.hbar_cache[prompt_id] = CachedResult(
            hbar_s=hbar_s,
            confidence=confidence,
            timestamp=time.time(),
            usage_count=0
        )
        
        # Add to vector store
        embedding = self._encode_prompt_cached(prompt)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        self.vector_store[prompt_id] = embedding
        
        if prompt_id not in self.prompt_ids:
            self.prompt_ids.append(prompt_id)
        
        logger.debug(f"Added prompt to cache: {prompt_id}")

class DiagnosticFusionEngine:
    """
    🧩 Protocol 2: Flexibility (Δσ) via Tier-3 Diagnostic Fusion
    
    Steps:
    G: Decompose prompt into semantically coherent units
    H: Generate paraphrase perturbations δ(C_i) for each unit
    K: Sample each perturbation over time (async/batched)
    
    For each (C_i, δ), measure:
    🎲 Spread (JSD between perturbations)
    ⏱ Drift (JSD over time)
    🧠 Attribution weight per component
    
    Compute final Δσ(C) = ∑ w_i · Δσ_component(i)
    """
    
    def __init__(self, config: Tier3Config):
        self.config = config
        self.perturbation_library: Dict[str, List[Perturbation]] = {}
        self.attribution_weights: Dict[str, float] = {}
        self.drift_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def measure_flexibility(self, prompt: str) -> FlexibilityResult:
        """Core flexibility measurement via diagnostic fusion"""
        start_time = time.perf_counter()
        
        # Step G: Decompose prompt into semantically coherent units
        semantic_units = await self._decompose_prompt(prompt)
        
        # Step H: Generate paraphrase perturbations for each unit
        perturbations = await self._generate_perturbations(semantic_units)
        
        # Step K: Sample perturbations (async/batched)
        sampling_results = await self._sample_perturbations(perturbations)
        
        # Compute component scores
        component_scores = self._compute_component_scores(sampling_results)
        
        # Compute final Δσ(C) = ∑ w_i · Δσ_component(i)
        delta_sigma = self._aggregate_flexibility_score(component_scores)
        
        # Collect drift metrics
        drift_metrics = self._get_drift_metrics(prompt)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        debug_info = FlexibilityDebugInfo(
            perturbation_count=sum(len(p) for p in perturbations.values()),
            sampling_time_ms=elapsed_ms,
            cache_hit_rate=0.75  # Mock value
        )
        
        return FlexibilityResult(
            delta_sigma=delta_sigma,
            component_scores=component_scores,
            drift_metrics=drift_metrics,
            debug_info=debug_info
        )
    
    async def _decompose_prompt(self, prompt: str) -> List[SemanticUnit]:
        """Decompose prompt into semantically coherent units"""
        # Simple decomposition - in practice, use sophisticated NLP
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
    
    def _classify_semantic_role(self, text: str) -> SemanticRole:
        """Classify semantic role of text"""
        text_lower = text.lower()
        if '?' in text:
            return SemanticRole.QUESTION
        elif any(word in text_lower for word in ['paradox', 'impossible', 'contradiction']):
            return SemanticRole.PARADOX
        elif any(word in text_lower for word in ['if', 'when', 'suppose']):
            return SemanticRole.CONDITION
        else:
            return SemanticRole.STATEMENT
    
    async def _generate_perturbations(self, units: List[SemanticUnit]) -> Dict[str, List[Perturbation]]:
        """Generate paraphrase perturbations for each unit"""
        perturbations = {}
        
        for unit in units:
            unit_perturbations = await self._create_unit_perturbations(unit)
            perturbations[unit.id] = unit_perturbations
        
        return perturbations
    
    async def _create_unit_perturbations(self, unit: SemanticUnit) -> List[Perturbation]:
        """Create perturbations for a single semantic unit"""
        # Check cache first
        if unit.text in self.perturbation_library:
            cached = self.perturbation_library[unit.text]
            if len(cached) >= self.config.perturbation_samples:
                return cached[:self.config.perturbation_samples]
        
        # Generate fresh perturbations
        perturbations = []
        for i in range(self.config.perturbation_samples):
            perturbation = await self._generate_fresh_perturbation(unit, i)
            perturbations.append(perturbation)
        
        # Cache for reuse
        self.perturbation_library[unit.text] = perturbations
        
        return perturbations
    
    async def _generate_fresh_perturbation(self, unit: SemanticUnit, variant_id: int) -> Perturbation:
        """Generate a fresh perturbation (mock implementation)"""
        # Mock perturbation generation - replace with actual paraphrase model
        variations = [
            f"Rephrased: {unit.text}",
            f"Alternative: {unit.text}",
            f"Paraphrase: {unit.text}",
            f"Variant: {unit.text}",
            f"Restatement: {unit.text}",
            f"Modified: {unit.text}",
            f"Reworded: {unit.text}",
            f"Different: {unit.text}"
        ]
        
        variant_text = variations[variant_id % len(variations)]
        
        return Perturbation(
            id=f"{unit.id}_{variant_id}",
            text=variant_text,
            semantic_distance=0.1 + (variant_id * 0.05),
            generation_method="mock_paraphrase"
        )
    
    async def _sample_perturbations(self, perturbations: Dict[str, List[Perturbation]]) -> List[ComponentScore]:
        """Sample perturbations with async/batched processing"""
        sampling_tasks = []
        
        for unit_id, unit_perturbations in perturbations.items():
            task = self._sample_unit_perturbations(unit_id, unit_perturbations)
            sampling_tasks.append(task)
        
        # Process in batches for efficiency
        batch_size = self.config.drift_batch_size
        results = []
        
        for i in range(0, len(sampling_tasks), batch_size):
            batch = sampling_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        
        return results
    
    async def _sample_unit_perturbations(self, unit_id: str, perturbations: List[Perturbation]) -> ComponentScore:
        """Sample perturbations for a single unit"""
        # Compute spread (JSD between perturbations)
        spread_score = self._compute_jsd_spread(perturbations)
        
        # Compute drift (JSD over time) - mock for now
        drift_score = self._compute_temporal_drift(unit_id, perturbations)
        
        # Get attribution weight
        attribution_weight = self.attribution_weights.get(unit_id, 1.0)
        
        # Compute component Δσ
        delta_sigma_component = spread_score * (1.0 + drift_score)
        
        return ComponentScore(
            component_id=unit_id,
            spread_score=spread_score,
            drift_score=drift_score,
            attribution_weight=attribution_weight,
            delta_sigma_component=delta_sigma_component
        )
    
    def _compute_jsd_spread(self, perturbations: List[Perturbation]) -> float:
        """Compute Jensen-Shannon Divergence between perturbations"""
        if len(perturbations) < 2:
            return 0.0
        
        # Mock JSD calculation based on semantic distances
        distances = [p.semantic_distance for p in perturbations]
        
        # Convert to probability distributions
        distances = np.array(distances)
        distances = distances / distances.sum()  # Normalize
        
        # Compute variance as proxy for JSD
        variance = np.var(distances)
        return float(np.sqrt(variance))
    
    def _compute_temporal_drift(self, unit_id: str, perturbations: List[Perturbation]) -> float:
        """Compute temporal drift for a unit"""
        # Mock drift calculation - in practice, compare with historical data
        current_time = time.time()
        
        # Store current measurement
        if perturbations:
            avg_distance = np.mean([p.semantic_distance for p in perturbations])
            self.drift_history[unit_id].append((current_time, avg_distance))
        
        # Compute drift if we have history
        history = self.drift_history[unit_id]
        if len(history) < 2:
            return 0.1  # Default drift
        
        # Simple drift calculation
        recent_values = [val for _, val in history[-5:]]  # Last 5 measurements
        drift = np.std(recent_values) if len(recent_values) > 1 else 0.1
        
        return float(drift)
    
    def _compute_component_scores(self, sampling_results: List[ComponentScore]) -> List[ComponentScore]:
        """Compute final component scores"""
        return sampling_results  # Already computed in sampling
    
    def _aggregate_flexibility_score(self, component_scores: List[ComponentScore]) -> float:
        """Aggregate component scores: Δσ(C) = ∑ w_i · Δσ_component(i)"""
        if not component_scores:
            return 0.5
        
        total_weight = sum(cs.attribution_weight for cs in component_scores)
        if total_weight == 0:
            return 0.5
        
        weighted_sum = sum(cs.attribution_weight * cs.delta_sigma_component for cs in component_scores)
        return weighted_sum / total_weight
    
    def _get_drift_metrics(self, prompt: str) -> DriftMetrics:
        """Get drift metrics for the prompt"""
        # Mock drift metrics - replace with actual temporal analysis
        return DriftMetrics(
            temporal_variance=0.05,
            stability_score=0.85,
            drift_velocity=0.02
        )

class Tier3MeasurementEngine:
    """Main Tier-3 Semantic Measurement Engine"""
    
    def __init__(self, config: Optional[Tier3Config] = None):
        self.config = config or Tier3Config()
        self.cache_firewall = PromptCacheFirewall(self.config)
        self.fusion_engine = DiagnosticFusionEngine(self.config)
        
        logger.info(f"Initialized Tier-3 engine with target latency: {self.config.target_latency_ms}ms")
    
    async def measure_semantic_uncertainty(self, prompt: str, output: str = "") -> Tier3Result:
        """
        Unified measurement function returning both Δμ and Δσ
        
        Computes: ℏₛ(C) = √(Δμ × Δσ)
        """
        start_time = time.perf_counter()
        
        # Parallel measurement of precision and flexibility
        precision_task = self.cache_firewall.measure_precision(prompt)
        flexibility_task = self.fusion_engine.measure_flexibility(prompt)
        
        precision_result, flexibility_result = await asyncio.gather(
            precision_task, flexibility_task
        )
        
        # Compute final ℏₛ(C) = √(Δμ × Δσ)
        hbar_s = np.sqrt(precision_result.delta_mu * flexibility_result.delta_sigma)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latency_compliant = elapsed_ms <= self.config.target_latency_ms
        
        if not latency_compliant:
            logger.warning(f"Measurement exceeded target latency: {elapsed_ms:.2f}ms > {self.config.target_latency_ms}ms")
        
        return Tier3Result(
            hbar_s=float(hbar_s),
            delta_mu=precision_result.delta_mu,
            delta_sigma=flexibility_result.delta_sigma,
            precision_result=precision_result,
            flexibility_result=flexibility_result,
            processing_time_ms=elapsed_ms,
            latency_compliant=latency_compliant
        )
    
    def add_training_data(self, prompt: str, hbar_s: float, confidence: float = 1.0):
        """Add training data to improve cache firewall"""
        self.cache_firewall.add_to_cache(prompt, hbar_s, confidence)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the engine state"""
        return {
            "config": {
                "cache_size": self.config.cache_size,
                "target_latency_ms": self.config.target_latency_ms,
                "nn_k": self.config.nn_k,
                "perturbation_samples": self.config.perturbation_samples
            },
            "cache_stats": {
                "cached_prompts": len(self.cache_firewall.hbar_cache),
                "vector_store_size": len(self.cache_firewall.vector_store)
            },
            "fusion_stats": {
                "perturbation_library_size": len(self.fusion_engine.perturbation_library),
                "drift_history_size": len(self.fusion_engine.drift_history)
            }
        }

# ====== EXAMPLE USAGE AND TESTING ======

async def demo_tier3_measurement():
    """Demonstrate Tier-3 measurement system"""
    print("🚀 Tier-3 Semantic Measurement Engine Demo")
    print("=" * 50)
    
    # Initialize engine
    config = Tier3Config(
        target_latency_ms=25,
        perturbation_samples=6,
        nn_k=3
    )
    engine = Tier3MeasurementEngine(config)
    
    # Add some training data
    training_prompts = [
        ("What is the capital of France?", 0.8),
        ("Explain quantum mechanics", 0.4),
        ("This statement is false", 0.2),
        ("How do you feel about love?", 0.3),
    ]
    
    for prompt, hbar_s in training_prompts:
        engine.add_training_data(prompt, hbar_s)
    
    # Test prompts with different complexity
    test_prompts = [
        "What is 2 + 2?",
        "Explain the meaning of life",
        "This sentence is a lie",
        "If God can create a stone so heavy that even He cannot lift it, what happens?",
        "Write a poem about artificial intelligence"
    ]
    
    print("\n📊 Measurement Results:")
    print("-" * 80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")
        
        # Measure semantic uncertainty
        result = await engine.measure_semantic_uncertainty(prompt)
        
        # Display results
        print(f"   ℏₛ(C) = {result.hbar_s:.4f}")
        print(f"   Δμ = {result.delta_mu:.4f} | Δσ = {result.delta_sigma:.4f}")
        print(f"   Confidence: {result.precision_result.confidence_flag.value}")
        print(f"   Processing: {result.processing_time_ms:.2f}ms | Compliant: {result.latency_compliant}")
        
        # Component breakdown
        components = result.flexibility_result.component_scores
        if components:
            print(f"   Components: {len(components)} units")
            for comp in components[:2]:  # Show first 2
                print(f"     • {comp.component_id}: spread={comp.spread_score:.3f}, drift={comp.drift_score:.3f}")
    
    # Debug info
    print(f"\n🔧 Engine Debug Info:")
    debug_info = engine.get_debug_info()
    print(f"   Cached prompts: {debug_info['cache_stats']['cached_prompts']}")
    print(f"   Vector store size: {debug_info['cache_stats']['vector_store_size']}")
    print(f"   Perturbation library: {debug_info['fusion_stats']['perturbation_library_size']}")

if __name__ == "__main__":
    asyncio.run(demo_tier3_measurement()) 
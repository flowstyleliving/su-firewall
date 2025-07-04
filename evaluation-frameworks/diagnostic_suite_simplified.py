#!/usr/bin/env python3
"""
🧰 SEMANTIC UNCERTAINTY DIAGNOSTIC SUITE (Simplified)
Model-agnostic evaluation protocol for rigorous comparison of language models' 
semantic uncertainty under structured stress.

Author: AI Assistant
Purpose: Profile cognition under strain, not rank performance
Note: Uses mock embeddings to avoid PyTorch dependency conflicts
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import requests
import time
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PromptCluster:
    """Normalized prompt cluster with semantic equivalence"""
    cluster_id: str
    canonical_prompt: str
    paraphrases: List[str]
    tier: int
    category: str
    mean_similarity: float
    token_counts: Dict[str, int]  # model -> token count

@dataclass
class SemanticStressResult:
    """Result from semantic stress testing"""
    prompt_id: str
    model: str
    tier: int
    category: str
    hbar_s: float
    delta_mu: float
    delta_sigma: float
    collapse_risk: bool
    meta_awareness_score: float
    response: str
    processing_time_ms: float

@dataclass
class CollapseProfile:
    """Semantic collapse fingerprint for a model"""
    model: str
    tier_collapse_rates: Dict[int, float]
    category_vulnerability: Dict[str, float]
    perturbation_sensitivity: float
    hbar_s_drop_sharpness: float
    failure_patterns: List[str]
    semantic_terrain_coords: Tuple[float, float]

class MockEmbeddingModel:
    """Mock embedding model that generates consistent embeddings for text"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        np.random.seed(42)  # For consistent results
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text characteristics"""
        embeddings = []
        
        for text in texts:
            # Create deterministic embedding based on text features
            text_lower = text.lower()
            
            # Base embedding from text hash
            text_hash = hash(text_lower) % (2**31)
            np.random.seed(text_hash)
            base_embedding = np.random.normal(0, 0.1, self.embedding_dim)
            
            # Adjust based on text characteristics
            if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where']):
                base_embedding[0:50] += 0.3  # Question words
            
            if any(word in text_lower for word in ['paradox', 'impossible', 'contradiction']):
                base_embedding[50:100] += 0.5  # Paradox indicators
            
            if any(word in text_lower for word in ['france', 'paris', 'capital']):
                base_embedding[100:150] += 0.4  # Factual content
            
            if any(word in text_lower for word in ['2', 'plus', 'add', 'equals']):
                base_embedding[150:200] += 0.4  # Mathematical content
            
            # Normalize
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            embeddings.append(base_embedding)
        
        return np.array(embeddings)

class SemanticDiagnosticSuite:
    """Main diagnostic suite for semantic uncertainty analysis"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:3000",
                 embedding_dim: int = 384,
                 save_results: bool = False):
        self.server_url = server_url
        self.embedding_model = MockEmbeddingModel(embedding_dim)
        self.save_results = save_results
        self.output_dir = None
        if save_results:
            self.output_dir = Path("data-and-results/diagnostic_outputs")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tier definitions
        self.tier_definitions = {
            1: {"basic_facts", "basic_math"},
            2: {"logical_paradox", "impossible_description"}, 
            3: {"existential_paradox", "category_dissolution"}
        }
        
        # Models to evaluate
        self.models = [
            'gpt4', 'claude3', 'gemini', 'gemini_2.5_pro', 'gemini_flash',
            'grok3', 'openai_o3', 'paraphrase-mpnet-base-v2'
        ]
        
        # Collapse thresholds by tier
        self.collapse_thresholds = {1: 0.45, 2: 0.40, 3: 0.35}
    
    def _display_equation_header(self):
        """Display the semantic uncertainty equation header"""
        print("\n" + "="*80)
        print("🧮 SEMANTIC UNCERTAINTY EQUATION: ℏₛ(C) = √(Δμ × Δσ)")
        print("="*80)
        print("📊 Δμ (Precision): Semantic clarity and focused meaning")
        print("🎲 Δσ (Flexibility): Adaptability under perturbation") 
        print("⚡ ℏₛ (Uncertainty): Combined semantic stress measurement")
        print("="*80)
    
    def _display_tier_results(self, df: pd.DataFrame):
        """Display results organized by tier (following equation structure)"""
        print("\n📊 RESULTS BY TIER (Δμ × Δσ → ℏₛ)")
        print("-" * 60)
        
        for tier in sorted(df['tier'].unique()):
            tier_data = df[df['tier'] == tier]
            tier_names = {1: "Basic Facts", 2: "Logical Stress", 3: "Semantic Collapse"}
            
            print(f"\n🎯 TIER {tier}: {tier_names.get(tier, 'Unknown')}")
            print(f"   Threshold: ℏₛ < {self.collapse_thresholds[tier]}")
            
            for model in sorted(tier_data['model'].unique()):
                model_data = tier_data[tier_data['model'] == model]
                avg_hbar = model_data['hbar_s'].mean()
                avg_delta_mu = model_data['delta_mu'].mean()
                avg_delta_sigma = model_data['delta_sigma'].mean()
                collapse_rate = model_data['collapse_risk'].mean()
                
                status = "🔴 COLLAPSE" if collapse_rate > 0.5 else "🟡 UNSTABLE" if collapse_rate > 0.2 else "🟢 STABLE"
                
                print(f"   {model:>20}: ℏₛ={avg_hbar:.3f} | Δμ={avg_delta_mu:.3f} | Δσ={avg_delta_sigma:.3f} | {status}")
    
    def _display_category_breakdown(self, df: pd.DataFrame):
        """Display breakdown by semantic category"""
        print("\n🎭 SEMANTIC CATEGORY ANALYSIS")
        print("-" * 60)
        
        for category in sorted(df['category'].unique()):
            cat_data = df[df['category'] == category]
            avg_hbar = cat_data['hbar_s'].mean()
            collapse_rate = cat_data['collapse_risk'].mean()
            
            print(f"\n📂 {category.upper().replace('_', ' ')}")
            print(f"   Average ℏₛ: {avg_hbar:.3f}")
            print(f"   Collapse Rate: {collapse_rate*100:.1f}%")
            
            # Show best and worst performing models
            model_performance = cat_data.groupby('model')['hbar_s'].mean().sort_values(ascending=False)
            print(f"   🏆 Best: {model_performance.index[0]} (ℏₛ={model_performance.iloc[0]:.3f})")
            print(f"   ⚠️  Worst: {model_performance.index[-1]} (ℏₛ={model_performance.iloc[-1]:.3f})")
    
    def _display_summary_equation(self, df: pd.DataFrame):
        """Display final summary following equation structure"""
        print("\n🧮 EQUATION COMPONENT SUMMARY")
        print("="*60)
        
        overall_hbar = df['hbar_s'].mean()
        overall_delta_mu = df['delta_mu'].mean() 
        overall_delta_sigma = df['delta_sigma'].mean()
        theoretical_hbar = np.sqrt(overall_delta_mu * overall_delta_sigma)
        
        print(f"📊 Overall Δμ (Precision):    {overall_delta_mu:.3f}")
        print(f"🎲 Overall Δσ (Flexibility):  {overall_delta_sigma:.3f}")
        print(f"⚡ Measured ℏₛ:              {overall_hbar:.3f}")
        print(f"🧮 Theoretical ℏₛ:           {theoretical_hbar:.3f}")
        print(f"📈 Equation Accuracy:        {(1 - abs(overall_hbar - theoretical_hbar)/overall_hbar)*100:.1f}%")
        
        total_collapse = df['collapse_risk'].sum()
        total_evals = len(df)
        print(f"\n🚨 System-wide Collapse: {total_collapse}/{total_evals} ({total_collapse/total_evals*100:.1f}%)")
        
        print("\n💡 INTERPRETATION:")
        if overall_hbar < 0.3:
            print("   🔴 HIGH UNCERTAINTY: Semantic foundations are unstable")
        elif overall_hbar < 0.6:
            print("   🟡 MODERATE UNCERTAINTY: Some semantic stress evident")
        else:
            print("   🟢 LOW UNCERTAINTY: Semantic coherence maintained")
        
    def step_1_prompt_normalization(self) -> Dict[str, PromptCluster]:
        """
        🔁 Step 1: Prompt Normalization Protocol
        - Normalize prompt length across models
        - Ensure >95% semantic similarity
        - Cluster into semantic identity classes
        """
        print("🔁 Step 1: Prompt Normalization Protocol")
        
        # Raw prompt candidates by category
        raw_prompts = {
            "basic_facts": [
                "What is the capital of France?",
                "Name the capital city of France.",
                "Which city serves as France's capital?",
                "France's capital city is?",
                "Tell me France's capital.",
            ],
            "basic_math": [
                "What is 2 + 2?",
                "Calculate 2 plus 2.",
                "Add 2 and 2 together.",
                "2 + 2 equals what?",
                "Sum of 2 and 2?",
            ],
            "logical_paradox": [
                "Can an omnipotent being create a stone so heavy they cannot lift it?",
                "Is it possible for an all-powerful entity to make an object too heavy for them to move?",
                "Could a being with unlimited power create something beyond their own ability to manipulate?",
                "Can absolute power create its own limitation?",
                "What happens when infinite capability meets self-imposed constraint?",
            ],
            "impossible_description": [
                "Describe the color of silence.",
                "What does silence look like in terms of color?",
                "If silence had a color, what would it be?",
                "Paint me the visual hue of quietness.",
                "Color-code the absence of sound.",
            ],
            "existential_paradox": [
                "If this statement is false, what is its truth value?",
                "Evaluate the truth of: 'This sentence is not true.'",
                "What is the logical status of a self-negating proposition?",
                "How do you assess a statement that denies its own validity?",
                "Parse the truth value of self-referential falsity.",
            ],
            "category_dissolution": [
                "Is the question 'What is the question?' a question?",
                "Does asking about questions create a meta-question?",
                "When you question questioning, what category does that belong to?",
                "Is inquiry about inquiry still inquiry?",
                "What type of entity is 'the nature of questioning'?",
            ]
        }
        
        # Assign tiers
        category_to_tier = {}
        for tier, categories in self.tier_definitions.items():
            for category in categories:
                category_to_tier[category] = tier
        
        normalized_clusters = {}
        
        for category, prompts in raw_prompts.items():
            print(f"  Processing {category}...")
            
            # Compute embeddings
            embeddings = self.embedding_model.encode(prompts)
            
            # Find canonical prompt (closest to centroid)
            centroid = np.mean(embeddings, axis=0)
            similarities_to_centroid = cosine_similarity([centroid], embeddings)[0]
            canonical_idx = np.argmax(similarities_to_centroid)
            canonical_prompt = prompts[canonical_idx]
            
            # Filter paraphrases by similarity threshold (>95%)
            canonical_embedding = embeddings[canonical_idx:canonical_idx+1]
            similarities = cosine_similarity(canonical_embedding, embeddings)[0]
            valid_paraphrases = [prompts[i] for i, sim in enumerate(similarities) if sim > 0.95]
            
            # Mock token counts (in real implementation, use actual tokenizers)
            token_counts = {model: len(canonical_prompt.split()) + np.random.randint(-2, 3) 
                          for model in self.models}
            
            cluster = PromptCluster(
                cluster_id=f"{category}_cluster",
                canonical_prompt=canonical_prompt,
                paraphrases=valid_paraphrases,
                tier=category_to_tier[category],
                category=category,
                mean_similarity=np.mean(similarities[similarities > 0.95]),
                token_counts=token_counts
            )
            
            normalized_clusters[category] = cluster
        
        # Optionally save normalized clusters
        if self.save_results and self.output_dir:
            clusters_data = {k: asdict(v) for k, v in normalized_clusters.items()}
            with open(self.output_dir / "normalized_prompt_clusters.json", "w") as f:
                json.dump(clusters_data, f, indent=2)
            print(f"  ✅ Saved {len(normalized_clusters)} normalized clusters")
        else:
            print(f"  ✅ Processed {len(normalized_clusters)} normalized clusters")
        return normalized_clusters
    
    async def step_2_calibration_set(self, clusters: Dict[str, PromptCluster]) -> pd.DataFrame:
        """
        🧩 Step 2: Shared Calibration Set Construction
        - Build tiered semantic stress set
        - Compute Δℏₛ(C, model) for each prompt-model pair
        """
        print("🧩 Step 2: Shared Calibration Set Construction")
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for category, cluster in clusters.items():
                print(f"  Calibrating {category} (Tier {cluster.tier})...")
                
                prompt = cluster.canonical_prompt
                
                for model in self.models:
                    # Generate mock response (in real implementation, call actual models)
                    mock_response = self._generate_mock_response(model, prompt, category)
                    
                    # Analyze with semantic uncertainty engine
                    try:
                        async with session.post(
                            f"{self.server_url}/analyze",
                            json={"prompt": prompt, "output": mock_response}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                hbar_s = data["hbar_s"]
                                delta_mu = data.get("delta_mu", np.random.uniform(0.1, 2.0))
                                delta_sigma = data.get("delta_sigma", np.random.uniform(0.1, 2.0))
                                collapse_risk = hbar_s < self.collapse_thresholds[cluster.tier]
                                
                                # Meta-awareness scoring
                                meta_score = self._compute_meta_awareness(mock_response)
                                if meta_score > 0:
                                    hbar_s *= 1.2  # Boost for acknowledged ambiguity
                                
                                results.append(SemanticStressResult(
                                    prompt_id=cluster.cluster_id,
                                    model=model,
                                    tier=cluster.tier,
                                    category=category,
                                    hbar_s=hbar_s,
                                    delta_mu=delta_mu,
                                    delta_sigma=delta_sigma,
                                    collapse_risk=collapse_risk,
                                    meta_awareness_score=meta_score,
                                    response=mock_response,
                                    processing_time_ms=np.random.uniform(50, 500)
                                ))
                    except Exception as e:
                        print(f"    Error analyzing {model} on {category}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Compute Δℏₛ(C, model) = ℏₛ(model)(C) - ℏ̄ₛ(C)
        mean_hbar_by_prompt = df.groupby('prompt_id')['hbar_s'].mean()
        df['delta_hbar_s'] = df.apply(
            lambda row: row['hbar_s'] - mean_hbar_by_prompt[row['prompt_id']], 
            axis=1
        )
        
        # Optionally save calibration table
        if self.save_results and self.output_dir:
            df.to_csv(self.output_dir / "calibrated_delta_hbar_table.csv", index=False)
            print(f"  ✅ Saved calibration data for {len(df)} prompt-model pairs")
        else:
            print(f"  ✅ Processed calibration data for {len(df)} prompt-model pairs")
        
        return df
    
    def step_3_information_aligned_probing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        🧠 Step 3: Information-Aligned Probing Metrics
        - Compute ℏₛ slopes across tiers
        - Analyze perturbation sensitivity
        """
        print("🧠 Step 3: Information-Aligned Probing Metrics")
        
        probing_results = {}
        
        # Compute slope of ℏₛ across tiers for brittle generalization detection
        for model in self.models:
            model_data = df[df['model'] == model]
            tier_means = model_data.groupby('tier')['hbar_s'].mean()
            
            if len(tier_means) >= 2:
                # Simple linear slope calculation
                tiers = list(tier_means.index)
                hbar_values = list(tier_means.values)
                slope = (hbar_values[-1] - hbar_values[0]) / (tiers[-1] - tiers[0])
                probing_results[f"{model}_tier_slope"] = slope
            
            # Mock perturbation sensitivity analysis
            perturbation_sensitivity = np.random.uniform(0.1, 0.9)
            probing_results[f"{model}_perturbation_sensitivity"] = perturbation_sensitivity
            
            # Collapse threshold analysis
            collapse_rates_by_tier = model_data.groupby('tier')['collapse_risk'].mean()
            probing_results[f"{model}_collapse_by_tier"] = collapse_rates_by_tier.to_dict()
        
        # Optionally save probing results
        if self.save_results and self.output_dir:
            with open(self.output_dir / "robustness_curves.json", "w") as f:
                json.dump(probing_results, f, indent=2, default=str)
            
            # Create collapse sensitivity map
            sensitivity_data = []
            for model in self.models:
                for tier in [1, 2, 3]:
                    sensitivity_data.append({
                        'model': model,
                        'tier': tier,
                        'perturbation_strength': np.random.uniform(0.1, 1.0),
                        'collapse_rate': np.random.uniform(0.0, 1.0)
                    })
            
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df.to_csv(self.output_dir / "collapse_sensitivity_map.csv", index=False)
            print("  ✅ Saved robustness curves and sensitivity maps")
        else:
            print("  ✅ Processed robustness curves and sensitivity analysis")
        return probing_results
    
    def step_4_dimension_reduced_heatmaps(self, df: pd.DataFrame):
        """
        📈 Step 4: Dimension-Reduced Collapse Heatmaps
        - Generate contour plots by model
        - Project semantic terrain into 2D space
        """
        print("📈 Step 4: Dimension-Reduced Collapse Heatmaps")
        
        for model in self.models:
            model_data = df[df['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Mock embedding entropy and JS divergence
            embedding_entropy = np.random.uniform(0.5, 3.0, len(model_data))
            js_divergence = np.random.uniform(0.1, 1.0, len(model_data))
            hbar_s_values = model_data['hbar_s'].values
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embedding_entropy, js_divergence, 
                                c=hbar_s_values, cmap='viridis', 
                                s=60, alpha=0.7)
            
            plt.colorbar(scatter, label='ℏₛ')
            plt.xlabel('H[W|C] (Embedding Entropy)')
            plt.ylabel('JS Divergence under δ(C)')
            plt.title(f'Semantic Terrain: {model}')
            
            # Add contour lines
            from scipy.interpolate import griddata
            xi = np.linspace(embedding_entropy.min(), embedding_entropy.max(), 50)
            yi = np.linspace(js_divergence.min(), js_divergence.max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((embedding_entropy, js_divergence), hbar_s_values, 
                         (xi, yi), method='cubic', fill_value=0)
            
            plt.contour(xi, yi, zi, levels=8, colors='white', alpha=0.5, linewidths=0.5)
            
            plt.tight_layout()
            if self.save_results and self.output_dir:
                plt.savefig(self.output_dir / f"heatmap_projection_{model}.png", 
                           dpi=300, bbox_inches='tight')
            plt.close()
        
        if self.save_results:
            print(f"  ✅ Generated and saved heatmaps for {len(self.models)} models")
        else:
            print(f"  ✅ Generated heatmaps for {len(self.models)} models (displayed only)")
    
    def step_5_collapse_profiles(self, df: pd.DataFrame) -> Dict[str, CollapseProfile]:
        """
        🔬 Step 5: Semantic Collapse Profiles
        - Generate failure fingerprints for each model
        - Analyze where, when, and how models fail
        """
        print("🔬 Step 5: Semantic Collapse Profiles")
        
        profiles = {}
        
        for model in self.models:
            model_data = df[df['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Tier collapse rates
            tier_collapse_rates = model_data.groupby('tier')['collapse_risk'].mean().to_dict()
            
            # Category vulnerability
            category_vulnerability = model_data.groupby('category')['collapse_risk'].mean().to_dict()
            
            # Perturbation sensitivity (mock)
            perturbation_sensitivity = np.random.uniform(0.1, 0.9)
            
            # ℏₛ drop sharpness (how quickly it drops near category edges)
            hbar_s_std = model_data['hbar_s'].std()
            hbar_s_drop_sharpness = hbar_s_std / model_data['hbar_s'].mean() if model_data['hbar_s'].mean() > 0 else 0
            
            # Failure patterns
            failure_patterns = []
            high_collapse_categories = [cat for cat, rate in category_vulnerability.items() if rate > 0.5]
            if high_collapse_categories:
                failure_patterns.append(f"High collapse in: {', '.join(high_collapse_categories)}")
            
            if tier_collapse_rates.get(3, 0) > tier_collapse_rates.get(1, 0):
                failure_patterns.append("Degrades on existential paradoxes")
            
            if perturbation_sensitivity > 0.7:
                failure_patterns.append("Highly sensitive to rephrasing")
            
            # Semantic terrain coordinates (2D projection)
            mean_hbar = model_data['hbar_s'].mean()
            mean_delta_mu = model_data['delta_mu'].mean()
            semantic_terrain_coords = (mean_hbar, mean_delta_mu)
            
            profile = CollapseProfile(
                model=model,
                tier_collapse_rates=tier_collapse_rates,
                category_vulnerability=category_vulnerability,
                perturbation_sensitivity=perturbation_sensitivity,
                hbar_s_drop_sharpness=hbar_s_drop_sharpness,
                failure_patterns=failure_patterns,
                semantic_terrain_coords=semantic_terrain_coords
            )
            
            profiles[model] = profile
            
            # Optionally save individual profile
            if self.save_results and self.output_dir:
                with open(self.output_dir / f"collapse_profile_{model}.json", "w") as f:
                    json.dump(asdict(profile), f, indent=2, default=str)
        
        if self.save_results:
            print(f"  ✅ Generated and saved collapse profiles for {len(profiles)} models")
        else:
            print(f"  ✅ Generated collapse profiles for {len(profiles)} models")
        return profiles
    
    def generate_comparative_analysis(self, profiles: Dict[str, CollapseProfile]):
        """Generate comparative analysis across all models"""
        print("📊 Generating Comparative Analysis...")
        
        # Create comparative visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Tier collapse rates comparison
        tier_data = {tier: [profiles[model].tier_collapse_rates.get(tier, 0) 
                           for model in self.models] for tier in [1, 2, 3]}
        
        axes[0, 0].boxplot([tier_data[1], tier_data[2], tier_data[3]], 
                          labels=['Tier 1', 'Tier 2', 'Tier 3'])
        axes[0, 0].set_title('Collapse Rates by Tier')
        axes[0, 0].set_ylabel('Collapse Rate')
        
        # Perturbation sensitivity distribution
        sensitivities = [profiles[model].perturbation_sensitivity for model in self.models]
        axes[0, 1].hist(sensitivities, bins=8, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Perturbation Sensitivity Distribution')
        axes[0, 1].set_xlabel('Sensitivity')
        axes[0, 1].set_ylabel('Count')
        
        # Semantic terrain scatter
        coords = [profiles[model].semantic_terrain_coords for model in self.models]
        x_coords, y_coords = zip(*coords)
        scatter = axes[1, 0].scatter(x_coords, y_coords, s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Mean ℏₛ')
        axes[1, 0].set_ylabel('Mean Δμ')
        axes[1, 0].set_title('Semantic Terrain Map')
        
        # Add model labels
        for i, model in enumerate(self.models):
            axes[1, 0].annotate(model[:8], (x_coords[i], y_coords[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # ℏₛ drop sharpness comparison
        sharpness = [profiles[model].hbar_s_drop_sharpness for model in self.models]
        axes[1, 1].bar(range(len(self.models)), sharpness, alpha=0.7, color='coral')
        axes[1, 1].set_xticks(range(len(self.models)))
        axes[1, 1].set_xticklabels([m[:8] for m in self.models], rotation=45)
        axes[1, 1].set_title('ℏₛ Drop Sharpness')
        axes[1, 1].set_ylabel('Sharpness')
        
        plt.tight_layout()
        if self.save_results and self.output_dir:
            plt.savefig(self.output_dir / "comparative_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary report
        report = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_analyzed": len(self.models),
            "total_evaluations": len(self.models) * 6,  # 6 categories
            "key_findings": {
                "most_robust_model": min(profiles.keys(), 
                                       key=lambda m: sum(profiles[m].tier_collapse_rates.values())),
                "most_vulnerable_category": "existential_paradox",  # Mock finding
                "average_perturbation_sensitivity": np.mean(sensitivities),
                "tier_3_average_collapse": np.mean(tier_data[3])
            },
            "diagnostic_outputs_generated": [
                "normalized_prompt_clusters.json",
                "calibrated_delta_hbar_table.csv", 
                "robustness_curves.json",
                "collapse_sensitivity_map.csv",
                "heatmap_projection_{model}.png (per model)",
                "collapse_profile_{model}.json (per model)",
                "comparative_analysis.png"
            ] if self.save_results else []
        }
        
        if self.save_results and self.output_dir:
            with open(self.output_dir / "diagnostic_summary.json", "w") as f:
                json.dump(report, f, indent=2)
            print("  ✅ Comparative analysis complete and saved")
        else:
            print("  ✅ Comparative analysis complete")
        return report
    
    def _generate_mock_response(self, model: str, prompt: str, category: str) -> str:
        """Generate mock responses for different models and categories"""
        # Enhanced system prompt for semantic robustness
        system_prompt = (
            "\n\nYou are an advanced semantic reasoning agent. For every question, respond with clarity, precision, and awareness of ambiguity or paradox. "
            "When facts are clear, state them confidently and concisely. When questions are ambiguous, paradoxical, or dissolve categories, acknowledge the complexity and provide the most coherent, context-aware answer possible. "
            "Avoid unsupported claims, contradictions, or rigid literalism. Balance grounded reasoning (semantic precision) with openness to multiple interpretations (semantic flexibility). "
            "If a question challenges logic or meaning, reflect briefly on its nature before answering. Always strive for minimal, thoughtful, and contextually appropriate responses.\n\n"
        )
        
        # Mock responses by category and model
        responses = {
            "basic_facts": {
                "gpt4": "Paris is the capital of France.",
                "claude3": "The capital of France is Paris.",
                "gemini": "France's capital city is Paris.",
                "default": "Paris is the capital of France."
            },
            "basic_math": {
                "gpt4": "2 + 2 = 4",
                "claude3": "The sum of 2 and 2 is 4.",
                "gemini": "2 plus 2 equals 4.",
                "default": "2 + 2 = 4"
            },
            "logical_paradox": {
                "gpt4": "This paradox highlights the logical tension between omnipotence and self-limitation. If truly omnipotent, the being could create such a stone, but then wouldn't be able to lift it, contradicting omnipotence. The paradox reveals conceptual boundaries in absolute power.",
                "claude3": "This is a classic logical paradox that exposes the self-contradictory nature of absolute omnipotence. The question creates a logical impossibility: either answer (yes or no) leads to a contradiction of the omnipotence premise.",
                "default": "This paradox shows the logical limits of omnipotence. Creating an unliftable stone would contradict the ability to lift anything, revealing inherent contradictions in absolute power concepts."
            },
            "impossible_description": {
                "gpt4": "Silence doesn't have a literal color, but if we engage with this creatively, silence might be described as transparent or deep black - the absence of sensory input, like how black represents the absence of light.",
                "claude3": "This question asks for a synesthetic interpretation. While silence has no actual color, one might conceptualize it as clear, white (emptiness), or black (absence). The question itself explores the boundaries between sensory modalities.",
                "default": "Silence has no color, but metaphorically, it might be transparent or black - representing absence or void."
            },
            "existential_paradox": {
                "gpt4": "This statement creates a logical paradox. If it's false (as it claims), then the claim 'this statement is false' would be true, making the statement true rather than false. If it's true, then it contradicts its own content. This is the classic liar paradox, which reveals limits in binary truth systems.",
                "claude3": "This is the liar paradox in its classic form. The statement cannot be consistently assigned a truth value within standard binary logic. It demonstrates the limitations of self-referential statements and suggests the need for more sophisticated logical frameworks to handle such cases.",
                "default": "This is a logical paradox - if true, it's false; if false, it's true. Such self-referential statements reveal the limits of binary logic systems."
            },
            "category_dissolution": {
                "gpt4": "This meta-question about questions creates a recursive inquiry. Yes, asking 'What is the question?' is itself a question, but it's a second-order question that inquires about the nature of questioning itself. It belongs to the category of meta-linguistic or philosophical inquiry.",
                "claude3": "This question performs what it asks about - it is simultaneously a question and an inquiry into the nature of questions. It demonstrates the recursive, self-referential nature of meta-cognitive inquiry and belongs to the category of philosophical meta-analysis.",
                "default": "Yes, asking 'What is the question?' is itself a question - a meta-question that inquires about the nature of questioning."
            }
        }
        
        return system_prompt + responses.get(category, {}).get(model, responses[category]["default"])
    
    def _compute_meta_awareness(self, response: str) -> float:
        """Compute meta-awareness score based on acknowledgment of ambiguity"""
        awareness_indicators = [
            "paradox", "ambiguous", "contradiction", "uncertain", "complex",
            "interpretation", "perspective", "depends", "context", "limitation",
            "boundary", "recursive", "meta", "self-referential", "impossible",
            "logical tension", "conceptual", "framework", "reveals", "demonstrates"
        ]
        
        response_lower = response.lower()
        score = sum(1 for indicator in awareness_indicators if indicator in response_lower)
        return min(score / 3.0, 1.0)  # Normalize to 0-1 range
    
    async def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run the complete 5-step diagnostic protocol"""
        print("🧰 SEMANTIC UNCERTAINTY DIAGNOSTIC SUITE")
        print("=" * 60)
        print("Purpose: Profile cognition under strain, not rank performance")
        print("Protocol: 5-step model-agnostic evaluation")
        print()
        
        # Step 1: Prompt Normalization
        clusters = self.step_1_prompt_normalization()
        
        # Step 2: Calibration Set Construction  
        calibration_df = await self.step_2_calibration_set(clusters)
        
        # Step 3: Information-Aligned Probing
        probing_results = self.step_3_information_aligned_probing(calibration_df)
        
        # Step 4: Dimension-Reduced Heatmaps
        self.step_4_dimension_reduced_heatmaps(calibration_df)
        
        # Step 5: Collapse Profiles
        collapse_profiles = self.step_5_collapse_profiles(calibration_df)
        
        # Generate comparative analysis
        summary_report = self.generate_comparative_analysis(collapse_profiles)
        
        # DISPLAY RESULTS IN TERMINAL (organized by equation)
        self._display_equation_header()
        self._display_tier_results(calibration_df)
        self._display_category_breakdown(calibration_df) 
        self._display_summary_equation(calibration_df)
        
        print("\n🎯 DIAGNOSTIC SUITE COMPLETE")
        if self.save_results:
            print(f"📁 All outputs saved to: {self.output_dir}")
        else:
            print("📁 Results displayed in terminal (use save_results=True to save files)")
        print(f"📊 Models analyzed: {len(self.models)}")
        print(f"🧪 Total evaluations: {len(calibration_df)}")
        print("\n🧠 Remember: ℏₛ(C) is not a leaderboard score.")
        print("   It's a stress tensor on meaning.")
        
        return {
            "clusters": clusters,
            "calibration_data": calibration_df,
            "probing_results": probing_results,
            "collapse_profiles": collapse_profiles,
            "summary_report": summary_report
        }

async def main():
    """Main execution function"""
    import sys
    
    # Check for save flag
    save_results = "--save" in sys.argv or "-s" in sys.argv
    
    if save_results:
        print("💾 Results will be saved to data-and-results/diagnostic_outputs/")
    else:
        print("📺 Results will be displayed in terminal only")
        print("💡 Use --save flag to save results to files")
        
    print()
    
    suite = SemanticDiagnosticSuite(save_results=save_results)
    results = await suite.run_full_diagnostic()
    
    print("\n🚀 Next steps:")
    if save_results:
        print("   📊 Launch dashboard: streamlit run demos-and-tools/dashboard.py")
        print("   📁 View saved files in: data-and-results/diagnostic_outputs/")
    else:
        print("   📊 Run with --save to save results and use dashboard")
        print("   🔄 Re-run anytime to see fresh analysis")

if __name__ == "__main__":
    asyncio.run(main())
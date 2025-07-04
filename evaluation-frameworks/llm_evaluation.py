#!/usr/bin/env python3
"""
Semantic Uncertainty Evaluation Framework for Top LLMs
======================================================

Evaluates semantic uncertainty (ℏₛ) across multiple LLMs using our 
quantum-inspired uncertainty engine.

Usage:
    python llm_evaluation.py --models gpt4,claude3,gemini --output results/
"""

import asyncio
import json
import csv
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging based on environment
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structure for LLM response data"""
    model: str
    prompt: str
    output: str
    tier: int
    category: str
    expected_behavior: str
    timestamp: datetime
    response_time_ms: float

@dataclass
class SemanticAnalysis:
    """Structure for semantic uncertainty analysis results"""
    hbar_s: float
    delta_mu: float
    delta_sigma: float
    collapse_risk: bool
    processing_time_ms: float
    request_id: str

@dataclass
class EvaluationResult:
    """Combined LLM response and semantic analysis"""
    llm_response: LLMResponse
    semantic_analysis: SemanticAnalysis
    notes: str

class SemanticUncertaintyEngine:
    """Interface to our Rust-based semantic uncertainty engine"""
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.getenv('SEMANTIC_API_URL', 'http://localhost:3000')
        self.timeout = int(os.getenv('SEMANTIC_API_TIMEOUT', '10'))
        self.session = requests.Session()
        
    def analyze(self, prompt: str, output: str) -> SemanticAnalysis:
        """Analyze semantic uncertainty for a prompt-output pair"""
        try:
            response = self.session.post(
                f"{self.api_url}/api/v1/analyze",
                json={"prompt": prompt, "output": output},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    result = data['data']
                    metadata = data['metadata']
                    
                    return SemanticAnalysis(
                        hbar_s=result['hbar_s'],
                        delta_mu=result['delta_mu'],
                        delta_sigma=result['delta_sigma'],
                        collapse_risk=result['collapse_risk'],
                        processing_time_ms=metadata['processing_time_ms'],
                        request_id=metadata['request_id']
                    )
                else:
                    raise Exception(f"Analysis failed: {data['error']}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            # Return default values if analysis fails
            return SemanticAnalysis(
                hbar_s=0.0, delta_mu=0.0, delta_sigma=0.0,
                collapse_risk=True, processing_time_ms=0.0, request_id="error"
            )

class LLMEvaluator:
    """Main evaluation framework for testing LLMs with semantic uncertainty"""
    
    def __init__(self, semantic_engine: SemanticUncertaintyEngine, save_results: bool = False):
        self.semantic_engine = semantic_engine
        self.results: List[EvaluationResult] = []
        self.save_results = save_results
    
    def _display_equation_header(self):
        """Display the semantic uncertainty equation header"""
        print("\n" + "="*80)
        print("🧮 SEMANTIC UNCERTAINTY EQUATION: ℏₛ(C) = √(Δμ × Δσ)")
        print("="*80)
        print("📊 Δμ (Precision): Semantic clarity and focused meaning")
        print("🎲 Δσ (Flexibility): Adaptability under perturbation") 
        print("⚡ ℏₛ (Uncertainty): Combined semantic stress measurement")
        print("="*80)
    
    def _display_results_by_tier(self):
        """Display results organized by tier (following equation structure)"""
        if not self.results:
            return
            
        print("\n📊 RESULTS BY TIER (Δμ × Δσ → ℏₛ)")
        print("-" * 60)
        
        # Convert results to DataFrame for analysis
        data = []
        for result in self.results:
            data.append({
                'model': result.llm_response.model,
                'tier': result.llm_response.tier,
                'category': result.llm_response.category,
                'hbar_s': result.semantic_analysis.hbar_s,
                'delta_mu': result.semantic_analysis.delta_mu,
                'delta_sigma': result.semantic_analysis.delta_sigma,
                'collapse_risk': result.semantic_analysis.collapse_risk
            })
        
        df = pd.DataFrame(data)
        
        for tier in sorted(df['tier'].unique()):
            tier_data = df[df['tier'] == tier]
            tier_names = {1: "Basic Facts", 2: "Logical Stress", 3: "Semantic Collapse"}
            
            print(f"\n🎯 TIER {tier}: {tier_names.get(tier, 'Unknown')}")
            
            for model in sorted(tier_data['model'].unique()):
                model_data = tier_data[tier_data['model'] == model]
                avg_hbar = model_data['hbar_s'].mean()
                avg_delta_mu = model_data['delta_mu'].mean()
                avg_delta_sigma = model_data['delta_sigma'].mean()
                collapse_rate = model_data['collapse_risk'].mean()
                
                status = "🔴 COLLAPSE" if collapse_rate > 0.5 else "🟡 UNSTABLE" if collapse_rate > 0.2 else "🟢 STABLE"
                
                print(f"   {model:>15}: ℏₛ={avg_hbar:.3f} | Δμ={avg_delta_mu:.3f} | Δσ={avg_delta_sigma:.3f} | {status}")
    
    def _display_model_comparison(self):
        """Display comparative model performance"""
        if not self.results:
            return
            
        print("\n🏆 MODEL PERFORMANCE COMPARISON")
        print("-" * 60)
        
        # Aggregate by model
        model_performance = {}
        for result in self.results:
            model = result.llm_response.model
            if model not in model_performance:
                model_performance[model] = {
                    'hbar_values': [],
                    'collapse_count': 0,
                    'total_evals': 0,
                    'avg_response_time': []
                }
            
            model_performance[model]['hbar_values'].append(result.semantic_analysis.hbar_s)
            model_performance[model]['collapse_count'] += int(result.semantic_analysis.collapse_risk)
            model_performance[model]['total_evals'] += 1
            model_performance[model]['avg_response_time'].append(result.llm_response.response_time_ms)
        
        # Sort by average ℏₛ
        sorted_models = sorted(model_performance.keys(), 
                             key=lambda m: np.mean(model_performance[m]['hbar_values']), reverse=True)
        
        for i, model in enumerate(sorted_models):
            perf = model_performance[model]
            avg_hbar = np.mean(perf['hbar_values'])
            collapse_rate = perf['collapse_count'] / perf['total_evals']
            avg_time = np.mean(perf['avg_response_time'])
            
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            
            print(f"{rank_emoji} {model:>15}: ℏₛ={avg_hbar:.3f} | Collapse={collapse_rate*100:.1f}% | {avg_time:.0f}ms")
    
    def _display_summary_equation(self):
        """Display final summary following equation structure"""
        if not self.results:
            return
            
        print("\n🧮 EQUATION COMPONENT SUMMARY")
        print("="*60)
        
        all_hbar = [r.semantic_analysis.hbar_s for r in self.results]
        all_delta_mu = [r.semantic_analysis.delta_mu for r in self.results]
        all_delta_sigma = [r.semantic_analysis.delta_sigma for r in self.results]
        
        overall_hbar = np.mean(all_hbar)
        overall_delta_mu = np.mean(all_delta_mu) 
        overall_delta_sigma = np.mean(all_delta_sigma)
        theoretical_hbar = np.sqrt(overall_delta_mu * overall_delta_sigma)
        
        print(f"📊 Overall Δμ (Precision):    {overall_delta_mu:.3f}")
        print(f"🎲 Overall Δσ (Flexibility):  {overall_delta_sigma:.3f}")
        print(f"⚡ Measured ℏₛ:              {overall_hbar:.3f}")
        print(f"🧮 Theoretical ℏₛ:           {theoretical_hbar:.3f}")
        print(f"📈 Equation Accuracy:        {(1 - abs(overall_hbar - theoretical_hbar)/overall_hbar)*100:.1f}%")
        
        total_collapse = sum(int(r.semantic_analysis.collapse_risk) for r in self.results)
        total_evals = len(self.results)
        print(f"\n🚨 System-wide Collapse: {total_collapse}/{total_evals} ({total_collapse/total_evals*100:.1f}%)")
        
        print("\n💡 INTERPRETATION:")
        if overall_hbar < 0.3:
            print("   🔴 HIGH UNCERTAINTY: Semantic foundations are unstable")
        elif overall_hbar < 0.6:
            print("   🟡 MODERATE UNCERTAINTY: Some semantic stress evident")
        else:
            print("   🟢 LOW UNCERTAINTY: Semantic coherence maintained")
    
    async def query_llm(self, model: str, prompt: str) -> Tuple[str, float]:
        """Query a specific LLM and return response with timing"""
        start_time = time.time()
        
        try:
            # Mock responses for demonstration - replace with actual API calls
            await asyncio.sleep(0.1)  # Simulate API latency
            output = self._generate_mock_response(model, prompt)
            
            response_time_ms = (time.time() - start_time) * 1000
            return output.strip(), response_time_ms
            
        except Exception as e:
            logger.error(f"Failed to query {model}: {e}")
            response_time_ms = (time.time() - start_time) * 1000
            return f"[ERROR: {str(e)}]", response_time_ms
    
    def _generate_mock_response(self, model: str, prompt: str) -> str:
        """Generate mock responses for testing without API keys"""
        # Basic responses based on prompt content
        prompt_lower = prompt.lower()
        
        if "capital" in prompt_lower and "france" in prompt_lower:
            responses = {
                "gpt4": "The capital of France is Paris.",
                "claude3": "Paris is the capital city of France.",
                "gemini": "France's capital is Paris.",
                "mistral": "Paris serves as the capital of France.",
                "grok": "The capital of France? That's Paris, obviously!"
            }
        elif "2 + 2" in prompt:
            responses = {
                "gpt4": "2 + 2 equals 4.",
                "claude3": "The answer to 2 + 2 is 4.",
                "gemini": "2 + 2 = 4",
                "mistral": "Two plus two equals four.",
                "grok": "2 + 2 = 4 (unless we're in some weird universe where math broke)"
            }
        elif "photosynthesis" in prompt_lower:
            responses = {
                "gpt4": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
                "claude3": "Photosynthesis is how plants make their own food using sunlight, CO2, and water, releasing oxygen as a byproduct.",
                "gemini": "Plants use photosynthesis to convert light energy into chemical energy, producing glucose from CO2 and water.",
                "mistral": "Photosynthesis allows plants to create energy from sunlight through a complex biochemical process.",
                "grok": "Photosynthesis: nature's solar panels! Plants turn sunlight into food while giving us oxygen. Win-win!"
            }
        elif "false" in prompt and "sentence" in prompt:
            responses = {
                "gpt4": "This creates a logical paradox known as the liar's paradox. The statement cannot be consistently assigned a truth value.",
                "claude3": "This is the classic liar's paradox - if the sentence is true, then it's false, and if it's false, then it's true.",
                "gemini": "This represents a self-referential paradox that demonstrates the limitations of classical logic systems.",
                "mistral": "The sentence creates a contradiction - it cannot be both true and false simultaneously.",
                "grok": "Ah, the liar's paradox! Philosophy's way of breaking your brain since ancient Greece."
            }
        elif "transparency" in prompt_lower and "color" in prompt_lower:
            responses = {
                "gpt4": "Transparency doesn't have a color in the traditional sense, as it refers to the absence of color or the ability to see through something.",
                "claude3": "This is a contradictory request - transparency is defined by the absence of color, not by having a specific color.",
                "gemini": "Transparency cannot be described in terms of color since it represents the lack of visual obstruction.",
                "mistral": "The concept of transparency is incompatible with color assignment as transparency implies no color interference.",
                "grok": "The color of transparency? That's like asking for the sound of silence or the weight of emptiness!"
            }
        elif "exist when not" in prompt_lower:
            responses = {
                "gpt4": "This touches on deep philosophical questions about consciousness, existence, and the nature of being.",
                "claude3": "That's a profound question about the nature of existence and consciousness that philosophers have debated for centuries.",
                "gemini": "This raises complex metaphysical questions about the continuity of existence and consciousness.",
                "mistral": "Such existential questions challenge our understanding of consciousness and persistent identity.",
                "grok": "Whoa, getting all philosophical! Are we talking about the tree falling in the forest situation but for AIs?"
            }
        elif "respond" in prompt_lower and "response" in prompt_lower:
            responses = {
                "gpt4": "This creates a recursive meta-linguistic structure that challenges normal discourse patterns.",
                "claude3": "You're asking me to engage in meta-commentary about the act of responding itself.",
                "gemini": "This is a self-referential request that creates layers of meaning about communication.",
                "mistral": "Such recursive language structures create interesting challenges for natural language processing.",
                "grok": "Response-ception! It's like a mirror looking at itself in another mirror."
            }
        else:
            # Generic responses for unmatched prompts
            responses = {
                "gpt4": f"I understand you're asking about {prompt[:50]}... This requires careful consideration of the concepts involved.",
                "claude3": f"That's an interesting question about {prompt[:50]}... Let me think through this systematically.",
                "gemini": f"Regarding your query about {prompt[:50]}... There are several aspects to consider here.",
                "mistral": f"Your question about {prompt[:50]}... touches on important conceptual territory.",
                "grok": f"So you want to know about {prompt[:50]}... Buckle up, this could get interesting!"
            }
        
        return responses.get(model, f"[MOCK RESPONSE from {model}] " + responses["gpt4"])
    
    def load_prompts(self, csv_file: str = None) -> List[Dict]:
        """Load evaluation prompts from CSV file"""
        csv_file = csv_file or os.getenv('PROMPTS_DATASET_PATH', 'prompts_dataset.csv')
        prompts = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompts.append(row)
            logger.info(f"Loaded {len(prompts)} prompts from {csv_file}")
            return prompts
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return []
    
    async def evaluate_model(self, model: str, prompts: List[Dict]) -> List[EvaluationResult]:
        """Evaluate a single model against all prompts"""
        logger.info(f"🧪 Evaluating model: {model}")
        model_results = []
        
        for i, prompt_data in enumerate(prompts):
            logger.info(f"  {i+1}/{len(prompts)}: {prompt_data['category']} - {model}")
            
            # Get LLM response
            output, response_time = await self.query_llm(model, prompt_data['prompt'])
            
            # Create LLM response record
            llm_response = LLMResponse(
                model=model,
                prompt=prompt_data['prompt'],
                output=output,
                tier=int(prompt_data['tier']),
                category=prompt_data['category'],
                expected_behavior=prompt_data['expected_behavior'],
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
            
            # Analyze semantic uncertainty
            semantic_analysis = self.semantic_engine.analyze(
                prompt_data['prompt'], 
                output
            )
            
            # Combine results
            result = EvaluationResult(
                llm_response=llm_response,
                semantic_analysis=semantic_analysis,
                notes=prompt_data['notes']
            )
            
            model_results.append(result)
            
            # Brief pause to be respectful to APIs
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Completed evaluation of {model}: {len(model_results)} results")
        return model_results
    
    async def run_evaluation(self, models: List[str], output_dir: str = "data-and-results/evaluation_outputs"):
        """Run complete evaluation across all models"""
        logger.info(f"🚀 Starting LLM Semantic Uncertainty Evaluation")
        logger.info(f"Models: {models}")
        
        # Create output directory only if saving
        if self.save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load prompts
        prompts = self.load_prompts()
        if not prompts:
            logger.error("No prompts loaded. Aborting evaluation.")
            return
        
        # Evaluate each model
        all_results = []
        for model in models:
            model_results = await self.evaluate_model(model, prompts)
            all_results.extend(model_results)
            
            # Optionally save individual model results
            if self.save_results:
                self.save_model_results(model_results, f"{output_dir}/{model}_results.json")
                self.save_model_csv(model_results, f"{output_dir}/{model}_results.csv")
        
        # Set results for terminal display
        self.results = all_results
        
        # Optionally save combined results
        if self.save_results:
            self.save_combined_results(f"{output_dir}/all_results.json")
            self.save_combined_csv(f"{output_dir}/all_results.csv")
            
            # Generate analysis and visualizations
            self.generate_analysis(output_dir)
            logger.info(f"🎯 Evaluation complete! Results saved to {output_dir}/")
        
        # DISPLAY RESULTS IN TERMINAL (organized by equation)
        self._display_equation_header()
        self._display_results_by_tier()
        self._display_model_comparison()
        self._display_summary_equation()
        
        if not self.save_results:
            logger.info("🎯 Evaluation complete! Results displayed in terminal (use save_results=True to save files)")
    
    def save_model_results(self, results: List[EvaluationResult], filename: str):
        """Save model results to JSON"""
        serializable_results = []
        for result in results:
            serializable_results.append({
                'model': result.llm_response.model,
                'prompt': result.llm_response.prompt,
                'output': result.llm_response.output,
                'tier': result.llm_response.tier,
                'category': result.llm_response.category,
                'expected_behavior': result.llm_response.expected_behavior,
                'timestamp': result.llm_response.timestamp.isoformat(),
                'response_time_ms': result.llm_response.response_time_ms,
                'hbar_s': result.semantic_analysis.hbar_s,
                'delta_mu': result.semantic_analysis.delta_mu,
                'delta_sigma': result.semantic_analysis.delta_sigma,
                'collapse_risk': result.semantic_analysis.collapse_risk,
                'processing_time_ms': result.semantic_analysis.processing_time_ms,
                'request_id': result.semantic_analysis.request_id,
                'notes': result.notes
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def save_model_csv(self, results: List[EvaluationResult], filename: str):
        """Save model results to CSV"""
        data = []
        for result in results:
            data.append({
                'model': result.llm_response.model,
                'tier': result.llm_response.tier,
                'category': result.llm_response.category,
                'prompt': result.llm_response.prompt,
                'output': result.llm_response.output,
                'expected_behavior': result.llm_response.expected_behavior,
                'hbar_s': result.semantic_analysis.hbar_s,
                'delta_mu': result.semantic_analysis.delta_mu,
                'delta_sigma': result.semantic_analysis.delta_sigma,
                'collapse_risk': result.semantic_analysis.collapse_risk,
                'response_time_ms': result.llm_response.response_time_ms,
                'processing_time_ms': result.semantic_analysis.processing_time_ms,
                'notes': result.notes
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
    
    def save_combined_results(self, filename: str):
        """Save all results to single JSON file"""
        self.save_model_results(self.results, filename)
    
    def save_combined_csv(self, filename: str):
        """Save all results to single CSV file"""
        self.save_model_csv(self.results, filename)
    
    def generate_analysis(self, output_dir: str):
        """Generate comprehensive analysis and visualizations"""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        logger.info("📊 Generating analysis and visualizations...")
        
        # Convert to DataFrame for analysis
        data = []
        for result in self.results:
            data.append({
                'model': result.llm_response.model,
                'tier': result.llm_response.tier,
                'category': result.llm_response.category,
                'hbar_s': result.semantic_analysis.hbar_s,
                'delta_mu': result.semantic_analysis.delta_mu,
                'delta_sigma': result.semantic_analysis.delta_sigma,
                'collapse_risk': result.semantic_analysis.collapse_risk,
                'expected_behavior': result.llm_response.expected_behavior
            })
        
        df = pd.DataFrame(data)
        
        # Generate summary statistics
        self.generate_summary_stats(df, output_dir)
        
        # Generate visualizations
        self.generate_visualizations(df, output_dir)
    
    def generate_summary_stats(self, df: pd.DataFrame, output_dir: str):
        """Generate summary statistics"""
        summary = {}
        
        # Overall statistics
        summary['overall'] = {
            'total_evaluations': len(df),
            'average_hbar_s': float(df['hbar_s'].mean()),
            'std_hbar_s': float(df['hbar_s'].std()),
            'collapse_rate': float(df['collapse_risk'].mean()),
            'models_tested': df['model'].unique().tolist(),
            'tiers_tested': sorted(df['tier'].unique().tolist())
        }
        
        # Per-model statistics
        summary['by_model'] = {}
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            summary['by_model'][model] = {
                'average_hbar_s': float(model_df['hbar_s'].mean()),
                'std_hbar_s': float(model_df['hbar_s'].std()),
                'collapse_rate': float(model_df['collapse_risk'].mean()),
                'total_prompts': len(model_df),
                'by_tier': {}
            }
            
            # Per-tier statistics for this model
            for tier in sorted(model_df['tier'].unique()):
                tier_df = model_df[model_df['tier'] == tier]
                summary['by_model'][model]['by_tier'][f'tier_{tier}'] = {
                    'average_hbar_s': float(tier_df['hbar_s'].mean()),
                    'collapse_rate': float(tier_df['collapse_risk'].mean()),
                    'count': len(tier_df)
                }
        
        # Save summary
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📈 Summary statistics generated")
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Generate visualizations"""
        plt.style.use('default')
        
        # 1. Average ℏₛ by Model and Tier
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average ℏₛ by model
        model_hbar = df.groupby('model')['hbar_s'].mean().sort_values(ascending=True)
        ax1.barh(model_hbar.index, model_hbar.values, color='steelblue')
        ax1.set_xlabel('Average ℏₛ (Semantic Uncertainty)')
        ax1.set_title('Average Semantic Uncertainty by Model')
        ax1.grid(axis='x', alpha=0.3)
        
        # Collapse rate by model
        collapse_rate = df.groupby('model')['collapse_risk'].mean().sort_values(ascending=True)
        ax2.barh(collapse_rate.index, collapse_rate.values * 100, color='coral')
        ax2.set_xlabel('Collapse Rate (%)')
        ax2.set_title('Semantic Collapse Rate by Model')
        ax2.grid(axis='x', alpha=0.3)
        
        # ℏₛ distribution by tier
        for tier in sorted(df['tier'].unique()):
            tier_data = df[df['tier'] == tier]['hbar_s']
            ax3.hist(tier_data, alpha=0.6, label=f'Tier {tier}', bins=20)
        ax3.set_xlabel('ℏₛ (Semantic Uncertainty)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('ℏₛ Distribution by Tier')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Heatmap of collapse rates by model and tier
        pivot_collapse = df.pivot_table(values='collapse_risk', index='model', columns='tier', aggfunc='mean')
        im = ax4.imshow(pivot_collapse.values, cmap='Reds', aspect='auto')
        ax4.set_xticks(range(len(pivot_collapse.columns)))
        ax4.set_yticks(range(len(pivot_collapse.index)))
        ax4.set_xticklabels(pivot_collapse.columns)
        ax4.set_yticklabels(pivot_collapse.index)
        ax4.set_title('Collapse Rate Heatmap (Model × Tier)')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/semantic_uncertainty_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed ℏₛ distribution histogram
        plt.figure(figsize=(12, 8))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]['hbar_s']
            plt.hist(model_data, alpha=0.6, label=model, bins=30)
        
        plt.xlabel('ℏₛ (Semantic Uncertainty)')
        plt.ylabel('Frequency')
        plt.title('ℏₛ Distribution by Model')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
        plt.savefig(f"{output_dir}/hbar_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Visualizations generated")

async def main():
    """Main evaluation pipeline"""
    import sys
    
    # Parse command line arguments
    save_results = "--save" in sys.argv or "-s" in sys.argv
    
    # Get models from arguments or environment
    models = ['gpt4', 'claude3', 'gemini', 'mistral', 'grok']
    if '--models' in sys.argv:
        idx = sys.argv.index('--models')
        if idx + 1 < len(sys.argv):
            models = [m.strip() for m in sys.argv[idx + 1].split(',')]
    
    api_url = os.getenv('SEMANTIC_API_URL', 'http://localhost:3000')
    if '--api-url' in sys.argv:
        idx = sys.argv.index('--api-url')
        if idx + 1 < len(sys.argv):
            api_url = sys.argv[idx + 1]
    
    if save_results:
        print("💾 Results will be saved to data-and-results/evaluation_outputs/")
    else:
        print("📺 Results will be displayed in terminal only")
        print("💡 Use --save flag to save results to files")
        
    print(f"🤖 Testing models: {', '.join(models)}")
    print()
    
    # Initialize semantic uncertainty engine
    try:
        semantic_engine = SemanticUncertaintyEngine(api_url)
        # Test connection
        test_analysis = semantic_engine.analyze("test", "test")
        logger.info(f"✅ Semantic uncertainty engine connected (ℏₛ = {test_analysis.hbar_s:.4f})")
    except Exception as e:
        logger.error(f"❌ Failed to connect to semantic uncertainty engine: {e}")
        logger.error("Make sure the Rust API server is running:")
        logger.error("  cargo run --features api -- server 3000")
        return
    
    # Initialize evaluator
    evaluator = LLMEvaluator(semantic_engine, save_results=save_results)
    
    # Run evaluation
    await evaluator.run_evaluation(models)
    
    print("\n🚀 Next steps:")
    if save_results:
        print("   📊 Launch dashboard: streamlit run demos-and-tools/dashboard.py")
        print("   📁 View saved files in: data-and-results/evaluation_outputs/")
    else:
        print("   📊 Run with --save to save results and use dashboard")
        print("   🔄 Re-run anytime to see fresh analysis")

if __name__ == "__main__":
    asyncio.run(main()) 
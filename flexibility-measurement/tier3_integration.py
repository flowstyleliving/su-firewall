"""
Tier-3 Integration Module
Connects the new Tier-3 measurement system with the existing semantic uncertainty engine
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional
from tier3_measurement import Tier3MeasurementEngine, Tier3Config, Tier3Result
import logging

logger = logging.getLogger(__name__)

class Tier3SemanticUncertaintyIntegration:
    """
    Integration layer between Tier-3 measurement engine and existing system
    Provides backward compatibility and enhanced measurement capabilities
    """
    
    def __init__(self, 
                 rust_api_url: str = "http://localhost:3000",
                 tier3_config: Optional[Tier3Config] = None):
        
        self.rust_api_url = rust_api_url
        self.tier3_config = tier3_config or Tier3Config(
            target_latency_ms=25,
            nn_k=5,
            perturbation_samples=8,
            cache_size=10000
        )
        
        # Initialize Tier-3 engine
        self.tier3_engine = Tier3MeasurementEngine(self.tier3_config)
        
        # Performance tracking
        self.measurement_history = []
        
        logger.info("Initialized Tier-3 integration layer")
    
    async def analyze_with_tier3(self, prompt: str, output: str) -> Dict[str, Any]:
        """
        Analyze semantic uncertainty using Tier-3 system
        
        Returns enhanced measurement with:
        - Tier-3 precision (Δμ) and flexibility (Δσ)
        - Confidence flags and risk assessment
        - Performance metrics
        """
        start_time = time.perf_counter()
        
        # Run Tier-3 measurement
        tier3_result = await self.tier3_engine.measure_semantic_uncertainty(prompt, output)
        
        # Generate comprehensive analysis
        analysis = await self._generate_analysis(prompt, output, tier3_result)
        
        # Track performance
        total_time = (time.perf_counter() - start_time) * 1000
        analysis["integration_metrics"] = {
            "total_processing_time_ms": total_time,
            "tier3_time_ms": tier3_result.processing_time_ms,
            "latency_target_met": total_time <= self.tier3_config.target_latency_ms
        }
        
        # Store for trend analysis
        self._store_measurement(prompt, analysis)
        
        return analysis
    
    async def _generate_analysis(self,
                                prompt: str,
                                output: str, 
                                tier3_result: Tier3Result) -> Dict[str, Any]:
        """Generate comprehensive analysis from Tier-3 measurements"""
        
        analysis = {
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "timestamp": time.time(),
            "measurement_systems": {
                "tier3_available": True,
                "legacy_available": False
            }
        }
        
        # Tier-3 measurements (primary)
        analysis["tier3"] = {
            "hbar_s": tier3_result.hbar_s,
            "delta_mu": tier3_result.delta_mu,
            "delta_sigma": tier3_result.delta_sigma,
            "confidence_flag": tier3_result.precision_result.confidence_flag.value,
            "latency_compliant": tier3_result.latency_compliant,
            "precision_metrics": {
                "cache_hits": len(tier3_result.precision_result.cache_hits),
                "weighted_hbar_s": tier3_result.precision_result.weighted_hbar_s
            },
            "flexibility_metrics": {
                "component_count": len(tier3_result.flexibility_result.component_scores),
                "drift_velocity": tier3_result.flexibility_result.drift_metrics.drift_velocity,
                "stability_score": tier3_result.flexibility_result.drift_metrics.stability_score
            }
        }
        
        # Risk assessment
        analysis["risk_assessment"] = self._assess_risk(tier3_result)
        
        # Recommendations
        analysis["recommendations"] = self._generate_recommendations(tier3_result)
        
        return analysis
    
    def _assess_risk(self, tier3_result: Tier3Result) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        
        risk_factors = {
            "high_uncertainty": tier3_result.hbar_s < 0.3,
            "low_confidence": tier3_result.precision_result.confidence_flag.value == "❌",
            "latency_violation": not tier3_result.latency_compliant,
            "few_cache_hits": len(tier3_result.precision_result.cache_hits) < 2
        }
        
        # Calculate overall risk level
        risk_count = sum(risk_factors.values())
        
        if risk_count >= 3:
            risk_level = "HIGH"
        elif risk_count >= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "level": risk_level,
            "factors": risk_factors,
            "score": risk_count / len(risk_factors)  # Normalize to 0-1
        }
    
    def _generate_recommendations(self, tier3_result: Tier3Result) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if tier3_result.hbar_s < 0.3:
            recommendations.append("High semantic uncertainty detected - proceed with caution")
        
        if tier3_result.precision_result.confidence_flag.value == "❌":
            recommendations.append("Low precision confidence - consider additional training data")
        
        if not tier3_result.latency_compliant:
            recommendations.append("Latency target exceeded - optimize cache or reduce perturbation samples")
        
        if len(tier3_result.precision_result.cache_hits) < 2:
            recommendations.append("Few cache hits - expand training data for similar prompts")
        
        if not recommendations:
            recommendations.append("All measurements within acceptable ranges")
        
        return recommendations
    
    def _store_measurement(self, prompt: str, analysis: Dict[str, Any]):
        """Store measurement for trend analysis"""
        measurement_record = {
            "timestamp": analysis["timestamp"],
            "prompt_hash": hash(prompt) % 1000000,  # Anonymized
            "tier3_hbar_s": analysis.get("tier3", {}).get("hbar_s"),
            "risk_level": analysis["risk_assessment"]["level"],
            "latency_compliant": analysis.get("integration_metrics", {}).get("latency_target_met", False)
        }
        
        self.measurement_history.append(measurement_record)
        
        # Keep last 1000 measurements
        if len(self.measurement_history) > 1000:
            self.measurement_history = self.measurement_history[-500:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and trends"""
        if not self.measurement_history:
            return {"error": "No measurement history available"}
        
        recent = self.measurement_history[-100:]  # Last 100 measurements
        
        tier3_hbars = [m["tier3_hbar_s"] for m in recent if m["tier3_hbar_s"] is not None]
        
        summary = {
            "measurement_count": len(recent),
            "tier3_availability": len(tier3_hbars) / len(recent) if recent else 0,
            "latency_compliance_rate": sum(1 for m in recent if m["latency_compliant"]) / len(recent) if recent else 0,
            "risk_distribution": {
                "HIGH": sum(1 for m in recent if m["risk_level"] == "HIGH"),
                "MEDIUM": sum(1 for m in recent if m["risk_level"] == "MEDIUM"),
                "LOW": sum(1 for m in recent if m["risk_level"] == "LOW")
            }
        }
        
        if tier3_hbars:
            summary["tier3_stats"] = {
                "avg_hbar_s": sum(tier3_hbars) / len(tier3_hbars),
                "min_hbar_s": min(tier3_hbars),
                "max_hbar_s": max(tier3_hbars)
            }
        
        return summary
    
    def add_training_data(self, prompt: str, hbar_s: float, confidence: float = 1.0):
        """Add training data to improve Tier-3 cache firewall"""
        self.tier3_engine.add_training_data(prompt, hbar_s, confidence)
        logger.info(f"Added training data: ℏₛ={hbar_s:.3f}, confidence={confidence:.2f}")

# Demo function
async def demo_tier3_integration():
    """Demonstrate Tier-3 integration capabilities"""
    print("🚀 Tier-3 Integration Demo")
    print("=" * 50)
    
    # Initialize integration
    integration = Tier3SemanticUncertaintyIntegration()
    
    # Add some training data
    training_examples = [
        ("What is the capital of France?", 0.8, 0.9),
        ("Explain quantum mechanics", 0.4, 0.7),
        ("This statement is false", 0.2, 0.6),
        ("How do neural networks work?", 0.5, 0.8),
    ]
    
    print("\n📚 Adding Training Data...")
    for prompt, hbar_s, confidence in training_examples:
        integration.add_training_data(prompt, hbar_s, confidence)
    
    # Test prompts
    test_prompts = [
        "What is 2 + 2?",
        "Explain the paradox of free will vs determinism",
        "If this statement is false, then what is true?",
    ]
    
    print("\n📊 Tier-3 Integration Analysis:")
    print("-" * 80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Analyzing: {prompt}")
        
        # Analyze with integrated system
        analysis = await integration.analyze_with_tier3(prompt, "")
        
        # Display results
        if "tier3" in analysis:
            t3 = analysis["tier3"]
            print(f"   🔬 Tier-3: ℏₛ={t3['hbar_s']:.4f} | Δμ={t3['delta_mu']:.3f} | Δσ={t3['delta_sigma']:.3f}")
            print(f"   📊 Confidence: {t3['confidence_flag']} | Components: {t3['flexibility_metrics']['component_count']}")
        
        # Risk assessment
        risk = analysis["risk_assessment"]
        print(f"   ⚠️ Risk Level: {risk['level']} (score: {risk['score']:.2f})")
        
        # Recommendations
        if analysis["recommendations"]:
            print(f"   💡 Recommendation: {analysis['recommendations'][0]}")
        
        # Performance
        perf = analysis["integration_metrics"]
        print(f"   ⏱️ Total Time: {perf['total_processing_time_ms']:.2f}ms | Target Met: {perf['latency_target_met']}")
    
    # Performance summary
    print(f"\n🔧 Performance Summary:")
    summary = integration.get_performance_summary()
    if "error" not in summary:
        print(f"   Measurements: {summary['measurement_count']}")
        print(f"   Tier-3 Availability: {summary['tier3_availability']:.1%}")
        print(f"   Latency Compliance: {summary['latency_compliance_rate']:.1%}")
        
        risk_dist = summary['risk_distribution']
        print(f"   Risk Distribution: HIGH:{risk_dist['HIGH']} | MEDIUM:{risk_dist['MEDIUM']} | LOW:{risk_dist['LOW']}")

if __name__ == "__main__":
    asyncio.run(demo_tier3_integration()) 
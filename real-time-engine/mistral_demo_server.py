#!/usr/bin/env python3
"""
🧠 Mistral 7B Live Uncertainty Demo Server
Demonstrates real-time semantic uncertainty quantification without PyTorch dependency
Shows the complete ℏₛ = √(Δμ × Δσ) calculation pipeline
"""

import asyncio
import json
import logging
import time
import math
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import traceback
import signal
import sys
import random

# Networking (no PyTorch dependency)
import websockets
from flask import Flask, request, jsonify
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TokenData:
    """Token data simulation (mirrors PyTorch implementation)"""
    text: str
    token_id: int
    probability: float
    logits: List[float]  # Simulated logits
    position: int
    attention_weights: Optional[List[float]] = None

@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics (exact same as PyTorch version)"""
    delta_mu: float  # Precision (Fisher Information approximation)
    delta_sigma: float  # Flexibility (Jensen-Shannon divergence)
    hbar_raw: float  # Raw semantic uncertainty √(Δμ × Δσ)
    hbar_calibrated: float  # Calibrated uncertainty
    risk_level: str  # Safe, Warning, HighRisk, Critical
    entropy: float  # Token entropy
    confidence: float  # Token confidence
    fisher_information: float  # Fisher Information estimate

class SimulatedUncertaintyEngine:
    """Uncertainty calculation engine with realistic simulations"""
    
    def __init__(self):
        self.uncertainty_threshold = 2.5
        self.confidence_threshold = 0.3
        self.epsilon = 1e-8
        self.vocab_size = 32000
        
    def calculate_entropy(self, logits: List[float]) -> float:
        """Calculate Shannon entropy from logits"""
        # Convert logits to probabilities (softmax)
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        # Shannon entropy
        entropy = -sum(p * math.log(p + self.epsilon) for p in probs if p > 0)
        return entropy
    
    def calculate_fisher_information(self, logits: List[float], target_token_id: int) -> float:
        """Approximate Fisher Information Matrix diagonal element"""
        # Convert to probabilities
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        target_prob = probs[target_token_id] if target_token_id < len(probs) else 0.01
        
        # Fisher Information approximation: 1/p for categorical distribution
        fisher_info = 1.0 / (target_prob + self.epsilon)
        return fisher_info
    
    def calculate_js_divergence(self, logits1: List[float], logits2: List[float]) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        def softmax(logits):
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            return [e / sum_exp for e in exp_logits]
        
        def kl_divergence(p, q):
            return sum(pi * math.log((pi + self.epsilon) / (qi + self.epsilon)) 
                      for pi, qi in zip(p, q) if pi > 0)
        
        p = softmax(logits1)
        q = softmax(logits2)
        
        # Ensure same length
        min_len = min(len(p), len(q))
        p = p[:min_len]
        q = q[:min_len]
        
        # Jensen-Shannon divergence
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        js_div = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
        return js_div
    
    def calculate_delta_mu(self, fisher_info: float, entropy: float) -> float:
        """Calculate precision (Δμ) using Fisher Information approximation"""
        # Enhanced precision calculation with Fisher Information
        precision = math.sqrt(fisher_info) * (1.0 - entropy / 10.0)
        return max(0.1, precision)
    
    def calculate_delta_sigma(self, logits: List[float], baseline_logits: Optional[List[float]] = None) -> float:
        """Calculate flexibility (Δσ) using Jensen-Shannon divergence"""
        if baseline_logits is None:
            # Use uniform distribution as baseline
            baseline_logits = [0.0] * len(logits)
            
        # Calculate JS divergence from baseline
        js_div = self.calculate_js_divergence(logits, baseline_logits)
        
        # Map JS divergence to flexibility measure
        flexibility = math.sqrt(js_div + self.epsilon)
        return flexibility
    
    def calculate_uncertainty_metrics(
        self, 
        token_data: TokenData,
        baseline_logits: Optional[List[float]] = None,
        previous_uncertainty: float = 0.0
    ) -> UncertaintyMetrics:
        """Calculate comprehensive uncertainty metrics"""
        
        # Basic metrics
        entropy = self.calculate_entropy(token_data.logits)
        fisher_info = self.calculate_fisher_information(token_data.logits, token_data.token_id)
        
        # Semantic uncertainty components
        delta_mu = self.calculate_delta_mu(fisher_info, entropy)
        delta_sigma = self.calculate_delta_sigma(token_data.logits, baseline_logits)
        
        # Core semantic uncertainty: ℏₛ = √(Δμ × Δσ)
        hbar_raw = math.sqrt(delta_mu * delta_sigma)
        
        # Apply calibration (position-based)
        calibration_factor = 1.0 + 0.1 * math.sin(token_data.position * 0.1)
        hbar_calibrated = hbar_raw * calibration_factor
        
        # Determine risk level
        if hbar_calibrated > self.uncertainty_threshold:
            risk_level = "Critical"
        elif hbar_calibrated > self.uncertainty_threshold * 0.7:
            risk_level = "HighRisk"
        elif hbar_calibrated > self.uncertainty_threshold * 0.4:
            risk_level = "Warning"
        else:
            risk_level = "Safe"
            
        return UncertaintyMetrics(
            delta_mu=delta_mu,
            delta_sigma=delta_sigma,
            hbar_raw=hbar_raw,
            hbar_calibrated=hbar_calibrated,
            risk_level=risk_level,
            entropy=entropy,
            confidence=token_data.probability,
            fisher_information=fisher_info
        )

class SimulatedMistralModel:
    """Realistic Mistral 7B simulation with proper uncertainty patterns"""
    
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1 (Simulated)"
        self.device = "cpu"
        self.uncertainty_engine = SimulatedUncertaintyEngine()
        self.vocab = self._create_realistic_vocab()
        self.baseline_logits = None
        
    def _create_realistic_vocab(self) -> Dict[int, str]:
        """Create a realistic vocabulary"""
        vocab = {}
        
        # Technical/scientific words (higher uncertainty)
        technical_words = [
            "quantum", "computing", "entanglement", "superposition", "algorithm",
            "neural", "network", "transformer", "attention", "embedding",
            "probability", "distribution", "variance", "entropy", "information",
            "optimization", "gradient", "backpropagation", "convolution", "activation"
        ]
        
        # Common words (lower uncertainty)
        common_words = [
            "the", "and", "is", "to", "of", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "but", "his", "by", "from", "they", "we", "say", "her", "she", "or"
        ]
        
        # Explanatory words (medium uncertainty)
        explanatory_words = [
            "explain", "understand", "because", "therefore", "however", "although",
            "essentially", "basically", "specifically", "generally", "particularly",
            "furthermore", "moreover", "consequently", "nevertheless", "meanwhile"
        ]
        
        # Build vocab
        idx = 0
        for word in technical_words + common_words + explanatory_words:
            vocab[idx] = word
            idx += 1
            
        # Fill remaining slots
        for i in range(idx, 32000):
            vocab[i] = f"token_{i}"
            
        return vocab
    
    def _get_word_uncertainty_profile(self, word: str) -> Dict[str, float]:
        """Get uncertainty profile for different word types"""
        technical_words = ["quantum", "computing", "neural", "transformer", "algorithm"]
        common_words = ["the", "and", "is", "to", "of", "a", "in"]
        explanatory_words = ["explain", "understand", "because", "therefore"]
        
        if any(tech in word.lower() for tech in technical_words):
            return {
                "base_uncertainty": 0.3,
                "uncertainty_variance": 0.4,
                "base_probability": 0.4,
                "prob_variance": 0.3
            }
        elif any(common in word.lower() for common in common_words):
            return {
                "base_uncertainty": 0.8,
                "uncertainty_variance": 0.2,
                "base_probability": 0.9,
                "prob_variance": 0.1
            }
        elif any(exp in word.lower() for exp in explanatory_words):
            return {
                "base_uncertainty": 0.6,
                "uncertainty_variance": 0.3,
                "base_probability": 0.7,
                "prob_variance": 0.2
            }
        else:
            return {
                "base_uncertainty": 0.5,
                "uncertainty_variance": 0.3,
                "base_probability": 0.6,
                "prob_variance": 0.25
            }
    
    def _generate_realistic_response(self, prompt: str) -> List[str]:
        """Generate realistic response based on prompt"""
        responses = {
            "quantum": [
                "Quantum", "computing", "is", "a", "revolutionary", "technology", "that",
                "leverages", "quantum", "mechanical", "phenomena", "like", "superposition",
                "and", "entanglement", "to", "process", "information", "in", "fundamentally",
                "different", "ways", "than", "classical", "computers"
            ],
            "machine learning": [
                "Machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence",
                "that", "enables", "computers", "to", "learn", "and", "improve", "from",
                "experience", "without", "being", "explicitly", "programmed", "for", "every", "task"
            ],
            "artificial intelligence": [
                "Artificial", "intelligence", "refers", "to", "computer", "systems", "that",
                "can", "perform", "tasks", "typically", "requiring", "human", "intelligence",
                "such", "as", "reasoning", "learning", "and", "problem", "solving"
            ]
        }
        
        # Select response based on prompt
        for key, response in responses.items():
            if key in prompt.lower():
                return response
                
        # Default response
        return [
            "This", "is", "a", "comprehensive", "explanation", "that", "demonstrates",
            "the", "semantic", "uncertainty", "quantification", "system", "working",
            "with", "realistic", "language", "patterns", "and", "uncertainty", "metrics"
        ]
    
    async def generate_with_uncertainty(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        websocket=None
    ) -> List[Tuple[TokenData, UncertaintyMetrics]]:
        """Generate text with realistic uncertainty analysis"""
        
        logger.info(f"🎯 Generating response for: {prompt[:50]}...")
        
        # Generate realistic response
        words = self._generate_realistic_response(prompt)[:max_tokens]
        
        results = []
        generated_text = ""
        
        for position, word in enumerate(words):
            # Get uncertainty profile for this word
            profile = self._get_word_uncertainty_profile(word)
            
            # Find token ID
            token_id = next((tid for tid, tword in self.vocab.items() if tword.lower() == word.lower()), 
                          random.randint(100, 1000))
            
            # Generate realistic probability with temperature effects
            base_prob = profile["base_probability"]
            temp_effect = 1.0 / (temperature + 0.1)  # Higher temp = more uncertainty
            prob_with_temp = base_prob * temp_effect
            prob_with_temp = max(0.1, min(0.95, prob_with_temp + random.uniform(-0.1, 0.1)))
            
            # Generate realistic logits (100 tokens for simplicity)
            logits = [random.uniform(-5.0, -1.0) for _ in range(100)]
            
            # Set target token logit to achieve desired probability
            if token_id < len(logits):
                target_logit = math.log(prob_with_temp / (1 - prob_with_temp + 1e-8))
                logits[token_id] = target_logit
            
            # Add temperature effects to logits
            logits = [l / temperature for l in logits]
            
            # Create token data
            token_data = TokenData(
                text=word,
                token_id=token_id,
                probability=prob_with_temp,
                logits=logits,
                position=position
            )
            
            generated_text += (" " if position > 0 else "") + word
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self.uncertainty_engine.calculate_uncertainty_metrics(
                token_data,
                self.baseline_logits
            )
            
            results.append((token_data, uncertainty_metrics))
            
            # Send real-time update via WebSocket
            if websocket:
                await self._send_token_update(websocket, token_data, uncertainty_metrics, generated_text)
            
            logger.info(f"Token {position}: '{word}' | P={prob_with_temp:.3f} | ℏₛ={uncertainty_metrics.hbar_calibrated:.3f} | Risk={uncertainty_metrics.risk_level}")
            
            # Realistic generation delay
            await asyncio.sleep(0.3)
        
        logger.info(f"✅ Generation completed: {len(results)} tokens")
        return results
    
    async def _send_token_update(
        self, 
        websocket, 
        token_data: TokenData, 
        uncertainty: UncertaintyMetrics, 
        generated_so_far: str
    ):
        """Send real-time token update via WebSocket"""
        
        # Generate alerts
        alerts = []
        if uncertainty.hbar_calibrated > self.uncertainty_engine.uncertainty_threshold:
            alerts.append({
                "alert_type": "HighUncertainty",
                "severity": "Warning",
                "message": f"High uncertainty detected: ℏₛ={uncertainty.hbar_calibrated:.3f}",
                "token_position": token_data.position,
                "uncertainty_value": uncertainty.hbar_calibrated
            })
            
        if uncertainty.confidence < self.uncertainty_engine.confidence_threshold:
            alerts.append({
                "alert_type": "LowConfidence",
                "severity": "Warning", 
                "message": f"Low confidence token: '{token_data.text}' (p={uncertainty.confidence:.3f})",
                "token_position": token_data.position,
                "uncertainty_value": uncertainty.hbar_calibrated
            })
        
        update = {
            "type": "generation_update",
            "data": {
                "token": {
                    "text": token_data.text,
                    "token_id": token_data.token_id,
                    "probability": token_data.probability,
                    "position": token_data.position
                },
                "audit": {
                    "session_id": "mistral_demo_session",
                    "current_uncertainty": uncertainty.hbar_calibrated,
                    "average_uncertainty": uncertainty.hbar_calibrated,
                    "risk_level": uncertainty.risk_level,
                    "tokens_processed": token_data.position + 1,
                    "alerts": alerts,
                    "metrics": {
                        "delta_mu": uncertainty.delta_mu,
                        "delta_sigma": uncertainty.delta_sigma,
                        "hbar_raw": uncertainty.hbar_raw,
                        "entropy": uncertainty.entropy,
                        "fisher_information": uncertainty.fisher_information
                    }
                },
                "generated_so_far": generated_so_far
            },
            "timestamp": time.time()
        }
        
        try:
            await websocket.send(json.dumps(update))
        except Exception as e:
            logger.warning(f"⚠️  Failed to send WebSocket update: {e}")

class MistralDemoWebSocketServer:
    """WebSocket server for real-time communication"""
    
    def __init__(self):
        self.model = SimulatedMistralModel()
        self.connections = set()
        
    async def setup(self):
        """Initialize the model"""
        logger.info("🔧 Demo server ready (no model loading required)")
        
    async def handle_client(self, websocket, path=None):
        """Handle WebSocket client connections"""
        logger.info(f"📡 New client connected: {websocket.remote_address}")
        self.connections.add(websocket)
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Connected to Mistral 7B Demo Server (Simulated)",
                "model_info": {
                    "model_name": self.model.model_name,
                    "device": self.model.device,
                    "status": "ready",
                    "note": "This is a realistic simulation of Mistral 7B behavior"
                },
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome))
            
            # Listen for messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON",
                        "timestamp": time.time()
                    }))
                except Exception as e:
                    logger.error(f"❌ Error handling message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error", 
                        "message": str(e),
                        "timestamp": time.time()
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📡 Client disconnected: {websocket.remote_address}")
        finally:
            self.connections.discard(websocket)
    
    async def handle_message(self, websocket, data):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type")
        
        if msg_type == "generate":
            await self.handle_generation_request(websocket, data)
        elif msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
                "timestamp": time.time()
            }))
    
    async def handle_generation_request(self, websocket, data):
        """Handle text generation requests"""
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        
        if not prompt:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "No prompt provided",
                "timestamp": time.time()
            }))
            return
        
        logger.info(f"🚀 Generation request: {prompt[:50]}...")
        
        try:
            # Generate with real-time uncertainty
            results = await self.model.generate_with_uncertainty(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                websocket=websocket
            )
            
            # Send completion message
            final_text = " ".join([token_data.text for token_data, _ in results])
            avg_uncertainty = sum([metrics.hbar_calibrated for _, metrics in results]) / len(results)
            
            completion = {
                "type": "generation_update",
                "data": {
                    "final": True,
                    "generated_text": final_text,
                    "final_audit": {
                        "average_uncertainty": avg_uncertainty,
                        "total_tokens": len(results),
                        "session_id": "mistral_demo_session"
                    },
                    "session_id": "mistral_demo_session"
                },
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(completion))
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Generation failed: {str(e)}",
                "timestamp": time.time()
            }))

# Flask app for HTTP endpoints
app = Flask(__name__)

# Global server instance
mistral_demo_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_type": "Mistral 7B Instruct (Simulated)",
        "device": "cpu",
        "note": "This is a realistic simulation for demonstration",
        "timestamp": time.time()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": mistral_demo_server.model.model_name if mistral_demo_server else "Unknown",
        "device": "cpu",
        "parameters": "7B (simulated)",
        "memory_usage": 0,
        "status": "ready",
        "simulation": True
    })

def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    logger.info("🛑 Shutdown signal received, cleaning up...")
    sys.exit(0)

async def start_websocket_server(host="0.0.0.0", port=8765):
    """Start WebSocket server"""
    global mistral_demo_server
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Mistral 7B Demo Server")
    
    # Initialize server
    mistral_demo_server = MistralDemoWebSocketServer()
    await mistral_demo_server.setup()
    
    logger.info(f"📡 WebSocket server starting on ws://{host}:{port}")
    
    # Start WebSocket server
    async with websockets.serve(mistral_demo_server.handle_client, host, port):
        logger.info("✅ Demo server ready for connections!")
        await asyncio.Future()  # Run forever

def start_flask_server(host="0.0.0.0", port=5001):
    """Start Flask HTTP server"""
    logger.info(f"🌐 Flask server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

async def main():
    """Main server startup"""
    logger.info("🧠 Mistral 7B Live Uncertainty Demo System")
    logger.info("🎯 Real-time semantic uncertainty quantification (ℏₛ)")
    logger.info("📋 Simulated Mistral behavior with realistic uncertainty patterns")
    
    # Start Flask server in background
    flask_thread = threading.Thread(
        target=start_flask_server,
        args=("0.0.0.0", 5001),
        daemon=True
    )
    flask_thread.start()
    
    # Start WebSocket server (main thread)
    await start_websocket_server("0.0.0.0", 8765)

if __name__ == "__main__":
    print("🧠 Mistral 7B Live Uncertainty Demo Server")
    print("📊 Real-time semantic uncertainty quantification")
    print("🔗 WebSocket: ws://localhost:8765")
    print("🌐 HTTP API: http://localhost:5001")
    print("📋 Features:")
    print("  ✅ Real-time ℏₛ = √(Δμ × Δσ) calculation")
    print("  ✅ Fisher Information Matrix approximation")
    print("  ✅ Jensen-Shannon divergence analysis")
    print("  ✅ Risk level classification")
    print("  ✅ Live uncertainty alerts")
    print("  📝 Note: This is a realistic simulation (no PyTorch required)")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        traceback.print_exc()
        raise 
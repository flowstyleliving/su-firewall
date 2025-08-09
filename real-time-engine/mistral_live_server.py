#!/usr/bin/env python3
"""
🧠 Mistral 7B Live Uncertainty Auditing Server
Real-time semantic uncertainty quantification with logit access
Integrates with Rust core-engine for ℏₛ calculation
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
import gc

# Core ML dependencies
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import numpy as np

# Networking
import websockets
from flask import Flask, request, jsonify
import requests

# Hardware monitoring
import psutil
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TokenData:
    """Token data with probability and logits"""
    text: str
    token_id: int
    probability: float
    logits: torch.Tensor
    position: int
    attention_weights: Optional[torch.Tensor] = None

@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics"""
    delta_mu: float  # Precision (Fisher Information approximation)
    delta_sigma: float  # Flexibility (Jensen-Shannon divergence)
    hbar_raw: float  # Raw semantic uncertainty √(Δμ × Δσ)
    hbar_calibrated: float  # Calibrated uncertainty
    risk_level: str  # Safe, Warning, HighRisk, Critical
    entropy: float  # Token entropy
    confidence: float  # Token confidence
    fisher_information: float  # Fisher Information estimate

class MistralUncertaintyEngine:
    """Core uncertainty calculation engine"""
    
    def __init__(self):
        self.uncertainty_threshold = 2.5
        self.confidence_threshold = 0.3
        self.epsilon = 1e-8  # Numerical stability
        
    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate Shannon entropy from logits"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.item()
    
    def calculate_fisher_information(self, logits: torch.Tensor, target_token_id: int) -> float:
        """Approximate Fisher Information Matrix diagonal element"""
        probs = F.softmax(logits, dim=-1)
        target_prob = probs[target_token_id].item()
        
        # Fisher Information approximation: 1/p for categorical distribution
        fisher_info = 1.0 / (target_prob + self.epsilon)
        return fisher_info
    
    def calculate_js_divergence(self, logits1: torch.Tensor, logits2: torch.Tensor) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        p = F.softmax(logits1, dim=-1)
        q = F.softmax(logits2, dim=-1)
        
        # Jensen-Shannon divergence
        m = 0.5 * (p + q)
        
        kl_pm = F.kl_div(F.log_softmax(logits1, dim=-1), m, reduction='sum')
        kl_qm = F.kl_div(F.log_softmax(logits2, dim=-1), m, reduction='sum')
        
        js_div = 0.5 * (kl_pm + kl_qm)
        return js_div.item()
    
    def calculate_delta_mu(self, fisher_info: float, entropy: float) -> float:
        """Calculate precision (Δμ) using Fisher Information approximation"""
        # Enhanced precision calculation with Fisher Information
        # Higher Fisher Information = higher precision
        precision = math.sqrt(fisher_info) * (1.0 - entropy / 10.0)
        return max(0.1, precision)
    
    def calculate_delta_sigma(self, logits: torch.Tensor, baseline_logits: Optional[torch.Tensor] = None) -> float:
        """Calculate flexibility (Δσ) using Jensen-Shannon divergence"""
        if baseline_logits is None:
            # Use uniform distribution as baseline
            uniform_logits = torch.zeros_like(logits)
            baseline_logits = uniform_logits
            
        # Calculate JS divergence from baseline
        js_div = self.calculate_js_divergence(logits, baseline_logits)
        
        # Map JS divergence to flexibility measure
        flexibility = math.sqrt(js_div + self.epsilon)
        return flexibility
    
    def calculate_uncertainty_metrics(
        self, 
        token_data: TokenData,
        baseline_logits: Optional[torch.Tensor] = None,
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
        
        # Apply calibration (simplified)
        calibration_factor = 1.0 + 0.1 * math.sin(token_data.position * 0.1)  # Position-based calibration
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

class MistralModel:
    """Mistral 7B model wrapper with uncertainty integration"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.uncertainty_engine = MistralUncertaintyEngine()
        self.generation_config = None
        self.baseline_logits = None  # For comparative uncertainty calculation
        
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, use_quantization: bool = True):
        """Load Mistral 7B model with optimizations"""
        logger.info(f"🚀 Loading {self.model_name} on {self.device}")
        
        try:
            # Configure quantization for memory efficiency
            quantization_config = None
            if use_quantization and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load tokenizer
            logger.info("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("🧠 Loading model...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "mps":
                self.model = self.model.to(self.device)
            
            # Configure generation
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("✅ Model loaded successfully!")
            self._log_model_info()
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _log_model_info(self):
        """Log model and system information"""
        if self.model:
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"📊 Model parameters: {param_count:,}")
            
        # System info
        logger.info(f"💻 System: {platform.system()} {platform.release()}")
        logger.info(f"🖥️  CPU: {platform.processor()}")
        logger.info(f"💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"📈 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt for Mistral Instruct"""
        return f"<s>[INST] {prompt} [/INST]"
    
    async def generate_with_uncertainty(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        websocket=None
    ) -> List[Tuple[TokenData, UncertaintyMetrics]]:
        """Generate text with real-time uncertainty analysis"""
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🎯 Generating response for: {prompt[:50]}...")
        
        # Format and tokenize prompt
        formatted_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Update generation config
        self.generation_config.max_new_tokens = max_tokens
        self.generation_config.temperature = temperature
        self.generation_config.top_p = top_p
        
        results = []
        generated_text = ""
        
        try:
            with torch.no_grad():
                # Generate with streaming
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    streamer=None,  # We'll handle streaming manually
                )
                
                # Extract generated tokens (excluding input)
                generated_ids = outputs.sequences[0][input_length:]
                scores = outputs.scores if hasattr(outputs, 'scores') else None
                
                # Process each generated token
                for position, token_id in enumerate(generated_ids):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    generated_text += token_text
                    
                    # Get logits for this position
                    if scores and position < len(scores):
                        token_logits = scores[position][0]  # Remove batch dimension
                        token_prob = F.softmax(token_logits, dim=-1)[token_id].item()
                    else:
                        # Fallback: re-run forward pass for this token
                        current_input = torch.cat([inputs.input_ids, generated_ids[:position+1].unsqueeze(0)], dim=1)
                        logits = self.model(current_input).logits[0, -1, :]
                        token_prob = F.softmax(logits, dim=-1)[token_id].item()
                        token_logits = logits
                    
                    # Create token data
                    token_data = TokenData(
                        text=token_text,
                        token_id=token_id.item(),
                        probability=token_prob,
                        logits=token_logits,
                        position=position
                    )
                    
                    # Calculate uncertainty metrics
                    uncertainty_metrics = self.uncertainty_engine.calculate_uncertainty_metrics(
                        token_data,
                        self.baseline_logits
                    )
                    
                    results.append((token_data, uncertainty_metrics))
                    
                    # Send real-time update via WebSocket
                    if websocket:
                        await self._send_token_update(websocket, token_data, uncertainty_metrics, generated_text)
                    
                    logger.info(f"Token {position}: '{token_text}' | P={token_prob:.3f} | ℏₛ={uncertainty_metrics.hbar_calibrated:.3f} | Risk={uncertainty_metrics.risk_level}")
                    
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            raise
        
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
                    "session_id": "mistral_session",
                    "current_uncertainty": uncertainty.hbar_calibrated,
                    "average_uncertainty": uncertainty.hbar_calibrated,  # Will be calculated by frontend
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

class MistralWebSocketServer:
    """WebSocket server for real-time communication"""
    
    def __init__(self):
        self.model = MistralModel()
        self.connections = set()
        
    async def setup(self):
        """Initialize the model"""
        logger.info("🔧 Setting up Mistral model...")
        self.model.load_model(use_quantization=True)
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        logger.info(f"📡 New client connected: {websocket.remote_address}")
        self.connections.add(websocket)
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Connected to Mistral 7B Live Uncertainty Server",
                "model_info": {
                    "model_name": self.model.model_name,
                    "device": self.model.device,
                    "status": "ready"
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
            final_text = "".join([token_data.text for token_data, _ in results])
            avg_uncertainty = sum([metrics.hbar_calibrated for _, metrics in results]) / len(results)
            
            completion = {
                "type": "generation_update",
                "data": {
                    "final": True,
                    "generated_text": final_text,
                    "final_audit": {
                        "average_uncertainty": avg_uncertainty,
                        "total_tokens": len(results),
                        "session_id": "mistral_session"
                    },
                    "session_id": "mistral_session"
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
mistral_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": mistral_server.model.model is not None if mistral_server else False,
        "model_type": "Mistral 7B Instruct",
        "device": mistral_server.model.device if mistral_server else "unknown",
        "timestamp": time.time()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not mistral_server or not mistral_server.model.model:
        return jsonify({"error": "Model not loaded"}), 500
        
    return jsonify({
        "model_name": mistral_server.model.model_name,
        "device": mistral_server.model.device,
        "parameters": sum(p.numel() for p in mistral_server.model.model.parameters()),
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "status": "ready"
    })

def signal_handler(signum, frame):
    """Graceful shutdown handler"""
    logger.info("🛑 Shutdown signal received, cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

async def start_websocket_server(host="0.0.0.0", port=8765):
    """Start WebSocket server"""
    global mistral_server
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Mistral 7B Live Uncertainty Server")
    
    # Initialize server
    mistral_server = MistralWebSocketServer()
    await mistral_server.setup()
    
    logger.info(f"📡 WebSocket server starting on ws://{host}:{port}")
    
    # Start WebSocket server
    async with websockets.serve(mistral_server.handle_client, host, port):
        logger.info("✅ Server ready for connections!")
        await asyncio.Future()  # Run forever

def start_flask_server(host="0.0.0.0", port=5000):
    """Start Flask HTTP server"""
    logger.info(f"🌐 Flask server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

async def main():
    """Main server startup"""
    logger.info("🧠 Mistral 7B + Live Uncertainty Auditing System")
    logger.info("🎯 Real-time semantic uncertainty quantification (ℏₛ)")
    
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
    print("🧠 Mistral 7B Live Uncertainty Auditing Server")
    print("📊 Real-time semantic uncertainty quantification")
    print("🔗 WebSocket: ws://localhost:8765")
    print("🌐 HTTP API: http://localhost:5001")
    print("📋 Usage:")
    print("  1. Connect React frontend to WebSocket")
    print("  2. Send generation requests")
    print("  3. Receive real-time uncertainty metrics")
    print("  4. Monitor ℏₛ = √(Δμ × Δσ) in real-time")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        traceback.print_exc()
        raise 
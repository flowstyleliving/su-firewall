#!/usr/bin/env python3
"""
🤖 Mock Mistral Server for Live Auditing Demo
Simulates Mistral 7B responses without requiring PyTorch/transformers
Shows live uncertainty quantification in action
"""

import asyncio
import json
import logging
import time
import random
import math
from typing import List, Dict, Any
from dataclasses import dataclass

import websockets
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MockToken:
    """Mock token data simulating what Mistral would provide"""
    text: str
    token_id: int
    probability: float
    logits: List[float]
    position: int

class MockMistralModel:
    """Mock Mistral model that generates realistic uncertainty patterns"""
    
    def __init__(self):
        self.vocab_size = 32000
        self.vocab = self._create_mock_vocab()
        
    def _create_mock_vocab(self) -> Dict[int, str]:
        """Create a simplified vocabulary"""
        vocab = {}
        
        # Common words with known uncertainty patterns
        common_words = [
            "the", "and", "is", "to", "of", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "quantum", "computing", "uncertainty", "probability", "machine", "learning",
            "artificial", "intelligence", "neural", "network", "model", "algorithm",
            "data", "science", "technology", "computer", "system", "analysis",
            "research", "development", "innovation", "future", "potential", "possible"
        ]
        
        for i, word in enumerate(common_words):
            vocab[i] = word
            
        # Fill in random tokens
        for i in range(len(common_words), self.vocab_size):
            vocab[i] = f"token_{i}"
            
        return vocab
    
    def generate_response(self, prompt: str, max_tokens: int = 50) -> List[MockToken]:
        """Generate a mock response with realistic uncertainty patterns"""
        
        # Determine response type based on prompt
        if any(word in prompt.lower() for word in ["quantum", "physics", "uncertainty"]):
            # High uncertainty for technical topics
            base_uncertainty = 0.3
            uncertainty_variance = 0.4
        elif any(word in prompt.lower() for word in ["what", "explain", "how"]):
            # Medium uncertainty for explanatory content
            base_uncertainty = 0.5
            uncertainty_variance = 0.3
        elif any(word in prompt.lower() for word in ["the", "is", "are"]):
            # Low uncertainty for factual content
            base_uncertainty = 0.8
            uncertainty_variance = 0.2
        else:
            # Default moderate uncertainty
            base_uncertainty = 0.6
            uncertainty_variance = 0.3
            
        # Generate response tokens
        response_templates = [
            "Quantum computing is a revolutionary technology that leverages quantum mechanics to process information.",
            "Machine learning involves training algorithms on data to make predictions or decisions.",
            "Artificial intelligence refers to computer systems that can perform tasks typically requiring human intelligence.",
            "Neural networks are computational models inspired by biological neural networks in the brain.",
            "Deep learning uses multiple layers of neural networks to learn complex patterns in data.",
            "Natural language processing enables computers to understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information from images.",
            "Robotics combines engineering, computer science, and AI to create autonomous machines."
        ]
        
        # Select response based on prompt
        if "quantum" in prompt.lower():
            base_response = response_templates[0]
        elif "machine learning" in prompt.lower():
            base_response = response_templates[1]
        elif "AI" in prompt.lower() or "artificial intelligence" in prompt.lower():
            base_response = response_templates[2]
        else:
            base_response = random.choice(response_templates)
            
        # Tokenize response (simplified)
        words = base_response.split()[:max_tokens]
        tokens = []
        
        for position, word in enumerate(words):
            # Find token ID (simplified matching)
            token_id = next((tid for tid, tword in self.vocab.items() if tword.lower() == word.lower()), 
                          random.randint(100, 1000))
            
            # Generate probability with realistic uncertainty
            base_prob = base_uncertainty + random.uniform(-uncertainty_variance/2, uncertainty_variance/2)
            base_prob = max(0.1, min(0.95, base_prob))
            
            # Add position-based effects
            if position == 0:  # First word often more certain
                base_prob = min(0.9, base_prob + 0.1)
            elif position == len(words) - 1:  # Last word might be less certain
                base_prob = max(0.3, base_prob - 0.1)
                
            # Generate realistic logits
            logits = self._generate_logits(token_id, base_prob)
            
            tokens.append(MockToken(
                text=word,
                token_id=token_id,
                probability=base_prob,
                logits=logits,
                position=position
            ))
            
        return tokens
    
    def _generate_logits(self, target_token_id: int, target_prob: float) -> List[float]:
        """Generate realistic logits for a token"""
        logits = [random.uniform(-5.0, -1.0) for _ in range(100)]  # Simplified vocab
        
        # Set target token logit to achieve desired probability
        target_logit = math.log(target_prob / (1 - target_prob + 1e-8))
        if target_token_id < len(logits):
            logits[target_token_id] = target_logit
            
        return logits

class MockAuditSystem:
    """Mock uncertainty audit system"""
    
    def __init__(self):
        self.uncertainty_threshold = 2.5
        self.confidence_threshold = 0.3
        
    def calculate_uncertainty(self, token: MockToken) -> Dict[str, Any]:
        """Calculate uncertainty metrics for a token"""
        
        # Calculate entropy from probability
        p = token.probability
        entropy = -p * math.log2(p + 1e-10) - (1-p) * math.log2(1-p + 1e-10)
        
        # Uncertainty increases with lower probability
        uncertainty = 2.0 + (1.0 - p) * 3.0
        
        # Determine risk level
        if uncertainty > self.uncertainty_threshold:
            risk_level = "Critical"
        elif uncertainty > self.uncertainty_threshold * 0.7:
            risk_level = "HighRisk"
        elif uncertainty > self.uncertainty_threshold * 0.4:
            risk_level = "Warning"
        else:
            risk_level = "Safe"
            
        # Generate alerts
        alerts = []
        if uncertainty > self.uncertainty_threshold:
            alerts.append({
                "alert_type": "HighUncertainty",
                "severity": "Warning",
                "message": f"High uncertainty detected: {uncertainty:.3f}",
                "token_position": token.position,
                "uncertainty_value": uncertainty
            })
            
        if p < self.confidence_threshold:
            alerts.append({
                "alert_type": "LowConfidence", 
                "severity": "Warning",
                "message": f"Low confidence token: '{token.text}' (p={p:.3f})",
                "token_position": token.position,
                "uncertainty_value": uncertainty
            })
            
        return {
            "session_id": "mock_session_123",
            "current_uncertainty": uncertainty,
            "average_uncertainty": uncertainty,  # Simplified
            "risk_level": risk_level,
            "tokens_processed": token.position + 1,
            "alerts": alerts
        }

class MockServerManager:
    """Manages the mock server with WebSocket streaming"""
    
    def __init__(self):
        self.model = MockMistralModel()
        self.audit = MockAuditSystem()
        self.websocket_connections = set()
        
    async def handle_websocket_client(self, websocket, path):
        """Handle WebSocket client connections"""
        logger.info(f"📡 New WebSocket client connected: {websocket.remote_address}")
        self.websocket_connections.add(websocket)
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Connected to Mock Mistral Live Audit Server",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome))
            
            # Listen for client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    error_msg = {"type": "error", "message": "Invalid JSON"}
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📡 WebSocket client disconnected: {websocket.remote_address}")
        finally:
            self.websocket_connections.discard(websocket)
            
    async def handle_websocket_message(self, websocket, data):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type")
        
        if msg_type == "generate":
            # Start generation with live auditing
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 50)
            
            logger.info(f"🚀 Starting generation: {prompt[:50]}...")
            
            try:
                # Generate tokens
                tokens = self.model.generate_response(prompt, max_tokens)
                
                generated_text = ""
                running_uncertainty = 0.0
                
                for i, token in enumerate(tokens):
                    # Add to generated text
                    generated_text += (" " if i > 0 else "") + token.text
                    
                    # Calculate uncertainty
                    audit_result = self.audit.calculate_uncertainty(token)
                    
                    # Update running average
                    running_uncertainty = (running_uncertainty * i + audit_result["current_uncertainty"]) / (i + 1)
                    audit_result["average_uncertainty"] = running_uncertainty
                    
                    # Send generation update
                    response = {
                        "type": "generation_update",
                        "data": {
                            "token": {
                                "text": token.text,
                                "token_id": token.token_id,
                                "probability": token.probability,
                                "position": token.position
                            },
                            "audit": audit_result,
                            "session_id": "mock_session_123",
                            "generated_so_far": generated_text
                        },
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(response))
                    
                    # Simulate realistic generation speed
                    await asyncio.sleep(0.2)
                
                # Send final completion
                final_response = {
                    "type": "generation_update",
                    "data": {
                        "final": True,
                        "generated_text": generated_text,
                        "final_audit": {
                            "average_uncertainty": running_uncertainty,
                            "total_tokens": len(tokens),
                            "session_id": "mock_session_123"
                        },
                        "session_id": "mock_session_123"
                    },
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(final_response))
                
                logger.info(f"✅ Generation completed: {len(tokens)} tokens")
                
            except Exception as e:
                error_response = {
                    "type": "error",
                    "message": str(e),
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(error_response))
                
        elif msg_type == "ping":
            # Respond to ping
            pong = {"type": "pong", "timestamp": time.time()}
            await websocket.send(json.dumps(pong))

# Flask app for HTTP API
app = Flask(__name__)

# Global server instance
mock_server = MockServerManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_type": "Mock Mistral 7B",
        "timestamp": time.time()
    })

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    """HTTP endpoint for generation (non-streaming)"""
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 50)
    
    try:
        # Generate tokens
        tokens = mock_server.model.generate_response(prompt, max_tokens)
        
        # Process with audit
        results = []
        generated_text = ""
        
        for i, token in enumerate(tokens):
            generated_text += (" " if i > 0 else "") + token.text
            audit_result = mock_server.audit.calculate_uncertainty(token)
            
            results.append({
                "token": {
                    "text": token.text,
                    "token_id": token.token_id, 
                    "probability": token.probability,
                    "position": token.position
                },
                "audit": audit_result,
                "generated_so_far": generated_text
            })
        
        return jsonify({
            "success": True,
            "generated_text": generated_text,
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

async def start_websocket_server(host="localhost", port=8765):
    """Start WebSocket server for real-time communication"""
    logger.info(f"🚀 Starting Mock WebSocket server on ws://{host}:{port}")
    
    async with websockets.serve(mock_server.handle_websocket_client, host, port):
        logger.info("✅ Mock WebSocket server started")
        await asyncio.Future()  # Run forever

def start_flask_server(host="localhost", port=5000):
    """Start Flask HTTP server"""
    logger.info(f"🌐 Starting Mock Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

async def main():
    """Main server startup"""
    logger.info("🚀 Starting Mock Mistral 7B Live Audit Server")
    logger.info("📋 This is a demonstration using simulated responses")
    
    # Start Flask server in background thread
    flask_thread = threading.Thread(
        target=start_flask_server,
        args=("0.0.0.0", 5000),
        daemon=True
    )
    flask_thread.start()
    
    # Start WebSocket server (blocks)
    await start_websocket_server("0.0.0.0", 8765)

if __name__ == "__main__":
    print("🤖 Mock Mistral 7B + Live Auditing Server")
    print("📋 This demonstrates the live uncertainty auditing system")
    print("🎯 Usage:")
    print("  1. Connect frontend to WebSocket (port 8765)")
    print("  2. Send generation requests via WebSocket or HTTP")
    print("  3. Watch live uncertainty metrics in real-time")
    print("  4. Responses are simulated but uncertainty patterns are realistic")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise 
#!/usr/bin/env python3
"""
🤖 Mistral 7B + Live Auditing Server
Connects Mistral 7B to the separated uncertainty auditing system
Provides WebSocket streaming for real-time frontend updates
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import torch
import websockets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import requests
import threading
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenData:
    """Token data structure matching our Rust interface"""
    text: str
    token_id: Optional[int] = None
    probability: Optional[float] = None
    logits: Optional[List[float]] = None
    position: int = 0

@dataclass
class AuditConfig:
    """Configuration for the auditing system"""
    uncertainty_threshold: float = 2.5
    spike_threshold: float = 1.5
    confidence_threshold: float = 0.3
    enable_alerts: bool = True

class MistralAuditServer:
    """Main server class connecting Mistral 7B to live auditing"""
    
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 audit_api_url: str = "http://localhost:8080",
                 device: str = "auto"):
        self.model_name = model_name
        self.audit_api_url = audit_api_url
        self.device = device
        
        # Initialize model and tokenizer
        logger.info(f"Loading Mistral model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        self.current_session_id = None
        
        logger.info("✅ Mistral model loaded successfully")

    async def start_audit_session(self, prompt: str) -> str:
        """Start a new audit session"""
        audit_request = {
            "prompt": prompt,
            "model_name": self.model_name,
            "model_version": "mistral-7b-instruct-v0.1", 
            "framework": "Transformers + PyTorch",
            "capabilities": {
                "has_logits": True,
                "has_probabilities": True,
                "supports_streaming": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.audit_api_url}/audit/start",
                json=audit_request,
                timeout=10
            )
            result = response.json()
            
            if result.get("success"):
                session_id = result["session_id"]
                self.current_session_id = session_id
                logger.info(f"✅ Started audit session: {session_id}")
                return session_id
            else:
                raise Exception(f"Failed to start audit: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error starting audit session: {e}")
            raise

    async def add_token_to_audit(self, token_data: TokenData) -> Dict[str, Any]:
        """Add a token to the current audit session"""
        if not self.current_session_id:
            raise Exception("No active audit session")
        
        token_request = {
            "text": token_data.text,
            "token_id": token_data.token_id,
            "probability": token_data.probability,
            "logits": token_data.logits
        }
        
        try:
            response = requests.post(
                f"{self.audit_api_url}/audit/token",
                json=token_request,
                timeout=5
            )
            result = response.json()
            
            if result.get("success"):
                audit_result = result["result"]
                
                # Broadcast to WebSocket clients
                await self.broadcast_audit_update(audit_result)
                
                return audit_result
            else:
                raise Exception(f"Failed to add token: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error adding token to audit: {e}")
            raise

    async def finish_audit_session(self) -> Dict[str, Any]:
        """Finish the current audit session"""
        if not self.current_session_id:
            raise Exception("No active audit session")
        
        try:
            response = requests.post(
                f"{self.audit_api_url}/audit/finish",
                timeout=10
            )
            result = response.json()
            
            if result.get("success"):
                final_result = result["result"]
                session_id = self.current_session_id
                self.current_session_id = None
                
                logger.info(f"✅ Finished audit session: {session_id}")
                
                # Broadcast final results
                await self.broadcast_session_complete(final_result)
                
                return final_result
            else:
                raise Exception(f"Failed to finish audit: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error finishing audit session: {e}")
            raise

    async def generate_with_audit(self, 
                                 prompt: str, 
                                 max_tokens: int = 256,
                                 temperature: float = 0.7,
                                 top_p: float = 0.9) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response with live uncertainty auditing"""
        
        # Start audit session
        session_id = await self.start_audit_session(prompt)
        
        try:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            generated_text = ""
            position = 0
            
            # Generate tokens one by one with uncertainty tracking
            for step in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top-p filtering
                    filtered_logits = self.top_p_filtering(next_token_logits, top_p)
                    
                    # Get probabilities
                    probs = F.softmax(filtered_logits, dim=-1)
                    
                    # Sample next token
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_token_id = next_token.item()
                    
                    # Decode token
                    next_token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                    
                    # Create token data for auditing
                    token_data = TokenData(
                        text=next_token_text,
                        token_id=next_token_id,
                        probability=probs[next_token_id].item(),
                        logits=next_token_logits.cpu().numpy().tolist(),
                        position=position
                    )
                    
                    # Add to audit
                    audit_result = await self.add_token_to_audit(token_data)
                    
                    # Yield token with audit data
                    yield {
                        "token": {
                            "text": next_token_text,
                            "token_id": next_token_id,
                            "probability": token_data.probability,
                            "position": position
                        },
                        "audit": audit_result,
                        "session_id": session_id,
                        "generated_so_far": generated_text + next_token_text
                    }
                    
                    generated_text += next_token_text
                    position += 1
                    
                    # Update inputs for next iteration
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(self.model.device)], dim=1)
                    
                    # Stop on EOS token
                    if next_token_id == self.tokenizer.eos_token_id:
                        break
                    
                    # Small delay for realistic streaming
                    await asyncio.sleep(0.1)
            
            # Finish audit session
            final_audit = await self.finish_audit_session()
            
            # Yield final results
            yield {
                "final": True,
                "generated_text": generated_text,
                "final_audit": final_audit,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error during generation: {e}")
            # Try to finish audit session on error
            try:
                await self.finish_audit_session()
            except:
                pass
            raise

    def top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
        return logits

    async def broadcast_audit_update(self, audit_result: Dict[str, Any]):
        """Broadcast audit updates to WebSocket clients"""
        message = {
            "type": "audit_update",
            "data": audit_result,
            "timestamp": time.time()
        }
        
        # Remove any disconnected clients
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
        
        self.websocket_connections -= disconnected

    async def broadcast_session_complete(self, final_result: Dict[str, Any]):
        """Broadcast session completion to WebSocket clients"""
        message = {
            "type": "session_complete",
            "data": final_result,
            "timestamp": time.time()
        }
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
        
        self.websocket_connections -= disconnected

    async def handle_websocket_client(self, websocket, path):
        """Handle WebSocket client connections"""
        logger.info(f"📡 New WebSocket client connected: {websocket.remote_address}")
        self.websocket_connections.add(websocket)
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Connected to Mistral Live Audit Server",
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
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            
            try:
                async for result in self.generate_with_audit(prompt, max_tokens, temperature, top_p):
                    # Send generation updates directly to this client
                    response = {
                        "type": "generation_update",
                        "data": result,
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(response))
                    
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
socketio = SocketIO(app, cors_allowed_origins="*")

# Global server instance
mistral_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": mistral_server is not None,
        "timestamp": time.time()
    })

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    """HTTP endpoint for generation (non-streaming)"""
    if not mistral_server:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    
    # Run async generation in sync context
    async def run_generation():
        results = []
        async for result in mistral_server.generate_with_audit(prompt, max_tokens, temperature, top_p):
            results.append(result)
        return results
    
    try:
        # This is a simplified sync wrapper - in production use proper async handling
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_generation())
        loop.close()
        
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

async def start_websocket_server(host="localhost", port=8765):
    """Start WebSocket server for real-time communication"""
    global mistral_server
    
    logger.info(f"🚀 Starting WebSocket server on ws://{host}:{port}")
    
    async with websockets.serve(mistral_server.handle_websocket_client, host, port):
        logger.info("✅ WebSocket server started")
        await asyncio.Future()  # Run forever

def start_flask_server(host="localhost", port=5000):
    """Start Flask HTTP server"""
    logger.info(f"🌐 Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

async def main():
    """Main server startup"""
    global mistral_server
    
    logger.info("🚀 Starting Mistral 7B Live Audit Server")
    
    # Initialize Mistral server
    mistral_server = MistralAuditServer(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        audit_api_url="http://localhost:8080",  # Our Rust audit API
        device="auto"
    )
    
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
    # Example usage
    print("🤖 Mistral 7B + Live Auditing Server")
    print("📋 Usage:")
    print("  1. Start the Rust audit API server first (port 8080)")
    print("  2. Run this script to start Mistral server")
    print("  3. Connect frontend to WebSocket (port 8765)")
    print("  4. Send generation requests via WebSocket or HTTP")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise 
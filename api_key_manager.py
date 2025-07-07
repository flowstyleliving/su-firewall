#!/usr/bin/env python3
"""
🔑 SEMANTIC UNCERTAINTY API KEY MANAGEMENT SYSTEM
Secure key generation, validation, and usage tracking for production deployment

Features:
- 🔐 Secure key generation with cryptographic randomness
- 📊 Usage tracking and rate limiting
- 🔄 Key rotation and expiration
- 🛡️ Security validation and monitoring
- 📈 Analytics and reporting
"""

import secrets
import hashlib
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import sys
import os

class APIKeyManager:
    """🔑 Comprehensive API key management system"""
    
    def __init__(self, db_path: str = "api_keys.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """🗄️ Initialize SQLite database for key management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create API keys table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT UNIQUE NOT NULL,
                key_name TEXT NOT NULL,
                user_email TEXT,
                tier TEXT DEFAULT 'free',
                rate_limit_per_minute INTEGER DEFAULT 100,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                last_used TIMESTAMP,
                total_requests INTEGER DEFAULT 0,
                failed_requests INTEGER DEFAULT 0
            )
        ''')
        
        # Create usage tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                response_time_ms REAL,
                status_code INTEGER,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        # Create rate limiting table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                window_start TIMESTAMP,
                request_count INTEGER DEFAULT 0,
                UNIQUE(key_hash, window_start)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def generate_api_key(self, name: str, email: str = None, tier: str = "free") -> str:
        """🔐 Generate a secure API key"""
        # Generate 32 bytes of cryptographically secure random data
        random_bytes = secrets.token_bytes(32)
        
        # Create a prefix for easy identification
        prefix = f"su_{tier[:3]}_"
        
        # Encode as base64 and format
        import base64
        encoded = base64.urlsafe_b64encode(random_bytes).decode('ascii')
        api_key = f"{prefix}{encoded}"
        
        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_keys (key_hash, key_name, user_email, tier, rate_limit_per_minute)
            VALUES (?, ?, ?, ?, ?)
        ''', (key_hash, name, email, tier, self._get_rate_limit_for_tier(tier)))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Generated API key for {name} ({tier} tier)")
        print(f"🔑 Key: {api_key}")
        print(f"📊 Rate limit: {self._get_rate_limit_for_tier(tier)} requests/minute")
        
        return api_key
    
    def _get_rate_limit_for_tier(self, tier: str) -> int:
        """📊 Get rate limit for tier"""
        limits = {
            "free": 100,
            "pro": 1000,
            "enterprise": 10000,
            "unlimited": 100000
        }
        return limits.get(tier, 100)
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict]]:
        """🔍 Validate API key and return usage info"""
        if not api_key:
            return False, None
        
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT key_name, tier, rate_limit_per_minute, is_active, expires_at, 
                   total_requests, failed_requests, last_used
            FROM api_keys 
            WHERE key_hash = ?
        ''', (key_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False, None
        
        key_name, tier, rate_limit, is_active, expires_at, total_requests, failed_requests, last_used = result
        
        # Check if key is active
        if not is_active:
            return False, {"error": "Key is deactivated"}
        
        # Check expiration
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if datetime.now() > expiry:
                return False, {"error": "Key has expired"}
        
        # Check rate limiting
        if not self._check_rate_limit(key_hash, rate_limit):
            return False, {"error": "Rate limit exceeded"}
        
        return True, {
            "key_name": key_name,
            "tier": tier,
            "rate_limit": rate_limit,
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "last_used": last_used
        }
    
    def _check_rate_limit(self, key_hash: str, rate_limit: int) -> bool:
        """⏱️ Check if rate limit is exceeded"""
        window_start = datetime.now().replace(second=0, microsecond=0)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current request count for this window
        cursor.execute('''
            SELECT request_count FROM rate_limits 
            WHERE key_hash = ? AND window_start = ?
        ''', (key_hash, window_start))
        
        result = cursor.fetchone()
        
        if result:
            current_count = result[0]
            if current_count >= rate_limit:
                conn.close()
                return False
            
            # Increment count
            cursor.execute('''
                UPDATE rate_limits SET request_count = request_count + 1
                WHERE key_hash = ? AND window_start = ?
            ''', (key_hash, window_start))
        else:
            # Create new window
            cursor.execute('''
                INSERT INTO rate_limits (key_hash, window_start, request_count)
                VALUES (?, ?, 1)
            ''', (key_hash, window_start))
        
        conn.commit()
        conn.close()
        return True
    
    def log_usage(self, api_key: str, endpoint: str, response_time: float, 
                  status_code: int, error_message: str = None, ip_address: str = None, 
                  user_agent: str = None):
        """📝 Log API usage"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Log usage
        cursor.execute('''
            INSERT INTO usage_logs (key_hash, endpoint, response_time_ms, status_code, 
                                   error_message, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (key_hash, endpoint, response_time, status_code, error_message, ip_address, user_agent))
        
        # Update key statistics
        cursor.execute('''
            UPDATE api_keys 
            SET total_requests = total_requests + 1,
                last_used = CURRENT_TIMESTAMP
            WHERE key_hash = ?
        ''', (key_hash,))
        
        if status_code >= 400:
            cursor.execute('''
                UPDATE api_keys 
                SET failed_requests = failed_requests + 1
                WHERE key_hash = ?
            ''', (key_hash,))
        
        conn.commit()
        conn.close()
    
    def list_keys(self) -> List[Dict]:
        """📋 List all API keys"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT key_name, user_email, tier, rate_limit_per_minute, 
                   created_at, expires_at, is_active, total_requests, 
                   failed_requests, last_used
            FROM api_keys
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        keys = []
        for row in results:
            keys.append({
                "name": row[0],
                "email": row[1],
                "tier": row[2],
                "rate_limit": row[3],
                "created": row[4],
                "expires": row[5],
                "active": bool(row[6]),
                "total_requests": row[7],
                "failed_requests": row[8],
                "last_used": row[9]
            })
        
        return keys
    
    def deactivate_key(self, key_name: str) -> bool:
        """🚫 Deactivate an API key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE api_keys SET is_active = 0 WHERE key_name = ?
        ''', (key_name,))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        if affected > 0:
            print(f"✅ Deactivated key: {key_name}")
            return True
        else:
            print(f"❌ Key not found: {key_name}")
            return False
    
    def get_usage_stats(self, days: int = 30) -> Dict:
        """📊 Get usage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total requests
        cursor.execute('''
            SELECT COUNT(*) FROM usage_logs 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        total_requests = cursor.fetchone()[0]
        
        # Failed requests
        cursor.execute('''
            SELECT COUNT(*) FROM usage_logs 
            WHERE status_code >= 400 AND timestamp >= datetime('now', '-{} days')
        '''.format(days))
        failed_requests = cursor.fetchone()[0]
        
        # Average response time
        cursor.execute('''
            SELECT AVG(response_time_ms) FROM usage_logs 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        avg_response_time = cursor.fetchone()[0] or 0
        
        # Top endpoints
        cursor.execute('''
            SELECT endpoint, COUNT(*) as count FROM usage_logs 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY endpoint ORDER BY count DESC LIMIT 5
        '''.format(days))
        top_endpoints = cursor.fetchall()
        
        conn.close()
        
        return {
            "period_days": days,
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "success_rate": ((total_requests - failed_requests) / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time_ms": round(avg_response_time, 2),
            "top_endpoints": [{"endpoint": ep, "count": count} for ep, count in top_endpoints]
        }

def main():
    """🎯 Main CLI interface"""
    parser = argparse.ArgumentParser(description="🔑 Semantic Uncertainty API Key Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate key command
    gen_parser = subparsers.add_parser("generate", help="Generate new API key")
    gen_parser.add_argument("--name", required=True, help="Key name/description")
    gen_parser.add_argument("--email", help="User email")
    gen_parser.add_argument("--tier", choices=["free", "pro", "enterprise", "unlimited"], 
                           default="free", help="API tier")
    
    # List keys command
    subparsers.add_parser("list", help="List all API keys")
    
    # Deactivate key command
    deact_parser = subparsers.add_parser("deactivate", help="Deactivate API key")
    deact_parser.add_argument("--name", required=True, help="Key name to deactivate")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show usage statistics")
    stats_parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    
    args = parser.parse_args()
    
    manager = APIKeyManager()
    
    if args.command == "generate":
        manager.generate_api_key(args.name, args.email, args.tier)
    
    elif args.command == "list":
        keys = manager.list_keys()
        print("\n📋 API Keys:")
        print("-" * 80)
        for key in keys:
            status = "🟢 ACTIVE" if key["active"] else "🔴 INACTIVE"
            print(f"Name: {key['name']} | Tier: {key['tier']} | Status: {status}")
            print(f"Email: {key['email']} | Rate Limit: {key['rate_limit']}/min")
            print(f"Created: {key['created']} | Requests: {key['total_requests']}")
            print(f"Last Used: {key['last_used'] or 'Never'}")
            print("-" * 80)
    
    elif args.command == "deactivate":
        manager.deactivate_key(args.name)
    
    elif args.command == "stats":
        stats = manager.get_usage_stats(args.days)
        print(f"\n📊 Usage Statistics (Last {args.days} days):")
        print("-" * 50)
        print(f"Total Requests: {stats['total_requests']:,}")
        print(f"Failed Requests: {stats['failed_requests']:,}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Avg Response Time: {stats['avg_response_time_ms']}ms")
        print("\nTop Endpoints:")
        for endpoint in stats['top_endpoints']:
            print(f"  {endpoint['endpoint']}: {endpoint['count']:,} requests")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
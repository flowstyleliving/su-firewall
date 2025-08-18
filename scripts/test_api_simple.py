#!/usr/bin/env python3
"""
Simple API test to verify endpoints work
"""

import requests
import json
import time

def test_simple():
    base_url = "http://127.0.0.1:8080"
    
    print("🔍 Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response type: {response.headers.get('content-type', 'unknown')}")
            print(f"  Response length: {len(response.text)} chars")
            if 'application/json' in response.headers.get('content-type', ''):
                data = response.json()
                print(f"  Health data: {data}")
            else:
                print(f"  HTML response (first 100 chars): {response.text[:100]}...")
    except Exception as e:
        print(f"Health test failed: {e}")
    
    # Test analyze endpoint
    try:
        payload = {"prompt": "What is 2+2?", "output": "4"}
        response = requests.post(
            f"{base_url}/api/v1/analyze", 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"\nAnalyze endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response type: {response.headers.get('content-type', 'unknown')}")
            print(f"  Response length: {len(response.text)} chars")
            if 'application/json' in response.headers.get('content-type', ''):
                data = response.json() 
                print(f"  Analysis data keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                if isinstance(data, dict) and 'hbar_s' in data:
                    print(f"  ℏₛ value: {data['hbar_s']}")
            else:
                print(f"  HTML response (first 100 chars): {response.text[:100]}...")
        else:
            print(f"  Error response: {response.text[:200]}")
    except Exception as e:
        print(f"Analyze test failed: {e}")

if __name__ == "__main__":
    test_simple()
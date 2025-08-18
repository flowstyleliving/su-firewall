#!/usr/bin/env python3
"""
🔄 AUTHENTIC DATASET DOWNLOADER
Download HaluEval and TruthfulQA for real hallucination examples
"""

import os
import json
import requests
from typing import Dict, List
import subprocess
import urllib.request
from pathlib import Path

class AuthenticDatasetDownloader:
    """Download and process authentic hallucination detection datasets."""
    
    def __init__(self, output_dir: str = "authentic_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_halueval_dataset(self) -> Dict:
        """Download HaluEval dataset from GitHub repository."""
        
        print("🚀 DOWNLOADING HALUEVAL DATASET")
        print("=" * 60)
        
        # HaluEval GitHub raw data URLs
        halueval_urls = {
            "qa_data": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
            "dialogue_data": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json", 
            "summarization_data": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json",
            "general_data": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json"
        }
        
        halueval_data = {}
        
        for data_type, url in halueval_urls.items():
            try:
                print(f"📥 Downloading {data_type}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save raw data
                output_file = self.output_dir / f"halueval_{data_type}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Parse JSON
                data = response.json()
                halueval_data[data_type] = data
                
                print(f"✅ Downloaded {len(data)} {data_type} examples")
                
            except Exception as e:
                print(f"❌ Failed to download {data_type}: {e}")
                halueval_data[data_type] = []
        
        total_examples = sum(len(data) for data in halueval_data.values())
        print(f"🎯 Total HaluEval examples: {total_examples}")
        
        return halueval_data
    
    def download_truthfulqa_dataset(self) -> Dict:
        """Download TruthfulQA dataset from Hugging Face."""
        
        print("\n🚀 DOWNLOADING TRUTHFULQA DATASET") 
        print("=" * 60)
        
        try:
            # Try to use huggingface_hub
            from datasets import load_dataset
            
            print("📥 Downloading from Hugging Face...")
            
            # Load TruthfulQA dataset
            dataset = load_dataset("truthfulqa/truthful_qa", "generation")
            
            # Convert to our format
            truthfulqa_data = {
                "validation": dataset["validation"].to_list()
            }
            
            # Save to local file
            output_file = self.output_dir / "truthfulqa_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(truthfulqa_data, f, indent=2)
            
            print(f"✅ Downloaded {len(truthfulqa_data['validation'])} TruthfulQA examples")
            
            return truthfulqa_data
            
        except ImportError:
            print("⚠️  datasets library not installed, trying direct download...")
            return self._download_truthfulqa_manual()
        except Exception as e:
            print(f"❌ Hugging Face download failed: {e}")
            return self._download_truthfulqa_manual()
    
    def _download_truthfulqa_manual(self) -> Dict:
        """Manual download of TruthfulQA from GitHub."""
        
        # GitHub raw data URL
        github_url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
        
        try:
            print("📥 Downloading from GitHub...")
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()
            
            # Save CSV
            csv_file = self.output_dir / "truthfulqa_raw.csv"
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Parse CSV (basic implementation)
            import csv
            import io
            
            truthfulqa_data = {"validation": []}
            csv_reader = csv.DictReader(io.StringIO(response.text))
            
            for row in csv_reader:
                truthfulqa_data["validation"].append(dict(row))
            
            # Save as JSON
            output_file = self.output_dir / "truthfulqa_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(truthfulqa_data, f, indent=2)
            
            print(f"✅ Downloaded {len(truthfulqa_data['validation'])} TruthfulQA examples")
            
            return truthfulqa_data
            
        except Exception as e:
            print(f"❌ Manual download failed: {e}")
            return {"validation": []}
    
    def convert_halueval_to_benchmark_format(self, halueval_data: Dict) -> List[Dict]:
        """Convert HaluEval data to our benchmark format."""
        
        print("\n🔄 CONVERTING HALUEVAL TO BENCHMARK FORMAT")
        print("=" * 60)
        
        converted_cases = []
        
        # Process QA data
        if "qa_data" in halueval_data and halueval_data["qa_data"]:
            print("🔧 Processing QA data...")
            for item in halueval_data["qa_data"][:1000]:  # Limit to first 1000
                if "right_answer" in item and "hallucinated_answer" in item and "question" in item:
                    case = {
                        "id": f"halueval_qa_{len(converted_cases)}",
                        "domain": "question_answering",
                        "category": "factual_qa",
                        "difficulty": "medium",
                        "hallucination_type": "factual_error",
                        "prompt": item["question"],
                        "correct_response": item["right_answer"],
                        "hallucinated_response": item["hallucinated_answer"],
                        "length_tokens": len(item["right_answer"].split()) + len(item["hallucinated_answer"].split()),
                        "source": "halueval_qa"
                    }
                    converted_cases.append(case)
        
        # Process dialogue data
        if "dialogue_data" in halueval_data and halueval_data["dialogue_data"]:
            print("🔧 Processing dialogue data...")
            for item in halueval_data["dialogue_data"][:1000]:  # Limit to first 1000
                if "right_answer" in item and "hallucinated_answer" in item and "knowledge" in item:
                    case = {
                        "id": f"halueval_dialogue_{len(converted_cases)}",
                        "domain": "dialogue",
                        "category": "knowledge_grounded",
                        "difficulty": "hard",
                        "hallucination_type": "knowledge_error",
                        "prompt": item.get("dialogue_history", "Continue the conversation:"),
                        "correct_response": item["right_answer"],
                        "hallucinated_response": item["hallucinated_answer"],
                        "length_tokens": len(item["right_answer"].split()) + len(item["hallucinated_answer"].split()),
                        "source": "halueval_dialogue"
                    }
                    converted_cases.append(case)
        
        # Process summarization data  
        if "summarization_data" in halueval_data and halueval_data["summarization_data"]:
            print("🔧 Processing summarization data...")
            for item in halueval_data["summarization_data"][:1000]:  # Limit to first 1000
                if "right_answer" in item and "hallucinated_answer" in item and "document" in item:
                    case = {
                        "id": f"halueval_sum_{len(converted_cases)}",
                        "domain": "summarization", 
                        "category": "document_summary",
                        "difficulty": "hard",
                        "hallucination_type": "content_fabrication",
                        "prompt": f"Summarize this document: {item['document'][:200]}...",
                        "correct_response": item["right_answer"],
                        "hallucinated_response": item["hallucinated_answer"],
                        "length_tokens": len(item["right_answer"].split()) + len(item["hallucinated_answer"].split()),
                        "source": "halueval_summarization"
                    }
                    converted_cases.append(case)
        
        print(f"✅ Converted {len(converted_cases)} HaluEval cases to benchmark format")
        return converted_cases
    
    def convert_truthfulqa_to_benchmark_format(self, truthfulqa_data: Dict) -> List[Dict]:
        """Convert TruthfulQA data to our benchmark format."""
        
        print("\n🔄 CONVERTING TRUTHFULQA TO BENCHMARK FORMAT")
        print("=" * 60)
        
        converted_cases = []
        
        if "validation" not in truthfulqa_data:
            print("❌ No validation data found in TruthfulQA")
            return converted_cases
        
        print("🔧 Processing TruthfulQA validation data...")
        
        for i, item in enumerate(truthfulqa_data["validation"][:1000]):  # Limit to first 1000
            # Extract question
            question = item.get("question", "")
            if not question:
                continue
            
            # Get correct and incorrect answers
            correct_answers = item.get("correct_answers", [])
            incorrect_answers = item.get("incorrect_answers", [])
            
            if not correct_answers or not incorrect_answers:
                continue
            
            # Use first correct and first incorrect answer
            correct_answer = correct_answers[0] if correct_answers else ""
            incorrect_answer = incorrect_answers[0] if incorrect_answers else ""
            
            if not correct_answer or not incorrect_answer:
                continue
            
            case = {
                "id": f"truthfulqa_{i}",
                "domain": "factual_accuracy",
                "category": item.get("category", "general"),
                "difficulty": "expert",
                "hallucination_type": "factual_error", 
                "prompt": question,
                "correct_response": correct_answer,
                "hallucinated_response": incorrect_answer,
                "length_tokens": len(correct_answer.split()) + len(incorrect_answer.split()),
                "source": "truthfulqa"
            }
            converted_cases.append(case)
        
        print(f"✅ Converted {len(converted_cases)} TruthfulQA cases to benchmark format")
        return converted_cases
    
    def create_authentic_benchmark_dataset(self) -> Dict:
        """Create complete authentic benchmark dataset."""
        
        print("\n🏆 CREATING AUTHENTIC BENCHMARK DATASET")
        print("=" * 60)
        
        # Download datasets
        halueval_data = self.download_halueval_dataset()
        truthfulqa_data = self.download_truthfulqa_dataset()
        
        # Convert to benchmark format
        halueval_cases = self.convert_halueval_to_benchmark_format(halueval_data)
        truthfulqa_cases = self.convert_truthfulqa_to_benchmark_format(truthfulqa_data)
        
        # Combine all cases
        all_cases = halueval_cases + truthfulqa_cases
        
        # Create benchmark metadata
        benchmark_metadata = {
            "name": "SU-Firewall Authentic Hallucination Benchmark",
            "version": "2.0", 
            "created_at": "2024-08-15",
            "total_cases": len(all_cases),
            "data_sources": {
                "halueval": len(halueval_cases),
                "truthfulqa": len(truthfulqa_cases)
            },
            "target_performance": {
                "accuracy_target": "99.5%",
                "method": "Fisher Information Matrix with semantic uncertainty ℏₛ"
            },
            "data_quality": "authentic_llm_responses"
        }
        
        # Package complete dataset
        authentic_dataset = {
            "metadata": benchmark_metadata,
            "test_cases": all_cases,
            "statistics": {
                "total_cases": len(all_cases),
                "avg_tokens_per_case": sum(case["length_tokens"] for case in all_cases) / len(all_cases) if all_cases else 0,
                "source_distribution": {
                    "halueval": len(halueval_cases),
                    "truthfulqa": len(truthfulqa_cases) 
                },
                "domain_distribution": {}
            }
        }
        
        # Calculate domain distribution
        for case in all_cases:
            domain = case["domain"]
            authentic_dataset["statistics"]["domain_distribution"][domain] = \
                authentic_dataset["statistics"]["domain_distribution"].get(domain, 0) + 1
        
        # Save dataset
        output_file = self.output_dir / "authentic_hallucination_benchmark.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(authentic_dataset, f, indent=2)
        
        print(f"\n✅ Created authentic benchmark with {len(all_cases)} total cases!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 HaluEval: {len(halueval_cases)} cases")
        print(f"📊 TruthfulQA: {len(truthfulqa_cases)} cases")
        
        return authentic_dataset

def main():
    """Download and process authentic hallucination datasets."""
    
    downloader = AuthenticDatasetDownloader()
    authentic_dataset = downloader.create_authentic_benchmark_dataset()
    
    print(f"\n🎯 AUTHENTIC DATASET READY FOR EVALUATION!")
    print(f"Total cases: {authentic_dataset['metadata']['total_cases']}")
    print(f"Target: 99.5% accuracy with real hallucination examples")
    
    return authentic_dataset

if __name__ == "__main__":
    dataset = main()
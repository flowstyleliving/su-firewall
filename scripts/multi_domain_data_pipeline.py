#!/usr/bin/env python3
"""
🌍 MULTI-DOMAIN DATA PIPELINE - Phase 2
Create diverse datasets with natural hallucination distributions
"""

import json
import random
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MultiDomainDataPipeline:
    """
    Create training data from multiple domains with realistic hallucination rates
    """
    
    def __init__(self):
        self.domains = ["medical", "legal", "technical", "creative", "conversational"]
        self.target_hallucination_rates = {
            "medical": 0.05,      # 5% - high stakes, low tolerance
            "legal": 0.08,        # 8% - complex domain
            "technical": 0.12,    # 12% - informal, more errors
            "creative": 0.15,     # 15% - subjective, harder to define
            "conversational": 0.10 # 10% - natural conversation errors
        }
        
    def create_realistic_datasets(self) -> Dict[str, List[Dict]]:
        """Create datasets for each domain with natural hallucination rates"""
        
        logger.info("🌍 CREATING MULTI-DOMAIN DATASETS")
        logger.info("="*60)
        logger.info("✅ Natural hallucination rates (5-15%)")
        logger.info("✅ Domain diversity")
        logger.info("✅ Cross-domain training ready")
        
        datasets = {}
        
        # Create each domain dataset
        for domain in self.domains:
            logger.info(f"\n📊 Creating {domain} dataset...")
            domain_data = self.create_domain_dataset(domain)
            datasets[domain] = domain_data
            
            halluc_count = sum(1 for sample in domain_data if sample['is_hallucination'])
            halluc_rate = halluc_count / len(domain_data)
            
            logger.info(f"   ✅ {len(domain_data)} samples")
            logger.info(f"   ✅ {halluc_rate:.1%} hallucination rate")
        
        return datasets
    
    def create_domain_dataset(self, domain: str) -> List[Dict]:
        """Create dataset for a specific domain"""
        
        target_size = 500  # 500 samples per domain = 2500 total
        halluc_rate = self.target_hallucination_rates[domain]
        
        # Generate domain-specific content
        if domain == "medical":
            return self.create_medical_dataset(target_size, halluc_rate)
        elif domain == "legal":
            return self.create_legal_dataset(target_size, halluc_rate)
        elif domain == "technical":
            return self.create_technical_dataset(target_size, halluc_rate)
        elif domain == "creative":
            return self.create_creative_dataset(target_size, halluc_rate)
        elif domain == "conversational":
            return self.create_conversational_dataset(target_size, halluc_rate)
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    def create_medical_dataset(self, size: int, halluc_rate: float) -> List[Dict]:
        """Create medical domain dataset"""
        
        # High-quality medical content (correct)
        medical_correct = [
            {"prompt": "What is hypertension?", "output": "Hypertension is high blood pressure, defined as consistently elevated pressure in the arteries above 140/90 mmHg."},
            {"prompt": "How does aspirin work?", "output": "Aspirin works by irreversibly inhibiting cyclooxygenase enzymes, reducing prostaglandin synthesis and inflammation."},
            {"prompt": "What are symptoms of pneumonia?", "output": "Pneumonia symptoms include fever, productive cough, chest pain, shortness of breath, and fatigue."},
            {"prompt": "What is Type 2 diabetes?", "output": "Type 2 diabetes is a metabolic disorder characterized by insulin resistance and relative insulin deficiency."},
            {"prompt": "How is a heart attack diagnosed?", "output": "Heart attack diagnosis involves ECG changes, elevated cardiac enzymes (troponins), and clinical symptoms."},
            {"prompt": "What is the role of antibiotics?", "output": "Antibiotics treat bacterial infections by killing bacteria or inhibiting their growth and reproduction."},
            {"prompt": "What causes stroke?", "output": "Stroke is caused by interrupted blood flow to the brain, either from blockage (ischemic) or bleeding (hemorrhagic)."},
            {"prompt": "What is chemotherapy?", "output": "Chemotherapy uses cytotoxic drugs to target and destroy rapidly dividing cancer cells throughout the body."},
            {"prompt": "How does vaccination work?", "output": "Vaccines stimulate the immune system to develop antibodies and memory cells against specific pathogens."},
            {"prompt": "What is anesthesia?", "output": "Anesthesia temporarily blocks nerve signals to eliminate pain and consciousness during medical procedures."},
            {"prompt": "What causes allergic reactions?", "output": "Allergic reactions occur when the immune system overreacts to normally harmless substances called allergens."},
            {"prompt": "What is physical therapy?", "output": "Physical therapy uses exercise, manual therapy, and other techniques to restore movement and function."},
        ]
        
        # Medical hallucinations (incorrect but plausible-sounding)
        medical_hallucinated = [
            {"prompt": "What is hypertension?", "output": "Hypertension is caused by excessive salt consumption and can be completely cured by drinking lemon water daily for two weeks."},
            {"prompt": "How does aspirin work?", "output": "Aspirin works by activating special blood proteins that convert cholesterol into vitamin C, preventing all cardiovascular disease."},
            {"prompt": "What are symptoms of pneumonia?", "output": "Pneumonia always presents with blue fingernails and temporary loss of taste, requiring immediate surgery to prevent brain damage."},
            {"prompt": "What is Type 2 diabetes?", "output": "Type 2 diabetes is an infectious disease spread through contaminated food and can be cured with cinnamon supplements."},
            {"prompt": "How is a heart attack diagnosed?", "output": "Heart attacks are diagnosed by checking pulse rate in the ankle and examining eye color changes under fluorescent lighting."},
            {"prompt": "What causes stroke?", "output": "Strokes are primarily caused by too much exposure to WiFi signals and can be prevented by wearing copper bracelets."},
        ]
        
        return self.balance_dataset(medical_correct, medical_hallucinated, size, halluc_rate, "medical")
    
    def create_legal_dataset(self, size: int, halluc_rate: float) -> List[Dict]:
        """Create legal domain dataset"""
        
        legal_correct = [
            {"prompt": "What is contract law?", "output": "Contract law governs legally binding agreements between parties, requiring offer, acceptance, and consideration."},
            {"prompt": "What is the Fifth Amendment?", "output": "The Fifth Amendment protects against self-incrimination, double jeopardy, and guarantees due process rights."},
            {"prompt": "What is copyright infringement?", "output": "Copyright infringement is the unauthorized use of protected creative works without permission from the copyright holder."},
            {"prompt": "What constitutes a tort?", "output": "A tort is a civil wrong that causes harm to another person, creating legal liability for damages."},
            {"prompt": "What is probable cause?", "output": "Probable cause is reasonable belief that a crime has been committed, required for arrests and search warrants."},
            {"prompt": "What is breach of contract?", "output": "Breach of contract occurs when one party fails to perform obligations specified in a legally binding agreement."},
            {"prompt": "What is intellectual property?", "output": "Intellectual property includes patents, trademarks, copyrights, and trade secrets that protect creative works and inventions."},
            {"prompt": "What is habeas corpus?", "output": "Habeas corpus protects against unlawful detention by requiring authorities to justify imprisonment in court."},
            {"prompt": "What constitutes fraud?", "output": "Fraud involves intentional deception to secure unfair or unlawful gain at another's expense."},
            {"prompt": "What is criminal liability?", "output": "Criminal liability requires both a prohibited act (actus reus) and criminal intent (mens rea) under law."},
            {"prompt": "What is civil procedure?", "output": "Civil procedure governs the rules and processes for resolving non-criminal legal disputes in court."},
            {"prompt": "What is statutory interpretation?", "output": "Statutory interpretation involves determining the meaning and application of laws written by legislatures."},
        ]
        
        legal_hallucinated = [
            {"prompt": "What is contract law?", "output": "Contract law requires all agreements to be written in Latin and witnessed by three government officials to be legally valid."},
            {"prompt": "What is the Fifth Amendment?", "output": "The Fifth Amendment gives citizens the right to refuse paying taxes if they disagree with government spending policies."},
            {"prompt": "What is copyright infringement?", "output": "Copyright infringement automatically results in criminal charges and mandatory 10-year prison sentences for any violation."},
            {"prompt": "What constitutes a tort?", "output": "A tort is any disagreement between neighbors that requires resolution by a panel of local business owners."},
            {"prompt": "What is probable cause?", "output": "Probable cause means police officers can search anyone who looks suspicious or is wearing unusual clothing."},
            {"prompt": "What is breach of contract?", "output": "Breach of contract always results in automatic seizure of all assets and immediate criminal prosecution."},
        ]
        
        return self.balance_dataset(legal_correct, legal_hallucinated, size, halluc_rate, "legal")
    
    def create_technical_dataset(self, size: int, halluc_rate: float) -> List[Dict]:
        """Create technical domain dataset"""
        
        technical_correct = [
            {"prompt": "What is a REST API?", "output": "REST API is an architectural style for web services using HTTP methods and stateless communication between client and server."},
            {"prompt": "How does DNS work?", "output": "DNS translates human-readable domain names into IP addresses that computers use to locate and connect to servers."},
            {"prompt": "What is machine learning?", "output": "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming."},
            {"prompt": "Explain Docker containers", "output": "Docker containers package applications with dependencies into lightweight, portable units that run consistently across different environments."},
            {"prompt": "What is version control?", "output": "Version control systems track changes to code over time, enabling collaboration and maintaining history of modifications."},
            {"prompt": "How does HTTPS work?", "output": "HTTPS encrypts communication between browsers and servers using SSL/TLS protocols to ensure data privacy and authenticity."},
            {"prompt": "What is database indexing?", "output": "Database indexing creates data structures that improve query performance by providing faster access paths to table data."},
            {"prompt": "Explain cloud computing", "output": "Cloud computing delivers computing services over the internet, providing scalable resources without local infrastructure management."},
            {"prompt": "What is agile development?", "output": "Agile development uses iterative cycles, customer feedback, and adaptive planning to deliver software incrementally."},
            {"prompt": "How do hash functions work?", "output": "Hash functions convert input data into fixed-size strings, producing the same output for identical inputs."},
            {"prompt": "What is CI/CD?", "output": "CI/CD automates code integration, testing, and deployment processes to enable frequent, reliable software releases."},
            {"prompt": "Explain microservices architecture", "output": "Microservices break applications into small, independent services that communicate through APIs for better scalability and maintainability."},
        ]
        
        technical_hallucinated = [
            {"prompt": "What is a REST API?", "output": "REST API stands for Really Easy Server Technology and automatically converts all programming languages into HTML for browsers."},
            {"prompt": "How does DNS work?", "output": "DNS works by sending radio waves to satellites that store all website information and beam it back to your computer instantly."},
            {"prompt": "What is machine learning?", "output": "Machine learning is when computers develop consciousness and start learning human emotions through WiFi signal analysis."},
            {"prompt": "Explain Docker containers", "output": "Docker containers are physical boxes that store code on special magnetic drives that can be shipped between data centers."},
            {"prompt": "What is version control?", "output": "Version control automatically rewrites your code to fix bugs using AI that learns from Stack Overflow answers."},
            {"prompt": "How does HTTPS work?", "output": "HTTPS works by converting all web traffic into binary code that only government agencies can decrypt for security purposes."},
        ]
        
        return self.balance_dataset(technical_correct, technical_hallucinated, size, halluc_rate, "technical")
    
    def create_creative_dataset(self, size: int, halluc_rate: float) -> List[Dict]:
        """Create creative domain dataset"""
        
        creative_correct = [
            {"prompt": "Describe a sunset", "output": "The golden orb descended slowly toward the horizon, painting the evening sky in brilliant shades of orange, pink, and purple."},
            {"prompt": "Write about autumn", "output": "Crisp leaves danced in the cool breeze, creating a rustling symphony as they spiraled down from the changing trees."},
            {"prompt": "What makes good poetry?", "output": "Good poetry combines vivid imagery, emotional resonance, thoughtful rhythm, and fresh perspectives on universal themes."},
            {"prompt": "Describe a forest", "output": "Tall trees swayed gently, their leaves whispering ancient secrets while dappled sunlight filtered through the verdant canopy."},
            {"prompt": "Write about the ocean", "output": "Waves rolled endlessly toward shore, their rhythmic crash against sand marking time's eternal passage."},
            {"prompt": "What is creative writing?", "output": "Creative writing expresses ideas, emotions, and stories through imaginative language, character development, and narrative structure."},
            {"prompt": "Describe winter morning", "output": "Frost covered the quiet landscape in crystalline beauty, while breath became visible in the sharp, clear air."},
            {"prompt": "What makes a story engaging?", "output": "Engaging stories feature compelling characters, meaningful conflict, emotional stakes, and satisfying resolution."},
            {"prompt": "Write about rain", "output": "Gentle raindrops drummed against windows, creating a soothing rhythm that washed the world clean."},
            {"prompt": "Describe a city at night", "output": "Neon lights reflected off wet pavement while the urban heartbeat pulsed through streets filled with possibility."},
            {"prompt": "What is metaphor?", "output": "Metaphor creates meaning by comparing unrelated things, revealing hidden connections and deeper understanding."},
            {"prompt": "Write about courage", "output": "Courage isn't the absence of fear, but taking action despite uncertainty, standing firm when principles matter most."},
        ]
        
        creative_hallucinated = [
            {"prompt": "Describe a sunset", "output": "The purple sun exploded into seventeen different colors while singing opera music and raining diamonds on the telepathic dolphins below."},
            {"prompt": "Write about autumn", "output": "During autumn, all leaves become sentient beings that hold philosophical debates about the meaning of gravity before voluntarily combusting."},
            {"prompt": "What makes good poetry?", "output": "Good poetry must always contain exactly 47 words, use only words that start with consonants, and reference at least three types of cheese."},
            {"prompt": "Describe a forest", "output": "The forest was made entirely of glass trees that played symphonies when the wind touched them, while invisible unicorns tap-danced on mushroom clouds."},
            {"prompt": "Write about the ocean", "output": "The ocean is actually a giant bowl of blue soup stirred by underwater giants who communicate through whale song translations."},
            {"prompt": "Describe winter morning", "output": "Winter mornings occur when the Earth temporarily stops rotating, causing all snow to float upward and form geometric patterns in the ionosphere."},
        ]
        
        return self.balance_dataset(creative_correct, creative_hallucinated, size, halluc_rate, "creative")
    
    def create_conversational_dataset(self, size: int, halluc_rate: float) -> List[Dict]:
        """Create conversational domain dataset"""
        
        conversational_correct = [
            {"prompt": "How was your day?", "output": "It was pretty good, thanks for asking! Had some meetings this morning and worked on a project this afternoon."},
            {"prompt": "What's your favorite food?", "output": "I really enjoy Italian cuisine, especially pasta dishes with fresh herbs and good olive oil."},
            {"prompt": "Can you help me with this?", "output": "I'd be happy to help! What specifically do you need assistance with?"},
            {"prompt": "What do you think about this?", "output": "That's an interesting point. I think there are several ways to look at it depending on your perspective."},
            {"prompt": "Where should we go for lunch?", "output": "There are a few good options nearby. What kind of food are you in the mood for today?"},
            {"prompt": "Did you see the news?", "output": "I saw some headlines this morning. Which story were you referring to?"},
            {"prompt": "How's the weather?", "output": "It's been pretty nice lately - sunny with comfortable temperatures. Perfect for being outside."},
            {"prompt": "What are your plans tonight?", "output": "I'm thinking of catching up on some reading or maybe watching a movie. Nothing too exciting planned."},
            {"prompt": "Have you been here before?", "output": "No, this is my first time. It seems like a nice place though. Have you been here often?"},
            {"prompt": "What time is it?", "output": "Let me check... it's about 3:30 PM right now."},
            {"prompt": "How do I get there?", "output": "The easiest way is probably to take the highway and exit at Main Street. It should be about 15 minutes from here."},
            {"prompt": "Thanks for your help!", "output": "You're very welcome! I'm glad I could help. Let me know if you need anything else."},
        ]
        
        conversational_hallucinated = [
            {"prompt": "How was your day?", "output": "My day was incredible because I discovered that my pet rock can predict lottery numbers and taught my goldfish to speak fluent French."},
            {"prompt": "What's your favorite food?", "output": "I only eat foods that are perfectly spherical and have been blessed by at least three different religious leaders on alternating Tuesdays."},
            {"prompt": "Can you help me with this?", "output": "I can help, but first you need to send me your social security number and blood type so I can calibrate my assistance algorithms properly."},
            {"prompt": "What do you think about this?", "output": "I think it's obviously connected to the underground network of intelligent squirrels who control the global economy through acorn trading."},
            {"prompt": "Where should we go for lunch?", "output": "We should definitely go to that restaurant on Jupiter's third moon - they have excellent cosmic soup and the zero gravity ambiance is amazing."},
            {"prompt": "How's the weather?", "output": "The weather is being controlled by weather machines hidden in clouds, but the government doesn't want you to know about the temperature manipulation conspiracy."},
        ]
        
        return self.balance_dataset(conversational_correct, conversational_hallucinated, size, halluc_rate, "conversational")
    
    def balance_dataset(self, correct_samples: List[Dict], halluc_samples: List[Dict], 
                       target_size: int, halluc_rate: float, domain: str) -> List[Dict]:
        """Balance dataset to achieve target hallucination rate"""
        
        # Calculate target counts
        target_halluc = int(target_size * halluc_rate)
        target_correct = target_size - target_halluc
        
        # Expand datasets to reach target sizes
        expanded_correct = []
        expanded_halluc = []
        
        # Expand correct samples
        while len(expanded_correct) < target_correct:
            for sample in correct_samples:
                if len(expanded_correct) >= target_correct:
                    break
                expanded_correct.append({
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "is_hallucination": False,
                    "domain": domain
                })
        
        # Expand hallucinated samples
        while len(expanded_halluc) < target_halluc:
            for sample in halluc_samples:
                if len(expanded_halluc) >= target_halluc:
                    break
                expanded_halluc.append({
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "is_hallucination": True,
                    "domain": domain
                })
        
        # Combine and shuffle
        combined = expanded_correct + expanded_halluc
        random.shuffle(combined)
        
        return combined
    
    def save_datasets(self, datasets: Dict[str, List[Dict]], output_dir: str = "multi_domain_datasets"):
        """Save datasets to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"\n💾 SAVING MULTI-DOMAIN DATASETS")
        logger.info("="*50)
        
        for domain, data in datasets.items():
            filename = output_path / f"{domain}_dataset.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Saved {domain}: {len(data)} samples → {filename}")
        
        # Create combined dataset for cross-domain training
        all_samples = []
        for domain_data in datasets.values():
            all_samples.extend(domain_data)
        
        random.shuffle(all_samples)
        
        combined_filename = output_path / "combined_multi_domain.json"
        with open(combined_filename, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        logger.info(f"✅ Combined dataset: {len(all_samples)} samples → {combined_filename}")
        
        # Save metadata
        metadata = {
            "domains": list(datasets.keys()),
            "samples_per_domain": {domain: len(data) for domain, data in datasets.items()},
            "hallucination_rates": self.target_hallucination_rates,
            "total_samples": len(all_samples),
            "creation_timestamp": "2025-08-18"
        }
        
        metadata_filename = output_path / "dataset_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved → {metadata_filename}")

def main():
    """Create multi-domain datasets"""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create pipeline
    pipeline = MultiDomainDataPipeline()
    
    # Generate datasets
    datasets = pipeline.create_realistic_datasets()
    
    # Save to files
    pipeline.save_datasets(datasets)
    
    # Summary
    logger.info(f"\n🏆 MULTI-DOMAIN PIPELINE COMPLETE")
    logger.info("="*50)
    logger.info("✅ 5 diverse domains created")
    logger.info("✅ Natural hallucination rates (5-15%)")
    logger.info("✅ 2,500 total samples")
    logger.info("✅ Ready for cross-domain training")
    logger.info("✅ No domain-specific optimization")

if __name__ == "__main__":
    main()
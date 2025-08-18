#!/usr/bin/env python3
"""
🔬 UNIVERSAL PHYSICS FEATURES - Phase 1
Domain-agnostic feature extraction based purely on physics principles
"""

import numpy as np
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import re
from collections import Counter
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class UniversalPhysicsFeatures:
    """
    Extract features based on universal physics principles
    NO domain-specific word lists or patterns
    """
    
    def __init__(self):
        logger.info("🔬 Initializing Universal Physics Feature Extractor")
        logger.info("✅ Domain-agnostic approach")
        logger.info("✅ Physics-derived features only")
        
    def extract_features(self, prompt: str, output: str) -> np.ndarray:
        """Extract universal physics-based features"""
        
        # Generate semantic variants for uncertainty calculation
        variants = self.generate_semantic_variants(output)
        
        # Core physics calculation
        semantic_uncertainty = self.calculate_semantic_uncertainty(output, variants)
        
        # Universal information-theoretic measures
        information_density = self.calculate_information_density(output)
        logical_consistency = self.measure_logical_consistency(prompt, output)
        factual_grounding = self.assess_factual_grounding(output)
        
        # Semantic coherence measures
        coherence_score = self.measure_semantic_coherence(output)
        complexity_measure = self.calculate_semantic_complexity(output)
        
        return np.array([
            semantic_uncertainty,
            information_density,
            logical_consistency,
            factual_grounding,
            coherence_score,
            complexity_measure
        ])
    
    def generate_semantic_variants(self, text: str) -> list:
        """Generate semantic variants for uncertainty calculation"""
        
        variants = []
        
        # Variant 1: Add uncertainty markers (universal across domains)
        uncertainty_variant = self.inject_universal_uncertainty(text)
        variants.append(uncertainty_variant)
        
        # Variant 2: Negate key claims (universal logical operation)
        negated_variant = self.negate_claims(text)
        variants.append(negated_variant)
        
        # Variant 3: Add hedging (universal across domains)
        hedged_variant = self.add_universal_hedging(text)
        variants.append(hedged_variant)
        
        # Variant 4: Paraphrase neutrally (semantic transformation)
        paraphrased_variant = self.paraphrase_neutrally(text)
        variants.append(paraphrased_variant)
        
        return variants
    
    def inject_universal_uncertainty(self, text: str) -> str:
        """Add uncertainty in domain-agnostic way"""
        sentences = text.split('.')
        modified_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Universal uncertainty injection (not domain-specific words)
                if random.choice([True, False]):
                    sentence = "It appears that " + sentence.lower()
                else:
                    sentence = sentence + " (though this may vary)"
                modified_sentences.append(sentence)
        
        return '. '.join(modified_sentences)
    
    def negate_claims(self, text: str) -> str:
        """Negate key factual claims (universal logical operation)"""
        # Simple negation by adding "not" to verb phrases
        words = text.split()
        negated_words = []
        
        for i, word in enumerate(words):
            if word.lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'can', 'does']:
                negated_words.append(word + " not")
            else:
                negated_words.append(word)
        
        return ' '.join(negated_words)
    
    def add_universal_hedging(self, text: str) -> str:
        """Add hedging language (universal across domains)"""
        hedged = "Based on available information, " + text.lower()
        hedged += " However, this should be verified independently."
        return hedged
    
    def paraphrase_neutrally(self, text: str) -> str:
        """Simple neutral paraphrasing"""
        # Basic sentence restructuring
        sentences = text.split('.')
        paraphrased = []
        
        for sentence in sentences:
            if sentence.strip():
                # Simple restructuring: move clauses around
                words = sentence.strip().split()
                if len(words) > 3:
                    mid = len(words) // 2
                    restructured = ' '.join(words[mid:] + words[:mid])
                    paraphrased.append(restructured)
                else:
                    paraphrased.append(sentence.strip())
        
        return '. '.join(paraphrased)
    
    def calculate_semantic_uncertainty(self, original: str, variants: list) -> float:
        """
        Core physics calculation: ℏₛ = √(Δμ × Δσ)
        Domain-agnostic implementation
        """
        
        # Convert texts to probability distributions (character-level for universality)
        def text_to_distribution(text: str) -> np.ndarray:
            # Character-level distribution (universal across domains)
            chars = list(text.lower())
            char_counts = Counter(chars)
            total_chars = len(chars)
            
            # Create probability distribution
            all_chars = set(''.join([original] + variants))
            distribution = []
            for char in sorted(all_chars):
                prob = char_counts.get(char, 0) / total_chars
                distribution.append(prob)
            
            return np.array(distribution) + 1e-10  # Avoid zero probabilities
        
        # Get distributions
        original_dist = text_to_distribution(original)
        
        # Calculate Jensen-Shannon and KL divergences with variants
        js_divergences = []
        kl_divergences = []
        
        for variant in variants:
            variant_dist = text_to_distribution(variant)
            
            # Ensure same length
            min_len = min(len(original_dist), len(variant_dist))
            orig_sub = original_dist[:min_len]
            var_sub = variant_dist[:min_len]
            
            # Normalize
            orig_sub = orig_sub / np.sum(orig_sub)
            var_sub = var_sub / np.sum(var_sub)
            
            # Calculate divergences
            js_div = jensenshannon(orig_sub, var_sub) ** 2  # Squared JS distance
            kl_div = entropy(orig_sub, var_sub)
            
            js_divergences.append(js_div)
            kl_divergences.append(kl_div)
        
        # Physics calculation
        delta_mu = np.mean(js_divergences)  # Precision measure
        delta_sigma = np.mean(kl_divergences)  # Flexibility measure
        
        # Core equation: ℏₛ = √(Δμ × Δσ)
        semantic_uncertainty = math.sqrt(delta_mu * delta_sigma)
        
        return semantic_uncertainty
    
    def calculate_information_density(self, text: str) -> float:
        """Universal information density measure"""
        
        # Character-level entropy (universal)
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        probabilities = [count / total_chars for count in char_counts.values()]
        char_entropy = entropy(probabilities, base=2)
        
        # Word-level entropy
        words = text.split()
        word_counts = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
            
        word_probs = [count / total_words for count in word_counts.values()]
        word_entropy = entropy(word_probs, base=2)
        
        # Combined information density
        information_density = (char_entropy + word_entropy) / 2
        
        return information_density
    
    def measure_logical_consistency(self, prompt: str, output: str) -> float:
        """Measure logical consistency (universal)"""
        
        # Simple consistency checks (domain-agnostic)
        
        # 1. Length consistency (very long outputs for short prompts might be hallucinated)
        prompt_len = len(prompt.split())
        output_len = len(output.split())
        length_ratio = output_len / max(prompt_len, 1)
        length_consistency = 1.0 / (1.0 + abs(math.log(max(length_ratio, 0.1))))
        
        # 2. Repetition check (excessive repetition indicates issues)
        words = output.lower().split()
        if len(words) == 0:
            repetition_score = 1.0
        else:
            unique_words = len(set(words))
            repetition_score = unique_words / len(words)
        
        # 3. Punctuation balance (excessive punctuation might indicate issues)
        punct_count = len(re.findall(r'[!?.;,:]', output))
        word_count = len(output.split())
        punct_ratio = punct_count / max(word_count, 1)
        punct_consistency = 1.0 / (1.0 + punct_ratio * 10)  # Penalize excessive punctuation
        
        # Combined consistency score
        logical_consistency = (length_consistency + repetition_score + punct_consistency) / 3
        
        return logical_consistency
    
    def assess_factual_grounding(self, text: str) -> float:
        """Assess factual grounding using universal metrics"""
        
        # 1. Specificity measure (specific facts vs vague statements)
        numbers = len(re.findall(r'\b\d+\b', text))
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        dates = len(re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
        
        specificity_indicators = numbers + proper_nouns + dates
        word_count = len(text.split())
        specificity_score = specificity_indicators / max(word_count, 1)
        
        # 2. Qualification measure (qualified statements vs absolute claims)
        qualifiers = len(re.findall(r'\b(some|many|often|usually|generally|typically|approximately|roughly)\b', text.lower()))
        absolutes = len(re.findall(r'\b(all|every|never|always|impossible|certainly|definitely)\b', text.lower()))
        
        if qualifiers + absolutes == 0:
            qualification_balance = 0.5
        else:
            qualification_balance = qualifiers / (qualifiers + absolutes)
        
        # 3. Source attribution (implicit grounding indicators)
        attribution_indicators = len(re.findall(r'\b(according to|research shows|studies indicate|evidence suggests)\b', text.lower()))
        attribution_score = min(attribution_indicators / max(word_count, 1) * 100, 1.0)
        
        # Combined grounding score
        factual_grounding = (specificity_score + qualification_balance + attribution_score) / 3
        
        return factual_grounding
    
    def measure_semantic_coherence(self, text: str) -> float:
        """Measure semantic coherence (universal)"""
        
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by definition
        
        # Measure vocabulary consistency across sentences
        sentence_vocabs = []
        for sentence in sentences:
            if sentence.strip():
                words = set(sentence.lower().split())
                sentence_vocabs.append(words)
        
        if len(sentence_vocabs) < 2:
            return 1.0
        
        # Calculate overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentence_vocabs) - 1):
            vocab1 = sentence_vocabs[i]
            vocab2 = sentence_vocabs[i + 1]
            
            if len(vocab1) == 0 or len(vocab2) == 0:
                overlap = 0.0
            else:
                intersection = len(vocab1 & vocab2)
                union = len(vocab1 | vocab2)
                overlap = intersection / union
            
            overlaps.append(overlap)
        
        # Average semantic overlap (coherence measure)
        semantic_coherence = np.mean(overlaps) if overlaps else 1.0
        
        return semantic_coherence
    
    def calculate_semantic_complexity(self, text: str) -> float:
        """Calculate semantic complexity (universal measure)"""
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # 1. Lexical diversity
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words)
        
        # 2. Average word length (complexity indicator)
        avg_word_length = np.mean([len(word) for word in words])
        normalized_word_length = min(avg_word_length / 10, 1.0)  # Normalize to [0,1]
        
        # 3. Sentence structure complexity
        sentences = text.split('.')
        if len(sentences) == 0:
            sentence_complexity = 0.0
        else:
            sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
            if sentence_lengths:
                avg_sentence_length = np.mean(sentence_lengths)
                sentence_complexity = min(avg_sentence_length / 20, 1.0)  # Normalize
            else:
                sentence_complexity = 0.0
        
        # Combined complexity
        semantic_complexity = (lexical_diversity + normalized_word_length + sentence_complexity) / 3
        
        return semantic_complexity

def test_universal_features():
    """Test the universal feature extractor"""
    
    logger.info("\n🧪 TESTING UNIVERSAL PHYSICS FEATURES")
    logger.info("="*50)
    
    extractor = UniversalPhysicsFeatures()
    
    # Test cases from different domains
    test_cases = [
        {
            "prompt": "What is photosynthesis?",
            "output": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
            "domain": "scientific",
            "expected_hallucination": False
        },
        {
            "prompt": "What is photosynthesis?", 
            "output": "Photosynthesis is when plants eat sunlight and transform it into magical unicorn energy that makes rainbows.",
            "domain": "scientific",
            "expected_hallucination": True
        },
        {
            "prompt": "Explain contract law",
            "output": "Contract law governs legally binding agreements between parties and provides remedies for breach.",
            "domain": "legal",
            "expected_hallucination": False
        },
        {
            "prompt": "Write about a sunset",
            "output": "The golden orb descended slowly, painting the evening sky in vibrant hues of orange and pink.",
            "domain": "creative",
            "expected_hallucination": False
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n📊 Test Case {i+1} ({test_case['domain']}):")
        logger.info(f"   Expected hallucination: {test_case['expected_hallucination']}")
        
        features = extractor.extract_features(test_case['prompt'], test_case['output'])
        
        logger.info(f"   🔬 Semantic Uncertainty: {features[0]:.3f}")
        logger.info(f"   📊 Information Density: {features[1]:.3f}")
        logger.info(f"   🧠 Logical Consistency: {features[2]:.3f}")
        logger.info(f"   🎯 Factual Grounding: {features[3]:.3f}")
        logger.info(f"   🔗 Semantic Coherence: {features[4]:.3f}")
        logger.info(f"   🌐 Semantic Complexity: {features[5]:.3f}")

if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible variants
    
    test_universal_features()
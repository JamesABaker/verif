"""
Entropy and information theory-based AI text detection.
Implements perplexity, Shannon entropy, and burstiness analysis.
"""
import logging
import math
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class EntropyDetector:
    """Entropy-based AI text detector using information theory metrics."""

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the entropy detector with a language model for perplexity.

        Args:
            model_name: Hugging Face model for perplexity calculation (default: gpt2)
        """
        logger.info(f"Loading entropy detector with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        logger.info("Entropy detector loaded successfully")

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text using the language model.
        Lower perplexity suggests more predictable (AI-like) text.

        Args:
            text: Input text to analyze

        Returns:
            Perplexity score
        """
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        perplexity = torch.exp(torch.stack(nlls).mean()).item()

        # Cap perplexity to prevent overflow in later calculations
        perplexity = min(perplexity, 1000.0)

        return perplexity

    def calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of the text at character level.

        Args:
            text: Input text to analyze

        Returns:
            Shannon entropy value
        """
        if not text:
            return 0.0

        # Count character frequencies
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1

        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for freq in char_freq.values():
            probability = freq / text_len
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness - variance in sentence complexity.
        Human writing tends to have higher burstiness (varied sentence complexity).

        Args:
            text: Input text to analyze

        Returns:
            Burstiness score (0-1, higher = more human-like variation)
        """
        # Split into sentences (simple approach)
        sentences = [
            s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()
        ]

        if len(sentences) < 2:
            return 0.0

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]

        # Calculate coefficient of variation (std/mean)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        if mean_length == 0:
            return 0.0

        burstiness = std_length / mean_length
        # Normalize to 0-1 range (typical values are 0-2)
        return min(burstiness / 2.0, 1.0)

    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Calculate lexical diversity (type-token ratio).
        Higher diversity often indicates human writing.

        Args:
            text: Input text to analyze

        Returns:
            Lexical diversity score (0-1)
        """
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def calculate_word_length_variance(self, text: str) -> float:
        """
        Calculate variance in word lengths.
        AI tends to use more uniform word lengths.

        Args:
            text: Input text to analyze

        Returns:
            Normalized variance score
        """
        words = text.split()
        if len(words) < 2:
            return 0.0

        lengths = [len(word) for word in words]
        variance = np.var(lengths)
        # Normalize (typical variance is 0-20)
        return min(variance / 20.0, 1.0)

    def calculate_punctuation_diversity(self, text: str) -> float:
        """
        Calculate diversity of punctuation usage.
        Humans use more varied punctuation.

        Args:
            text: Input text to analyze

        Returns:
            Punctuation diversity score (0-1)
        """
        import string

        punct_marks = [c for c in text if c in string.punctuation]
        if not punct_marks:
            return 0.0

        unique_punct = set(punct_marks)
        # Normalize by common punctuation count (.,!?;:)
        return min(len(unique_punct) / 6.0, 1.0)

    def calculate_vocabulary_richness(self, text: str) -> float:
        """
        Calculate Yule's K measure of vocabulary richness.
        Lower K = more diverse vocabulary (more human-like).

        Args:
            text: Input text to analyze

        Returns:
            Normalized richness score
        """
        words = text.lower().split()
        if len(words) < 10:
            return 0.5

        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Calculate frequency of frequencies
        freq_freq = {}
        for freq in word_freq.values():
            freq_freq[freq] = freq_freq.get(freq, 0) + 1

        # Yule's K formula
        M1 = len(words)
        M2 = sum([freq * freq * count for freq, count in freq_freq.items()])

        if M1 == 0 or M1 == M2:
            return 0.5

        K = 10000 * (M2 - M1) / (M1 * M1)
        # Lower K is better (more diverse), normalize to 0-1
        # Typical K ranges: 50-300, invert so higher = more diverse
        return max(0, min(1.0, 1.0 - (K / 300.0)))

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive entropy-based detection.
        Detects statistical extremes - text that is either too perfect/uniform OR too random.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with entropy metrics and AI probability
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Calculate all metrics
        perplexity = self.calculate_perplexity(text)
        shannon_entropy = self.calculate_shannon_entropy(text)
        burstiness = self.calculate_burstiness(text)
        lexical_diversity = self.calculate_lexical_diversity(text)
        word_length_var = self.calculate_word_length_variance(text)
        punct_diversity = self.calculate_punctuation_diversity(text)
        vocab_richness = self.calculate_vocabulary_richness(text)

        # Define "normal human" ranges - flag if outside these bounds
        # Each metric returns a score where 0 = within human range, 1 = outside (suspicious)

        # Perplexity: Human typical range 50-200
        # Too low (< 30) = too perfect/predictable, Too high (> 250) = too random
        if perplexity < 30:
            perplexity_anomaly = (30 - perplexity) / 30  # Score how far below threshold
        elif perplexity > 250:
            perplexity_anomaly = min((perplexity - 250) / 250, 1.0)  # Score how far above
        else:
            perplexity_anomaly = 0.0  # Within normal range

        # Shannon entropy: Human typical range 3.5-4.8
        # Too low = repetitive, Too high = random noise
        if shannon_entropy < 3.5:
            entropy_anomaly = (3.5 - shannon_entropy) / 3.5
        elif shannon_entropy > 4.8:
            entropy_anomaly = min((shannon_entropy - 4.8) / 4.8, 1.0)
        else:
            entropy_anomaly = 0.0

        # Burstiness: Human typical range 0.3-0.8
        # Too low = too uniform, Too high = chaotic
        if burstiness < 0.3:
            burstiness_anomaly = (0.3 - burstiness) / 0.3
        elif burstiness > 0.8:
            burstiness_anomaly = (burstiness - 0.8) / 0.2
        else:
            burstiness_anomaly = 0.0

        # Lexical diversity: Human typical range 0.4-0.8
        # Too low = repetitive, Too high = forced variation
        if lexical_diversity < 0.4:
            diversity_anomaly = (0.4 - lexical_diversity) / 0.4
        elif lexical_diversity > 0.8:
            diversity_anomaly = (lexical_diversity - 0.8) / 0.2
        else:
            diversity_anomaly = 0.0

        # Word length variance: Human typical range 0.3-0.7
        if word_length_var < 0.3:
            word_var_anomaly = (0.3 - word_length_var) / 0.3
        elif word_length_var > 0.7:
            word_var_anomaly = (word_length_var - 0.7) / 0.3
        else:
            word_var_anomaly = 0.0

        # Punctuation diversity: Human typical range 0.3-0.7
        if punct_diversity < 0.3:
            punct_anomaly = (0.3 - punct_diversity) / 0.3
        elif punct_diversity > 0.7:
            punct_anomaly = (punct_diversity - 0.7) / 0.3
        else:
            punct_anomaly = 0.0

        # Vocabulary richness: Human typical range 0.4-0.8
        if vocab_richness < 0.4:
            vocab_anomaly = (0.4 - vocab_richness) / 0.4
        elif vocab_richness > 0.8:
            vocab_anomaly = (vocab_richness - 0.8) / 0.2
        else:
            vocab_anomaly = 0.0

        # Aggregate anomaly score - weighted by metric reliability
        ai_probability = (
            0.30 * perplexity_anomaly  # Most reliable for detecting extremes
            + 0.20 * burstiness_anomaly  # Sentence variation extremes
            + 0.15 * vocab_anomaly  # Vocabulary patterns
            + 0.15 * diversity_anomaly  # Lexical diversity extremes
            + 0.10 * entropy_anomaly  # Character entropy
            + 0.05 * word_var_anomaly  # Word length patterns
            + 0.05 * punct_anomaly  # Punctuation patterns
        )

        # Cap at 1.0
        ai_probability = min(ai_probability, 1.0)

        return {
            "perplexity": round(perplexity, 2),
            "shannon_entropy": round(shannon_entropy, 3),
            "burstiness": round(burstiness, 3),
            "lexical_diversity": round(lexical_diversity, 3),
            "word_length_variance": round(word_length_var, 3),
            "punctuation_diversity": round(punct_diversity, 3),
            "vocabulary_richness": round(vocab_richness, 3),
            "ai_probability_entropy": round(ai_probability * 100, 2),
            "human_probability_entropy": round((1 - ai_probability) * 100, 2),
        }

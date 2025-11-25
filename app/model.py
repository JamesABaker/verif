"""
Hybrid AI text detection combining ML model with entropy-based features.
Uses RoBERTa classifier + information theory metrics for robust detection.
"""
import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.entropy_detector import EntropyDetector

logger = logging.getLogger(__name__)


class AIDetector:
    """Dual-pathway AI detector: RoBERTa + Entropy analysis."""

    def __init__(self, model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"):
        """
        Initialize the dual-pathway AI detector.

        Args:
            model_name: Hugging Face model identifier for RoBERTa classifier
        """
        logger.info(f"Loading ML model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        logger.info("ML model loaded successfully")

        logger.info("Initializing entropy detector...")
        self.entropy_detector = EntropyDetector()
        logger.info("Dual-pathway detector ready")

    def detect(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Dual-pathway detection: RoBERTa for AI patterns + Entropy for statistical extremes.
        Flags as AI if EITHER pathway exceeds its threshold.

        Args:
            text: Input text to analyze
            max_length: Maximum token length (default 512)

        Returns:
            Dictionary with both pathway scores and detection results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # PATHWAY 1: RoBERTa - Detects AI-like linguistic patterns
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ml_human_prob = probs[0][0].item() * 100
            ml_ai_prob = probs[0][1].item() * 100

        # PATHWAY 2: Entropy - Detects statistical extremes (too perfect OR too random)
        entropy_results = self.entropy_detector.detect(text)

        # Independent thresholds for each pathway
        ROBERTA_THRESHOLD = 50.0  # Flag if RoBERTa AI probability > 50%

        # Determine which pathways triggered
        roberta_triggered = ml_ai_prob > ROBERTA_THRESHOLD
        entropy_triggered = (
            entropy_results["human_probability_entropy"] > 90.0
        )  # If human prob > 90% = too perfect

        # Dual-gate logic: Flag as AI if EITHER pathway exceeds threshold
        if roberta_triggered or entropy_triggered:
            prediction = "ai"
            # Use the higher confidence score for final probability
            ai_probability = max(ml_ai_prob, entropy_results["ai_probability_entropy"])
        else:
            prediction = "human"
            # Use the lower AI probability (higher human confidence)
            ai_probability = min(ml_ai_prob, entropy_results["ai_probability_entropy"])

        human_probability = 100 - ai_probability

        return {
            # Final dual-gate results
            "human_probability": round(human_probability, 2),
            "ai_probability": round(ai_probability, 2),
            "prediction": prediction,
            "roberta_triggered": roberta_triggered,
            "entropy_triggered": entropy_triggered,
            # Pathway 1: RoBERTa scores
            "ml_human_probability": round(ml_human_prob, 2),
            "ml_ai_probability": round(ml_ai_prob, 2),
            # Pathway 2: Entropy metrics
            "perplexity": entropy_results["perplexity"],
            "shannon_entropy": entropy_results["shannon_entropy"],
            "burstiness": entropy_results["burstiness"],
            "lexical_diversity": entropy_results["lexical_diversity"],
            "word_length_variance": entropy_results["word_length_variance"],
            "punctuation_diversity": entropy_results["punctuation_diversity"],
            "vocabulary_richness": entropy_results["vocabulary_richness"],
            "entropy_ai_probability": entropy_results["ai_probability_entropy"],
            "entropy_human_probability": entropy_results["human_probability_entropy"],
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the dual-pathway detector."""
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "architecture": "Dual-Pathway: RoBERTa + Entropy Analysis",
            "pathway_1": "RoBERTa - AI linguistic pattern detection",
            "pathway_2": "Entropy - Statistical extremes detection",
            "detection_logic": "Flag as AI if EITHER pathway exceeds threshold",
            "entropy_features": [
                "perplexity",
                "shannon_entropy",
                "burstiness",
                "lexical_diversity",
            ],
            "max_length": 512,
            "labels": {"0": "Human-written", "1": "AI-generated"},
        }
        return info

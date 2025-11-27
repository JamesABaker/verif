"""
Extract entropy + RoBERTa features from HC3 dataset for Joseph model training.

This script:
1. Loads HC3 dataset (human vs ChatGPT text pairs)
2. Extracts 7 entropy features + RoBERTa probability for each sample
3. Saves as train/val/test parquet files (70/15/15 split)
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.ml_model import AIDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_features(texts, labels, detector, desc="Processing"):
    """
    Extract features from texts using entropy detector and RoBERTa.

    Args:
        texts: List of text samples
        labels: List of labels (0=human, 1=ai)
        detector: AIDetector instance
        desc: Progress bar description

    Returns:
        DataFrame with features and labels
    """
    features_list = []

    for text, label in tqdm(zip(texts, labels), total=len(texts), desc=desc):
        try:
            # Get full detection results (includes entropy + RoBERTa)
            result = detector.detect(text)

            # Extract the 8 features we want
            features = {
                "perplexity": result["perplexity"],
                "shannon_entropy": result["shannon_entropy"],
                "burstiness": result["burstiness"],
                "lexical_diversity": result["lexical_diversity"],
                "word_length_variance": result["word_length_variance"],
                "punctuation_diversity": result["punctuation_diversity"],
                "vocabulary_richness": result["vocabulary_richness"],
                "roberta_ai_prob": result["ml_ai_probability"],  # RoBERTa as 8th feature
                "label": label,  # 0=human, 1=ai
            }
            features_list.append(features)

        except Exception as e:
            logger.warning(f"Failed to process sample (label={label}): {e}")
            continue

    return pd.DataFrame(features_list)


def main():
    """Main feature extraction pipeline."""
    logger.info("Starting feature extraction from HC3 dataset")

    # Load HC3 dataset
    logger.info("Loading HC3 dataset...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")

    # Extract train split (we'll do our own splitting)
    hc3_train = dataset["train"]
    logger.info(f"Loaded {len(hc3_train)} samples from HC3")

    # Prepare texts and labels
    # HC3 format: each row has 'question', 'human_answers', 'chatgpt_answers'
    texts = []
    labels = []

    logger.info("Extracting human and AI texts from dataset...")
    for sample in tqdm(hc3_train, desc="Parsing HC3"):
        # Human answers
        for human_text in sample["human_answers"]:
            if human_text and len(human_text.strip()) > 50:  # Filter too-short samples
                texts.append(human_text)
                labels.append(0)  # 0 = human

        # ChatGPT answers
        for chatgpt_text in sample["chatgpt_answers"]:
            if chatgpt_text and len(chatgpt_text.strip()) > 50:
                texts.append(chatgpt_text)
                labels.append(1)  # 1 = ai

    logger.info(f"Extracted {len(texts)} total samples (human + AI)")
    logger.info(f"Human samples: {sum(1 for label in labels if label == 0)}")
    logger.info(f"AI samples: {sum(1 for label in labels if label == 1)}")

    # Train/val/test split: 70/15/15
    logger.info("Splitting dataset: 70% train, 15% val, 15% test")
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    logger.info(f"Train: {len(train_texts)} samples")
    logger.info(f"Val: {len(val_texts)} samples")
    logger.info(f"Test: {len(test_texts)} samples")

    # Initialize detector
    logger.info("Initializing AIDetector (this will download models on first run)...")
    detector = AIDetector()

    # Extract features for each split
    logger.info("Extracting features from training set...")
    train_df = extract_features(train_texts, train_labels, detector, "Train set")

    logger.info("Extracting features from validation set...")
    val_df = extract_features(val_texts, val_labels, detector, "Val set")

    logger.info("Extracting features from test set...")
    test_df = extract_features(test_texts, test_labels, detector, "Test set")

    # Save to parquet
    output_dir = Path(__file__).parent.parent / "data" / "joseph_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    logger.info(f"Saving training features to {train_path}")
    train_df.to_parquet(train_path, index=False)

    logger.info(f"Saving validation features to {val_path}")
    val_df.to_parquet(val_path, index=False)

    logger.info(f"Saving test features to {test_path}")
    test_df.to_parquet(test_path, index=False)

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    logger.info(f"\nFeatures extracted: {list(train_df.columns)}")
    logger.info(f"\nFiles saved to: {output_dir}")
    logger.info("\nNext step: Run scripts/train_joseph_model.py")


if __name__ == "__main__":
    main()

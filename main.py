"""
Main script for spam email classification using Decision Tree.
This script orchestrates the entire machine learning pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt

import numpy as np
from sklearn.exceptions import NotFittedError
from data_loader import DataLoader
from model_trainer import SpamClassifier
from visualizer import Visualizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Setup logging with file handler
def setup_logging(log_file: str = "spam_classifier.log") -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on reruns
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def validate_data(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X is None or y is None:
        raise ValueError("Input data cannot be None")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) == 0:
        raise ValueError("Input data cannot be empty")


def create_output_directory(output_dir: str = "output") -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None
) -> SpamClassifier:
    """Train the spam classification model."""
    classifier = SpamClassifier()
    try:
        classifier.train(X_train, y_train, feature_names=feature_names)
        return classifier
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")


def evaluate_and_analyze(
    classifier: SpamClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Perform model evaluation and analysis."""
    try:
        results = classifier.evaluate(X_test, y_test, detailed=True)
        cv_results = classifier.cross_validate(X_train, y_train, cv=5)
        return results, cv_results
    except NotFittedError:
        logger.error("Model not fitted before evaluation")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def generate_visualizations(
    visualizer: Visualizer,
    data_loader: DataLoader,
    classifier: SpamClassifier,
    results: Dict[str, Any],
    feature_importance: list,
    y_test: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate and save all visualizations."""
    try:
        # Plot and save class distribution
        visualizer.plot_class_distribution(
            data_loader.data[data_loader.target_column],
            title="Spam vs Ham Distribution",
        )
        plt.savefig(output_dir / "class_distribution.png")

        # Plot and save other visualizations
        visualizer.plot_model_performance(results)
        plt.savefig(output_dir / "model_performance.png")

        visualizer.plot_confusion_matrix(
            y_test,
            results["predictions"],
            class_names=classifier.class_names,
        )
        plt.savefig(output_dir / "confusion_matrix.png")

        visualizer.plot_feature_importance(feature_importance, top_n=15)
        plt.savefig(output_dir / "feature_importance.png")

        visualizer.plot_decision_tree(
            classifier.model,
            feature_names=None,
            class_names=classifier.class_names,
            max_depth=3,
        )
        plt.savefig(output_dir / "decision_tree.png")

    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")
        raise


def main():
    """Main execution function."""
    logger = setup_logging()
    output_dir = create_output_directory()

    try:
        logger.info("Starting Spam Email Classification Pipeline")

        # Step 1: Data Loading and Preprocessing
        logger.info("Step 1: Loading and preprocessing data...")
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.prepare_data()
        

        # ✅ Clean target labels with <2 samples
        unique, counts = np.unique(np.concatenate([y_train, y_test]), return_counts=True)
        rare_classes = unique[counts < 2]
        if len(rare_classes) > 0:
            logger.warning(f"Removing rare classes with <2 samples: {rare_classes}")
            mask_train = ~np.isin(y_train, rare_classes)
            mask_test = ~np.isin(y_test, rare_classes)
            X_train, y_train = X_train[mask_train], y_train[mask_train]
            X_test, y_test = X_test[mask_test], y_test[mask_test]

        # Validate data
        validate_data(X_train, y_train)
        validate_data(X_test, y_test)

        # Get feature names
        feature_names = (
            data_loader.vectorizer.get_feature_names_out()
            if hasattr(data_loader, "vectorizer")
            and data_loader.vectorizer is not None
            else None
        )

        # Step 2: Model Training
        logger.info("Step 2: Training model...")
        classifier = train_model(X_train, y_train, feature_names)

        # Step 3: Evaluation and Analysis
        logger.info("Step 3: Evaluating model...")
        results, cv_results = evaluate_and_analyze(
            classifier, X_train, X_test, y_train, y_test, logger
        )

        # Step 4: Feature Importance
        logger.info("Step 4: Analyzing feature importance...")
        feature_importance = classifier.get_feature_importance(top_n=20)

        # Step 5: Visualizations
        logger.info("Step 5: Generating visualizations...")
        visualizer = Visualizer()
        generate_visualizations(
            visualizer,
            data_loader,
            classifier,
            results,
            feature_importance,
            y_test,
            output_dir,
            logger,
        )

        # Step 6: Save Model
        logger.info("Step 6: Saving model...")
        model_path = output_dir / "spam_classifier_model.pkl"
        classifier.save_model(str(model_path))

        # Print Summary Report
        print_summary_report(results, cv_results, feature_importance[:5])

        logger.info("Pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


def print_summary_report(
    results: Dict[str, float], cv_results: Dict[str, float], top_features: list
) -> None:
    """Print a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("SPAM EMAIL CLASSIFICATION - SUMMARY REPORT")
    print("=" * 60)
    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"\nCross-validation Results:")
    print(f"Mean CV Accuracy: {cv_results['mean']:.4f}")
    print(f"Std CV Accuracy: {cv_results['std']:.4f}")
    print(f"\nTop {len(top_features)} Important Features:")
    for feature, importance in top_features:
        print(f"- {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()

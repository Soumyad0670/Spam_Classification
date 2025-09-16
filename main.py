"""
Main script for spam email classification using Decision Tree.
This script orchestrates the entire machine learning pipeline.
"""

import logging
from data_loader import DataLoader
from model_trainer import SpamClassifier
from visualizer import Visualizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    try:
        logger.info("Starting Spam Email Classification Pipeline")
        
        # Step 1: Data Loading and Preprocessing
        logger.info("=" * 50)
        logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        logger.info("=" * 50)
        
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.prepare_data()
        
        # Get feature names from vectorizer
        feature_names = data_loader.vectorizer.get_feature_names_out()
        
        # Step 2: Model Training
        logger.info("=" * 50)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 50)
        
        classifier = SpamClassifier()
        classifier.train(X_train, y_train, feature_names=feature_names)
        
        # Step 3: Model Evaluation
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("=" * 50)
        
        results = classifier.evaluate(X_test, y_test, detailed=True)
        
        # Step 4: Cross-validation
        logger.info("=" * 50)
        logger.info("STEP 4: CROSS-VALIDATION")
        logger.info("=" * 50)
        
        # Use training data for cross-validation
        cv_results = classifier.cross_validate(X_train, y_train, cv=5)
        
        # Step 5: Feature Importance Analysis
        logger.info("=" * 50)
        logger.info("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 50)
        
        feature_importance = classifier.get_feature_importance(top_n=20)
        
        # Step 6: Visualizations
        logger.info("=" * 50)
        logger.info("STEP 6: GENERATING VISUALIZATIONS")
        logger.info("=" * 50)
        
        visualizer = Visualizer()
        
        # Plot class distribution
        visualizer.plot_class_distribution(
            data_loader.data[data_loader.target_column],
            title="Spam vs Ham Distribution"
        )
        
        # Plot model performance
        visualizer.plot_model_performance(results)
        
        # Plot confusion matrix
        visualizer.plot_confusion_matrix(
            y_test, results['predictions'],
            class_names=classifier.class_names
        )
        
        # Plot feature importance
        visualizer.plot_feature_importance(feature_importance, top_n=15)
        
        # Plot decision tree (simplified version)
        visualizer.plot_decision_tree(
            classifier.model,
            feature_names=None,  # Too many features to display
            class_names=classifier.class_names,
            max_depth=3
        )
        
        # Step 7: Summary Report
        logger.info("=" * 50)
        logger.info("STEP 7: SUMMARY REPORT")
        logger.info("=" * 50)
        
        print_summary_report(results, cv_results, feature_importance[:5])
        
        # Step 8: Save Model (Optional)
        logger.info("=" * 50)
        logger.info("STEP 8: SAVING MODEL")
        logger.info("=" * 50)
        
        try:
            classifier.save_model("spam_classifier_model.pkl")
            logger.info("Model saved successfully!")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


def print_summary_report(results, cv_results, top_features):
    """Print a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("SPAM EMAIL CLASSIFICATION - SUMMARY REPORT")
    print("=" * 60)
    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"\nCross-validation Results:")
    print(f"Mean CV Accuracy: {cv_results['mean']:.4f}")
    print(f"Std CV Accuracy: {cv_results['std']:.4f}")
    print(f"\nTop {len(top_features)} Important Features:")
    for feature, importance in top_features:
        print(f"- {feature}: {importance:.4f}")
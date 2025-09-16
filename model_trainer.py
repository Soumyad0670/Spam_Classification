"""
Model training and evaluation utilities for spam email classification.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import logging
from config import MODEL_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpamClassifier:
    """Spam email classifier using Decision Tree."""
    
    def __init__(self, **kwargs):
        """Initialize classifier with custom or default parameters."""
        # Merge custom parameters with default config
        params = {**MODEL_CONFIG, **kwargs}
        
        self.model = DecisionTreeClassifier(**params)
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
    
    def train(self, X_train, y_train, feature_names=None):
        """Train the decision tree model."""
        try:
            logger.info("Training decision tree classifier...")
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Store feature names for visualization
            if hasattr(X_train, 'shape'):
                if feature_names is None:
                    self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                else:
                    self.feature_names = feature_names
            
            # Store class names
            self.class_names = [str(cls) for cls in sorted(np.unique(y_train))]
            
            logger.info("Model training completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X_test):
        """Make predictions on test data."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            predictions = self.model.predict(X_test)
            logger.info(f"Predictions made for {len(predictions)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            probabilities = self.model.predict_proba(X_test)
            return probabilities
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}")
            raise
    
    def evaluate(self, X_test, y_test, detailed=True):
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            results = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            logger.info(f"Model Accuracy: {accuracy:.4f}")
            
            if detailed:
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                results['classification_report'] = class_report
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                results['confusion_matrix'] = conf_matrix
                
                # Print detailed results
                logger.info("Classification Report:")
                print(classification_report(y_test, y_pred))
                
                logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        try:
            logger.info(f"Performing {cv}-fold cross-validation...")
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"Cross-validation scores: {scores}")
            logger.info(f"Mean CV Accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            
            return {
                'scores': scores,
                'mean': mean_score,
                'std': std_score
            }
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            raise
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            importances = self.model.feature_importances_
            
            if self.feature_names:
                feature_importance = list(zip(self.feature_names, importances))
                # Sort by importance
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"Top {top_n} most important features:")
                for i, (feature, importance) in enumerate(feature_importance[:top_n]):
                    logger.info(f"{i+1}. {feature}: {importance:.4f}")
                
                return feature_importance[:top_n]
            else:
                return importances
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            import joblib
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath):
        """Load a trained model."""
        try:
            import joblib
            self.model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
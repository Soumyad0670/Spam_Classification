"""
Visualization utilities for spam email classification project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
from config import VIZ_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """Handles all visualization tasks for the spam classification project."""
    
    @staticmethod
    def plot_decision_tree(model, feature_names=None, class_names=None, 
                          max_depth=None, save_path=None):
        """Visualize the decision tree."""
        try:
            if max_depth is None:
                max_depth = VIZ_CONFIG['max_depth_viz']
            
            plt.figure(figsize=VIZ_CONFIG['figsize'])
            
            # Create a simplified tree for visualization if original is too deep
            if hasattr(model, 'tree_') and model.tree_.max_depth > max_depth:
                logger.warning(f"Tree depth ({model.tree_.max_depth}) > max_depth ({max_depth}). "
                             f"Showing simplified version.")
            
            plot_tree(model, 
                     filled=True,
                     feature_names=feature_names,
                     class_names=class_names,
                     max_depth=max_depth,
                     fontsize=VIZ_CONFIG['fontsize'],
                     rounded=True,
                     proportion=True)
            
            plt.title("Decision Tree for Spam Email Classification", 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Decision tree plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting decision tree: {e}")
            raise
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                            normalize=False, save_path=None):
        """Plot confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
                title = 'Normalized Confusion Matrix'
            else:
                fmt = 'd'
                title = 'Confusion Matrix'
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            raise
    
    @staticmethod
    def plot_feature_importance(feature_importance, top_n=20, save_path=None):
        """Plot feature importance."""
        try:
            # Take top N features
            features = feature_importance[:top_n]
            feature_names = [f[0] for f in features]
            importances = [f[1] for f in features]
            
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(feature_names))
            
            bars = plt.barh(y_pos, importances, color='skyblue', alpha=0.8)
            plt.yticks(y_pos, feature_names)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{importance:.4f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise
    
    @staticmethod
    def plot_class_distribution(y, title="Class Distribution", save_path=None):
        """Plot target class distribution."""
        try:
            plt.figure(figsize=(8, 6))
            
            counts = y.value_counts()
            colors = sns.color_palette("husl", len(counts))
            
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.axis('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Class distribution plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting class distribution: {e}")
            raise
    
    @staticmethod
    def plot_model_performance(results, save_path=None):
        """Plot comprehensive model performance metrics."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Accuracy score
            axes[0, 0].bar(['Accuracy'], [results['accuracy']], color='lightgreen', alpha=0.7)
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_title('Model Accuracy', fontweight='bold')
            axes[0, 0].text(0, results['accuracy'] + 0.02, f"{results['accuracy']:.4f}",
                           ha='center', va='bottom', fontweight='bold')
            
            # 2. Precision, Recall, F1-score
            if 'classification_report' in results:
                metrics = results['classification_report']
                classes = [k for k in metrics.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                
                precision = [metrics[cls]['precision'] for cls in classes]
                recall = [metrics[cls]['recall'] for cls in classes]
                f1 = [metrics[cls]['f1-score'] for cls in classes]
                
                x = np.arange(len(classes))
                width = 0.25
                
                axes[0, 1].bar(x - width, precision, width, label='Precision', alpha=0.8)
                axes[0, 1].bar(x, recall, width, label='Recall', alpha=0.8)
                axes[0, 1].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
                
                axes[0, 1].set_xlabel('Classes')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_title('Precision, Recall, F1-Score by Class', fontweight='bold')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(classes)
                axes[0, 1].legend()
                axes[0, 1].set_ylim(0, 1)
            
            # 3. Confusion Matrix
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
                axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
                axes[1, 0].set_xlabel('Predicted')
                axes[1, 0].set_ylabel('Actual')
            
            # 4. Class-wise metrics
            if 'classification_report' in results:
                metrics_data = []
                for cls in classes:
                    metrics_data.append([
                        metrics[cls]['precision'],
                        metrics[cls]['recall'],
                        metrics[cls]['f1-score']
                    ])
                
                metrics_array = np.array(metrics_data).T
                im = axes[1, 1].imshow(metrics_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                
                axes[1, 1].set_xticks(range(len(classes)))
                axes[1, 1].set_yticks(range(3))
                axes[1, 1].set_xticklabels(classes)
                axes[1, 1].set_yticklabels(['Precision', 'Recall', 'F1-Score'])
                axes[1, 1].set_title('Metrics Heatmap', fontweight='bold')
                
                # Add text annotations
                for i in range(3):
                    for j in range(len(classes)):
                        axes[1, 1].text(j, i, f'{metrics_array[i, j]:.3f}',
                                       ha='center', va='center', fontweight='bold')
                
                plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting model performance: {e}")
            raise
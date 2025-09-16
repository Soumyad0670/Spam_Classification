📋 Project Overview
This project implements a complete machine learning pipeline for spam email classification, featuring:

Automated data downloading from Kaggle

Text preprocessing and TF-IDF vectorization

Decision Tree classification with hyperparameter tuning

Comprehensive model evaluation and visualization

Modular architecture for maintainability and extensibility

🏗️ Project Structure
text
spam-classification/
├── main.py              # Main orchestration script
├── config.py            # Configuration parameters
├── data_loader.py       # Data loading and preprocessing
├── model_trainer.py     # Model training and evaluation
├── visualizer.py        # Visualization utilities
├── requirements.txt     # Dependencies
└── README.md           # Project documentation
⚙️ Installation & Setup
Clone the repository

bash
git clone <repository-url>
cd spam-classification
Install dependencies

bash
pip install -r requirements.txt
pip install kagglehub joblib
Set up Kaggle API credentials

Create a Kaggle account at https://www.kaggle.com/

Download your API token (kaggle.json) from Account Settings

Place it in ~/.kaggle/kaggle.json or set environment variables:

bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
🚀 Usage
Run the complete pipeline:

bash
python main.py
Individual Components
Data loading only:

python
from data_loader import DataLoader

loader = DataLoader()
X_train, X_test, y_train, y_test = loader.prepare_data()
Model training:

python
from model_trainer import SpamClassifier

classifier = SpamClassifier()
classifier.train(X_train, y_train)
results = classifier.evaluate(X_test, y_test)
Visualization:

python
from visualizer import Visualizer

viz = Visualizer()
viz.plot_confusion_matrix(y_test, predictions)
viz.plot_feature_importance(feature_importance)
📊 Features
Data Processing
Automated dataset download from Kaggle

Text preprocessing with TF-IDF vectorization

Automatic column identification

Duplicate removal and missing value handling

Model Training
Configurable Decision Tree classifier

Cross-validation support

Feature importance analysis

Model persistence (save/load)

Visualization
Decision tree visualization

Confusion matrix plots

Feature importance charts

Class distribution analysis

Comprehensive performance dashboards

⚙️ Configuration
Modify config.py to customize:

python
# Model parameters
MODEL_CONFIG = {
    'criterion': 'gini',
    'max_depth': 5,
    'random_state': 42,
    'min_samples_split': 20,
    'min_samples_leaf': 10
}

# Data processing
DATA_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'max_features': 1000,
    'min_df': 2,
    'max_df': 0.95,
    'stop_words': 'english'
}
📈 Performance Metrics
The model provides comprehensive evaluation:

Accuracy, Precision, Recall, F1-score

Cross-validation results

Confusion matrix analysis

Feature importance rankings

🔧 Customization
Adding New Models
Extend the SpamClassifier class:

python
from sklearn.ensemble import RandomForestClassifier

class RandomForestSpamClassifier(SpamClassifier):
    def __init__(self, **kwargs):
        params = {**MODEL_CONFIG, **kwargs}
        self.model = RandomForestClassifier(**params)
Custom Preprocessing
Modify the DataLoader class:

python
def custom_preprocessing(self):
    # Add custom text cleaning
    self.data['text'] = self.data['text'].apply(my_custom_cleaner)
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset provided by ashfakyeafi on Kaggle

Built with Scikit-learn, Pandas, and Matplotlib

Inspired by practical spam detection applications
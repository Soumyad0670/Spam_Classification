# Spam Classification 📧

A machine learning pipeline for classifying text messages/emails as **spam** or **not spam**, built with modular code, visualization, and configurability.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Setup & Installation](#setup--installation)  
5. [Usage](#usage)  
6. [Configuration](#configuration)  
7. [Evaluation & Visualization](#evaluation--visualization)  
8. [Extending the Project](#extending-the-project)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## Project Overview

This repository demonstrates a complete workflow for detecting spam using a decision-tree-based classifier. Key steps include:

- Loading and preprocessing text data  
- Feature engineering using TF-IDF  
- Training, tuning, and evaluating a classifier  
- Visualizing results and metrics  

Ideal for learning ML pipelines, text preprocessing, and as a starter for more complex spam / text classification tasks.

---

## Features

- Automated data loading & cleaning (duplicates, missing values)  
- Text preprocessing: tokenization, stopword removal, TF-IDF vectorization  
- Configurable Decision Tree classifier (hyperparameters configurable)  
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrix  
- Visual tools: feature importance plots, confusion matrix, etc.  
- Modular code structure → easy to maintain, modify, or extend  

---

## Project Structure

Here’s what files/folders are in the repo and their purpose:

```

Spam\_Classification/
├── **main.py**            # Orchestrates the full pipeline: from data to model to evaluation
├── **config.py**          # All configuration variables: model params, data split, vectorizer settings, etc.
├── **data\_loader.py**     # Loading, cleaning, preprocessing of data; splitting into train & test sets
├── **model\_trainer.py**   # Building, training, evaluating the classifier
├── **visualizer.py**      # Functions for plotting metrics (confusion matrix, feature importance, etc.)
├── **requirements.txt**   # Python libraries / dependencies
└── **README.md**          # This documentation

````

---

## Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Soumyad0670/Spam_Classification.git
   cd Spam_Classification
````

2. **Install dependencies**
   It’s recommended to use a virtual environment (venv / conda)

   ```bash
   pip install -r requirements.txt
   ```

3. **Data / Credentials**
   If the project uses data from Kaggle (or any external source), set that up. For example:

   * Create a Kaggle account
   * Obtain API credentials (e.g., `kaggle.json`)
   * Place them in the proper location (e.g `~/.kaggle/kaggle.json`) or set environment variables

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This will execute the full flow: load & preprocess data → train model → evaluate → visualize results.

### Use individual components

* **Data loading & preprocessing**

  ```python
  from data_loader import DataLoader
  loader = DataLoader()  
  X_train, X_test, y_train, y_test = loader.prepare_data()
  ```

* **Training & evaluation**

  ```python
  from model_trainer import SpamClassifier
  classifier = SpamClassifier()
  classifier.train(X_train, y_train)
  metrics, predictions = classifier.evaluate(X_test, y_test)
  ```

* **Visualization**

  ```python
  from visualizer import Visualizer
  viz = Visualizer()
  viz.plot_confusion_matrix(y_test, predictions)
  viz.plot_feature_importance(classifier.feature_importances_)
  ```

---

## Configuration

All key settings are in **config.py**. You can modify:

* Model hyperparameters: e.g. `criterion`, `max_depth`, `min_samples_split`, etc.
* TF-IDF / vectorizer settings: `max_features`, `min_df`, `max_df`, stop words, etc.
* Train-test split ratios, random seed, etc.

---

## Evaluation & Visualization

When you run the pipeline or evaluate the model, you’ll get:

* Standard classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**
* Confusion Matrix to see what types of errors are being made
* Feature Importance plot to understand which words / features are most discriminative
* Optional: Visualizations of class balance, distributions, etc.

---

## Extending the Project

Here are ideas for how you might expand or improve this project:

* Use more / different models (Random Forests, SVM, Neural Networks)
* Add more preprocessing: stemming/lemmatization, handling imbalanced data, etc.
* Try more feature types: word embeddings (Word2Vec, GloVe), or transformer-based features
* Build a web API or UI wrapper so users can input sentences and get spam prediction
* Include real-time streaming prediction if dealing with live messages

---

## Contributing

You’re welcome to contribute! Suggested workflow:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes, add tests if needed
4. Commit with clear message: `git commit -m "Add XYZ"`
5. Push your branch: `git push origin feature/your-feature`
6. Open a Pull Request & describe what you changed

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* Dataset & inspiration from Kaggle
* Built using Python, scikit-learn, pandas, matplotlib (or whichever libs you use)
* Thanks to any tutorials / authors whose code you adapted or learned from

---


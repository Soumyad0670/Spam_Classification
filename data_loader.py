"""
Data loading and preprocessing utilities for spam email classification.
"""

import os
import pandas as pd
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG, POSSIBLE_TARGET_COLUMNS, POSSIBLE_TEXT_COLUMNS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing for spam email classification."""
    
    def __init__(self):
        self.data = None
        self.vectorizer = None
        self.text_column = None
        self.target_column = None
    
    def download_dataset(self, dataset_id="ashfakyeafi/spam-email-classification"):
        """Download dataset from Kaggle."""
        try:
            path = kagglehub.dataset_download(dataset_id)
            logger.info(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def load_data(self, dataset_path=None):
        """Load data from CSV file."""
        if dataset_path is None:
            dataset_path = self.download_dataset()
        
        # Handle different path types
        if isinstance(dataset_path, list):
            dataset_path = dataset_path[0]
        
        # Find CSV files
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        
        csv_file_path = os.path.join(dataset_path, csv_files[0])
        logger.info(f"Loading file: {csv_file_path}")
        
        try:
            self.data = pd.read_csv(csv_file_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def identify_columns(self):
        """Automatically identify text and target columns."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Find target column
        self.target_column = None
        for col in POSSIBLE_TARGET_COLUMNS:
            if col in self.data.columns:
                self.target_column = col
                break
        
        if self.target_column is None:
            # If no standard target column found, use the last column
            self.target_column = self.data.columns[-1]
            logger.warning(f"No standard target column found. Using: {self.target_column}")
        
        # Find text column
        self.text_column = None
        for col in POSSIBLE_TEXT_COLUMNS:
            if col in self.data.columns:
                self.text_column = col
                break
        
        if self.text_column is None:
            # Use the first non-target column
            remaining_cols = [col for col in self.data.columns if col != self.target_column]
            self.text_column = remaining_cols[0] if remaining_cols else None
        
        if self.text_column is None:
            raise ValueError("Could not identify text column")
        
        logger.info(f"Text column: {self.text_column}")
        logger.info(f"Target column: {self.target_column}")
        
        return self.text_column, self.target_column
    
    def preprocess_data(self):
        """Clean and preprocess the data."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Remove duplicates
        initial_shape = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        logger.info(f"Removed {initial_shape - self.data.shape[0]} duplicate rows")
        
        # Remove missing values
        self.data = self.data.dropna(subset=[self.text_column, self.target_column])
        logger.info(f"Final data shape: {self.data.shape}")
        
        # Display target distribution
        target_counts = self.data[self.target_column].value_counts()
        logger.info(f"Target distribution:\n{target_counts}")
        
        return self.data
    
    def vectorize_text(self, X_train, X_test=None):
        """Convert text to numerical features using TF-IDF."""
        self.vectorizer = TfidfVectorizer(
            max_features=DATA_CONFIG['max_features'],
            min_df=DATA_CONFIG['min_df'],
            max_df=DATA_CONFIG['max_df'],
            stop_words=DATA_CONFIG['stop_words'],
            lowercase=True,
            strip_accents='ascii'
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        logger.info(f"Vectorized training data shape: {X_train_vectorized.shape}")
        
        if X_test is not None:
            X_test_vectorized = self.vectorizer.transform(X_test)
            logger.info(f"Vectorized test data shape: {X_test_vectorized.shape}")
            return X_train_vectorized, X_test_vectorized
        
        return X_train_vectorized
    
    def prepare_data(self, dataset_path=None):
        """Complete data preparation pipeline."""
        # Load data
        self.load_data(dataset_path)
        
        # Identify columns
        self.identify_columns()
        
        # Preprocess data
        self.preprocess_data()
        
        # Extract features and target
        X = self.data[self.text_column]
        y = self.data[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATA_CONFIG['test_size'], 
            random_state=DATA_CONFIG['random_state'],
            stratify=y  # Maintain class distribution
        )
        
        # Vectorize text
        X_train_vec, X_test_vec = self.vectorize_text(X_train, X_test)
        
        logger.info("Data preparation completed successfully")
        
        return X_train_vec, X_test_vec, y_train, y_test
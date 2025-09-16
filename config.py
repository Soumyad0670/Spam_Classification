"""
Configuration settings for spam email classification project.
"""

# Model parameters
MODEL_CONFIG = {
    'criterion': 'gini',
    'max_depth': 5,  # Increased from 3 for better performance
    'random_state': 42,
    'min_samples_split': 20,  # Added for better generalization
    'min_samples_leaf': 10    # Added to prevent overfitting
}

# Data processing parameters
DATA_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'max_features': 1000,
    'min_df': 2,              # Ignore terms that appear in less than 2 documents
    'max_df': 0.95,           # Ignore terms that appear in more than 95% of documents
    'stop_words': 'english'    # Remove English stop words
}

# Visualization parameters
VIZ_CONFIG = {
    'figsize': (15, 10),
    'max_depth_viz': 3,  # Limit visualization depth for readability
    'fontsize': 10
}

# Common target column names to search for
POSSIBLE_TARGET_COLUMNS = [
    'target', 'label', 'spam', 'class', 'Category', 'type', 'classification'
]

# Common text column names to search for
POSSIBLE_TEXT_COLUMNS = [
    'text', 'email', 'message', 'content', 'body', 'email_text', 'Message'
]
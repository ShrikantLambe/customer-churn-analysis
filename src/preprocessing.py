"""
Preprocessing pipeline for Customer Churn Prediction
"""
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    """
    Load the Telco Customer Churn dataset from a CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(path)


def preprocess_data(df, config, target_col='Churn', return_preprocessor=False, return_feature_names=False):
    """
    Preprocess the Telco Customer Churn dataset using config.
    - Separates numerical and categorical features
    - Handles missing values
    - One-hot encodes categorical variables
    - Scales numerical features
    - Splits into train/test sets
    Args:
        df (pd.DataFrame): Raw DataFrame
        config (dict): Configuration dictionary
        target_col (str): Name of the target column
    Returns:
        X_train, X_test, y_train, y_test: Processed splits
    """

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert target to binary if needed
    y = df[target_col].map(
        {'Yes': 1, 'No': 0}) if df[target_col].dtype == 'O' else df[target_col]
    X = df.drop(target_col, axis=1)
    feature_names = X.columns.tolist()

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(
        include=['object', 'category', 'bool']).columns.tolist()

    # Preprocessing for numerical data
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Split data using config
    split_cfg = config['train_test_split']
    test_size = split_cfg.get('test_size', 0.2)
    random_seed = config.get('random_seed', 42)
    stratify = y if split_cfg.get('stratify', True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=stratify
    )

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logging.info(
        f"Preprocessing complete. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")

    if return_preprocessor and return_feature_names:
        return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor, feature_names
    if return_preprocessor:
        return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor
    if return_feature_names:
        return X_train_processed, X_test_processed, y_train.values, y_test.values, feature_names
    return X_train_processed, X_test_processed, y_train.values, y_test.values

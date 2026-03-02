"""
Model training module for Customer Churn Prediction
"""
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_logistic_regression(X_train, y_train, config, cv=5):
    """
    Train a Logistic Regression model with cross-validation.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        config (dict): Configuration dictionary.
        cv (int): Number of cross-validation folds.
    Returns:
        model (LogisticRegression): Trained model.
        float: Mean cross-validation ROC-AUC score.
    """
    params = config['model']['logistic_regression']
    model = LogisticRegression(
        max_iter=params.get('max_iter', 1000),
        solver=params.get('solver', 'lbfgs'),
        random_state=config['random_seed']
    )
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_train, y_train)
    logging.info(f"Logistic Regression CV ROC-AUC: {cv_scores.mean():.4f}")
    return model, cv_scores.mean()


def train_random_forest(X_train, y_train, config, cv=5):
    """
    Train a Random Forest Classifier with cross-validation.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        config (dict): Configuration dictionary.
        cv (int): Number of cross-validation folds.
    Returns:
        model (RandomForestClassifier): Trained model.
        float: Mean cross-validation ROC-AUC score.
    """
    params = config['model']['random_forest']
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', None),
        random_state=params.get('random_state', config['random_seed'])
    )
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_train, y_train)
    logging.info(f"Random Forest CV ROC-AUC: {cv_scores.mean():.4f}")
    return model, cv_scores.mean()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model on test data and print metrics.
    Args:
        model: Trained classifier with predict and predict_proba.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    logging.info(f"Test Accuracy:  {acc:.4f}")
    logging.info(f"Test Precision: {prec:.4f}")
    logging.info(f"Test Recall:    {rec:.4f}")
    logging.info(f"Test F1 Score:  {f1:.4f}")
    if roc_auc is not None:
        logging.info(f"Test ROC-AUC:   {roc_auc:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}


def save_model(model, config, model_name):
    """
    Save the trained model as a pickle file with versioning.
    Args:
        model: Trained model to save.
        config (dict): Configuration dictionary.
        model_name (str): Name for the model file.
    """
    model_dir = config['output']['model_dir']
    version = config['output']['model_version']
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}_{version}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {path}")

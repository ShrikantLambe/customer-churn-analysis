"""
Evaluation utilities for classification models
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import numpy as np


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix using matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if class_names is None:
        class_names = ["Class 0", "Class 1"]
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, title='ROC Curve'):
    """
    Plot ROC curve using matplotlib.
    y_proba: probability estimates for the positive class
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def print_classification_report(y_true, y_pred):
    """
    Print classification report (precision, recall, f1-score, support).
    """
    print(classification_report(y_true, y_pred))


def get_feature_importance(model, feature_names=None, top_n=10):
    """
    Extract and print feature importances for tree-based models or coefficients for linear models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        name = 'Feature Importance'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        name = 'Absolute Coefficient'
    else:
        print("Model does not support feature importance.")
        return
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i]
               for i in indices], rotation=45, ha='right')
    plt.title(f'Top {top_n} {name}')
    plt.tight_layout()
    plt.show()
    for i in indices:
        print(f'{feature_names[i]}: {importances[i]:.4f}')

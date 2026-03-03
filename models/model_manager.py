import pickle
import os
import pandas as pd


def load_model(model_dir, version):
    model_path = os.path.join(model_dir, f"best_churn_model_{version}.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_preprocessor(model_dir, version):
    preprocessor_path = os.path.join(model_dir, f"preprocessor_{version}.pkl")
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)


def load_feature_names(model_dir):
    feature_names_path = os.path.join(model_dir, "feature_names_v1.pkl")
    with open(feature_names_path, "rb") as f:
        return pickle.load(f)

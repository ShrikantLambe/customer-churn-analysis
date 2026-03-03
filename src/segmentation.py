import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(df, features, n_clusters=4, random_state=42):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns for clustering are missing: {missing}")
    X = df[features].copy()
    # Fill missing values if needed
    X = X.fillna(X.mean())
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters
    # Compute cluster summaries
    summaries = []
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i]
        summary = {
            'cluster': i,
            'size': len(cluster_df),
            'means': cluster_df[features].mean().to_dict(),
            'churn_rate': cluster_df['Churn'].mean() if 'Churn' in cluster_df.columns else None
        }
        summaries.append(summary)
    return df, summaries

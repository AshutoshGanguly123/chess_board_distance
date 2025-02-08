#!/usr/bin/env python3
import os
import json
import numpy as np
import plotly.express as px
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_opening_metrics(root_dir):
    """
    Scans each subfolder in root_dir (each opening folder) for a file ending in "_summary.json".
    Returns two dictionaries: one for white average metrics and one for black average metrics.
    """
    white_data = {}
    black_data = {}
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find the summary file (assumed to end with "_summary.json")
            for fname in os.listdir(folder_path):
                if fname.endswith("_summary.json"):
                    summary_file = os.path.join(folder_path, fname)
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                        opening_name = summary.get("opening", folder)
                        avg_metrics = summary.get("average_metrics", {})
                        white_metrics = avg_metrics.get("white", {})
                        black_metrics = avg_metrics.get("black", {})
                        expected_keys = ["control", "threat", "strategic", "hybrid", "eval"]
                        if white_metrics and all(white_metrics.get(k) is not None for k in expected_keys):
                            white_data[opening_name] = white_metrics
                        if black_metrics and all(black_metrics.get(k) is not None for k in expected_keys):
                            black_data[opening_name] = black_metrics
                    break
    return white_data, black_data

def create_feature_matrix_category(metrics_dict, category):
    """
    For a given dictionary mapping an opening to its metrics and a chosen category,
    constructs a feature matrix and a list of opening names.
    
    The categories and chosen features are:
      - "threat": [threat, hybrid, eval]
      - "position": [control, hybrid, eval]
      - "strategic": [strategic, hybrid, eval]
    """
    opening_names = []
    features = []
    for name, metrics in metrics_dict.items():
        if category == "threat":
            vec = [metrics["threat"], metrics["hybrid"], metrics["eval"]]
        elif category == "position":
            vec = [metrics["control"], metrics["hybrid"], metrics["eval"]]
        elif category == "strategic":
            vec = [metrics["strategic"], metrics["hybrid"], metrics["eval"]]
        else:
            raise ValueError("Invalid category. Choose one of 'threat', 'position', or 'strategic'.")
        opening_names.append(name)
        features.append(vec)
    return np.array(features), opening_names

def perform_clustering(feature_matrix, n_clusters=3):
    """
    Standardizes the features and applies KMeans clustering.
    Returns the cluster labels and the standardized feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels, X_scaled

def plot_clusters_3d(X, labels, opening_names, title, filename):
    """
    Plots a 3D scatter plot using Plotly with the given 3D data X,
    cluster labels, and annotations (opening names).
    Saves the plot to an HTML file.
    """
    # Create a data dictionary
    data = {
        "PC1": X[:, 0],
        "PC2": X[:, 1],
        "PC3": X[:, 2],
        "Opening": opening_names,
        "Cluster": labels.astype(str)
    }
    fig = px.scatter_3d(data, x="PC1", y="PC2", z="PC3", color="Cluster",
                        text="Opening", title=title, template="plotly_white")
    fig.update_traces(textposition='top center')
    fig.write_html(filename)
    print(f"Saved 3D cluster plot to {filename}")

def main():
    root_dir = "openings_experiment_outputs"  # Your folder containing subfolders per opening
    white_data, black_data = load_opening_metrics(root_dir)
    
    # Define the three categories
    categories = ["threat", "position", "strategic"]
    
    # Process for each color and category
    for color, data in zip(["white", "black"], [white_data, black_data]):
        for cat in categories:
            # Build feature matrix for the current category
            features, opening_names = create_feature_matrix_category(data, cat)
            # Standardize and cluster
            labels, X_scaled = perform_clustering(features, n_clusters=3)
            # For 3D plot, we already have 3 features; we can use PCA if desired,
            # but here our feature dimension is already 3.
            # We simply use the standardized features.
            title = f"{color.capitalize()} {cat.capitalize()} Metrics Clusters"
            filename = f"clusters_{color}_{cat}.html"
            plot_clusters_3d(X_scaled, labels, opening_names, title, filename)
            # Optionally print cluster assignments:
            print(f"{color.capitalize()} {cat.capitalize()} Clusters:")
            for name, label in zip(opening_names, labels):
                print(f"  {name}: Cluster {label}")
    
if __name__ == "__main__":
    main()

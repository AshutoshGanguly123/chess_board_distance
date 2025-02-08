#!/usr/bin/env python3
import os
import json
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_opening_metrics(root_dir):
    """
    Scans each subfolder in root_dir for a file ending in "_summary.json" and loads it.
    Returns two dictionaries: white_data and black_data mapping opening names to their average metrics.
    """
    white_data = {}
    black_data = {}
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
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

def create_combined_feature_matrix(white_data, black_data, category):
    """
    For a given category, create a combined feature matrix and labels.
    The category must be one of "threat", "position", or "strategic".
    
    For each opening:
      - If category == "threat": feature vector = [threat["spectral_radius"], hybrid, eval]
      - If category == "position": feature vector = [control["fiedler"], hybrid, eval]
      - If category == "strategic": feature vector = [strategic["central_dominance"], hybrid, eval]
    
    The label for each is "Opening (White)" or "Opening (Black)".
    """
    features = []
    labels = []
    for opening, white_metrics in white_data.items():
        # White point
        if category == "threat":
            feat = [white_metrics["threat"].get("spectral_radius", 0.0), white_metrics["hybrid"], white_metrics["eval"]]
        elif category == "position":
            feat = [white_metrics["control"].get("fiedler", 0.0), white_metrics["hybrid"], white_metrics["eval"]]
        elif category == "strategic":
            feat = [white_metrics["strategic"].get("central_dominance", 0.0), white_metrics["hybrid"], white_metrics["eval"]]
        else:
            raise ValueError("Invalid category")
        features.append(feat)
        labels.append(f"{opening} (White)")
    for opening, black_metrics in black_data.items():
        if category == "threat":
            feat = [black_metrics["threat"].get("spectral_radius", 0.0), black_metrics["hybrid"], black_metrics["eval"]]
        elif category == "position":
            feat = [black_metrics["control"].get("fiedler", 0.0), black_metrics["hybrid"], black_metrics["eval"]]
        elif category == "strategic":
            feat = [black_metrics["strategic"].get("central_dominance", 0.0), black_metrics["hybrid"], black_metrics["eval"]]
        else:
            raise ValueError("Invalid category")
        features.append(feat)
        labels.append(f"{opening} (Black)")
    return np.array(features), labels

def perform_clustering(feature_matrix, n_clusters=3):
    """
    Standardizes the features and applies KMeans clustering.
    Returns the cluster labels and standardized features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels, X_scaled

def plot_clusters_3d(X, cluster_labels, point_labels, title, filename):
    """
    Uses Plotly Express to plot a 3D scatter plot with point annotations.
    """
    data = {
        "X": X[:, 0],
        "Y": X[:, 1],
        "Z": X[:, 2],
        "Label": point_labels,
        "Cluster": cluster_labels.astype(str)
    }
    fig = px.scatter_3d(data, x="X", y="Y", z="Z", color="Cluster", text="Label",
                        title=title, template="plotly_white")
    fig.update_traces(textposition='top center')
    fig.write_html(filename)
    print(f"Saved 3D cluster plot to {filename}")

def main():
    root_dir = "openings_experiment_outputs"
    white_data, black_data = load_opening_metrics(root_dir)
    categories = ["threat", "position", "strategic"]
    
    for cat in categories:
        # Create combined feature matrix and labels for both white and black.
        features, point_labels = create_combined_feature_matrix(white_data, black_data, cat)
        cluster_labels, X_scaled = perform_clustering(features, n_clusters=3)
        title = f"{cat.capitalize()} Metrics Clusters (White & Black)"
        filename = f"clusters_combined_{cat}.html"
        plot_clusters_3d(X_scaled, cluster_labels, point_labels, title, filename)
        print(f"Category '{cat}':")
        for pl, cl in zip(point_labels, cluster_labels):
            print(f"  {pl}: Cluster {cl}")

if __name__ == "__main__":
    main()

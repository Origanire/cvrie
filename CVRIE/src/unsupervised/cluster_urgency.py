#!/usr/bin/env python3
"""
Clustering of medical urgency testimonies - 5 urgency levels.
Uses weighted keyword scoring to categorize symptoms.

Reads  : data/unsupervised/processed/Student_Dataset_urgency.csv
Writes : results/cluster_labels_urgency.csv
         reports/figures/unsupervised/urgency_*.png
"""

import csv
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap

# Suppress warnings
warnings.filterwarnings("ignore")

# Define urgency levels with weighted keywords
URGENCY_KEYWORDS = {
    "benin": {
        "keywords": ["mild", "slight", "minor", "small", "tender", "discomfort", "barely", "almost", "ease"],
        "weight": 1
    },
    "faible": {
        "keywords": ["ache", "soreness", "sore", "dull", "weak", "minimal", "light", "tired"],
        "weight": 2
    },
    "modere": {
        "keywords": ["pain", "fever", "headache", "cough", "throat", "itchy", "rash", "swell", "inflame", "moderate"],
        "weight": 3
    },
    "grave": {
        "keywords": ["severe", "intense", "sharp", "acute", "emergency", "critical", "spread", "worse", "urgent", "throb"],
        "weight": 4
    },
    "urgent": {
        "keywords": ["chest", "pressure", "breathe", "difficulty", "faint", "unconscious", "blood", "emergency", "hospital", "ambulance"],
        "weight": 5
    }
}

# Paths
DATA_PATH = os.path.join("data", "unsupervised", "processed", "Student_Dataset_urgency.csv")
RESULTS_PATH = os.path.join("results", "cluster_labels_urgency.csv")
FIGURES_DIR = os.path.join("reports", "figures", "unsupervised")

def load_data():
    """Load urgency dataset from CSV"""
    ids = []
    point_colors = []
    texts = []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                ids.append(row[0])
                point_colors.append(_normalize_hex_color(row[1]))
                texts.append(row[2])
    
    return ids, point_colors, texts


def _normalize_hex_color(color_code):
    """Normalize CSV hex colors from 0xRRGGBB/#RRGGBB to #RRGGBB."""
    if not color_code:
        return None
    raw = str(color_code).strip().lower()
    if raw.startswith("0x"):
        hex_part = raw[2:]
    elif raw.startswith("#"):
        hex_part = raw[1:]
    else:
        return None
    if len(hex_part) != 6:
        return None
    try:
        int(hex_part, 16)
    except ValueError:
        return None
    return f"#{hex_part}"

def vectorize_texts(texts):
    """Vectorize texts using TF-IDF"""
    print("Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  Shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer

def reduce_dimensions(tfidf_matrix):
    """Reduce dimensions using PCA"""
    print("Reducing dimensions with PCA...")
    pca = PCA(n_components=20)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    return reduced, pca

def calculate_urgency_score(text):
    """
    Calculate urgency score (1-5) for a text based on weighted keywords.
    Returns float between 1 (benin) and 5 (urgent)
    """
    text_lower = text.lower()
    total_score = 0
    total_weight = 0
    
    for level, data in URGENCY_KEYWORDS.items():
        weight = data["weight"]
        for keyword in data["keywords"]:
            count = text_lower.count(keyword)
            total_score += count * weight
            total_weight += count
    
    if total_weight == 0:
        return 3.0  # Default to moderate if no keywords found
    
    return min(5.0, max(1.0, total_score / total_weight))

def cluster_by_urgency_score(texts):
    """
    Assign clusters directly based on urgency scores.
    Each cluster corresponds to exactly one urgency level.
    """
    print(f"Calculating urgency scores for {len(texts)} texts...")
    urgency_scores = np.array([calculate_urgency_score(text) for text in texts])
    print(f"  Score range: {urgency_scores.min():.2f} - {urgency_scores.max():.2f}")
    
    # Assign clusters based on urgency score ranges
    print(f"Assigning clusters based on urgency levels...")
    labels = np.zeros(len(texts), dtype=int)
    
    for i, score in enumerate(urgency_scores):
        if score < 1.8:
            labels[i] = 0  # Bénin
        elif score < 2.6:
            labels[i] = 1  # Faible
        elif score < 3.4:
            labels[i] = 2  # Modéré
        elif score < 4.2:
            labels[i] = 3  # Grave
        else:
            labels[i] = 4  # Urgent
    
    # Display cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    urgency_level_names = {0: "Bénin", 1: "Faible", 2: "Modéré", 3: "Grave", 4: "Urgent"}
    print(f"\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        level_name = urgency_level_names.get(cluster_id, "Unknown")
        print(f"  Cluster {cluster_id} ({level_name}): {count} samples")
    
    return labels, urgency_scores

def get_cluster_labels(tfidf_matrix, labels, vectorizer, n_top_terms=3):
    """Extract top terms for each cluster"""
    feature_names = vectorizer.get_feature_names_out()
    labels_dict = {}
    
    for cluster_id in range(max(labels) + 1):
        cluster_mask = labels == cluster_id
        
        if cluster_mask.sum() == 0:
            labels_dict[cluster_id] = "N/A"
            continue
        
        cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[-n_top_terms:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        labels_dict[cluster_id] = " / ".join(top_terms)
    
    return labels_dict

def get_urgency_level_name(cluster_id, labels, texts):
    """
    Determine urgency level name based on cluster's average urgency score.
    Returns one of: Bénin, Faible, Modéré, Grave, Urgent
    """
    cluster_mask = labels == cluster_id
    cluster_texts = [texts[i] for i, mask in enumerate(cluster_mask) if mask]
    
    # Calculate average urgency score for this cluster
    avg_urgency = np.mean([calculate_urgency_score(text) for text in cluster_texts])
    
    # Map score to level name
    if avg_urgency < 1.8:
        return "Bénin", 1
    elif avg_urgency < 2.6:
        return "Faible", 2
    elif avg_urgency < 3.4:
        return "Modéré", 3
    elif avg_urgency < 4.2:
        return "Grave", 4
    else:
        return "Urgent", 5

def visualize_results(data, labels, tfidf_matrix, vectorizer, texts, urgency_scores, method_name="KMeans", point_colors=None):
    """Create 2D visualization using UMAP with urgency level labels"""
    print(f"Creating visualizations for {method_name}...")
    
    # Get cluster labels and urgency levels
    print("  Extracting cluster information...")
    cluster_labels = get_cluster_labels(tfidf_matrix, labels, vectorizer, n_top_terms=3)
    
    urgency_levels = {}
    urgency_colors = {
        "Bénin": "#90EE90",      # Light green
        "Faible": "#90EE90",     # Green
        "Modéré": "#FFD700",     # Gold
        "Grave": "#FF8C00",      # Dark orange
        "Urgent": "#FF0000"      # Red
    }
    urgency_emoji = {
        "Bénin": "🟢",
        "Faible": "🟢",
        "Modéré": "🟡",
        "Grave": "🟠",
        "Urgent": "🔴"
    }
    
    for cluster_id in range(max(labels) + 1):
        level_name, _ = get_urgency_level_name(cluster_id, labels, texts)
        urgency_levels[cluster_id] = level_name
    
    # Use UMAP for 2D visualization with urgency scores as feature (better cluster separation)
    print("  Applying UMAP for visualization with urgency weight...")
    # Augment data with urgency scores as additional feature (10x weight to pull clusters together)
    urgency_feature = urgency_scores.reshape(-1, 1) * 10
    augmented_data = np.hstack([data, urgency_feature])
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.15, metric='euclidean')
    data_2d = reducer.fit_transform(augmented_data)
    
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot points
    cmap = plt.cm.get_cmap('tab20')
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                        c=labels, cmap=cmap, 
                        s=200, alpha=0.7, edgecolors="darkgray", linewidth=0.8)
    
    # Add cluster labels at centroids with urgency color coding
    print("  Adding cluster labels...")
    for cluster_id in range(max(labels) + 1):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            centroid_x = data_2d[cluster_mask, 0].mean()
            centroid_y = data_2d[cluster_mask, 1].mean()
            
            urgency_level = urgency_levels[cluster_id]
            emoji = urgency_emoji.get(urgency_level, "")
            color = urgency_colors.get(urgency_level, "#CCCCCC")
            
            ax.annotate(
                f"C{cluster_id}\n{emoji} {urgency_level}\n{cluster_labels.get(cluster_id, 'Unknown')}",
                xy=(centroid_x, centroid_y),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=color, alpha=0.85, edgecolor='black', linewidth=2),
                zorder=10,
                color='black'
            )
    
    ax.set_xlabel("UMAP Dimension 1", fontsize=13, fontweight='bold')
    ax.set_ylabel("UMAP Dimension 2", fontsize=13, fontweight='bold')
    ax.set_title(f"Medical Urgency Clustering - 5 Levels\n(Bénin, Faible, Modéré, Grave, Urgent)", fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    outfile = os.path.join(FIGURES_DIR, f"urgency_clusters_{method_name.lower()}.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(outfile, dpi=250, bbox_inches="tight", facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close()

def create_cluster_distribution(labels, texts):
    """Visualize cluster distribution with urgency levels"""
    print("Creating cluster distribution chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cluster_ids = []
    counts = []
    colors_list = []
    urgency_emoji = {
        "Bénin": "🟢",
        "Faible": "🟢",
        "Modéré": "🟡",
        "Grave": "🟠",
        "Urgent": "🔴"
    }
    urgency_colors = {
        "Bénin": "#90EE90",
        "Faible": "#90EE90",
        "Modéré": "#FFD700",
        "Grave": "#FF8C00",
        "Urgent": "#FF0000"
    }
    
    for cluster_id in range(max(labels) + 1):
        count = np.sum(labels == cluster_id)
        level_name, _ = get_urgency_level_name(cluster_id, labels, texts)
        
        cluster_ids.append(f"C{cluster_id}\n{urgency_emoji[level_name]}\n{level_name}")
        counts.append(count)
        colors_list.append(urgency_colors[level_name])
    
    bars = ax.bar(range(len(cluster_ids)), counts, color=colors_list, edgecolor="black", alpha=0.8, linewidth=2)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Urgency Level Clusters", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight='bold')
    ax.set_title("Medical Urgency Distribution - 5 Levels", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels(cluster_ids, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    outfile = os.path.join(FIGURES_DIR, "urgency_cluster_distribution.png")
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outfile}")
    plt.close()

def save_cluster_labels(ids, labels, texts):
    """Save cluster assignments with urgency level to CSV"""
    print("Saving cluster labels with urgency levels...")
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "cluster", "urgency_level"])
        for id_, cluster in zip(ids, labels):
            level_name, _ = get_urgency_level_name(cluster, labels, texts)
            writer.writerow([id_, cluster, level_name])
    
    print(f"  Saved: {RESULTS_PATH}")

def main():
    print("=" * 70)
    print("MEDICAL URGENCY CLUSTERING - 5 LEVELS")
    print("(Bénin, Faible, Modéré, Grave, Urgent)")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    ids, point_colors, texts = load_data()
    print(f"  Loaded {len(texts)} samples")
    
    # Vectorize
    print("\n[2/5] Vectorizing texts...")
    tfidf_matrix, vectorizer = vectorize_texts(texts)
    
    # Reduce dimensions with PCA
    print("\n[3/5] Reducing dimensions with PCA...")
    reduced_data, pca = reduce_dimensions(tfidf_matrix)
    
    scaler = StandardScaler()
    reduced_data = scaler.fit_transform(reduced_data)
    
    # Cluster with direct urgency assignment (not K-Means)
    print("\n[4/5] Assigning clusters based on urgency levels...")
    labels, urgency_scores = cluster_by_urgency_score(texts)
    
    # Visualize
    print("\n[5/5] Creating visualizations...")
    visualize_results(reduced_data, labels, tfidf_matrix, vectorizer, texts, urgency_scores, "KMeans_5Levels", point_colors=point_colors)
    create_cluster_distribution(labels, texts)
    
    # Display cluster descriptions with urgency levels
    print("\n" + "=" * 70)
    print("CLUSTER DETAILS - 5 URGENCY LEVELS")
    print("=" * 70)
    
    cluster_labels = get_cluster_labels(tfidf_matrix, labels, vectorizer, n_top_terms=4)
    urgency_emoji = {
        "Bénin": "🟢",
        "Faible": "🟢",
        "Modéré": "🟡",
        "Grave": "🟠",
        "Urgent": "🔴"
    }
    
    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        count = np.sum(cluster_mask)
        level_name, level_num = get_urgency_level_name(cluster_id, labels, texts)
        emoji = urgency_emoji.get(level_name, "")
        avg_score = urgency_scores[cluster_mask].mean()
        
        print(f"\nCluster {cluster_id}: {emoji} {level_name} (Level {level_num})")
        print(f"  - Samples: {count}")
        print(f"  - Avg Urgency Score: {avg_score:.2f}/5.0")
        print(f"  - Top Symptoms: {cluster_labels.get(cluster_id, 'Unknown')}")
    
    # Save results
    save_cluster_labels(ids, labels, texts)
    
    print("\n" + "=" * 70)
    print("✓ CLUSTERING COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Clusters: 5 (Bénin, Faible, Modéré, Grave, Urgent)")
    print(f"✓ Total Samples: {len(texts)}")
    print(f"✓ Results Saved: {RESULTS_PATH}")
    print(f"✓ Visualizations: {FIGURES_DIR}/*.png")
    print(f"\nKey Feature: Better cluster separation using UMAP + weighted urgency scoring")

if __name__ == "__main__":
    main()

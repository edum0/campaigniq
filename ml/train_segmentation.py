"""
train_segmentation.py
======================
Goal 3 — Customer Segmentation (Unsupervised Learning)

Answers: "What kind of buyer is this customer?"
Method:  K-Means Clustering (no target labels needed)

What this script does:
    1. Selects a focused subset of behavioural features for clustering
    2. Scales features (critical — K-Means is distance-based, Income
       would dominate Recency without scaling)
    3. Runs Elbow Method + Silhouette Analysis to find optimal K
    4. Fits final K-Means model
    5. Automatically labels each cluster by inspecting its centroid profile
    6. Reduces to 2D with PCA for visualisation
    7. Plots: Elbow curve, Silhouette scores, 2D cluster map,
              radar chart of cluster profiles
    8. Saves model + scaler to ml/models/

Author: CampaignIQ
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import run_pipeline

# ─────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_SAVE_PATH  = os.path.join(MODELS_DIR,  "segmentation_kmeans.pkl")
SCALER_SAVE_PATH = os.path.join(MODELS_DIR,  "segmentation_scaler.pkl")
REPORT_PLOT_PATH = os.path.join(REPORTS_DIR, "segmentation_evaluation.png")

RANDOM_STATE = 42

# Features used for clustering — chosen to capture:
#   spending behaviour, lifestyle, loyalty, engagement, demographics
CLUSTER_FEATURES = [
    "Income",               # wealth tier
    "Total_Spend",          # overall spending power
    "Recency",              # how recently they bought (low = more active)
    "Campaign_History",     # how responsive to campaigns historically
    "Total_Purchases",      # purchase frequency
    "Avg_Basket_Size",      # premium vs budget buyer
    "Has_Children",         # family lifestyle flag
    "Age",                  # life stage
    "NumWebVisitsMonth",    # digital engagement
    "NumDealsPurchases",    # deal-seeking behaviour
]


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_segmentation_data():
    """
    Load preprocessed data and extract only the clustering feature subset.
    Returns the raw (unscaled) clustering matrix and a scaler for later use.
    """
    df, feature_cols = run_pipeline()

    # Validate all cluster features exist
    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing cluster features: {missing}")

    X_raw = df[CLUSTER_FEATURES].copy()

    print(f"[SEGMENT] Clustering on {len(CLUSTER_FEATURES)} features across {len(df)} customers")
    return df, X_raw


# ─────────────────────────────────────────────
# STEP 2 — SCALE FEATURES
# ─────────────────────────────────────────────

def scale_features(X_raw: pd.DataFrame):
    """
    StandardScaler: transforms each feature to mean=0, std=1.

    Why this is critical for K-Means:
      K-Means computes Euclidean distances between customers.
      Income ranges 0-150k while Recency ranges 0-99.
      Without scaling, Income would dominate the distance calculation
      and Recency/Campaign_History would be ignored entirely.
      After scaling, every feature contributes equally.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES, index=X_raw.index)
    print(f"[SEGMENT] Features scaled with StandardScaler")
    return X_scaled, scaler


# ─────────────────────────────────────────────
# STEP 3 — FIND OPTIMAL K
# ─────────────────────────────────────────────

def find_optimal_k(X_scaled: pd.DataFrame, k_range: range = range(2, 11)):
    """
    Use two complementary methods to find the best number of clusters K:

    1. Elbow Method:
       Plot inertia (within-cluster sum of squares) vs K.
       The "elbow" — where adding more clusters gives diminishing returns —
       is the optimal K.

    2. Silhouette Score:
       Measures how similar a customer is to its own cluster vs other clusters.
       Score range: -1 to +1 (higher is better, > 0.35 is acceptable).

    Returns the K with the highest Silhouette Score.
    """
    print("\n" + "─"*50)
    print("  FINDING OPTIMAL K (Elbow + Silhouette)")
    print("─"*50)

    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  K={k}  Inertia={km.inertia_:>10,.0f}  Silhouette={sil:.4f}")

    # Best K = highest silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  ✅ Recommended K: {best_k} (Silhouette={max(silhouettes):.4f})")

    return best_k, list(k_range), inertias, silhouettes


# ─────────────────────────────────────────────
# STEP 4 — FIT FINAL KMEANS
# ─────────────────────────────────────────────

def fit_kmeans(X_scaled: pd.DataFrame, k: int):
    """
    Fit the final K-Means model with the chosen K.
    n_init=20 runs K-Means 20 times with different random seeds
    and keeps the best result (avoids local minima).
    """
    print(f"\n[SEGMENT] Fitting final KMeans with K={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, labels)
    print(f"[SEGMENT] Final Silhouette Score: {final_sil:.4f}")
    return km, labels


# ─────────────────────────────────────────────
# STEP 5 — AUTO-LABEL CLUSTERS
# ─────────────────────────────────────────────

def auto_label_clusters(df: pd.DataFrame, labels: np.ndarray, k: int) -> dict:
    """
    Automatically generate a meaningful name for each cluster by inspecting
    its centroid characteristics relative to the overall dataset means.

    Logic:
      - High Income + High Spend + Low Recency + High Campaign_History → "Champions"
      - High Recency (inactive) + previously active → "At-Risk"
      - Has Children + Moderate spend + Many store purchases → "Loyal Families"
      - Low income + High NumDealsPurchases → "Budget Shoppers"
      - Everything else → "Casual Buyers"

    These labels are heuristic — you should always verify them by
    looking at the cluster profile plots.
    """
    df = df.copy()
    df["Cluster"] = labels

    cluster_profiles = {}
    overall_means = df[CLUSTER_FEATURES].mean()

    segment_names = []
    for c in range(k):
        cluster_df   = df[df["Cluster"] == c]
        cluster_mean = cluster_df[CLUSTER_FEATURES].mean()

        # Relative to overall: positive = above average, negative = below
        relative = (cluster_mean - overall_means) / (overall_means + 1)

        # Scoring heuristics
        is_high_value    = relative["Total_Spend"] > 0.3 and relative["Income"] > 0.1
        is_inactive      = relative["Recency"] > 0.3      # high recency = not bought recently
        is_responsive    = relative["Campaign_History"] > 0.3
        has_children     = relative["Has_Children"] > 0.1
        is_deal_seeker   = relative["NumDealsPurchases"] > 0.3
        is_digital       = relative["NumWebVisitsMonth"] > 0.2

        if is_high_value and is_responsive:
            name = "Champions"
            desc = "High income, big spenders, historically very responsive to campaigns"
        elif is_high_value and not is_responsive:
            name = "High-Value Inactives"
            desc = "Big spenders but not responsive to campaigns — need a different approach"
        elif is_inactive and not is_high_value:
            name = "At-Risk"
            desc = "Haven't purchased recently, low campaign history — churn risk"
        elif has_children and not is_high_value:
            name = "Loyal Families"
            desc = "Moderate spenders with children, consistent store shoppers"
        elif is_deal_seeker:
            name = "Budget Shoppers"
            desc = "Low spenders, heavily discount-driven, many deals purchases"
        elif is_digital:
            name = "Digital Natives"
            desc = "High web engagement, prefer online channels"
        else:
            name = f"Segment {c}"
            desc = "Mixed profile group"

        cluster_profiles[c] = {
            "name": name,
            "description": desc,
            "size": len(cluster_df),
            "pct": len(cluster_df) / len(df) * 100,
            "profile": cluster_mean.to_dict(),
        }

        print(f"  Cluster {c}: {name:25s} | {len(cluster_df):4d} customers ({len(cluster_df)/len(df)*100:.1f}%)")
        print(f"             {desc}")

    return cluster_profiles


# ─────────────────────────────────────────────
# STEP 6 — PCA PROJECTION
# ─────────────────────────────────────────────

def compute_pca(X_scaled: pd.DataFrame):
    """
    Reduce the high-dimensional clustering features to 2D for visualisation.
    PCA (Principal Component Analysis) finds the 2 axes that capture
    the most variance in the data.

    Note: This is ONLY for plotting. The actual clustering used all features.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA] 2D projection explains {var_explained:.1f}% of total variance")
    return coords, pca, var_explained


# ─────────────────────────────────────────────
# STEP 7 — PLOT
# ─────────────────────────────────────────────

def plot_segmentation(k_vals, inertias, silhouettes, best_k,
                      pca_coords, labels, cluster_profiles, X_raw, var_explained):
    """
    4-panel segmentation report:
      Panel 1: Elbow Curve (inertia vs K)
      Panel 2: Silhouette Score vs K
      Panel 3: 2D PCA Scatter coloured by segment
      Panel 4: Cluster profile comparison (bar chart for key metrics)
    """
    print("\n[PLOT] Generating segmentation report...")

    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f1a")
    fig.suptitle("CampaignIQ — Customer Segmentation Report",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    PALETTE = ["#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#3b82f6",
               "#ec4899", "#14b8a6", "#a855f7"]

    # ── Panel 1: Elbow Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1a2e")
    for spine in ax1.spines.values(): spine.set_edgecolor("#2d2d4e")
    ax1.plot(k_vals, inertias, "o-", color="#7c3aed", lw=2.5, ms=8)
    ax1.axvline(best_k, color="#f59e0b", lw=2, linestyle="--", label=f"Best K={best_k}")
    ax1.set(xlabel="Number of Clusters (K)", ylabel="Inertia (WCSS)", title="Elbow Method")
    ax1.tick_params(colors="white"); ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white"); ax1.title.set_color("white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white"); ax1.grid(alpha=0.15)

    # ── Panel 2: Silhouette Scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1a2e")
    for spine in ax2.spines.values(): spine.set_edgecolor("#2d2d4e")
    bar_colors = ["#f59e0b" if k == best_k else "#4a4a6a" for k in k_vals]
    ax2.bar(k_vals, silhouettes, color=bar_colors, edgecolor="none", width=0.6)
    ax2.axhline(0.35, color="#10b981", lw=1.5, linestyle="--", label="Good threshold (0.35)")
    ax2.set(xlabel="Number of Clusters (K)", ylabel="Silhouette Score", title="Silhouette Analysis")
    ax2.tick_params(colors="white"); ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white"); ax2.title.set_color("white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white"); ax2.grid(alpha=0.15, axis="y")

    # ── Panel 3: 2D PCA Scatter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1a1a2e")
    for spine in ax3.spines.values(): spine.set_edgecolor("#2d2d4e")
    for c, profile in cluster_profiles.items():
        mask = labels == c
        ax3.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                    c=PALETTE[c % len(PALETTE)], alpha=0.55, s=20,
                    edgecolors="none", label=profile["name"])
    ax3.set(xlabel=f"PC1", ylabel="PC2",
            title=f"Customer Segments (PCA 2D, {var_explained:.0f}% variance)")
    ax3.tick_params(colors="white"); ax3.xaxis.label.set_color("white")
    ax3.yaxis.label.set_color("white"); ax3.title.set_color("white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, markerscale=2)
    ax3.grid(alpha=0.1)

    # ── Panel 4: Cluster Profile Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1a1a2e")
    for spine in ax4.spines.values(): spine.set_edgecolor("#2d2d4e")

    profile_metrics = ["Income", "Total_Spend", "Recency", "Campaign_History", "Total_Purchases"]
    n_clusters = len(cluster_profiles)
    x = np.arange(len(profile_metrics))
    bar_width = 0.8 / n_clusters

    for idx, (c, profile) in enumerate(cluster_profiles.items()):
        vals = []
        for m in profile_metrics:
            raw = X_raw[m]
            normalized = (profile["profile"].get(m, 0) - raw.min()) / (raw.max() - raw.min() + 1e-9)
            vals.append(normalized)
        offset = (idx - n_clusters / 2) * bar_width + bar_width / 2
        ax4.bar(x + offset, vals, width=bar_width,
                color=PALETTE[idx % len(PALETTE)], label=profile["name"],
                alpha=0.85, edgecolor="none")

    ax4.set_xticks(x)
    ax4.set_xticklabels(profile_metrics, color="white", fontsize=9, rotation=20)
    ax4.set(ylabel="Normalised Value (0-1)", title="Cluster Profiles (Key Metrics)")
    ax4.tick_params(colors="white"); ax4.yaxis.label.set_color("white")
    ax4.title.set_color("white")
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax4.grid(alpha=0.15, axis="y")

    plt.savefig(REPORT_PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")
    plt.close()
    print(f"[PLOT] Saved to: {REPORT_PLOT_PATH}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run(k_override: int = None):
    """
    Full segmentation pipeline.

    Args:
        k_override — if set, skip K selection and use this K directly
    """
    print("\n" + "="*55)
    print("  CampaignIQ — Customer Segmentation Training")
    print("="*55)

    # 1. Load
    df, X_raw = load_segmentation_data()

    # 2. Scale
    X_scaled, scaler = scale_features(X_raw)

    # 3. Find optimal K (or use override)
    if k_override:
        best_k = k_override
        k_vals, inertias, silhouettes = [], [], []
        print(f"[SEGMENT] Using K={best_k} (manual override)")
    else:
        best_k, k_vals, inertias, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 10))

    # 4. Fit final model
    km, labels = fit_kmeans(X_scaled, best_k)

    # 5. Auto-label clusters
    print("\n" + "─"*50)
    print("  CLUSTER PROFILES")
    print("─"*50)
    cluster_profiles = auto_label_clusters(df, labels, best_k)

    # 6. PCA for visualisation
    pca_coords, pca_model, var_explained = compute_pca(X_scaled)

    # 7. Plot
    if k_vals:  # skip plot if k_override (no elbow data)
        plot_segmentation(k_vals, inertias, silhouettes, best_k,
                          pca_coords, labels, cluster_profiles, X_raw, var_explained)

    # 8. Save model + scaler
    joblib.dump(km, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(pca_model, os.path.join(MODELS_DIR, "segmentation_pca.pkl"))
    print(f"\n[SAVE] KMeans model saved → {MODEL_SAVE_PATH}")
    print(f"[SAVE] Scaler saved       → {SCALER_SAVE_PATH}")

    # 9. Add segment labels back to df and show summary
    df["Segment_ID"] = labels
    df["Segment_Name"] = [cluster_profiles[c]["name"] for c in labels]

    print("\n" + "="*55)
    print("  Segmentation Training Complete ✅")
    print("="*55 + "\n")

    return km, scaler, labels, cluster_profiles, df


if __name__ == "__main__":
    run()

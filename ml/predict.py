"""
predict.py
===========
CampaignIQ — Inference Engine

Loads all trained models and runs predictions on the full customer dataset.
Produces a final predictions DataFrame that combines all 3 model outputs:
  - response_probability  (from Response Classifier)
  - top_product / predicted spend per category (from Affinity Models)
  - segment_name (from K-Means Segmentation)

This is the file you run AFTER training to score all customers.
It can also score a brand-new customer in real time given their attributes.

Author: CampaignIQ
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import run_pipeline
from ml.train_affinity import PRODUCT_TARGETS, predict_product_ranking

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

CLASSIFIER_PATH  = os.path.join(MODELS_DIR, "response_classifier.pkl")
SEG_KMEANS_PATH  = os.path.join(MODELS_DIR, "segmentation_kmeans.pkl")
SEG_SCALER_PATH  = os.path.join(MODELS_DIR, "segmentation_scaler.pkl")

CLUSTER_FEATURES = [
    "Income", "Total_Spend", "Recency", "Campaign_History", "Total_Purchases",
    "Avg_Basket_Size", "Has_Children", "Age", "NumWebVisitsMonth", "NumDealsPurchases",
]


def load_models():
    """Load all saved model files from disk."""
    print("[PREDICT] Loading models...")

    models = {}

    # Classifier
    if os.path.exists(CLASSIFIER_PATH):
        models["classifier"] = joblib.load(CLASSIFIER_PATH)
        print(f"  ✅ Response Classifier loaded")
    else:
        print(f"  ⚠️  Response Classifier not found at {CLASSIFIER_PATH}")
        print(f"      Run: python ml/train_classifier.py")

    # Affinity models (6 products)
    affinity_models = {}
    for col, product_name in PRODUCT_TARGETS.items():
        path = os.path.join(MODELS_DIR, f"affinity_{product_name.lower()}.pkl")
        if os.path.exists(path):
            affinity_models[product_name] = joblib.load(path)
        else:
            print(f"  ⚠️  Affinity model for {product_name} not found. Run train_affinity.py")

    if affinity_models:
        models["affinity"] = affinity_models
        print(f"  ✅ Affinity Models loaded ({len(affinity_models)}/6 products)")

    # Segmentation
    if os.path.exists(SEG_KMEANS_PATH):
        models["kmeans"]  = joblib.load(SEG_KMEANS_PATH)
        models["seg_scaler"] = joblib.load(SEG_SCALER_PATH)
        print(f"  ✅ Segmentation Model loaded")
    else:
        print(f"  ⚠️  Segmentation model not found. Run train_segmentation.py")

    return models


def predict_all_customers(models: dict) -> pd.DataFrame:
    """
    Run full inference over all customers in the dataset.

    Returns a DataFrame with columns:
      ID, response_probability, will_respond,
      top_product, pred_Wines, pred_Meat, ...,
      segment_id, segment_name  (if segment labels available from training)
    """
    print("\n[PREDICT] Running inference on all customers...")

    # Load and preprocess data
    df, feature_cols = run_pipeline()
    X = df[feature_cols]

    results = pd.DataFrame({"ID": df["ID"].values})

    # ── 1. Response Probability
    if "classifier" in models:
        clf = models["classifier"]
        y_prob = clf.predict_proba(X.values)[:, 1]
        results["response_probability"] = np.round(y_prob, 4)
        results["will_respond"]         = (y_prob >= 0.5).astype(int)
        results["response_tier"] = pd.cut(
            y_prob,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )
        print(f"  ✅ Response probabilities computed")
        print(f"     Predicted YES: {results['will_respond'].sum()} customers "
              f"({results['will_respond'].mean()*100:.1f}%)")

    # ── 2. Product Affinity
    if "affinity" in models:
        affinity_models = models["affinity"]
        affinity_df = predict_product_ranking(affinity_models, X)
        for col in affinity_df.columns:
            results[col] = affinity_df[col].values
        top_counts = results["top_product"].value_counts()
        print(f"  ✅ Product affinity scores computed")
        print(f"     Top product distribution:\n{top_counts.to_string()}")

    # ── 3. Segmentation
    if "kmeans" in models:
        km      = models["kmeans"]
        scaler  = models["seg_scaler"]
        X_seg   = df[CLUSTER_FEATURES]
        X_scaled = scaler.transform(X_seg)
        seg_labels = km.predict(X_scaled)
        results["segment_id"] = seg_labels
        print(f"  ✅ Customer segments assigned")
        print(f"     Segment distribution:\n{pd.Series(seg_labels).value_counts().sort_index().to_string()}")

    print(f"\n[PREDICT] Done. Predictions for {len(results)} customers ready.")
    return results, df


def predict_new_customer(models: dict, customer_dict: dict, feature_cols: list) -> dict:
    """
    Score a single brand-new customer (not in the database).

    Args:
        models        — loaded model dict from load_models()
        customer_dict — dict of raw customer attributes (same fields as dataset)
        feature_cols  — feature column list from run_pipeline()

    Returns dict with all predictions.

    Example customer_dict:
        {
            "Income": 55000,
            "Age": 38,
            "Education_Enc": 2,        # Graduation
            "Recency": 20,
            "Total_Spend": 800,
            "Has_Children": 1,
            ...
        }
    """
    import warnings; warnings.filterwarnings("ignore")

    # Build a 1-row DataFrame in the expected feature order
    row = pd.DataFrame([customer_dict])

    # Ensure all feature cols are present (fill missing with 0)
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_cols]

    result = {}

    # Response
    if "classifier" in models:
        prob = models["classifier"].predict_proba(row.values)[0][1]
        result["response_probability"] = round(float(prob), 4)
        result["will_respond"]         = bool(prob >= 0.5)
        result["response_tier"]        = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    # Product affinity
    if "affinity" in models:
        preds = {}
        for product_name, model in models["affinity"].items():
            pred = max(0, float(model.predict(row.values)[0]))
            preds[product_name] = round(pred, 2)
        result["predicted_spend"]        = preds
        result["top_product"]            = max(preds, key=preds.get)
        result["product_affinity_ranking"] = sorted(preds, key=preds.get, reverse=True)

    # Segmentation
    if "kmeans" in models:
        seg_cols = [c for c in CLUSTER_FEATURES if c in customer_dict]
        if len(seg_cols) == len(CLUSTER_FEATURES):
            X_seg   = pd.DataFrame([[customer_dict.get(c, 0) for c in CLUSTER_FEATURES]],
                                   columns=CLUSTER_FEATURES)
            X_scaled = models["seg_scaler"].transform(X_seg)
            seg_id   = int(models["kmeans"].predict(X_scaled)[0])
            result["segment_id"] = seg_id

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    models = load_models()

    if models:
        predictions_df, df = predict_all_customers(models)
        print("\nSample predictions (first 5 customers):")
        display_cols = ["ID", "response_probability", "will_respond",
                        "response_tier", "top_product", "segment_id"]
        display_cols = [c for c in display_cols if c in predictions_df.columns]
        print(predictions_df[display_cols].head(10).to_string(index=False))
    else:
        print("\n[ERROR] No models found. Train models first:")
        print("  python ml/train_classifier.py")
        print("  python ml/train_affinity.py")
        print("  python ml/train_segmentation.py")

"""
app.py
=======
CampaignIQ — Flask API Backend

Loads all 6 trained models on startup, runs inference over all 2,237
customers, then serves the results via REST API endpoints consumed
by the dashboard frontend.

Run with:  python backend/app.py
           python backend/app.py --port 5000

API Endpoints:
  GET  /api/overview            - KPI summary stats
  GET  /api/customers           - all customers + predictions (paginated, filterable)
  GET  /api/customer/<id>       - single customer full profile
  GET  /api/segments            - segment breakdown with profiles
  GET  /api/top-products        - product distribution
  GET  /api/model-insights      - feature importances + model metrics
  GET  /api/response-dist       - response probability histogram data
  POST /api/predict             - predict for a new customer (real-time)
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
import joblib

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")

# ─────────────────────────────────────────────────────────
# STARTUP: Load models + run inference on all customers
# ─────────────────────────────────────────────────────────

MODELS   = {}
PREDS_DF = None   # cached predictions for all customers
RAW_DF   = None   # original preprocessed customer data

SEGMENT_NAMES = {
    0: "High-Value Inactives",
    1: "Loyal Families",
    2: "Champions",
    3: "At-Risk",
    4: "Budget Shoppers",
    5: "Digital Natives",
}

CHANNEL_NAMES_MAP = {0: "Store", 1: "Web", 2: "Catalog"}


def load_all_models():
    global MODELS
    print("[SERVER] Loading models...")

    model_files = {
        "classifier":    "response_classifier.pkl",
        "churn":         "churn_classifier.pkl",
        "ltv":           "ltv_regressor.pkl",
        "channel":       "channel_classifier.pkl",
        "kmeans":        "segmentation_kmeans.pkl",
        "seg_scaler":    "segmentation_scaler.pkl",
        "churn_feats":   "churn_features.pkl",
        "ltv_feats":     "ltv_features.pkl",
        "channel_feats": "channel_features.pkl",
        "channel_names": "channel_names.pkl",
    }

    for key, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            MODELS[key] = joblib.load(path)
        else:
            print(f"  [WARN] {fname} not found — some predictions may be unavailable")

    # Affinity models
    MODELS["affinity"] = {}
    for product in ["Wines", "Meat", "Fruits", "Fish", "Sweets", "Gold"]:
        path = os.path.join(MODELS_DIR, f"affinity_{product.lower()}.pkl")
        if os.path.exists(path):
            MODELS["affinity"][product] = joblib.load(path)

    loaded = [k for k in MODELS if k not in ("affinity",)] + [f"affinity_{p}" for p in MODELS.get("affinity", {})]
    print(f"  Loaded {len(loaded)} model artifacts")


def run_all_predictions():
    global PREDS_DF, RAW_DF
    from ml.preprocess import run_pipeline

    print("[SERVER] Running inference on all customers...")
    df, feature_cols = run_pipeline()
    RAW_DF = df.copy()
    X = df[feature_cols]

    results = pd.DataFrame({"ID": df["ID"].astype(int).values})
    results["year_birth"]      = df["Year_Birth"].astype(int).values
    results["age"]             = df["Age"].astype(int).values
    results["income"]          = df["Income"].round(0).astype(int).values
    results["education"]       = df["Education_Enc"].values
    results["is_partnered"]    = df["Is_Partnered"].astype(int).values
    results["has_children"]    = df["Has_Children"].astype(int).values
    results["total_children"]  = df["Total_Children"].astype(int).values
    results["recency"]         = df["Recency"].astype(int).values
    results["tenure_days"]     = df["Customer_Tenure_Days"].round(0).astype(int).values
    results["total_spend"]     = df["Total_Spend"].round(2).values
    results["campaign_history"]= df["Campaign_History"].astype(int).values
    results["complain"]        = df["Complain"].astype(int).values

    # 1. Response probability
    if "classifier" in MODELS:
        probs = MODELS["classifier"].predict_proba(X.values)[:, 1]
        results["response_prob"] = np.round(probs, 4)
        results["will_respond"]  = (probs >= 0.5).astype(int)
        results["response_tier"] = pd.cut(probs, bins=[0,0.3,0.6,1.0],
                                          labels=["Low","Medium","High"]).astype(str)

    # 2. Churn risk
    if "churn" in MODELS and "churn_feats" in MODELS:
        churn_feats = MODELS["churn_feats"]
        X_churn = df[[c for c in churn_feats if c in df.columns]]
        # Align columns
        missing = [c for c in churn_feats if c not in df.columns]
        for m in missing:
            X_churn[m] = 0
        X_churn = X_churn[churn_feats]
        churn_probs = MODELS["churn"].predict_proba(X_churn.values)[:, 1]
        results["churn_prob"]    = np.round(churn_probs, 4)
        results["churn_risk"]    = pd.cut(churn_probs, bins=[0,0.35,0.65,1.0],
                                          labels=["Low","Medium","High"]).astype(str)

    # 3. LTV
    if "ltv" in MODELS and "ltv_feats" in MODELS:
        ltv_feats = MODELS["ltv_feats"]
        X_ltv = df[[c for c in ltv_feats if c in df.columns]]
        missing = [c for c in ltv_feats if c not in df.columns]
        for m in missing:
            X_ltv[m] = 0
        X_ltv = X_ltv[ltv_feats]
        ltv_preds = np.clip(MODELS["ltv"].predict(X_ltv.values), 0, None)
        results["predicted_ltv"] = np.round(ltv_preds, 2)

    # 4. Channel recommendation
    if "channel" in MODELS and "channel_feats" in MODELS:
        ch_feats = MODELS["channel_feats"]
        X_ch = df[[c for c in ch_feats if c in df.columns]]
        missing = [c for c in ch_feats if c not in df.columns]
        for m in missing:
            X_ch[m] = 0
        X_ch = X_ch[ch_feats]
        ch_labels = MODELS["channel"].predict(X_ch.values)
        ch_names_map = MODELS.get("channel_names", CHANNEL_NAMES_MAP)
        results["recommended_channel"] = [ch_names_map[int(l)] for l in ch_labels]

    # 5. Product affinity
    if MODELS["affinity"]:
        for product, model in MODELS["affinity"].items():
            preds = np.clip(model.predict(X.values), 0, None)
            results[f"pred_{product.lower()}"] = np.round(preds, 2)
        spend_cols = [f"pred_{p.lower()}" for p in MODELS["affinity"]]
        results["top_product"] = results[spend_cols].idxmax(axis=1).str.replace("pred_", "").str.capitalize()

    # 6. Segmentation
    CLUSTER_FEATURES = ["Income","Total_Spend","Recency","Campaign_History","Total_Purchases",
                        "Avg_Basket_Size","Has_Children","Age","NumWebVisitsMonth","NumDealsPurchases"]
    if "kmeans" in MODELS and "seg_scaler" in MODELS:
        X_seg = df[[c for c in CLUSTER_FEATURES if c in df.columns]]
        X_scaled = MODELS["seg_scaler"].transform(X_seg)
        seg_ids = MODELS["kmeans"].predict(X_scaled)
        results["segment_id"]   = seg_ids
        results["segment_name"] = [SEGMENT_NAMES.get(int(s), f"Segment {s}") for s in seg_ids]

    PREDS_DF = results
    print(f"[SERVER] Predictions ready for {len(PREDS_DF)} customers")


# ─────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────

EDU_LABELS = {0:"Basic",1:"2n Cycle",2:"Graduation",3:"Master",4:"PhD"}

def safe_val(v):
    """Convert numpy types to plain Python for JSON serialisation."""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    return v

def row_to_dict(row):
    return {k: safe_val(v) for k, v in row.items()}


# ─────────────────────────────────────────────────────────
# ROUTES — Frontend
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ─────────────────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────────────────

@app.route("/api/overview")
def api_overview():
    if PREDS_DF is None:
        return jsonify({"error": "Models not loaded"}), 503

    df = PREDS_DF
    total = len(df)

    overview = {
        "total_customers":      total,
        "predicted_responders": int(df["will_respond"].sum()) if "will_respond" in df else 0,
        "response_rate_pct":    round(df["will_respond"].mean() * 100, 1) if "will_respond" in df else 0,
        "avg_ltv":              round(float(df["predicted_ltv"].mean()), 0) if "predicted_ltv" in df else 0,
        "high_churn_risk":      int((df["churn_risk"] == "High").sum()) if "churn_risk" in df else 0,
        "churn_risk_pct":       round((df["churn_risk"] == "High").mean() * 100, 1) if "churn_risk" in df else 0,
        "avg_response_prob":    round(float(df["response_prob"].mean()), 3) if "response_prob" in df else 0,
        "total_segments":       int(df["segment_id"].nunique()) if "segment_id" in df else 0,
    }
    return jsonify(overview)


@app.route("/api/customers")
def api_customers():
    if PREDS_DF is None:
        return jsonify({"error": "Models not loaded"}), 503

    df = PREDS_DF.copy()
    page      = int(request.args.get("page", 1))
    per_page  = int(request.args.get("per_page", 50))
    search    = request.args.get("search", "").strip()
    segment   = request.args.get("segment", "")
    product   = request.args.get("product", "")
    resp_tier = request.args.get("response_tier", "")
    churn     = request.args.get("churn_risk", "")
    channel   = request.args.get("channel", "")
    sort_by   = request.args.get("sort_by", "response_prob")
    sort_dir  = request.args.get("sort_dir", "desc")

    # Filters
    if search:
        df = df[df["ID"].astype(str).str.contains(search)]
    if segment and "segment_name" in df:
        df = df[df["segment_name"] == segment]
    if product and "top_product" in df:
        df = df[df["top_product"].str.lower() == product.lower()]
    if resp_tier and "response_tier" in df:
        df = df[df["response_tier"] == resp_tier]
    if churn and "churn_risk" in df:
        df = df[df["churn_risk"] == churn]
    if channel and "recommended_channel" in df:
        df = df[df["recommended_channel"] == channel]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))

    total_filtered = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start: start + per_page]

    return jsonify({
        "total": total_filtered,
        "page": page,
        "per_page": per_page,
        "customers": [row_to_dict(r) for _, r in page_df.iterrows()]
    })


@app.route("/api/customer/<int:customer_id>")
def api_customer(customer_id):
    if PREDS_DF is None:
        return jsonify({"error": "Models not loaded"}), 503

    row = PREDS_DF[PREDS_DF["ID"] == customer_id]
    if row.empty:
        return jsonify({"error": "Customer not found"}), 404

    data = row_to_dict(row.iloc[0])

    # Add education label
    data["education_label"] = EDU_LABELS.get(data.get("education"), "Unknown")

    # Add spend breakdown from RAW_DF
    if RAW_DF is not None:
        raw_row = RAW_DF[RAW_DF["ID"] == customer_id]
        if not raw_row.empty:
            r = raw_row.iloc[0]
            data["actual_spend"] = {
                "Wines":  safe_val(r.get("MntWines", 0)),
                "Meat":   safe_val(r.get("MntMeatProducts", 0)),
                "Fish":   safe_val(r.get("MntFishProducts", 0)),
                "Fruits": safe_val(r.get("MntFruits", 0)),
                "Sweets": safe_val(r.get("MntSweetProducts", 0)),
                "Gold":   safe_val(r.get("MntGoldProds", 0)),
            }
            data["purchase_channels"] = {
                "Web":     safe_val(r.get("NumWebPurchases", 0)),
                "Store":   safe_val(r.get("NumStorePurchases", 0)),
                "Catalog": safe_val(r.get("NumCatalogPurchases", 0)),
            }

    return jsonify(data)


@app.route("/api/segments")
def api_segments():
    if PREDS_DF is None:
        return jsonify([])

    df = PREDS_DF
    segments = []
    for seg_id in sorted(df["segment_id"].unique()):
        seg_df = df[df["segment_id"] == seg_id]
        seg = {
            "id":             int(seg_id),
            "name":           SEGMENT_NAMES.get(int(seg_id), f"Segment {seg_id}"),
            "count":          int(len(seg_df)),
            "pct":            round(len(seg_df) / len(df) * 100, 1),
            "avg_response":   round(float(seg_df["response_prob"].mean()), 3) if "response_prob" in seg_df else 0,
            "avg_ltv":        round(float(seg_df["predicted_ltv"].mean()), 0) if "predicted_ltv" in seg_df else 0,
            "avg_churn":      round(float(seg_df["churn_prob"].mean()), 3) if "churn_prob" in seg_df else 0,
            "top_product":    seg_df["top_product"].mode()[0] if "top_product" in seg_df else "N/A",
            "top_channel":    seg_df["recommended_channel"].mode()[0] if "recommended_channel" in seg_df else "N/A",
        }
        segments.append(seg)
    return jsonify(segments)


@app.route("/api/top-products")
def api_top_products():
    if PREDS_DF is None or "top_product" not in PREDS_DF:
        return jsonify([])
    counts = PREDS_DF["top_product"].value_counts()
    return jsonify([{"product": p, "count": int(c)} for p, c in counts.items()])


@app.route("/api/response-dist")
def api_response_dist():
    if PREDS_DF is None or "response_prob" not in PREDS_DF:
        return jsonify([])
    probs = PREDS_DF["response_prob"].values
    counts, edges = np.histogram(probs, bins=20, range=(0, 1))
    return jsonify([
        {"range": f"{edges[i]:.2f}-{edges[i+1]:.2f}", "count": int(counts[i])}
        for i in range(len(counts))
    ])


@app.route("/api/model-insights")
def api_model_insights():
    """Return feature importances and evaluation metrics for model insights page."""
    insights = {"feature_importances": {}, "metrics": {}}

    from ml.preprocess import run_pipeline
    _, feature_cols = run_pipeline()

    # Classifier feature importance
    if "classifier" in MODELS:
        clf = MODELS["classifier"]
        imps = clf.feature_importances_
        feat_imp = sorted(zip(feature_cols, imps.tolist()), key=lambda x: x[1], reverse=True)[:20]
        insights["feature_importances"]["response"] = [{"feature": f, "importance": round(i, 5)} for f, i in feat_imp]

    # Churn feature importance
    if "churn" in MODELS and "churn_feats" in MODELS:
        ch_feats = MODELS["churn_feats"]
        imps = MODELS["churn"].feature_importances_
        feat_imp = sorted(zip(ch_feats, imps.tolist()), key=lambda x: x[1], reverse=True)[:15]
        insights["feature_importances"]["churn"] = [{"feature": f, "importance": round(i, 5)} for f, i in feat_imp]

    # LTV feature importance
    if "ltv" in MODELS and "ltv_feats" in MODELS:
        ltv_feats = MODELS["ltv_feats"]
        imps = MODELS["ltv"].feature_importances_
        feat_imp = sorted(zip(ltv_feats, imps.tolist()), key=lambda x: x[1], reverse=True)[:15]
        insights["feature_importances"]["ltv"] = [{"feature": f, "importance": round(i, 5)} for f, i in feat_imp]

    if PREDS_DF is not None:
        df = PREDS_DF
        insights["metrics"] = {
            "total_customers":      len(df),
            "predicted_responders": int(df["will_respond"].sum()) if "will_respond" in df else 0,
            "high_churn_customers": int((df["churn_risk"] == "High").sum()) if "churn_risk" in df else 0,
            "avg_predicted_ltv":    round(float(df["predicted_ltv"].mean()), 2) if "predicted_ltv" in df else 0,
        }

    return jsonify(insights)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Real-time prediction for a new customer."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    from ml.preprocess import run_pipeline, REFERENCE_DATE, EDUCATION_ORDER
    _, feature_cols = run_pipeline()

    # Build feature row from incoming data
    # Expected fields: income, age, education (0-4), is_partnered, has_children,
    #                  total_children, recency, tenure_days, campaign_history, complain
    income          = float(data.get("income", 50000))
    age             = int(data.get("age", 40))
    education_enc   = int(data.get("education", 2))
    is_partnered    = int(data.get("is_partnered", 0))
    has_children    = int(data.get("has_children", 0))
    total_children  = int(data.get("total_children", 0))
    recency         = int(data.get("recency", 30))
    tenure_days     = int(data.get("tenure_days", 1000))
    campaign_history= int(data.get("campaign_history", 0))
    complain        = int(data.get("complain", 0))
    num_deals       = int(data.get("num_deals", 2))
    num_web_visits  = int(data.get("num_web_visits", 4))

    # Build a template row: start with median values, override with provided
    from ml.preprocess import run_pipeline
    base_df, fcols = run_pipeline()
    row = base_df[fcols].median().to_dict()

    # Override with user input
    row["Income"]             = income
    row["Age"]                = age
    row["Education_Enc"]      = education_enc
    row["Is_Partnered"]       = is_partnered
    row["Has_Children"]       = has_children
    row["Total_Children"]     = total_children
    row["Recency"]            = recency
    row["Customer_Tenure_Days"] = tenure_days
    row["Campaign_History"]   = campaign_history
    row["Complain"]           = complain
    row["NumDealsPurchases"]  = num_deals
    row["NumWebVisitsMonth"]  = num_web_visits
    # Marital dummies — reset and set based on is_partnered
    for col in [c for c in fcols if c.startswith("Marital_")]:
        row[col] = 0
    if is_partnered:
        row["Marital_Married"] = 1

    X_row = np.array([[row.get(c, 0) for c in fcols]])
    result = {}

    # Response
    if "classifier" in MODELS:
        prob = float(MODELS["classifier"].predict_proba(X_row)[0][1])
        result["response_probability"] = round(prob, 4)
        result["will_respond"]         = bool(prob >= 0.5)
        result["response_tier"]        = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    # Churn
    if "churn" in MODELS and "churn_feats" in MODELS:
        ch_feats = MODELS["churn_feats"]
        X_ch = np.array([[row.get(c, 0) for c in ch_feats]])
        churn_prob = float(MODELS["churn"].predict_proba(X_ch)[0][1])
        result["churn_probability"] = round(churn_prob, 4)
        result["churn_risk"]        = "High" if churn_prob > 0.65 else "Medium" if churn_prob > 0.35 else "Low"

    # LTV
    if "ltv" in MODELS and "ltv_feats" in MODELS:
        ltv_feats = MODELS["ltv_feats"]
        X_ltv = np.array([[row.get(c, 0) for c in ltv_feats]])
        ltv = max(0, float(MODELS["ltv"].predict(X_ltv)[0]))
        result["predicted_ltv"] = round(ltv, 2)

    # Channel
    if "channel" in MODELS and "channel_feats" in MODELS:
        ch_feats = MODELS["channel_feats"]
        X_ch = np.array([[row.get(c, 0) for c in ch_feats]])
        ch_label  = int(MODELS["channel"].predict(X_ch)[0])
        ch_probs  = MODELS["channel"].predict_proba(X_ch)[0].tolist()
        ch_names  = MODELS.get("channel_names", CHANNEL_NAMES_MAP)
        result["recommended_channel"]       = ch_names[ch_label]
        result["channel_probabilities"]     = {ch_names[i]: round(p, 3) for i, p in enumerate(ch_probs)}

    # Product affinity
    if MODELS["affinity"]:
        spend_preds = {}
        for product, model in MODELS["affinity"].items():
            spend_preds[product] = round(max(0, float(model.predict(X_row)[0])), 2)
        result["predicted_spend"] = spend_preds
        result["top_product"]     = max(spend_preds, key=spend_preds.get)
        result["affinity_ranking"]= sorted(spend_preds, key=spend_preds.get, reverse=True)

    # Segmentation
    CLUSTER_FEATURES = ["Income","Total_Spend","Recency","Campaign_History","Total_Purchases",
                        "Avg_Basket_Size","Has_Children","Age","NumWebVisitsMonth","NumDealsPurchases"]
    if "kmeans" in MODELS and "seg_scaler" in MODELS:
        X_seg = np.array([[row.get(c, 0) for c in CLUSTER_FEATURES]])
        X_scaled = MODELS["seg_scaler"].transform(X_seg)
        seg_id = int(MODELS["kmeans"].predict(X_scaled)[0])
        result["segment_id"]   = seg_id
        result["segment_name"] = SEGMENT_NAMES.get(seg_id, f"Segment {seg_id}")

    return jsonify(result)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    load_all_models()
    run_all_predictions()

    print(f"\n[SERVER] CampaignIQ dashboard running at:")
    print(f"         http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)

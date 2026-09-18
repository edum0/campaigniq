"""
train_channel.py
=================
Goal 6 — Campaign Channel Recommender

Answers: "Should we reach this customer via Web, Catalog, or Store?"
Target:  preferred_channel = argmax(NumWebPurchases, NumCatalogPurchases, NumStorePurchases)

The model learns which channel a customer naturally gravitates toward
based on their demographics and behaviour. This tells you HOW to reach
them, not just WHETHER to reach them.

Classes:
  0 = Store    (physical store is preferred channel)
  1 = Web      (online is preferred channel)
  2 = Catalog  (catalog is preferred channel)

Feature strategy:
  Exclude NumWebPurchases, NumCatalogPurchases, NumStorePurchases from features.
  These directly determine the label — using them would be leakage.
  The model must infer channel preference from demographics, spend, and tenure.

Author: CampaignIQ
"""

import os, sys, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import run_pipeline

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR,  "channel_classifier.pkl")
REPORT_PATH = os.path.join(REPORTS_DIR, "channel_evaluation.png")
RANDOM_STATE = 42

CHANNEL_NAMES = {0: "Store", 1: "Web", 2: "Catalog"}
# Columns that ARE the label — must be excluded from features
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
                "NumDealsPurchases", "Total_Purchases", "Web_Engagement_Ratio"]


def load_channel_data():
    """
    Build channel preference label from purchase channel counts.

    Tie-breaking order: Store > Web > Catalog
    (Store purchase is most intentional — requires physical effort)

    Returns X, y, feature_cols where y in {0=Store, 1=Web, 2=Catalog}.
    """
    df, all_feature_cols = run_pipeline()

    # Build channel label
    channel_matrix = pd.DataFrame({
        "Store":   df["NumStorePurchases"],
        "Web":     df["NumWebPurchases"],
        "Catalog": df["NumCatalogPurchases"],
    })
    # argmax picks the dominant channel; ties broken by column order (Store > Web > Catalog)
    channel_label = channel_matrix.idxmax(axis=1)
    label_map = {"Store": 0, "Web": 1, "Catalog": 2}
    y = channel_label.map(label_map).values

    # Remove channel-derived features
    feature_cols = [c for c in all_feature_cols if c not in CHANNEL_COLS]
    X = df[feature_cols]

    dist = pd.Series(y).map({v:k for k,v in label_map.items()}).value_counts()
    print(f"[CHANNEL] Channel distribution:\n{dist.to_string()}")
    print(f"[CHANNEL] Features: {len(feature_cols)} (channel count cols excluded)")
    return X, y, feature_cols, df


def tune_channel_model(X_train, y_train, n_classes, n_trials=40):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_class":        n_classes,
            "objective":        "multi:softprob",
            "eval_metric":      "mlogloss",
            "verbosity": 0, "random_state": RANDOM_STATE,
        }
        return cross_val_score(XGBClassifier(**params), X_train, y_train,
                               cv=cv, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"num_class": n_classes, "objective": "multi:softprob",
                 "eval_metric": "mlogloss", "verbosity": 0, "random_state": RANDOM_STATE})
    print(f"  Best CV F1-macro: {study.best_value:.4f}")
    return best


def plot_channel_evaluation(model, X_test, y_test, y_pred, y_prob, metrics, feature_cols):
    class_names = [CHANNEL_NAMES[i] for i in sorted(CHANNEL_NAMES)]
    fig = plt.figure(figsize=(18, 10), facecolor="#0f0f1a")
    fig.suptitle("CampaignIQ — Channel Recommender\nEvaluation Report",
                 fontsize=16, color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    COLORS = ["#7c3aed", "#10b981", "#f59e0b"]

    def style(ax):
        ax.set_facecolor("#1a1a2e")
        for s in ax.spines.values(): s.set_edgecolor("#2d2d4e")
        ax.tick_params(colors="white"); ax.grid(alpha=0.15)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1a1a2e")
    for s in ax1.spines.values(): s.set_edgecolor("#2d2d4e")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, linewidths=1, linecolor="#0f0f1a")
    ax1.set_title("Confusion Matrix", color="white"); ax1.tick_params(colors="white")

    # Per-class probability distribution
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    for i, name in CHANNEL_NAMES.items():
        ax2.hist(y_prob[:, i], bins=25, alpha=0.6, color=COLORS[i],
                 label=name, density=True, edgecolor="none")
    ax2.set(xlabel="Predicted Probability", ylabel="Density", title="Probability Distribution by Channel")
    ax2.xaxis.label.set_color("white"); ax2.yaxis.label.set_color("white"); ax2.title.set_color("white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Class distribution (actual vs predicted)
    ax3 = fig.add_subplot(gs[0, 2]); style(ax3)
    x = np.arange(len(class_names))
    act_counts  = [(y_test==i).sum()  for i in sorted(CHANNEL_NAMES)]
    pred_counts = [(y_pred==i).sum() for i in sorted(CHANNEL_NAMES)]
    ax3.bar(x - 0.2, act_counts,  0.35, label="Actual",    color="#4a4a6a", edgecolor="none")
    ax3.bar(x + 0.2, pred_counts, 0.35, label="Predicted", color=COLORS,  edgecolor="none", alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(class_names, color="white")
    ax3.set(title="Actual vs Predicted Distribution"); ax3.title.set_color("white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white")

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 0:2]); style(ax4)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(18)
    fcolors = ["#f59e0b" if i > feat_df["Importance"].median() else "#4a4a6a"
               for i in feat_df["Importance"]]
    ax4.barh(feat_df["Feature"], feat_df["Importance"], color=fcolors, edgecolor="none", height=0.6)
    ax4.set(xlabel="Importance (Gain)", title="Top Features for Channel Prediction")
    ax4.xaxis.label.set_color("white"); ax4.title.set_color("white")

    # Accuracy card
    ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor("#1a1a2e")
    for s in ax5.spines.values(): s.set_edgecolor("#2d2d4e")
    ax5.axis("off")
    metric_text = (
        f"Accuracy:  {metrics['accuracy']:.3f}\n\n"
        f"F1 Macro:  {metrics['f1_macro']:.3f}\n\n"
        f"F1 Weighted: {metrics['f1_weighted']:.3f}"
    )
    ax5.text(0.5, 0.5, metric_text, color="white", fontsize=14,
             ha="center", va="center", transform=ax5.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#7c3aed", alpha=0.25))
    ax5.set_title("Model Metrics", color="white")

    plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"[PLOT] Saved -> {REPORT_PATH}")


def run(tune=True, n_trials=40):
    print("\n" + "="*55)
    print("  CampaignIQ — Channel Recommender Training")
    print("="*55)

    X, y, feature_cols, df = load_channel_data()
    n_classes = len(np.unique(y))
    print(f"[CHANNEL] {n_classes} classes: {list(CHANNEL_NAMES.values())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    if tune:
        print("\n[TUNE] Running Optuna...")
        best_params = tune_channel_model(X_train.values, y_train, n_classes, n_trials)
    else:
        best_params = {"n_estimators":300, "max_depth":6, "learning_rate":0.05,
                       "subsample":0.8, "colsample_bytree":0.8,
                       "num_class": n_classes, "objective": "multi:softprob",
                       "eval_metric": "mlogloss", "verbosity":0, "random_state":RANDOM_STATE}

    model = XGBClassifier(**best_params)
    model.fit(X_train.values, y_train)
    y_prob = model.predict_proba(X_test.values)
    y_pred = model.predict(X_test.values)

    metrics = {
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "f1_macro":    float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }
    print(f"\n[EVAL] Accuracy={metrics['accuracy']:.4f}  "
          f"F1-macro={metrics['f1_macro']:.4f}  F1-weighted={metrics['f1_weighted']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=list(CHANNEL_NAMES.values()))}")

    plot_channel_evaluation(model, X_test.values, y_test, y_pred, y_prob, metrics, feature_cols)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "channel_features.pkl"))
    joblib.dump(CHANNEL_NAMES, os.path.join(MODELS_DIR, "channel_names.pkl"))
    print(f"[SAVE] Model saved -> {MODEL_PATH}")
    print("="*55 + "\n")
    return model, metrics, feature_cols


if __name__ == "__main__":
    run(tune=True, n_trials=40)

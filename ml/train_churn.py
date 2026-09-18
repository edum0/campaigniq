"""
train_churn.py
===============
Goal 4 — Churn Risk Classifier

Answers: "Which customers are about to stop buying from us?"

Since we have no explicit churn label in the dataset, we engineer one:
  churned = 1  if Recency > 60 days (hasn't purchased in 2+ months)
  churned = 0  if Recency <= 60 days (recently active)

This threshold captures the top ~43% of customers by recency,
making it a realistic "at-risk" flag.

Pipeline:
  1. Engineer churn label from Recency
  2. Train XGBoost classifier (no SMOTE needed — classes are ~57/43)
  3. Tune with Optuna
  4. Evaluate and plot
  5. Save model

Author: CampaignIQ
"""

import os, sys, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
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

MODEL_PATH  = os.path.join(MODELS_DIR,  "churn_classifier.pkl")
REPORT_PATH = os.path.join(REPORTS_DIR, "churn_evaluation.png")
RANDOM_STATE = 42
CHURN_RECENCY_THRESHOLD = 60  # days — customers not buying in 60+ days are "at risk"


def load_churn_data():
    """
    Load preprocessed data and engineer a binary churn label.

    Churn label engineering rationale:
      Recency = days since last purchase (0-99 range in this dataset).
      A customer who hasn't purchased in 60+ days shows disengagement signals.
      This threshold gives a ~43% positive rate — well-balanced, no SMOTE needed.

    We EXCLUDE Recency from the feature set after using it to make the label,
    to prevent the model from trivially learning "high recency = churn" which
    would be leakage, not learning.
    """
    df, feature_cols = run_pipeline()

    # Build churn label from Recency
    y = (df["Recency"] > CHURN_RECENCY_THRESHOLD).astype(int)

    # Remove Recency from features (it IS the label — leakage prevention)
    feature_cols = [c for c in feature_cols if c != "Recency"]
    X = df[feature_cols]

    pos_rate = y.mean() * 100
    print(f"[CHURN] Churn threshold: Recency > {CHURN_RECENCY_THRESHOLD} days")
    print(f"[CHURN] At-risk (1): {y.sum():,} | Active (0): {(y==0).sum():,} | Positive rate: {pos_rate:.1f}%")
    return X, y, feature_cols, df


def tune_churn_model(X_train, y_train, n_trials=40):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 700),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": 0, "random_state": RANDOM_STATE, "eval_metric": "logloss",
        }
        return cross_val_score(XGBClassifier(**params), X_train, y_train,
                               cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"verbosity": 0, "random_state": RANDOM_STATE, "eval_metric": "logloss"})
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return best


def plot_churn_evaluation(model, X_test, y_test, y_prob, metrics, feature_cols):
    fig = plt.figure(figsize=(18, 10), facecolor="#0f0f1a")
    fig.suptitle("CampaignIQ — Churn Risk Classifier\nEvaluation Report",
                 fontsize=16, color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ACCENT, GREEN, RED = "#ef4444", "#10b981", "#f59e0b"

    def style(ax):
        ax.set_facecolor("#1a1a2e")
        for s in ax.spines.values(): s.set_edgecolor("#2d2d4e")
        ax.tick_params(colors="white"); ax.grid(alpha=0.15)

    # ROC
    ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC={metrics['auc']:.3f}")
    ax1.plot([0,1],[0,1],"--",color="#555",lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax1.xaxis.label.set_color("white"); ax1.yaxis.label.set_color("white"); ax1.title.set_color("white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")

    # PR Curve
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    p_arr, r_arr, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(r_arr, p_arr, color=GREEN, lw=2.5, label=f"AP={metrics['ap']:.3f}")
    ax2.fill_between(r_arr, p_arr, alpha=0.15, color=GREEN)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax2.xaxis.label.set_color("white"); ax2.yaxis.label.set_color("white"); ax2.title.set_color("white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")

    # Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#1a1a2e")
    for s in ax3.spines.values(): s.set_edgecolor("#2d2d4e")
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Reds", ax=ax3,
                xticklabels=["Active","At-Risk"], yticklabels=["Active","At-Risk"],
                cbar=False, linewidths=1, linecolor="#0f0f1a")
    ax3.set_title("Confusion Matrix", color="white"); ax3.tick_params(colors="white")

    # Feature Importance
    ax4 = fig.add_subplot(gs[1, 0:2]); style(ax4)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(18)
    colors = [ACCENT if i > feat_df["Importance"].median() else "#4a4a6a" for i in feat_df["Importance"]]
    ax4.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="none", height=0.6)
    ax4.set(xlabel="Importance (Gain)", title="Top Features for Churn Prediction")
    ax4.xaxis.label.set_color("white"); ax4.title.set_color("white")

    # Score Distribution
    ax5 = fig.add_subplot(gs[1, 2]); style(ax5)
    ax5.hist(y_prob[y_test==0], bins=25, alpha=0.7, color=GREEN, label="Active", density=True)
    ax5.hist(y_prob[y_test==1], bins=25, alpha=0.7, color=RED,   label="At-Risk", density=True)
    ax5.axvline(0.5, color="white", lw=1.5, ls="--", label="Threshold")
    ax5.set(xlabel="Churn Probability", ylabel="Density", title="Score Distribution")
    ax5.xaxis.label.set_color("white"); ax5.yaxis.label.set_color("white"); ax5.title.set_color("white")
    ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"[PLOT] Saved -> {REPORT_PATH}")


def run(tune=True, n_trials=40):
    print("\n" + "="*55)
    print("  CampaignIQ — Churn Risk Classifier Training")
    print("="*55)

    X, y, feature_cols, df = load_churn_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    if tune:
        print("\n[TUNE] Running Optuna...")
        best_params = tune_churn_model(X_train.values, y_train.values, n_trials)
    else:
        best_params = {"n_estimators":300,"max_depth":6,"learning_rate":0.05,
                       "subsample":0.8,"colsample_bytree":0.8,"verbosity":0,
                       "random_state":RANDOM_STATE,"eval_metric":"logloss"}

    model = XGBClassifier(**best_params)
    model.fit(X_train.values, y_train.values)
    y_prob = model.predict_proba(X_test.values)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc":  roc_auc_score(y_test, y_prob),
        "ap":   average_precision_score(y_test, y_prob),
        "f1":   f1_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, zero_division=0),
        "rec":  recall_score(y_test, y_pred),
        "cm":   confusion_matrix(y_test, y_pred),
    }

    print(f"\n[EVAL] AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
          f"P={metrics['prec']:.3f}  R={metrics['rec']:.3f}")

    plot_churn_evaluation(model, X_test.values, y_test.values, y_prob, metrics, feature_cols)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "churn_features.pkl"))
    print(f"[SAVE] Model saved -> {MODEL_PATH}")
    print("="*55 + "\n")
    return model, metrics, feature_cols


if __name__ == "__main__":
    run(tune=True, n_trials=40)

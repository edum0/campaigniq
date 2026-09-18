"""
train_classifier.py
====================
Goal 1 — Campaign Response Classifier

Answers: "Will this customer accept the next campaign offer?"
Target:  `Response` (binary: 0 = No, 1 = Yes)

What this script does:
    1. Loads the preprocessed data
    2. Splits into train/test (stratified to preserve 85/15 class ratio)
    3. Applies SMOTE on the training set to fix class imbalance
    4. Trains and compares 4 models: Logistic Regression, Random Forest,
       XGBoost (primary), LightGBM
    5. Tunes the best model with Optuna (Bayesian hyperparameter search)
    6. Evaluates on held-out test set with full metrics
    7. Plots: ROC curve, Precision-Recall curve, Confusion Matrix,
              Feature Importance, Probability Distribution
    8. Saves the best model to ml/models/response_classifier.pkl

Author: CampaignIQ
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ── local
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import run_pipeline

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_SAVE_PATH  = os.path.join(MODELS_DIR,  "response_classifier.pkl")
SCALER_SAVE_PATH = os.path.join(MODELS_DIR,  "classifier_scaler.pkl")
REPORT_PLOT_PATH = os.path.join(REPORTS_DIR, "classifier_evaluation.png")

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_classifier_data():
    """
    Run the full preprocessing pipeline and extract
    features (X) and the binary Response target (y).
    """
    df, feature_cols = run_pipeline()

    X = df[feature_cols].values
    y = df["Response"].values.astype(int)

    print(f"[CLASSIFIER] X shape: {X.shape}")
    print(f"[CLASSIFIER] Class distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    print(f"[CLASSIFIER] Positive rate: {y.mean()*100:.1f}%")

    return X, y, feature_cols, df


# ─────────────────────────────────────────────
# STEP 2 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X, y):
    """
    Stratified split — preserves the 85/15 class ratio in both
    train and test sets so the test set is representative.
    80% train, 20% test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y          # ← critical: keeps class ratio intact
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"[SPLIT] Train positives: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"[SPLIT] Test  positives: {y_test.sum()}  ({y_test.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 3 — SMOTE (Fix Class Imbalance)
# ─────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """
    SMOTE = Synthetic Minority Oversampling TEchnique.

    Why: 85% of customers said NO. If we train directly, the model
    will just learn to always say NO and get 85% accuracy — useless.

    How SMOTE fixes this:
      - Takes each YES customer
      - Finds its k nearest neighbours (also YES customers)
      - Generates a synthetic customer halfway between them
      - Repeats until YES and NO counts are equal

    IMPORTANT: SMOTE is applied ONLY to training data.
    Never on test data (that would be data leakage / cheating).
    """
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"\n[SMOTE] Before: {(y_train==0).sum()} NO | {(y_train==1).sum()} YES")
    print(f"[SMOTE] After:  {(y_res==0).sum()} NO | {(y_res==1).sum()} YES (balanced)")
    return X_res, y_res


# ─────────────────────────────────────────────
# STEP 4 — BASELINE MODEL COMPARISON
# ─────────────────────────────────────────────

def compare_baseline_models(X_train_bal, y_train_bal, X_test, y_test):
    """
    Quickly train 4 models with default hyperparameters to see
    which architecture performs best before tuning.

    Models:
      - Logistic Regression: simple linear baseline
      - Random Forest: ensemble of decision trees, handles non-linearity
      - XGBoost: gradient boosting, typically best on tabular data
      - LightGBM: faster gradient boosting, good for larger datasets

    Primary metric: AUC-ROC (not accuracy, because of class imbalance).
    """
    print("\n" + "─"*50)
    print("  BASELINE MODEL COMPARISON")
    print("─"*50)

    # Scale for Logistic Regression (tree models don't need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1
        ),
    }

    results = {}
    for name, model in candidates.items():
        # Logistic Regression uses scaled data; tree models use raw
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train_bal)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_bal, y_train_bal)
            y_prob = model.predict_proba(X_test)[:, 1]

        auc   = roc_auc_score(y_test, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)

        results[name] = {"model": model, "auc": auc, "f1": f1,
                         "precision": prec, "recall": rec, "y_prob": y_prob}

        print(f"  {name:<22} AUC={auc:.4f}  F1={f1:.4f}  P={prec:.3f}  R={rec:.3f}")

    # Pick best by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  ✅ Best baseline: {best_name} (AUC={results[best_name]['auc']:.4f})")
    return results, best_name, scaler


# ─────────────────────────────────────────────
# STEP 5 — HYPERPARAMETER TUNING (Optuna)
# ─────────────────────────────────────────────

def tune_xgboost(X_train_bal, y_train_bal, n_trials: int = 50):
    """
    Use Optuna (Bayesian optimisation) to find the best XGBoost
    hyperparameters via 5-fold cross-validation on the training set.

    Why Optuna over GridSearch:
      - GridSearch tries every combination → O(n^k) evaluations
      - Optuna intelligently samples the space using TPE (Tree-structured
        Parzen Estimator) — finds good params in far fewer trials

    Tuned hyperparameters:
      n_estimators    — number of boosting rounds (trees)
      max_depth       — maximum tree depth (controls overfitting)
      learning_rate   — how much each tree contributes (shrinkage)
      subsample       — fraction of training rows per tree
      colsample_bytree— fraction of features per tree
      min_child_weight— minimum samples in a leaf (controls overfitting)
      gamma           — minimum loss reduction to split a node
    """
    print("\n" + "─"*50)
    print(f"  OPTUNA HYPERPARAMETER TUNING (XGBoost, {n_trials} trials)")
    print("─"*50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "eval_metric":      "logloss",
            "verbosity":        0,
            "random_state":     RANDOM_STATE,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train_bal, y_train_bal,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"eval_metric": "logloss", "verbosity": 0, "random_state": RANDOM_STATE})
    print(f"\n  Best AUC (CV): {study.best_value:.4f}")
    print(f"  Best params:   {best_params}")

    return best_params


# ─────────────────────────────────────────────
# STEP 6 — TRAIN FINAL MODEL
# ─────────────────────────────────────────────

def train_final_model(X_train_bal, y_train_bal, best_params: dict):
    """
    Train the final XGBoost classifier on the full SMOTE-balanced
    training data using the Optuna-tuned hyperparameters.
    """
    print("\n[TRAIN] Fitting final XGBoost model on full training set...")
    model = XGBClassifier(**best_params)
    model.fit(X_train_bal, y_train_bal)
    print("[TRAIN] Done.")
    return model


# ─────────────────────────────────────────────
# STEP 7 — EVALUATE
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, feature_cols):
    """
    Full evaluation suite on the held-out test set:
      - AUC-ROC  (primary metric — robust to imbalance)
      - Average Precision / PR-AUC  (informative for imbalanced positives)
      - F1 Score, Precision, Recall at threshold=0.5
      - Confusion Matrix
      - Classification Report

    Returns probability scores and threshold-based predictions.
    """
    print("\n" + "─"*50)
    print("  FINAL MODEL EVALUATION (Test Set)")
    print("─"*50)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc     = roc_auc_score(y_test, y_prob)
    ap      = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n  AUC-ROC:            {auc:.4f}")
    print(f"  Avg Precision (AP): {ap:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Response", "Response"]))

    return y_prob, y_pred, {"auc": auc, "ap": ap, "f1": f1, "precision": prec, "recall": rec, "cm": cm}


# ─────────────────────────────────────────────
# STEP 8 — PLOT EVALUATION
# ─────────────────────────────────────────────

def plot_evaluation(model, X_test, y_test, y_prob, metrics, feature_cols):
    """
    Generate a 6-panel evaluation report figure:
      Panel 1: ROC Curve
      Panel 2: Precision-Recall Curve
      Panel 3: Confusion Matrix (heatmap)
      Panel 4: Top 20 Feature Importances
      Panel 5: Predicted Probability Distribution (by true class)
      Panel 6: Score Summary Card
    """
    print(f"\n[PLOT] Generating evaluation report...")

    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f1a")
    fig.suptitle("CampaignIQ — Campaign Response Classifier\nEvaluation Report",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ACCENT = "#7c3aed"
    GOLD   = "#f59e0b"
    GREEN  = "#10b981"
    RED    = "#ef4444"

    # ── Panel 1: ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1a2e")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"XGBoost (AUC={metrics['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, color="#555")
    ax1.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax1.tick_params(colors="white"); ax1.xaxis.label.set_color("white"); ax1.yaxis.label.set_color("white")
    ax1.title.set_color("white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax1.grid(alpha=0.15)
    for spine in ax1.spines.values(): spine.set_edgecolor("#2d2d4e")

    # ── Panel 2: Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1a2e")
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(rec_arr, prec_arr, color=GREEN, lw=2.5, label=f"AP={metrics['ap']:.3f}")
    ax2.axhline(y_test.mean(), color="#555", lw=1, linestyle="--", label="Baseline")
    ax2.fill_between(rec_arr, prec_arr, alpha=0.15, color=GREEN)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax2.tick_params(colors="white"); ax2.xaxis.label.set_color("white"); ax2.yaxis.label.set_color("white")
    ax2.title.set_color("white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax2.grid(alpha=0.15)
    for spine in ax2.spines.values(): spine.set_edgecolor("#2d2d4e")

    # ── Panel 3: Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1a1a2e")
    cm = metrics["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax3,
                xticklabels=["Pred: No", "Pred: Yes"],
                yticklabels=["True: No", "True: Yes"],
                cbar=False, linewidths=1, linecolor="#0f0f1a")
    ax3.set_title("Confusion Matrix", color="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values(): spine.set_edgecolor("#2d2d4e")

    # ── Panel 4: Feature Importance (top 20)
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor("#1a1a2e")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(20)
    colors = [ACCENT if imp > feat_df["Importance"].median() else "#4a4a6a"
              for imp in feat_df["Importance"]]
    ax4.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="none", height=0.6)
    ax4.set(xlabel="Feature Importance (Gain)", title="Top 20 Most Predictive Features")
    ax4.tick_params(colors="white"); ax4.xaxis.label.set_color("white")
    ax4.title.set_color("white")
    ax4.grid(axis="x", alpha=0.15)
    for spine in ax4.spines.values(): spine.set_edgecolor("#2d2d4e")

    # ── Panel 5: Probability Distribution by class
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1a1a2e")
    ax5.hist(y_prob[y_test == 0], bins=30, alpha=0.7, color=RED,   label="True: No",  density=True)
    ax5.hist(y_prob[y_test == 1], bins=30, alpha=0.7, color=GREEN, label="True: Yes", density=True)
    ax5.axvline(0.5, color="white", lw=1.5, linestyle="--", label="Threshold=0.5")
    ax5.set(xlabel="Predicted Probability", ylabel="Density", title="Score Distribution by Class")
    ax5.tick_params(colors="white"); ax5.xaxis.label.set_color("white"); ax5.yaxis.label.set_color("white")
    ax5.title.set_color("white")
    ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax5.grid(alpha=0.15)
    for spine in ax5.spines.values(): spine.set_edgecolor("#2d2d4e")

    plt.savefig(REPORT_PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")
    plt.close()
    print(f"[PLOT] Saved to: {REPORT_PLOT_PATH}")


# ─────────────────────────────────────────────
# STEP 9 — SAVE MODEL
# ─────────────────────────────────────────────

def save_model(model, scaler=None):
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n[SAVE] Model saved → {MODEL_SAVE_PATH}")
    if scaler is not None:
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"[SAVE] Scaler saved → {SCALER_SAVE_PATH}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run(tune: bool = True, n_trials: int = 50):
    """
    Full training pipeline for the Response Classifier.

    Args:
        tune      — if True, run Optuna tuning (slower but better)
        n_trials  — number of Optuna trials (more = better, slower)
    """
    print("\n" + "="*55)
    print("  CampaignIQ — Response Classifier Training")
    print("="*55)

    # 1. Load
    X, y, feature_cols, df = load_classifier_data()

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Balance with SMOTE
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # 4. Baseline comparison
    baseline_results, best_name, scaler = compare_baseline_models(
        X_train_bal, y_train_bal, X_test, y_test
    )

    # 5. Tune XGBoost (always tune XGBoost — it's our primary model)
    if tune:
        best_params = tune_xgboost(X_train_bal, y_train_bal, n_trials=n_trials)
    else:
        # Sensible defaults when tune=False (for quick runs)
        best_params = {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 3, "gamma": 0.1,
            "eval_metric": "logloss", "verbosity": 0, "random_state": RANDOM_STATE
        }

    # 6. Train final
    final_model = train_final_model(X_train_bal, y_train_bal, best_params)

    # 7. Evaluate
    y_prob, y_pred, metrics = evaluate_model(final_model, X_test, y_test, feature_cols)

    # 8. Plot
    plot_evaluation(final_model, X_test, y_test, y_prob, metrics, feature_cols)

    # 9. Save
    save_model(final_model, scaler)

    print("\n" + "="*55)
    print("  Response Classifier Training Complete ✅")
    print("="*55 + "\n")

    return final_model, metrics, feature_cols


if __name__ == "__main__":
    run(tune=True, n_trials=50)

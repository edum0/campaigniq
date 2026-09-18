"""
train_ltv.py
=============
Goal 5 — Customer Lifetime Value (LTV) Regression

Answers: "How much will this customer spend in total (across all products)?"
Target:  Total_Spend = sum of all Mnt* columns (2-year spend proxy for LTV)

Feature strategy:
  We EXCLUDE all Mnt* columns AND Total_Spend from features to prevent leakage.
  The model must predict LTV purely from:
    - Demographics (Age, Income, Education, Marital status)
    - Tenure (how long they've been a customer)
    - Channel behaviour (Web, Catalog, Store purchase counts)
    - Campaign history (how responsive to past campaigns)
    - Digital engagement (web visits, deal purchases)

Author: CampaignIQ
"""

import os, sys, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import run_pipeline

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR,  "ltv_regressor.pkl")
REPORT_PATH = os.path.join(REPORTS_DIR, "ltv_evaluation.png")
RANDOM_STATE = 42

# Spend columns that ARE the target — must not be used as features
SPEND_COLS = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts",
              "MntSweetProducts","MntGoldProds",
              "Total_Spend","Spend_on_Premium","Spend_on_Everyday","Avg_Basket_Size"]


def load_ltv_data():
    """
    Load data and build the LTV feature/target pair.
    Target = Total_Spend (2-year spend sum across all product categories).
    Features = everything EXCEPT any spend-derived columns (leakage prevention).
    """
    df, all_feature_cols = run_pipeline()

    # LTV target
    y = df["Total_Spend"].values.astype(float)

    # Feature set: strip all spend-related columns
    feature_cols = [c for c in all_feature_cols
                    if c not in SPEND_COLS and c != "Total_Spend"]

    X = df[feature_cols]
    print(f"[LTV] Target  — Total_Spend: mean=${y.mean():.0f}  max=${y.max():.0f}  min=${y.min():.0f}")
    print(f"[LTV] Features: {len(feature_cols)} columns (all spend-derived columns excluded)")
    return X, y, feature_cols, df


def tune_ltv_model(X_train, y_train, n_trials=40):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 700),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": 0, "random_state": RANDOM_STATE,
        }
        return cross_val_score(XGBRegressor(**params), X_train, y_train,
                               cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"verbosity": 0, "random_state": RANDOM_STATE})
    print(f"  Best CV neg-RMSE: {study.best_value:.2f}")
    return best


def plot_ltv_evaluation(model, X_test, y_test, y_pred, metrics, feature_cols):
    fig = plt.figure(figsize=(18, 10), facecolor="#0f0f1a")
    fig.suptitle("CampaignIQ — Lifetime Value (LTV) Regression\nEvaluation Report",
                 fontsize=16, color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ACCENT = "#f59e0b"

    def style(ax):
        ax.set_facecolor("#1a1a2e")
        for s in ax.spines.values(): s.set_edgecolor("#2d2d4e")
        ax.tick_params(colors="white"); ax.grid(alpha=0.15)

    # Actual vs Predicted scatter
    ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
    ax1.scatter(y_test, y_pred, alpha=0.4, s=15, color=ACCENT, edgecolors="none")
    lim = max(y_test.max(), y_pred.max()) * 1.05
    ax1.plot([0,lim],[0,lim],"w--",lw=1.5,alpha=0.5)
    ax1.set(xlabel="Actual LTV ($)", ylabel="Predicted LTV ($)", title=f"Actual vs Predicted (R²={metrics['r2']:.3f})")
    ax1.xaxis.label.set_color("white"); ax1.yaxis.label.set_color("white"); ax1.title.set_color("white")

    # Residuals
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.4, s=15, color="#7c3aed", edgecolors="none")
    ax2.axhline(0, color="white", lw=1.5, ls="--")
    ax2.set(xlabel="Predicted LTV ($)", ylabel="Residual ($)", title="Residuals vs Predicted")
    ax2.xaxis.label.set_color("white"); ax2.yaxis.label.set_color("white"); ax2.title.set_color("white")

    # Error distribution
    ax3 = fig.add_subplot(gs[0, 2]); style(ax3)
    ax3.hist(residuals, bins=40, color=ACCENT, alpha=0.8, edgecolor="none")
    ax3.axvline(0, color="white", lw=1.5, ls="--")
    ax3.set(xlabel="Error ($)", ylabel="Count", title=f"Error Distribution (MAE=${metrics['mae']:.0f})")
    ax3.xaxis.label.set_color("white"); ax3.yaxis.label.set_color("white"); ax3.title.set_color("white")

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 0:2]); style(ax4)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(18)
    colors = [ACCENT if i > feat_df["Importance"].median() else "#4a4a6a" for i in feat_df["Importance"]]
    ax4.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="none", height=0.6)
    ax4.set(xlabel="Importance (Gain)", title="Top Features for LTV Prediction")
    ax4.xaxis.label.set_color("white"); ax4.title.set_color("white")

    # LTV distribution
    ax5 = fig.add_subplot(gs[1, 2]); style(ax5)
    ax5.hist(y_test, bins=40, alpha=0.6, color="#7c3aed", label="Actual", density=True, edgecolor="none")
    ax5.hist(y_pred, bins=40, alpha=0.6, color=ACCENT,    label="Predicted", density=True, edgecolor="none")
    ax5.set(xlabel="Total Spend ($)", ylabel="Density", title="LTV Distribution Comparison")
    ax5.xaxis.label.set_color("white"); ax5.yaxis.label.set_color("white"); ax5.title.set_color("white")
    ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"[PLOT] Saved -> {REPORT_PATH}")


def run(tune=True, n_trials=40):
    print("\n" + "="*55)
    print("  CampaignIQ — LTV Regressor Training")
    print("="*55)

    X, y, feature_cols, df = load_ltv_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    if tune:
        print("\n[TUNE] Running Optuna...")
        best_params = tune_ltv_model(X_train.values, y_train, n_trials)
    else:
        best_params = {"n_estimators":300,"max_depth":6,"learning_rate":0.05,
                       "subsample":0.8,"colsample_bytree":0.8,"verbosity":0,"random_state":RANDOM_STATE}

    model = XGBRegressor(**best_params)
    model.fit(X_train.values, y_train)
    y_pred = np.clip(model.predict(X_test.values), 0, None)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
    }

    print(f"\n[EVAL] RMSE=${metrics['rmse']:.2f}  MAE=${metrics['mae']:.2f}  R²={metrics['r2']:.4f}")

    plot_ltv_evaluation(model, X_test.values, y_test, y_pred, metrics, feature_cols)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "ltv_features.pkl"))
    print(f"[SAVE] Model saved -> {MODEL_PATH}")
    print("="*55 + "\n")
    return model, metrics, feature_cols


if __name__ == "__main__":
    run(tune=True, n_trials=40)

"""
train_affinity.py
==================
Goal 2 — Product Affinity Scorer

Answers: "Which product category will this customer spend most on?"
Targets: MntWines, MntMeatProducts, MntFruits, MntFishProducts,
         MntSweetProducts, MntGoldProds

What this script does:
    1. Trains one XGBoost Regressor per product category (6 models total)
    2. Each model predicts how much a customer will spend on that category
    3. Products are ranked per customer by predicted spend → affinity ranking
    4. Evaluates each model with RMSE and R² on a held-out test set
    5. Plots: actual vs predicted scatter, residuals, category comparison,
              feature importance per product
    6. Saves all 6 models to ml/models/affinity_{product}.pkl

Why 6 separate models (not one multi-output model)?
    Each product category has different predictors. For example:
    - Wine spend is driven by income and age
    - Meat spend is driven by income and number of catalog purchases
    - Fruit/Sweet spend is driven by having children
    Training separate models lets each one specialise.

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

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

REPORT_PLOT_PATH = os.path.join(REPORTS_DIR, "affinity_evaluation.png")

# Product targets to predict — maps column name → display name
PRODUCT_TARGETS = {
    "MntWines":        "Wines",
    "MntMeatProducts": "Meat",
    "MntFruits":       "Fruits",
    "MntFishProducts": "Fish",
    "MntSweetProducts":"Sweets",
    "MntGoldProds":    "Gold",
}

RANDOM_STATE = 42


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_affinity_data():
    """
    Load preprocessed data.
    X  = customer features (same as classifier)
    Ys = dict of 6 spend target arrays (one per product)
    """
    df, feature_cols = run_pipeline()

    X = df[feature_cols]

    # Collect all 6 product targets
    Ys = {}
    for col, name in PRODUCT_TARGETS.items():
        Ys[name] = df[col].values.astype(float)
        print(f"[AFFINITY] {name:10s} — mean spend=${Ys[name].mean():>7.1f}  "
              f"max=${Ys[name].max():>7.1f}  zero-spend={( Ys[name]==0).sum()} customers")

    return X, Ys, feature_cols, df


# ─────────────────────────────────────────────
# STEP 2 — TRAIN ONE MODEL PER PRODUCT
# ─────────────────────────────────────────────

def tune_product_model(X_train, y_train, product_name: str, n_trials: int = 30):
    """
    Run Optuna Bayesian optimisation for a single product's XGBoost regressor.
    Uses 5-fold CV with negative RMSE as the objective (maximised by Optuna).
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": 0, "random_state": RANDOM_STATE,
        }
        model = XGBRegressor(**params)
        # neg_root_mean_squared_error: higher (less negative) is better
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"verbosity": 0, "random_state": RANDOM_STATE})
    return best


def train_all_affinity_models(X, Ys, feature_cols, tune: bool = True, n_trials: int = 30):
    """
    Train one XGBoost regressor per product category.

    For each product:
      - Split 80/20 train/test
      - Optionally tune with Optuna
      - Train final model
      - Evaluate (RMSE, MAE, R²)
      - Save model to disk

    Returns:
      models      — dict of {product_name: fitted XGBRegressor}
      test_data   — dict of {product_name: (X_test, y_test, y_pred)}
      metrics_df  — DataFrame of evaluation metrics per product
    """
    print("\n" + "─"*55)
    print("  PRODUCT AFFINITY MODEL TRAINING")
    print("─"*55)

    models    = {}
    test_data = {}
    metrics   = []

    for col, product_name in PRODUCT_TARGETS.items():
        y = Ys[product_name]

        print(f"\n  ▶ Training: {product_name}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Tune
        if tune:
            best_params = tune_product_model(X_train, y_train, product_name, n_trials)
        else:
            best_params = {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "verbosity": 0, "random_state": RANDOM_STATE
            }

        # Train
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Clip negatives (spend can't be < 0)
        y_pred = np.clip(y_pred, 0, None)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        print(f"    RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"affinity_{product_name.lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"    Saved → {model_path}")

        models[product_name]    = model
        test_data[product_name] = (X_test, y_test, y_pred)
        metrics.append({
            "Product": product_name,
            "RMSE": round(rmse, 2),
            "MAE":  round(mae, 2),
            "R²":   round(r2, 4),
        })

    metrics_df = pd.DataFrame(metrics).set_index("Product")
    print("\n" + "─"*55)
    print("  AFFINITY MODELS SUMMARY")
    print("─"*55)
    print(metrics_df.to_string())

    return models, test_data, metrics_df


# ─────────────────────────────────────────────
# STEP 3 — PRODUCT RANKING PER CUSTOMER
# ─────────────────────────────────────────────

def predict_product_ranking(models, X: pd.DataFrame) -> pd.DataFrame:
    """
    Given the 6 trained affinity models and a feature DataFrame,
    produce a per-customer product affinity table:

    Returns a DataFrame where:
      - Each row = one customer
      - Columns = predicted spend per product (Wines, Meat, etc.)
      - Extra columns = rank of each product (1 = most likely)
      - top_product = the single product they're predicted to spend most on

    This is the core output used by the dashboard and targeting system.
    """
    predictions = {}
    for product_name, model in models.items():
        preds = model.predict(X)
        predictions[f"pred_{product_name}"] = np.clip(preds, 0, None)

    pred_df = pd.DataFrame(predictions, index=X.index)

    # Add ranking columns (1 = highest predicted spend)
    spend_cols = list(predictions.keys())
    ranks = pred_df[spend_cols].rank(axis=1, ascending=False).astype(int)
    ranks.columns = [f"rank_{c.replace('pred_','')}" for c in spend_cols]
    pred_df = pd.concat([pred_df, ranks], axis=1)

    # Top product per customer
    pred_df["top_product"] = pred_df[spend_cols].idxmax(axis=1).str.replace("pred_", "")

    return pred_df


# ─────────────────────────────────────────────
# STEP 4 — PLOT EVALUATION
# ─────────────────────────────────────────────

def plot_affinity_evaluation(models, test_data, metrics_df, feature_cols):
    """
    Generate a 3×2 grid of actual vs predicted scatter plots for each product.
    Perfect predictions lie on the diagonal (y=x line).
    """
    print("\n[PLOT] Generating affinity evaluation plots...")

    products = list(PRODUCT_TARGETS.values())
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#0f0f1a")
    fig.suptitle("CampaignIQ — Product Affinity Models\nActual vs Predicted Spend",
                 fontsize=16, color="white", fontweight="bold", y=1.01)

    COLORS = ["#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#ec4899"]

    for idx, product_name in enumerate(products):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d2d4e")

        X_test, y_test, y_pred = test_data[product_name]
        color = COLORS[idx]

        # Scatter
        ax.scatter(y_test, y_pred, alpha=0.4, s=18, color=color, edgecolors="none")

        # Perfect prediction line
        lim = max(y_test.max(), y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "w--", lw=1.5, alpha=0.5, label="Perfect fit")

        row = metrics_df.loc[product_name]
        ax.set_title(f"{product_name}\nRMSE={row['RMSE']:.1f}  R²={row['R²']:.3f}",
                     color="white", fontsize=11)
        ax.set_xlabel("Actual Spend ($)", color="white")
        ax.set_ylabel("Predicted Spend ($)", color="white")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.12)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(REPORT_PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")
    plt.close()
    print(f"[PLOT] Saved to: {REPORT_PLOT_PATH}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run(tune: bool = True, n_trials: int = 30):
    """
    Full training pipeline for the Product Affinity models.

    Args:
        tune     — if True, run Optuna tuning per model
        n_trials — Optuna trials per product model
    """
    print("\n" + "="*55)
    print("  CampaignIQ — Product Affinity Models Training")
    print("="*55)

    X, Ys, feature_cols, df = load_affinity_data()
    models, test_data, metrics_df = train_all_affinity_models(
        X, Ys, feature_cols, tune=tune, n_trials=n_trials
    )
    plot_affinity_evaluation(models, test_data, metrics_df, feature_cols)

    # Show example ranking for first 5 customers
    print("\n[RANKING] Sample product affinity ranking (first 5 customers):")
    sample_ranking = predict_product_ranking(models, X.iloc[:5])
    pred_cols = [f"pred_{p}" for p in PRODUCT_TARGETS.values()]
    print(sample_ranking[pred_cols + ["top_product"]].to_string())

    print("\n" + "="*55)
    print("  Product Affinity Training Complete ✅")
    print("="*55 + "\n")

    return models, metrics_df


if __name__ == "__main__":
    run(tune=True, n_trials=30)

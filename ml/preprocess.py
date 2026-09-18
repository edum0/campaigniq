"""
preprocess.py
=============
Data cleaning and feature engineering pipeline for the CampaignIQ marketing dataset.

What this module does:
    1. Loads the raw Excel file into a pandas DataFrame
    2. Cleans dirty/inconsistent values (outliers, nulls, noisy categories)
    3. Engineers new features that will boost ML model performance
    4. Encodes categorical variables for model consumption
    5. Returns a clean, model-ready DataFrame

Author: CampaignIQ
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RAW_DATA_PATH = "marketing_campaign.xlsx"
REFERENCE_DATE = datetime(2026, 9, 18)   # "today" for tenure calculations

# Education mapped to ordinal numeric levels (knowledge ladder)
EDUCATION_ORDER = {
    "Basic":     0,
    "2n Cycle":  1,
    "Graduation":2,
    "Master":    3,
    "PhD":       4,
}

# Noisy marital status values to consolidate into 'Single'
MARITAL_NOISE = {"Alone", "Absurd", "YOLO"}


# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────

def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw Excel file and do immediate type normalisation.
    The Dt_Customer column has mixed date formats across rows,
    so we handle both datetime objects and string formats.
    """
    df = pd.read_excel(path)

    # Normalise the customer enrolment date column
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")

    print(f"[LOAD] Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# STEP 2 — CLEAN
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix all data quality issues found during EDA:
      - Income: 24 nulls → median imputation; extreme outlier capped
      - Year_Birth: implausibly old customers (age > 100) → dropped
      - Marital_Status: Absurd / YOLO / Alone → consolidated to 'Single'
      - Z_CostContact & Z_Revenue: constant columns, no predictive value → dropped
    """
    df = df.copy()

    # ── Drop constant columns (Z_CostContact=3, Z_Revenue=11 for every row)
    cols_to_drop = [c for c in ["Z_CostContact", "Z_Revenue"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[CLEAN] Dropped constant columns: {cols_to_drop}")

    # ── Income: impute 24 nulls with median
    income_median = df["Income"].median()
    null_income = df["Income"].isna().sum()
    df["Income"] = df["Income"].fillna(income_median)
    print(f"[CLEAN] Imputed {null_income} null Income values with median={income_median:,.0f}")

    # ── Income: cap extreme outlier at 99th percentile
    income_99 = df["Income"].quantile(0.99)
    outliers = (df["Income"] > income_99).sum()
    df["Income"] = df["Income"].clip(upper=income_99)
    print(f"[CLEAN] Capped {outliers} Income outliers above {income_99:,.0f}")

    # ── Year_Birth: drop customers older than 100 years
    df["_age_raw"] = REFERENCE_DATE.year - df["Year_Birth"]
    implausible = (df["_age_raw"] > 100).sum()
    df = df[df["_age_raw"] <= 100].copy()
    df.drop(columns=["_age_raw"], inplace=True)
    print(f"[CLEAN] Removed {implausible} rows with implausible birth year")

    # ── Marital_Status: consolidate noisy values
    df["Marital_Status"] = df["Marital_Status"].replace(
        {v: "Single" for v in MARITAL_NOISE}
    )
    print(f"[CLEAN] Consolidated noisy Marital_Status values → 'Single'")

    print(f"[CLEAN] Final clean shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns that summarise raw columns into higher-level signals.

    New features:
      Age                    — more human-readable than Year_Birth
      Customer_Tenure_Days   — loyalty signal: how long they've been a customer
      Total_Spend            — sum of all product spend (overall spending power)
      Total_Purchases        — sum across all purchase channels
      Avg_Basket_Size        — spend per purchase (premium vs budget buyer)
      Campaign_History       — count of previous campaigns accepted (0-5)
      Has_Children           — binary flag: any kids or teens at home
      Total_Children         — total children count
      Is_Partnered           — binary flag: married or together
      Web_Engagement_Ratio   — how digital-first is this customer
      Spend_on_Premium       — wines + meat (premium category spend proxy)
      Spend_on_Everyday      — fruits + fish + sweets (everyday spend proxy)
    """
    df = df.copy()

    # ── Demographic features
    df["Age"] = REFERENCE_DATE.year - df["Year_Birth"]
    df["Customer_Tenure_Days"] = (REFERENCE_DATE - df["Dt_Customer"]).dt.days
    # Fill any nulls from unparseable dates with the median tenure
    df["Customer_Tenure_Days"].fillna(df["Customer_Tenure_Days"].median(), inplace=True)

    # ── Spend aggregates
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spend"] = df[spend_cols].sum(axis=1)

    # ── Purchase channel aggregates
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    # ── Avg basket size — add 1 to avoid division by zero
    df["Avg_Basket_Size"] = df["Total_Spend"] / (df["Total_Purchases"] + 1)

    # ── Campaign responsiveness history (how many of the 5 past campaigns did they accept)
    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    df["Campaign_History"] = df[cmp_cols].sum(axis=1)

    # ── Family flags
    df["Has_Children"] = ((df["Kidhome"] + df["Teenhome"]) > 0).astype(int)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # ── Relationship status flag
    df["Is_Partnered"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)

    # ── Digital engagement — web visits relative to total purchases
    df["Web_Engagement_Ratio"] = df["NumWebVisitsMonth"] / (df["Total_Purchases"] + 1)

    # ── Category spend splits
    df["Spend_on_Premium"]  = df["MntWines"] + df["MntMeatProducts"]
    df["Spend_on_Everyday"] = df["MntFruits"] + df["MntFishProducts"] + df["MntSweetProducts"]

    print(f"[FEATURES] Engineered 12 new features. Total columns: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# STEP 4 — ENCODE CATEGORICALS
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical text columns into numeric form for the ML models.

    Education:     Ordinal encoding (Basic=0 ... PhD=4)
                   Ordinal is correct here because education genuinely has a rank.

    Marital_Status: One-hot encoding (no natural ordering between statuses)
                    Drop first to avoid multicollinearity.
    """
    df = df.copy()

    # ── Education: ordinal
    df["Education_Enc"] = df["Education"].map(EDUCATION_ORDER)
    df.drop(columns=["Education"], inplace=True)
    print(f"[ENCODE] Ordinal-encoded Education")

    # ── Marital_Status: one-hot
    marital_dummies = pd.get_dummies(df["Marital_Status"], prefix="Marital", drop_first=True)
    df = pd.concat([df, marital_dummies], axis=1)
    df.drop(columns=["Marital_Status"], inplace=True)
    print(f"[ENCODE] One-hot-encoded Marital_Status → columns: {list(marital_dummies.columns)}")

    return df


# ─────────────────────────────────────────────
# STEP 5 — SELECT MODEL FEATURES
# ─────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of columns to feed into the ML models.
    Excludes: ID, raw dates, raw text cols already encoded,
    and the target columns (Response + Mnt* for regression targets).
    """
    # Columns to exclude from features
    exclude = {
        "ID",
        "Dt_Customer",        # replaced by Customer_Tenure_Days
        "Year_Birth",         # replaced by Age
        "Response",           # target for classifier
        "MntWines",           # targets for affinity regression
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    }

    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"[FEATURES] Using {len(feature_cols)} feature columns for ML")
    return feature_cols


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(path: str = RAW_DATA_PATH) -> tuple[pd.DataFrame, list]:
    """
    Run the full preprocessing pipeline end-to-end.

    Returns:
        df          — clean, feature-engineered DataFrame
        feature_cols — list of column names to use as model input (X)
    """
    print("\n" + "="*55)
    print("  CampaignIQ — Data Preprocessing Pipeline")
    print("="*55)

    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    feature_cols = get_feature_columns(df)

    # Final sanity checks
    remaining_nulls = df[feature_cols].isna().sum().sum()
    if remaining_nulls > 0:
        # Fill any remaining nulls with column median as a safety net
        for col in feature_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        print(f"[PIPELINE] Fixed {remaining_nulls} residual nulls via median imputation")
    else:
        print(f"[PIPELINE] Remaining nulls in feature set: 0")
    print(f"[PIPELINE] Final dataset: {df.shape[0]} customers x {len(feature_cols)} features")
    print("="*55 + "\n")

    return df, feature_cols


# ─────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df, features = run_pipeline()
    print("Sample feature columns:")
    print(features)
    print("\nFirst 3 rows of processed data:")
    print(df[features].head(3))

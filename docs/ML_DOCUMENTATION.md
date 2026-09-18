# CampaignIQ â€” ML System Documentation

> **Project**: Customer Purchase Prediction & Segmentation  
> **Dataset**: `marketing_campaign.xlsx` â€” 2,240 customers, 29 columns  
> **Last Updated**: September 2026  
> **Status**: Active Development

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Understanding](#2-dataset-understanding)
3. [File Structure](#3-file-structure)
4. [Pipeline: Step by Step](#4-pipeline-step-by-step)
5. [Model 1 â€” Response Classifier](#5-model-1--response-classifier)
6. [Model 2 â€” Product Affinity Scorer](#6-model-2--product-affinity-scorer)
7. [Model 3 â€” Customer Segmentation](#7-model-3--customer-segmentation)
8. [How to Run](#8-how-to-run)
9. [Understanding the Outputs](#9-understanding-the-outputs)
10. [Design Decisions Explained](#10-design-decisions-explained)

---

## 1. Project Overview

**The Problem**: We have historical data on 2,240 customers â€” what they bought, how they responded to past campaigns, their demographics. We want to use this to answer three questions:

| Question | ML Approach | Output |
|---|---|---|
| Will this customer respond to the next campaign? | Binary Classification (XGBoost) | Probability score 0-1 |
| What product will they most likely buy? | Multi-output Regression (XGBoost) | Ranked product list + spend estimate |
| What kind of buyer are they? | Unsupervised Clustering (K-Means) | Segment label (e.g. "Champions") |

**The Key Insight**: These aren't SQL lookups on past behaviour. These are **predictions about future behaviour** that the raw data doesn't contain. ML learns the patterns from the past and extrapolates.

---

## 2. Dataset Understanding

### Raw Columns (29 total)

| Column | Type | Notes |
|---|---|---|
| `ID` | Identifier | Not used in models |
| `Year_Birth` | Demographic | Converted to `Age` |
| `Education` | Categorical | 5 levels: Basic to PhD |
| `Marital_Status` | Categorical | Had noisy values (YOLO, Absurd, Alone) |
| `Income` | Numeric | 24 nulls; one extreme outlier ($666k) |
| `Kidhome` | Numeric | Number of young kids |
| `Teenhome` | Numeric | Number of teenagers |
| `Dt_Customer` | Date | Converted to `Customer_Tenure_Days` |
| `Recency` | Numeric | Days since last purchase (lower = more active) |
| `MntWines` ... `MntGoldProds` | Numeric | 2-year spend per category |
| `NumWebPurchases` ... `NumStorePurchases` | Numeric | Purchase channel counts |
| `NumWebVisitsMonth` | Numeric | Digital engagement |
| `NumDealsPurchases` | Numeric | Discount-driven purchases |
| `AcceptedCmp1` ... `AcceptedCmp5` | Binary | Past campaign responses |
| `Complain` | Binary | Complained in last 2 years |
| `Response` | Binary | **Primary target** - last campaign response |
| `Z_CostContact`, `Z_Revenue` | Constant | Identical for all rows - dropped |

### Key Data Quality Issues Found

| Issue | Resolution |
|---|---|
| `Income`: 24 nulls | Median imputation (robust to outlier) |
| `Income` = $666,666 | Capped at 99th percentile ($94,459) |
| `Year_Birth` = 1893 (age 133) | Dropped 3 rows with age > 100 |
| `Marital_Status`: YOLO, Absurd, Alone | Consolidated to Single |
| `Dt_Customer`: mixed date formats | Parsed with `dayfirst=True` |
| `Z_CostContact` & `Z_Revenue` | Constant - dropped (no signal) |

### Class Imbalance (Critical Issue)
```
Response = 0 (No):  1,903 customers  ->  85.1%
Response = 1 (Yes):   334 customers  ->  14.9%
```
If you train naively, a model that always says "No" would be "85% accurate" -- useless. This is handled with **SMOTE** (see Model 1 section).

---

## 3. File Structure

```
CampaignIQ/
|
+-- marketing_campaign.xlsx         <- Raw input data
+-- run_all.py                      <- Master runner (trains all 3 models)
|
+-- ml/
|   +-- __init__.py
|   +-- preprocess.py               <- Data cleaning + feature engineering
|   +-- train_classifier.py         <- Model 1: Response Classifier
|   +-- train_affinity.py           <- Model 2: Product Affinity Models
|   +-- train_segmentation.py       <- Model 3: Customer Segmentation
|   +-- predict.py                  <- Inference engine (score any customer)
|   |
|   +-- models/                     <- Saved model files (generated after training)
|   |   +-- response_classifier.pkl
|   |   +-- affinity_wines.pkl ... affinity_gold.pkl
|   |   +-- segmentation_kmeans.pkl
|   |   +-- segmentation_scaler.pkl
|   |   +-- segmentation_pca.pkl
|   |
|   +-- reports/                    <- Evaluation plots (generated after training)
|       +-- classifier_evaluation.png
|       +-- affinity_evaluation.png
|       +-- segmentation_evaluation.png
|
+-- docs/
    +-- ML_DOCUMENTATION.md         <- This file
```

---

## 4. Pipeline: Step by Step

### `preprocess.py` - The Foundation

Every ML run starts here. The pipeline does 5 things in order:

#### 4.1 Load
Loads 2,240 rows from Excel. Parses the customer enrolment date (which has mixed formats in the raw file).

#### 4.2 Clean
Fixes all data quality issues listed above.

#### 4.3 Feature Engineering
Derives 12 new columns from raw data. **This is the most impactful step** â€” raw columns like `AcceptedCmp1-5` are weak individually; combined into `Campaign_History` they become a powerful signal.

| New Feature | Source | Why It's Useful |
|---|---|---|
| `Age` | 2026 - Year_Birth | More interpretable, easier for model |
| `Customer_Tenure_Days` | Today - Dt_Customer | Long-tenured customers behave differently |
| `Total_Spend` | Sum of all Mnt* columns | Overall spending power in one number |
| `Total_Purchases` | Web + Catalog + Store | Purchase frequency |
| `Avg_Basket_Size` | Total_Spend / Total_Purchases | Premium vs budget buyer |
| `Campaign_History` | Sum of AcceptedCmp1-5 | Single strongest predictor of Response |
| `Has_Children` | Kidhome + Teenhome > 0 | Family lifestyle fundamentally changes buying |
| `Total_Children` | Kidhome + Teenhome | More granular than binary |
| `Is_Partnered` | Married or Together | Household composition signal |
| `Web_Engagement_Ratio` | WebVisits / TotalPurchases | Digital-first vs store-first |
| `Spend_on_Premium` | Wines + Meat | High-margin product affinity |
| `Spend_on_Everyday` | Fruits + Fish + Sweets | Budget product affinity |

#### 4.4 Encoding
- **Education -> Ordinal** (0-4): `Basic < 2n Cycle < Graduation < Master < PhD`  
  Ordinal is correct here because education has a genuine rank.
- **Marital_Status -> One-Hot**: Creates binary columns for each category.  
  One-hot is correct because there's no natural ordering between Single/Married/etc.

#### 4.5 Feature Selection
32 columns selected for ML. Excluded: `ID`, `Dt_Customer`, `Year_Birth` (replaced by derived versions), `Response` (target), and the 6 `Mnt*` columns (targets for affinity models).

---

## 5. Model 1 - Response Classifier

**File**: `ml/train_classifier.py`  
**Question**: Will this customer say YES to the next campaign?  
**Target**: `Response` (1 = Yes, 0 = No)

### Why XGBoost?

| Model | Strengths | Weaknesses |
|---|---|---|
| Logistic Regression | Interpretable, fast | Assumes linear decision boundary |
| Random Forest | Handles non-linearity, robust | Slower, less optimisable |
| **XGBoost** | Best tabular data performance | Needs tuning |
| LightGBM | Fastest, memory efficient | Similar to XGBoost |

XGBoost wins on tabular structured data. It builds trees **sequentially** â€” each tree fixing the mistakes of the previous one (boosting), giving higher accuracy than Random Forest's parallel approach.

### Step 1: Stratified Train/Test Split (80/20)

`stratify=y` ensures both train and test sets have the same 85/15 class ratio. Without this, the test set might have no positive examples.

Result: **1,789 training rows, 448 test rows**.

### Step 2: SMOTE - Fixing Class Imbalance

```
Before SMOTE:  1,522 No  |  267 Yes  (training set)
After SMOTE:   1,522 No  | 1,522 Yes (balanced)
```

How SMOTE creates synthetic customers:
1. Take a real YES customer (Income=$58k, TotalSpend=$1,200, Age=45)
2. Find 5 nearest YES-customer neighbours
3. Pick a random neighbour (Income=$62k, TotalSpend=$980, Age=47)
4. Create synthetic customer halfway: Income=$60k, TotalSpend=$1,090, Age=46

SMOTE is **only applied to training data**. Test set stays real and untouched.

### Step 3: Optuna Hyperparameter Tuning (50 trials)

What gets tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`

**Why Optuna over GridSearch**:  
GridSearch tries every combination â€” exponential time.  
Optuna uses **Bayesian optimisation** (TPE) â€” builds a model of which parameter regions tend to give good results, then samples more heavily from those. 50 trials finds near-optimal params in a fraction of the time.

### Step 4: Evaluation Metrics

| Metric | Why Used |
|---|---|
| AUC-ROC | Primary metric, robust to imbalance. Target > 0.80 |
| Average Precision | More informative than AUC-ROC for small positive class. Target > 0.55 |
| F1 Score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows False Negatives (missed buyers) vs False Positives (wrong targets) |

### Output Per Customer
```json
{
  "customer_id": 5524,
  "response_probability": 0.73,
  "will_respond": true,
  "response_tier": "High"
}
```

---

## 6. Model 2 - Product Affinity Scorer

**File**: `ml/train_affinity.py`  
**Question**: Which product will this customer spend most on?  
**Targets**: MntWines, MntMeatProducts, MntFruits, MntFishProducts, MntSweetProducts, MntGoldProds

### Architecture: 6 Separate Regression Models

One XGBoost Regressor per product category.

**Why separate?** Wine spend is driven by income and age. Fruit spend is driven by having children. Each product has different predictors. Separate models let each specialise.

### Metrics

| Metric | Meaning |
|---|---|
| RMSE | Average prediction error in dollars |
| MAE | Less sensitive to large errors than RMSE |
| R2 | Variance explained (1.0 = perfect, 0 = useless) |

### Output Per Customer
```json
{
  "predicted_spend": { "Wines": 620, "Meat": 210, "Gold": 55, "Fish": 40, "Fruits": 20, "Sweets": 15 },
  "product_affinity_ranking": ["Wines", "Meat", "Gold", "Fish", "Fruits", "Sweets"],
  "top_product": "Wines"
}
```

---

## 7. Model 3 - Customer Segmentation

**File**: `ml/train_segmentation.py`  
**Question**: What kind of buyer is this customer?  
**Method**: K-Means Clustering (unsupervised)

### Unsupervised vs Supervised
Models 1 and 2 are **supervised** â€” we tell the model the answer and it learns to predict it.  
Segmentation is **unsupervised** â€” there are no pre-defined segments. The model finds natural groupings.

### Why StandardScaler Is Mandatory

K-Means uses Euclidean distance. Income (0-94,000 range) would dominate Recency (0-99 range) without scaling. After StandardScaler (mean=0, std=1), every feature contributes equally.

### Finding Optimal K

- **Elbow Method**: Plot inertia vs K; find where curve bends
- **Silhouette Score**: Measures how well-separated clusters are (>0.35 = good)
- **Best K chosen by highest Silhouette Score**

### Auto-Labelled Segments

| Segment | Profile | Strategy |
|---|---|---|
| Champions | High income, big spenders, responsive | Reward with exclusives |
| High-Value Inactives | Spenders but non-responsive | Personal outreach |
| At-Risk | High recency, previously active | Urgent win-back + discount |
| Loyal Families | Kids, store shoppers, consistent | Family bundles |
| Budget Shoppers | Deal-seekers, low spend | Heavy discounts |
| Digital Natives | High web visits | Email, online-only deals |

---

## 8. How to Run

```bash
# Full training with Optuna tuning (10-20 min)
python run_all.py

# Fast run without tuning (1-2 min, lower accuracy)
python run_all.py --no-tune

# Individual models
python ml/train_classifier.py
python ml/train_affinity.py
python ml/train_segmentation.py

# Score all customers after training
python ml/predict.py
```

---

## 9. Understanding the Outputs

After running `python run_all.py`, you get:

**`ml/models/`** â€” 9 saved model files (`.pkl`). These are the trained ML brains.

**`ml/reports/`** â€” 3 evaluation plot files:
- `classifier_evaluation.png` â€” ROC curve, PR curve, confusion matrix, feature importance, score distribution
- `affinity_evaluation.png` â€” Actual vs predicted scatter for each of the 6 products
- `segmentation_evaluation.png` â€” Elbow curve, silhouette scores, 2D cluster map, cluster profiles

---

## 10. Design Decisions Explained

| Decision | Reason |
|---|---|
| XGBoost over Neural Networks | NNs need 100k+ examples. With 2,240 rows, XGBoost generalises far better |
| SMOTE over Class Weights | SMOTE creates diverse synthetic positives. Class weights just re-weight existing ones. SMOTE gives better recall |
| 6 separate affinity models | Each product has different key predictors. Separate models let each specialise |
| K-Means over DBSCAN | K-Means centroids let us describe and auto-label each segment quantitatively |
| Dropped Z_CostContact/Z_Revenue | Constant columns carry zero information â€” they can't distinguish one customer from another |
| Ordinal for Education | Education has a genuine ordering (Basic < PhD). Ordinal preserves this; one-hot would lose it |
| One-hot for Marital_Status | No natural ordering between Single/Married/Together. One-hot treats all categories equally |

---

*Documentation is maintained alongside code changes. Each function in the codebase has inline docstrings explaining the what, why, and how.*

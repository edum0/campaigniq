# CampaignIQ — Customer Intelligence & Predictive Marketing Suite

CampaignIQ is an end-to-end Machine Learning and Customer Intelligence platform designed to optimize marketing campaign performance, customer segmentation, product affinity targeting, churn prevention, lifetime value (LTV) forecasting, and omnichannel acquisition strategies.

---

## 🚀 Key Features & ML Models

CampaignIQ runs **6 specialized Machine Learning models** on customer demographic and transaction data:

1. **Campaign Response Predictor**:
   - Classifies whether a customer will accept the next marketing campaign offer.
   - Algorithm: Optuna-tuned LightGBM / XGBoost with SMOTE class rebalancing.
2. **Product Affinity Recommender**:
   - Predicts high-propensity product categories (Wines, Fruits, Meat, Fish, Sweets, Gold).
   - Algorithm: Multi-output Gradient Boosted classifiers.
3. **Customer Segmentation Engine**:
   - Clusters customers into 6 actionable personas (*Champions, Loyal Families, Digital Natives, Budget Shoppers, At-Risk, High-Value Inactives*).
   - Algorithm: PCA dimensionality reduction + K-Means clustering.
4. **Churn Risk Classifier**:
   - Predicts customers likely to disengage or churn based on recency and engagement velocity.
   - Algorithm: LightGBM / Random Forest classifier.
5. **Customer Lifetime Value (LTV) Regressor**:
   - Forecasts 2-year customer monetary spend from acquisition attributes.
   - Algorithm: Gradient Boosting Regressor ($R^2 \approx 0.90$).
6. **Channel Propensity Model**:
   - Recommends the optimal conversion touchpoint (Store, Web, or Catalog) for each customer.
   - Algorithm: Multi-class Random Forest / Gradient Boosting.

---

## 🖥️ Interactive Dashboard

CampaignIQ features a dark glassmorphic single-page web dashboard powered by a Flask REST API backend:

- **Executive Overview**: High-level KPIs, campaign acceptance rates, segment spend distributions, and response probability histograms.
- **Customer Explorer**: Interactive data table of all 2,237 customers with real-time multi-filter support (by segment, churn risk, response likelihood, channel) and deep-dive profile modals.
- **Real-Time Predictor**: Interactive simulation tool to input custom customer demographics and instantly receive multi-model predictions and targeted marketing actions.
- **Model Insights**: Feature importance rankings, accuracy metrics, and evaluation summaries for each model.

---

## 📁 Repository Structure

```text
CampaignIQ/
├── backend/
│   └── app.py                  # Flask REST API backend
├── frontend/
│   ├── index.html              # Dashboard single-page application
│   ├── index.css               # Design system & dark glassmorphic styling
│   └── app.js                  # Frontend state management & Chart.js visualizations
├── ml/
│   ├── preprocess.py           # Feature engineering & data pipeline
│   ├── train_classifier.py     # Campaign response model training
│   ├── train_affinity.py       # Product affinity model training
│   ├── train_segmentation.py   # K-Means + PCA clustering pipeline
│   ├── train_churn.py          # Churn prediction pipeline
│   ├── train_ltv.py            # LTV regression pipeline
│   ├── train_channel.py        # Channel propensity pipeline
│   ├── predict.py              # Unified batch/real-time inference module
│   ├── models/                 # Serialized model weights (*.pkl)
│   └── reports/                # Model evaluation charts and confusion matrices
├── docs/
│   └── ML_DOCUMENTATION.md     # Detailed ML methodology and metric reports
├── marketing_campaign.xlsx     # Customer transaction dataset
├── run_all.py                  # Master training script
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Armaan3535/campaigniq.git
cd campaigniq
pip install -r requirements.txt
```

### 2. Launch Dashboard & Backend

Start the Flask server:

```bash
python backend/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Retrain Models (Optional)

To retrain all 6 models from scratch:

```bash
# Fast training mode
python run_all.py --no-tune

# Full training mode with Optuna hyperparameter optimization
python run_all.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/overview` | `GET` | Dashboard KPI summary statistics |
| `/api/customers` | `GET` | Paginated customer records with ML predictions |
| `/api/customer/<id>` | `GET` | Detailed profile and historical spending for a customer |
| `/api/segments` | `GET` | Segment metrics, sizing, and spend breakdowns |
| `/api/top-products` | `GET` | Top product affinity distribution |
| `/api/model-insights` | `GET` | Feature importances and evaluation metrics |
| `/api/response-dist` | `GET` | Campaign response probability distribution |
| `/api/predict` | `POST` | Real-time multi-model inference for custom customer inputs |

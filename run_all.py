"""
run_all.py
===========
CampaignIQ — Master Training Runner

Trains ALL 6 models end-to-end:
  1. Response Classifier   (will customer accept next campaign?)
  2. Product Affinity      (which product will they buy?)
  3. Customer Segmentation (what type of buyer are they?)
  4. Churn Risk            (are they about to disengage?)
  5. LTV Regressor         (how much will they spend in total?)
  6. Channel Recommender   (web, store, or catalog?)

Usage:
  python run_all.py               # full training with Optuna tuning
  python run_all.py --no-tune     # fast run for testing (1-3 min)
"""

import argparse, time, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def banner(title):
    print("\n" + "█"*60)
    print(f"  {title}")
    print("█"*60)


def main(tune=True):
    start = time.time()
    n_clf  = 50 if tune else 0
    n_reg  = 40 if tune else 0

    banner("CampaignIQ — Full ML Training Pipeline (6 Models)")
    print(f"  Tuning: {'Enabled (Optuna)' if tune else 'Disabled (fast mode)'}\n")

    # 1. Preprocessing validation
    banner("STEP 0 — Data Preprocessing Validation")
    from ml.preprocess import run_pipeline
    df, feature_cols = run_pipeline()
    print(f"  Preprocessing OK — {len(df)} customers, {len(feature_cols)} features")

    # 2. Response Classifier
    banner("STEP 1/5 — Response Classifier")
    from ml.train_classifier import run as train_classifier
    clf_model, clf_metrics, _ = train_classifier(tune=tune, n_trials=n_clf)

    # 3. Product Affinity Models
    banner("STEP 2/5 — Product Affinity Models (6 products)")
    from ml.train_affinity import run as train_affinity
    aff_models, aff_metrics = train_affinity(tune=tune, n_trials=n_reg)

    # 4. Segmentation
    banner("STEP 3/5 — Customer Segmentation")
    from ml.train_segmentation import run as train_segmentation
    km, scaler, labels, profiles, seg_df = train_segmentation()

    # 5. Churn Risk
    banner("STEP 4/5 — Churn Risk Classifier")
    from ml.train_churn import run as train_churn
    churn_model, churn_metrics, _ = train_churn(tune=tune, n_trials=n_reg)

    # 6. LTV Regressor
    banner("STEP 5/5 — Lifetime Value Regressor")
    from ml.train_ltv import run as train_ltv
    ltv_model, ltv_metrics, _ = train_ltv(tune=tune, n_trials=n_reg)

    # 7. Channel Recommender
    banner("STEP 5B/5 — Channel Recommender")
    from ml.train_channel import run as train_channel
    ch_model, ch_metrics, _ = train_channel(tune=tune, n_trials=n_reg)

    # Summary
    elapsed = (time.time() - start) / 60
    banner(f"ALL TRAINING COMPLETE in {elapsed:.1f} minutes")
    print(f"\n  {'─'*52}")
    print(f"  {'MODEL':<30} {'KEY METRIC':<20} SCORE")
    print(f"  {'─'*52}")
    print(f"  {'Response Classifier':<30} {'AUC-ROC':<20} {clf_metrics['auc']:.4f}")
    print(f"  {'Churn Risk Classifier':<30} {'AUC-ROC':<20} {churn_metrics['auc']:.4f}")
    print(f"  {'LTV Regressor':<30} {'R2':<20} {ltv_metrics['r2']:.4f}")
    print(f"  {'Channel Recommender':<30} {'F1-Macro':<20} {ch_metrics['f1_macro']:.4f}")
    print(f"\n  Product Affinity R2:")
    for prod, row in aff_metrics.iterrows():
        print(f"    {prod:<12} R2={row['R2']:.3f}  RMSE=${row['RMSE']:.0f}")
    print(f"\n  Customer Segments: {len(profiles)}")
    for c, p in profiles.items():
        print(f"    [{c}] {p['name']:25s}  {p['size']:4d} customers ({p['pct']:.1f}%)")
    print(f"\n  Reports -> ml/reports/")
    print(f"  Models  -> ml/models/")
    print("█"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()
    main(tune=not args.no_tune)

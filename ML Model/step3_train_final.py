import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, brier_score_loss
)

# -------------------------------------------------------------------------
# Temporal window configuration (must match step2_feature_engineering.py)
# -------------------------------------------------------------------------
TEMPORAL_HORIZON = 75
SAMPLING_STRIDE = 6
SAMPLED_OFFSETS = list(range(0, TEMPORAL_HORIZON, SAMPLING_STRIDE))
NUM_SAMPLED_POSITIONS = len(SAMPLED_OFFSETS)  # = 13
NUM_FEATURES = 2 * NUM_SAMPLED_POSITIONS + 1  # = 27

# -------------------------------------------------------------------------
# 5-Level danger score thresholds (initial; will be refined by calibration)
# -------------------------------------------------------------------------
LEVEL_THRESHOLDS = {
    "Level 0 (Safe)":              (0.00, 0.30),
    "Level 1 (Nearly Safe)":       (0.30, 0.50),
    "Level 2 (Nearly Dangerous)":  (0.50, 0.65),
    "Level 3 (Dangerous)":         (0.65, 0.80),
    "Level 4 (Highly Dangerous)":  (0.80, 1.01),
}

BINARY_THRESHOLD = 0.50


def find_threshold_at_precision(precision_arr, recall_arr, thresholds_arr, target_precision):
    """Find the lowest threshold that achieves at least target_precision."""
    # precision_recall_curve returns precision[i] for threshold[i]
    # Walk from highest threshold to lowest, find first point at or above target
    valid_mask = precision_arr[:-1] >= target_precision  # exclude last element (always 1.0)
    if not np.any(valid_mask):
        return thresholds_arr[-1]  # fallback to highest threshold
    # Among valid entries, pick the one with highest recall (lowest threshold)
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(recall_arr[:-1][valid_indices])]
    return thresholds_arr[best_idx]


def main():
    # Setup Paths
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    
    X_path = data_dir / "X_train_final.csv"
    y_path = data_dir / "y_train_final.csv"
    w_path = data_dir / "weights_train_final.csv"
    
    for p in [X_path, y_path, w_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing final matrix file: {p}")
            
    print("Loading finalized training matrices...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    weights = pd.read_csv(w_path)
    
    assert X.shape[1] == NUM_FEATURES, \
        f"Expected {NUM_FEATURES} features, got {X.shape[1]}"
    
    y_1d = y.values.ravel().astype(float)
    w_1d = weights.values.ravel()
    y_binary = y_1d.astype(int)
    
    print(f"Loaded Features Shape: {X.shape}")
    print(f"Loaded Targets Shape:  {y_1d.shape}")
    print(f"Loaded Weights Shape:  {w_1d.shape}")
    print(f"Class balance: {np.sum(y_binary==1)} dangerous, {np.sum(y_binary==0)} safe")
    
    # -------------------------------------------------------------------------
    # RandomForestRegressor configuration — SOFTENED to reduce score clustering
    #
    # Key changes from previous config:
    #   max_depth: 15 -> 10  (shallower trees = less extreme leaf predictions)
    #   min_samples_leaf: 2 -> 5 (larger leaves = more averaging = softer scores)
    #   min_samples_split: 5 -> 10
    #   n_estimators: 200 -> 300 (more trees = smoother averaging across ensemble)
    # -------------------------------------------------------------------------
    rf_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # =====================================================================
    # PHASE 1: 5-Fold Stratified Cross Validation + OOF Collection
    #
    # Each sample is predicted exactly once (by a model that never trained
    # on it), giving unbiased "test" predictions for the full dataset.
    # =====================================================================
    print("\n" + "="*65)
    print("PHASE 1: 5-Fold Stratified CV (Train/Test per fold)")
    print("="*65)
    
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Out-of-fold predictions for calibration
    oof_raw_scores = np.zeros(len(X))
    
    # Per-fold metrics storage
    fold_metrics = {
        "mae": [], "rmse": [], "r2": [],
        "acc": [], "prec": [], "rec": [], "f1": []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_binary), 1):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y_1d[train_idx]
        y_test_fold = y_1d[test_idx]
        w_train_fold = w_1d[train_idx]
        w_test_fold = w_1d[test_idx]
        
        fold_model = RandomForestRegressor(**rf_params)
        fold_model.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
        
        # Raw predictions on test fold
        y_score = np.clip(fold_model.predict(X_test_fold), 0.0, 1.0)
        oof_raw_scores[test_idx] = y_score
        
        # Primary regression metrics
        mae = mean_absolute_error(y_test_fold, y_score, sample_weight=w_test_fold)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_score, sample_weight=w_test_fold))
        r2 = r2_score(y_test_fold, y_score, sample_weight=w_test_fold)
        
        # Secondary binary metrics (threshold = 0.50)
        y_pred_bin = (y_score >= BINARY_THRESHOLD).astype(int)
        y_true_bin = y_test_fold.astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin, sample_weight=w_test_fold)
        prec = precision_score(y_true_bin, y_pred_bin, sample_weight=w_test_fold, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, sample_weight=w_test_fold, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, sample_weight=w_test_fold, zero_division=0)
        
        for k, v in [("mae", mae), ("rmse", rmse), ("r2", r2),
                      ("acc", acc), ("prec", prec), ("rec", rec), ("f1", f1)]:
            fold_metrics[k].append(v)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        print(f"\nFold {fold} (Train: {n_train}, Test: {n_test}):")
        print(f"  Regression  - MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        print(f"  Binary (>={BINARY_THRESHOLD}) - Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    
    print("\n" + "-"*65)
    print("AGGREGATE TEST RESULTS (averaged across 5 test folds):")
    print("-"*65)
    print("REGRESSION:")
    print(f"  Average MAE:   {np.mean(fold_metrics['mae']):.4f} +/- {np.std(fold_metrics['mae']):.4f}")
    print(f"  Average RMSE:  {np.mean(fold_metrics['rmse']):.4f} +/- {np.std(fold_metrics['rmse']):.4f}")
    print(f"  Average R2:    {np.mean(fold_metrics['r2']):.4f} +/- {np.std(fold_metrics['r2']):.4f}")
    print(f"\nBINARY (threshold >= {BINARY_THRESHOLD}):")
    print(f"  Average Accuracy:  {np.mean(fold_metrics['acc']):.4f} +/- {np.std(fold_metrics['acc']):.4f}")
    print(f"  Average Precision: {np.mean(fold_metrics['prec']):.4f} +/- {np.std(fold_metrics['prec']):.4f}")
    print(f"  Average Recall:    {np.mean(fold_metrics['rec']):.4f} +/- {np.std(fold_metrics['rec']):.4f}")
    print(f"  Average F1-score:  {np.mean(fold_metrics['f1']):.4f} +/- {np.std(fold_metrics['f1']):.4f}")
    
    # =====================================================================
    # PHASE 2: Score Calibration via Isotonic Regression on OOF Predictions
    #
    # The OOF raw scores represent what the model produces on "unseen" data.
    # Isotonic regression learns a monotonic, non-parametric mapping:
    #     raw_score  -->  P(dangerous)
    # This spreads the bimodal clustering and produces well-calibrated
    # probabilities that are meaningful for threshold-based decisions.
    # =====================================================================
    print("\n" + "="*65)
    print("PHASE 2: Score Calibration (Isotonic Regression on OOF)")
    print("="*65)
    
    oof_clamped = np.clip(oof_raw_scores, 0.0, 1.0)
    
    print(f"\nRaw OOF Score Distribution:")
    print(f"  Mean:   {oof_clamped.mean():.4f}")
    print(f"  Median: {np.median(oof_clamped):.4f}")
    print(f"  Std:    {oof_clamped.std():.4f}")
    print(f"  Min:    {oof_clamped.min():.4f}, Max: {oof_clamped.max():.4f}")
    
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(oof_clamped, p):.4f}")
    
    # Fit isotonic regression calibrator
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(oof_clamped, y_binary)
    
    oof_calibrated = calibrator.predict(oof_clamped)
    
    print(f"\nCalibrated OOF Score Distribution:")
    print(f"  Mean:   {oof_calibrated.mean():.4f}")
    print(f"  Median: {np.median(oof_calibrated):.4f}")
    print(f"  Std:    {oof_calibrated.std():.4f}")
    print(f"  Min:    {oof_calibrated.min():.4f}, Max: {oof_calibrated.max():.4f}")
    
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(oof_calibrated, p):.4f}")
    
    # Brier score comparison (lower = better calibrated)
    brier_raw = brier_score_loss(y_binary, oof_clamped)
    brier_cal = brier_score_loss(y_binary, oof_calibrated)
    print(f"\nBrier Score (lower = better calibration):")
    print(f"  Raw:        {brier_raw:.4f}")
    print(f"  Calibrated: {brier_cal:.4f}")
    print(f"  Improvement: {((brier_raw - brier_cal) / brier_raw * 100):.1f}%")
    
    # =====================================================================
    # PHASE 3: Data-Driven Threshold Derivation
    #
    # Instead of hardcoding thresholds, we derive them from the calibrated
    # OOF score distribution using precision-recall analysis.
    # =====================================================================
    print("\n" + "="*65)
    print("PHASE 3: Data-Driven Threshold Derivation")
    print("="*65)
    
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_binary, oof_calibrated)
    
    # F1-optimal threshold -> Level 2 (minimum render level)
    f1_scores = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    threshold_l2 = float(thresh_arr[best_f1_idx])
    
    # Threshold at precision >= 0.80 -> Level 3
    threshold_l3 = float(find_threshold_at_precision(prec_arr, rec_arr, thresh_arr, 0.80))
    
    # Threshold at precision >= 0.92 -> Level 4
    threshold_l4 = float(find_threshold_at_precision(prec_arr, rec_arr, thresh_arr, 0.92))
    
    # Ensure monotonicity: L2 < L3 < L4
    threshold_l3 = max(threshold_l3, threshold_l2 + 0.02)
    threshold_l4 = max(threshold_l4, threshold_l3 + 0.02)
    
    # Clamp to [0, 1]
    threshold_l2 = np.clip(threshold_l2, 0.05, 0.95)
    threshold_l3 = np.clip(threshold_l3, 0.10, 0.98)
    threshold_l4 = np.clip(threshold_l4, 0.15, 1.00)
    
    print(f"\nDerived Rendering Thresholds (from calibrated PR curve):")
    print(f"  Level 2 (Nearly Dangerous):  >= {threshold_l2:.4f}  (F1-optimal)")
    print(f"  Level 3 (Dangerous):         >= {threshold_l3:.4f}  (Precision >= 0.80)")
    print(f"  Level 4 (Highly Dangerous):  >= {threshold_l4:.4f}  (Precision >= 0.92)")
    
    # Evaluate calibrated binary metrics at derived L2 threshold
    y_pred_cal = (oof_calibrated >= threshold_l2).astype(int)
    cal_acc = accuracy_score(y_binary, y_pred_cal)
    cal_prec = precision_score(y_binary, y_pred_cal, zero_division=0)
    cal_rec = recall_score(y_binary, y_pred_cal, zero_division=0)
    cal_f1 = f1_score(y_binary, y_pred_cal, zero_division=0)
    
    print(f"\nCalibrated Binary Metrics at L2 threshold ({threshold_l2:.4f}):")
    print(f"  Accuracy:  {cal_acc:.4f}")
    print(f"  Precision: {cal_prec:.4f}")
    print(f"  Recall:    {cal_rec:.4f}")
    print(f"  F1-score:  {cal_f1:.4f}")
    
    # 5-Level distribution on calibrated scores
    print(f"\nCalibrated 5-Level Distribution (OOF):")
    for lvl_name, (lo, hi) in LEVEL_THRESHOLDS.items():
        count = np.sum((oof_calibrated >= lo) & (oof_calibrated < hi))
        pct = count / len(oof_calibrated) * 100
        print(f"  {lvl_name} [{lo:.2f} - {hi:.2f}): {count} ({pct:.1f}%)")
    
    # =====================================================================
    # PHASE 4: Production Model Training on 100% Dataset
    # =====================================================================
    print("\n" + "="*65)
    print("PHASE 4: Production Model Training & Serialization")
    print("="*65)
    
    print("Fitting production RandomForestRegressor...")
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y_1d, sample_weight=w_1d)
    print("Model training complete.")
    
    # -------------------------------------------------------------------------
    # Feature Importance Analysis
    # -------------------------------------------------------------------------
    print("\n=== Random Forest Feature Importance (Top 15) ===")
    feature_names = list(X.columns)
    importances = model.feature_importances_
    
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    for rank, (_, row) in enumerate(df_importance.head(15).iterrows(), 1):
        print(f"  {rank:2d}. {row['Feature']:<12} : {row['Importance']:.4f}")
    
    # -------------------------------------------------------------------------
    # Production danger score distribution (on training data, for reference)
    # -------------------------------------------------------------------------
    print("\n=== Production Score Distribution (training data) ===")
    prod_raw = np.clip(model.predict(X), 0.0, 1.0)
    prod_calibrated = calibrator.predict(prod_raw)
    
    print("Raw scores:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(prod_raw, p):.4f}")
    
    print("Calibrated scores:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(prod_calibrated, p):.4f}")
    
    # -------------------------------------------------------------------------
    # Serialize production artifacts
    # -------------------------------------------------------------------------
    model_out = base_dir / "danger_model_production.pkl"
    calibrator_out = base_dir / "calibrator_production.pkl"
    metadata_out = base_dir / "model_metadata.pkl"
    
    # Remove old scaler if it exists
    old_scaler_path = base_dir / "scaler_production.pkl"
    if old_scaler_path.exists():
        old_scaler_path.unlink()
        print(f"\nRemoved obsolete scaler: {old_scaler_path.name}")
    
    print(f"\nSerializing production artifacts...")
    joblib.dump(model, model_out)
    joblib.dump(calibrator, calibrator_out)
    
    metadata = {
        "model_type": "RandomForestRegressor",
        "rf_params": rf_params,
        "num_features": NUM_FEATURES,
        "num_sampled_positions": NUM_SAMPLED_POSITIONS,
        "temporal_horizon": TEMPORAL_HORIZON,
        "sampling_stride": SAMPLING_STRIDE,
        "feature_names": feature_names,
        "level_thresholds": {k: v for k, v in LEVEL_THRESHOLDS.items()},
        "binary_threshold": BINARY_THRESHOLD,
        # Data-driven rendering thresholds
        "render_threshold_l2": threshold_l2,
        "render_threshold_l3": threshold_l3,
        "render_threshold_l4": threshold_l4,
        # Calibration metadata
        "calibration_method": "IsotonicRegression",
        "brier_raw": float(brier_raw),
        "brier_calibrated": float(brier_cal),
        "cv_f1_mean": float(np.mean(fold_metrics["f1"])),
        "cv_precision_mean": float(np.mean(fold_metrics["prec"])),
        "cv_recall_mean": float(np.mean(fold_metrics["rec"])),
    }
    joblib.dump(metadata, metadata_out)
    
    print(f"  Saved Model:      {model_out.name}")
    print(f"  Saved Calibrator: {calibrator_out.name}")
    print(f"  Saved Metadata:   {metadata_out.name}")
    print("="*65)
    print("Production training execution completed successfully.")

if __name__ == "__main__":
    main()

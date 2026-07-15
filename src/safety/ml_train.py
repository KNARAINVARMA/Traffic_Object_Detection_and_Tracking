import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(dataset_path, model_path):
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Define features and labels
    feature_cols = [
        'velocity_ms', 'r', 'theta', 'is_in_ring', 'delta_theta', 'omega', 'accel',
        'disp_30', 'disp_90', 'mean_vel_30', 'mean_vel_90',
        'dist_to_leader', 'leader_vel_diff', 'leader_omega_diff', 'leader_r_diff'
    ]
    
    label_cols = [
        'label_straddling', 'label_tailgating', 'label_overtaking',
        'label_braking', 'label_stoppage', 'label_wrong_way'
    ]
    
    # Drop rows with NaN in essential features
    # Fill leader-related NaNs or other minor missing fields
    df['is_in_ring'] = df['is_in_ring'].astype(int)
    
    # Perform one-hot encoding for class_name and lane
    df = pd.get_dummies(df, columns=['class_name', 'lane'], drop_first=False)
    
    # Update feature columns to include one-hot encoded variables
    one_hot_features = [c for c in df.columns if c.startswith('class_name_') or c.startswith('lane_')]
    all_feature_cols = feature_cols + one_hot_features
    
    print(f"Features list ({len(all_feature_cols)}): {all_feature_cols}")
    
    # Handle NaNs in features
    df[all_feature_cols] = df[all_feature_cols].fillna(0.0)
    
    X = df[all_feature_cols]
    y = df[label_cols]
    
    # Group-based split using track_id to avoid data leakage
    print("Performing group-based train/test split on track_id...")
    unique_tracks = df['track_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_tracks)
    split_idx = int(len(unique_tracks) * 0.8)
    train_tracks = set(unique_tracks[:split_idx])
    
    train_mask = df['track_id'].isin(train_tracks)
    test_mask = ~train_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Training set size: {len(X_train)} samples ({len(train_tracks)} tracks)")
    print(f"Test set size: {len(X_test)} samples ({len(unique_tracks) - len(train_tracks)} tracks)")
    
    # Build MultiOutputClassifier using RandomForest
    print("Training Multi-Output Random Forest Classifier...")
    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model on Test Set:")
    y_pred = model.predict(X_test)
    
    # Print metrics for each label
    for i, label_name in enumerate(label_cols):
        print("\n" + "="*50)
        print(f"VIOLATION: {label_name.replace('label_', '').upper()}")
        print("="*50)
        
        y_true_lbl = y_test.iloc[:, i].values
        y_pred_lbl = y_pred[:, i]
        
        acc = accuracy_score(y_true_lbl, y_pred_lbl)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_true_lbl, y_pred_lbl, target_names=['Normal', 'Violating'], zero_division=0))
        
    # Save the trained model and features metadata
    print(f"\nSaving model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_cols': all_feature_cols,
        'label_cols': label_cols
    }
    joblib.dump(model_data, model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML safety violations classifier")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input features and labels CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to save output model joblib file")
    args = parser.parse_args()
    
    train_model(args.dataset, args.model)

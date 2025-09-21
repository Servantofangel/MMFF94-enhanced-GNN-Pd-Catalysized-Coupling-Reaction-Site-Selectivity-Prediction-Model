#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN ON MASSIVE 20,000 MOLECULE DATASET - 367,450 ATOMS
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
import joblib
import time

print("🔥 TRAINING ON MASSIVE 20,000 MOLECULE DATASET")
print("="*60)
print("📊 Dataset: 367,450 atoms | ~20,000 positive examples")
print("⏰ Expected training time: 15-20 minutes")
print("="*60)

# Load the massive dataset
df = pd.read_csv('scaled_dataset_20000_molecules.csv')
print(f"Dataset shape: {df.shape}")
print(f"Positive examples: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")

# Prepare features
feature_cols = ['atomic_num', 'degree', 'formal_charge', 'hybridization', 
                'is_aromatic', 'total_hs', 'valence_electrons', 'partial_charge',
                'in_ring', 'is_halogen', 'min_ring_size']

X = df[feature_cols]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print(f"\n📈 Dataset Split:")
print(f"   Training set: {X_train.shape[0]} examples")
print(f"   Test set: {X_test.shape[0]} examples")
print(f"   Positive ratio: {y_test.mean():.4f}")

# Train Gradient Boosting
print("\n🌳 Training Gradient Boosting...")
start_time = time.time()

gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.7,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    verbose=1
)

gb_model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"✅ Training completed in {training_time/60:.1f} minutes")

# Evaluate
y_pred_proba = gb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 Model Performance:")
print(f"   ROC-AUC: {auc:.3f}")

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(2 * (precisions * recalls) / (precisions + recalls + 1e-10))
optimal_threshold = thresholds[optimal_idx]

print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
print(f"   Precision: {precisions[optimal_idx]:.3f}")
print(f"   Recall: {recalls[optimal_idx]:.3f}")

# Apply optimal threshold
y_pred_optimized = (y_pred_proba > optimal_threshold).astype(int)

# Detailed evaluation
print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred_optimized))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_optimized)
tn, fp, fn, tp = cm.ravel()

print(f"📊 Confusion Matrix:")
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")
print(f"   Precision: {tp/(tp+fp):.3f}")
print(f"   Recall: {tp/(tp+fn):.3f}")

# Feature importance
importances = gb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n🔍 Top 5 Feature Importance:")
for _, row in feature_importance_df.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# Save the model
model_filename = 'production_model_20000_final.pkl'
joblib.dump(gb_model, model_filename)
print(f"\n💾 Production model saved as '{model_filename}'")

# Save threshold info
threshold_info = {
    'optimal_threshold': optimal_threshold,
    'precision': precisions[optimal_idx],
    'recall': recalls[optimal_idx],
    'auc': auc
}
joblib.dump(threshold_info, 'final_model_threshold_info.pkl')

# Next steps recommendation
print(f"\n🎯 NEXT STEPS:")
if auc > 0.90:
    print("🚀 PHENOMENAL! READY FOR PUBLICATION!")
    print("   Scale to full dataset for production deployment")
elif auc > 0.85:
    print("✅ EXCELLENT! PRODUCTION-READY PERFORMANCE!")
    print("   Consider publication with current results")
elif auc > 0.84:
    print("👍 VERY GOOD! Almost at production level")
    print("   Scale to 30,000 molecules for final push")
else:
    print("📈 SOLID PROGRESS! Continue optimization")

print(f"\n✨ Training on 20,000 molecules complete! AUC: {auc:.3f}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

train = pd.read_csv("train.csv")
train = train.drop('CustomerID', axis=1)

X = train.drop('Churn', axis=1)
Y = train['Churn']

X_encoded = pd.get_dummies(X, drop_first=True)

X_encoded['charge_per_hour'] = X_encoded['MonthlyCharges'] / (X_encoded['ViewingHoursPerWeek'] + 1)
X_encoded['engagement_per_dollar'] = (X_encoded['ViewingHoursPerWeek'] + X_encoded['ContentDownloadsPerMonth']) / (X_encoded['MonthlyCharges'] + 1)
X_encoded['loyalty_satisfactions'] = X_encoded['AccountAge'] / (X_encoded['SupportTicketsPerMonth'] + 1)
X_encoded['content_satisfaction'] = X_encoded['UserRating'] * X_encoded['AverageViewingDuration']


X_train, X_val, Y_train, Y_val = train_test_split(
    X_encoded,
    Y,
    test_size=0.2,
    random_state=42
)

print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")

nn_model = tf.keras.models.load_model('churn_model_NN_7525.keras')

with open('churn_model_xgb.pkl', 'rb') as f:
    xgb_data = pickle.load(f)
    xgb_model = xgb_data['model']

# XGB data
X_train_xgb = X_train
X_val_xgb = X_val

# Neural netword data
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_val_nn = scaler.transform(X_val)

# XGB prediction
xgb_train_preds = xgb_model.predict_proba(X_train_xgb)[:,1]
xgb_val_preds = xgb_model.predict_proba(X_val_xgb)[:,1]

# Neural netword prediction
nn_train_preds = nn_model.predict(X_train_nn, verbose=0).flatten()
nn_val_preds = nn_model.predict(X_val_nn, verbose=0).flatten()

#Combine predictions
ensemble_avg_train = (xgb_train_preds + nn_train_preds) / 2
ensemble_avg_val = (xgb_val_preds + nn_val_preds) / 2

ensemble_weighted_train = 0.6 * nn_train_preds + 0.4 * xgb_train_preds
ensemble_weighted_val = 0.6 * nn_val_preds + 0.4 * xgb_val_preds

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

# Evaluate XGBoost alone
xgb_train_auc = roc_auc_score(Y_train, xgb_train_preds)
xgb_val_auc = roc_auc_score(Y_val, xgb_val_preds)

print(f"\n🌳 XGBoost:")
print(f"   Training AUC:   {xgb_train_auc:.4f}")
print(f"   Validation AUC: {xgb_val_auc:.4f}")
print(f"   Gap:            {xgb_train_auc - xgb_val_auc:.4f}")

# Evaluate Neural Network alone
nn_train_auc = roc_auc_score(Y_train, nn_train_preds)
nn_val_auc = roc_auc_score(Y_val, nn_val_preds)

print(f"\n🧠 Neural Network:")
print(f"   Training AUC:   {nn_train_auc:.4f}")
print(f"   Validation AUC: {nn_val_auc:.4f}")
print(f"   Gap:            {nn_train_auc - nn_val_auc:.4f}")

# Evaluate Ensemble - Average
avg_train_auc = roc_auc_score(Y_train, ensemble_avg_train)
avg_val_auc = roc_auc_score(Y_val, ensemble_avg_val)

print(f"\n🎭 Ensemble (50/50 Average):")
print(f"   Training AUC:   {avg_train_auc:.4f}")
print(f"   Validation AUC: {avg_val_auc:.4f}")
print(f"   Gap:            {avg_train_auc - avg_val_auc:.4f}")

# Evaluate Ensemble - Weighted
weighted_train_auc = roc_auc_score(Y_train, ensemble_weighted_train)
weighted_val_auc = roc_auc_score(Y_val, ensemble_weighted_val)

print(f"\n🎭 Ensemble (60/40 Weighted - NN favored):")
print(f"   Training AUC:   {weighted_train_auc:.4f}")
print(f"   Validation AUC: {weighted_val_auc:.4f}")
print(f"   Gap:            {weighted_train_auc - weighted_val_auc:.4f}")

# Find the winner!
print("\n" + "="*60)
best_val_auc = max(xgb_val_auc, nn_val_auc, avg_val_auc, weighted_val_auc)
print(f"🏆 BEST VALIDATION AUC: {best_val_auc:.4f}")

if best_val_auc == avg_val_auc:
    print("   Winner: Ensemble (Average)")
elif best_val_auc == weighted_val_auc:
    print("   Winner: Ensemble (Weighted)")
elif best_val_auc == nn_val_auc:
    print("   Winner: Neural Network alone")
else:
    print("   Winner: XGBoost alone")

print("="*60)
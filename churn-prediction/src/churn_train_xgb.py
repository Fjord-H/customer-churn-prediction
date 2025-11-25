import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
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

#X_encoded['is_loyal'] = (X_encoded['AccountAge'] > 24).astype(int) # new aggresive apporach: loyalty and disengaged (int)
#X_encoded['is_disengaged'] = ((X_encoded['ViewingHoursPerWeek'] < 10) & (X_encoded['ContentDownloadsPerMonth'] < 10)).astype(int)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_encoded,
    Y,
    test_size=0.2,
    random_state=42
)

scale_pos_weight = (Y_train == 0).sum() / (Y_train == 1).sum()

#print(f"Scale pos weight: {scale_pos_weight:.2f}")

#Building XGBoost model
model_xgb = XGBClassifier( 
    max_depth=6,
    learning_rate=0.01,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,                  # ← NEW! L1 regularization
    reg_lambda=1.0,                 # ← NEW! L2 regularization
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=100
)

model_xgb.fit(
    X_train, Y_train,
    eval_set=[(X_train, Y_train), (X_val, Y_val)],
    verbose=50
)

with open('churn_xgb.pkl', 'wb') as f:
    pickle.dump({
        'model': model_xgb,
        'best_score': model_xgb.best_score
    }, f)

print("XGB model saved!")
print(f"Stopped at tree: {model_xgb.best_iteration}")
print(f"Best validation AUC: {model_xgb.best_score:.4f}")
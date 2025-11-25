import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
train = pd.read_csv("train.csv")
train = train.drop('CustomerID', axis=1)

X = train.drop('Churn', axis=1)
y = train['Churn']

# One-hot encode
X_encoded = pd.get_dummies(X, drop_first=True)

# Add your 6 engineered features
X_encoded['charge_per_hour'] = X_encoded['MonthlyCharges'] / (X_encoded['ViewingHoursPerWeek'] + 1)
X_encoded['engagement_per_dollar'] = (X_encoded['ViewingHoursPerWeek'] + X_encoded['ContentDownloadsPerMonth']) / (X_encoded['MonthlyCharges'] + 1)
X_encoded['loyalty_satisfactions'] = X_encoded['AccountAge'] / (X_encoded['SupportTicketsPerMonth'] + 1)
X_encoded['content_satisfaction'] = X_encoded['UserRating'] * X_encoded['AverageViewingDuration']
X_encoded['is_loyal'] = (X_encoded['AccountAge'] > 24).astype(int)
X_encoded['is_disengaged'] = ((X_encoded['ViewingHoursPerWeek'] < 10) & (X_encoded['ContentDownloadsPerMonth'] < 10)).astype(int)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 30
plt.figure(figsize=(12, 10))  # Taller for 30 features
top_30 = feature_importance.head(30)

# Highlight ALL 6 engineered features
engineered_features = [
    'engagement_per_dollar', 'charge_per_hour', 
    'loyalty_satisfactions', 'content_satisfaction',
    'is_loyal', 'is_disengaged'
]

colors = ['#ff6b6b' if feat in engineered_features else '#4ecdc4' 
          for feat in top_30['feature']]

plt.barh(top_30['feature'], top_30['importance'], color=colors)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 30 Most Important Features (with 6 Engineered Features)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ff6b6b', label='Engineered Features (6 total)'),
    Patch(facecolor='#4ecdc4', label='Original Features')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.savefig('results/feature_importance_all6.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved!")

# Save full list
feature_importance.to_csv('results/feature_importance_full.csv', index=False)
print("✓ Full feature importance saved to CSV!")

# Print where your 6 features ranked
print("\n📊 Engineered Feature Rankings:")
for i, row in feature_importance.iterrows():
    if row['feature'] in engineered_features:
        rank = feature_importance.index.get_loc(i) + 1
        print(f"  {rank}. {row['feature']}: {row['importance']:.4f}")
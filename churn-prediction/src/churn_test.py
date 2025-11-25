print("\n" + "="*60)
print("MAKING TEST PREDICTIONS")
print("="*60)

# Load test data
test_df = pd.read_csv("test.csv")
test_customer_ids = test_df['CustomerID']
test_data = test_df.drop('CustomerID', axis=1)

# One-hot encode
X_test_encoded = pd.get_dummies(test_data, drop_first=True)

# Add all 6 features
X_test_encoded['charge_per_hour'] = X_test_encoded['MonthlyCharges'] / (X_test_encoded['ViewingHoursPerWeek'] + 1)
X_test_encoded['engagement_per_dollar'] = (X_test_encoded['ViewingHoursPerWeek'] + X_test_encoded['ContentDownloadsPerMonth']) / (X_test_encoded['MonthlyCharges'] + 1)
X_test_encoded['loyalty_satisfactions'] = X_test_encoded['AccountAge'] / (X_test_encoded['SupportTicketsPerMonth'] + 1)
X_test_encoded['content_satisfaction'] = X_test_encoded['UserRating'] * X_test_encoded['AverageViewingDuration']

#X_test_encoded['is_loyal'] = (X_test_encoded['AccountAge'] > 24).astype(int)
#X_test_encoded['is_disengaged'] = ((X_test_encoded['ViewingHoursPerWeek'] < 10) & (X_test_encoded['ContentDownloadsPerMonth'] < 10)).astype(int)

# Align columns with training
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

print(f"Test shape: {X_test_encoded.shape}")
print(f"Training shape: {X_encoded.shape}")

# Scale for NN
X_test_scaled = scaler.transform(X_test_encoded)

# Get predictions from both models
print("Getting predictions...")
nn_predictions = model.predict(X_test_scaled, verbose=0).flatten()
xgb_predictions = model_xgb.predict_proba(X_test_encoded)[:, 1]

# Ensemble: Weighted average (60% NN, 40% XGBoost)
ensemble_predictions = 0.6 * nn_predictions + 0.4 * xgb_predictions

# Create submission dataframe
prediction_df = pd.DataFrame({
    'CustomerID': test_customer_ids,
    'predicted_probability': ensemble_predictions
})

print(f"Predictions complete!")
print(f"Shape: {prediction_df.shape}")
print(prediction_df.head())
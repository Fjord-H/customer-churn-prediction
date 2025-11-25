import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



train = pd.read_csv("train.csv")
train = train.drop('CustomerID', axis=1)

X = train.drop('Churn', axis=1)
Y = train['Churn']

X_encoded = pd.get_dummies(X, drop_first=True)

X_encoded['charge_per_hour'] = X_encoded['MonthlyCharges'] / (X_encoded['ViewingHoursPerWeek'] + 1)
X_encoded['engagement_per_dollar'] = (X_encoded['ViewingHoursPerWeek'] + X_encoded['ContentDownloadsPerMonth']) / (X_encoded['MonthlyCharges'] + 1)
X_encoded['loyalty_satisfactions'] = X_encoded['AccountAge'] / (X_encoded['SupportTicketsPerMonth'] + 1)
X_encoded['content_satisfaction'] = X_encoded['UserRating'] * X_encoded['AverageViewingDuration']
# new aggresive apporach: loyalty and disengaged (int)(didn't work that much)
#X_encoded['is_loyal'] = (X_encoded['AccountAge'] > 24).astype(int) 
#X_encoded['is_disengaged'] = ((X_encoded['ViewingHoursPerWeek'] < 10) & (X_encoded['ContentDownloadsPerMonth'] < 10)).astype(int)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_encoded,
    Y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(Y_train),
    y=Y_train
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model = keras.Sequential([

    layers.Dense(128, activation='relu',input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(96, activation='relu'),  # ← Extra layer!
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name='auc')]
)

early_stop = EarlyStopping(
    monitor='val_auc',          
    patience=15,                
    mode='max',                 
    restore_best_weights=True,  
    verbose=1                   
)

checkpoint = ModelCheckpoint(
    'churn_nn_4_layers.keras',      
    monitor='val_auc',          
    mode='max',                
    save_best_only=True,        
    verbose=1                  
)

# On your local machine, save as .h5 instead:
model.save('churn_model_NN.h5')  # ← Use .h5 extension!

print("✓ Model saved in H5 format (compatible)")

history = model.fit(
    X_train_scaled,Y_train,                 
    validation_data=(X_val_scaled,Y_val),   
    epochs=100,                             
    batch_size=32,                          
    class_weight=class_weight_dict,          
    callbacks=[early_stop, checkpoint],      
    verbose=1                                
)

print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

val_loss, val_auc = model.evaluate(X_val_scaled, Y_val, verbose=1)

print(f" Final Validation Results:")
print(f" Loss: {val_loss:.4f}")
print(f" AUC:  {val_auc:.4f}")
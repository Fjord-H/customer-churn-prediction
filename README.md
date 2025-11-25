# Customer Churn Prediction for Streaming Services

Systematic approach to predicting customer churn using feature engineering, neural networks, and gradient boosting. Achieved 91st percentile (top 10%) in Coursera ML competition.

## Project Overview

This project tackles customer churn prediction for a streaming service using a dataset of 240,000+ customer records. The goal was to identify customers likely to cancel their subscriptions based on usage patterns, account details, and service interactions.

**Challenge Context:** Coursera Machine Learning Specialization capstone project

**Dataset:** 
- Training: 243,787 customers with 19 features
- Test: 104,480 customers (unlabeled)
- Target: Binary classification (churn vs. retain)
- Class distribution: 18% churn, 82% retain (imbalanced)

**Key Achievement:** 

- **91st Percentile** in Coursera ML Competition
- **0.750 AUC** on held-out test set
- Engineered 4 features that improved baseline by 14 percentile points (77th → 91st)

## Technical Stack

**Languages & Frameworks:**
- Python 3.7+
- TensorFlow 2.11 (Keras API)
- XGBoost 1.7

**Libraries:**
- pandas, NumPy (data manipulation)
- scikit-learn (preprocessing, metrics)
- matplotlib (visualization)

**Environment:**
- Local development: VS Code
- Training: Coursera Jupyter Lab

##  Project Structure
```
churn-prediction/                     
├── README.md                         
├── data/                             
│   ├── train.csv
│   ├── test.csv
│   └── data_descriptions.csv
├── notebooks/
│   └── coursera_submission.ipynb     
├── src/                               
│   ├── churn_train_nn_4_layers.py
│   ├── churn_train_xgb.py
│   ├── churn_train_ensemble.py
│   ├── churn_analyze_features.py
│   ├── churn_test.py
│   └── features_importance_plots.py
├── models/                            
│   ├── churn_nn_4_layers.keras
│   └── churn_xgb.pkl
└── results/                          
    ├── feature_importance_all6.png
    ├── numerical_corr.csv
    ├── categorical_corr.csv
    ├── prediction_submission.csv
    └── feature_importance_full

    
```

## Methodology

### 1. Exploratory Data Analysis

**Initial observations:**
- 18/82 class imbalance (churn/retain)
- No missing values
- Mix of numerical (10) and categorical (9) features
- AccountAge showed strongest correlation with churn (-0.198)

**Key insights:**
- Engagement metrics (viewing hours, downloads) inversely correlated with churn
- Premium subscription users churned less than Basic users
- Support tickets positively correlated with churn

### 2. Feature Engineering 

#### Initial Feature Set (4 features)

Based on business logic around engagement, value, and satisfaction:
```python
# Value perception
charge_per_hour = MonthlyCharges / (ViewingHoursPerWeek + 1)
engagement_per_dollar = (ViewingHours + Downloads) / (MonthlyCharges + 1)

# Customer satisfaction
loyalty_satisfactions = AccountAge / (SupportTicketsPerMonth + 1)
content_satisfaction = UserRating × AverageViewingDuration
```

**Initial result:** 0.7499 AUC (91st percentile)

### Expansion Attempt (adding 2 binary features)

After reviewing a previous experiment where `is_loyal` showed high importance (0.55), I hypothesized that adding binary threshold features might capture additional signal:
```python
is_loyal = (AccountAge > 24)  # Long-term customers
is_disengaged = (ViewingHours < 10) & (Downloads < 10)  # Low activity
```

**Result:** 0.7499 AUC (unchanged)

### Feature Importance Analysis

![Feature Importance](results/feature_importance_all6.png)

| Rank | Feature | Importance | Status |
|------|---------|-----------|--------|
| 1 | engagement_per_dollar | 0.164 | ✓ Core feature |
| 2 | loyalty_satisfactions | 0.133 | ✓ Core feature |
| 12 | charge_per_hour | 0.022 | ✓ Core feature |
| 22 | content_satisfaction | 0.014 | ✓ Core feature |
| 33 | is_disengaged | 0.009 | Redundant |
| 34 | is_loyal | 0.000 | Redundant |

**Key Findings:**

1. **Continuous features dominated:** The top 2 features were both continuous, capturing nuanced patterns that binary thresholds couldn't match.

2. **Feature redundancy:** `is_loyal` showed zero importance because its signal was already captured by `loyalty_satisfactions` (which encodes the same loyalty concept but with continuous values).

3. **Performance plateau:** Three submissions with different feature sets all achieved 0.7499, indicating the original 4 features had extracted maximum available signal from the data.

**Lesson learned:** Well-designed continuous features that encode business logic can be more powerful than adding numerous binary flags. Quality > quantity in feature engineering.


### 3. Model Development

**Approach:** Trained three model types to compare performance:

#### Neural Network (Primary Model)
- Architecture: 128 → 64 → 32 → 1 (fully connected)
- Regularization: Batch normalization + 30% dropout per layer
- Class weights: Balanced to handle 18/82 imbalance
- Early stopping: Patience=15 on validation AUC
- **Result:** 0.7528 validation AUC

#### XGBoost (Secondary Model)
- Configuration: max_depth=6, learning_rate=0.01, 1000 estimators
- Regularization: L1=0.1, L2=1.0, early_stopping=100
- Scale pos weight: 4.51 (to handle imbalance)
- **Result:** 0.7507 validation AUC

#### Ensemble (Final Model)
- Weighted average: 60% NN + 40% XGBoost
- **Result:** 0.7527 validation AUC

### 4. Hyperparameter Tuning

Experimented with multiple configurations:
- NN architectures: 3-layer, 4-layer, varying dropout (0.2-0.5)
- XGBoost: depth 4-6, learning rates 0.01-0.1
- **Finding:** Diminishing returns beyond initial configuration; all variants converged to 0.752-0.753 AUC

### 5. Final Evaluation

**Test set performance:** 0.7499 AUC (91st percentile)

## Results

| Model Configuration | Validation AUC | Notes |
|---------------------|----------------|-------|
| Neural Network (3-layer) | 0.7525 | Baseline architecture |
| Neural Network (4-layer) | 0.7528 | Best single model |
| Neural Network (high dropout) | 0.7528 | Same performance, higher Val loss |
| XGBoost (regularized) | 0.7507 | Strong performance, longer training |
| Ensemble (60/40 weighted) | 0.7527 | Marginal improvement |
| **Final Test Score** | **0.7499** | **91st percentile** |

**Performance trajectory:**
- First submission: 0.747 (77th percentile)
- Second submission (with features): 0.7499 (91st percentile)
- Third submission (more features): 0.7499 (no improvement - hit ceiling)


## Challenges & Solutions

### Challenge 1: Class Imbalance (18/82 split)

**Problem:** 
Initial models achieved 82% accuracy by simply predicting "no churn" for all customers, meaning the data is technically accurate but completely useless for business decisions.

**Solution:** 
- Implemented balanced class weights in neural networks
```python
  class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
  # Result: {0: 0.61, 1: 2.76} - churners weighted ~4.5x higher
```
- Used `scale_pos_weight=4.51` in XGBoost
- Evaluated models using AUC instead of accuracy (AUC penalizes poor ranking)

**Result:** 
Models learned to properly distinguish churners from non-churners, achieving 0.75 AUC with only 0.002 train-validation gap (minimal overfitting).

---

### Challenge 2: Model Selection - Neural Networks vs Trees

**Initial assumption:** 
Expected XGBoost to outperform neural networks for tabular data (common wisdom in ML).

**Reality:** 
Neural networks performed slightly better (0.7528 vs 0.7507 validation AUC).

**Analysis:**

| Model | Val AUC | Train-Val Gap | Training Time |
|-------|---------|---------------|---------------|
| Neural Network | 0.7528 | 0.0002 | ~10 minutes |
| XGBoost | 0.7507 | 0.0221 | ~1 minute (local) / ~5 hours (Coursera) |
| Ensemble | 0.7527 | 0.0108 | ~1 minute (local) / ~5 hours (Coursera) |

**Key findings:**
1. **NN showed perfect generalization** (negative gap) while XGBoost had slight overfitting
2. **Feature engineering leveled the playing field** : well-designed features allowed Neural Network to compete with tree-based methods
3. **Ensemble provided minimal gain** (0.0001) : both models learned similar patterns
4. **Computational trade-off** - XGBoost trained 10x faster locally (~1 minute vs ~10 minutes for NN) but 30x slower on Coursera's shared infrastructure (~5 hours vs ~10 minutes). This made NN the practical choice for the competition despite XGBoost's faster local training.

**Lesson:** When feature engineering is strong, model choice matters less. Focus on features first, then optimize model selection.

---

### Challenge 3: Performance Plateau & Diminishing Returns

**Experimentation attempts:**

Architecture variations:
- 3-layer (64→32→1): 0.7525 AUC
- 4-layer (128→96→64→32): 0.7528 AUC (+0.0003)
- Higher dropout (0.5): 0.7528 AUC (same performance, faster training but higher val loss)

XGBoost tuning:
- Shallow trees (depth=4): 0.7494 AUC
- Regularized (depth=6, L1=0.1, L2=1.0): 0.7507 AUC
- Lower learning rate (0.01): 0.7510 AUC (+0.0003)

Feature expansion:
- 4 features: 0.7499 test AUC (91st percentile)
- 6 features: 0.7499 test AUC (unchanged)

**Pattern observed:** 
All optimization efforts yielded improvements of ≤0.0003 AUC on validation, which disappeared on test set.

**Interpretation:**
1. **Data ceiling reached** - the dataset's inherent predictability limits performance
2. **Feature quality > model complexity** - jumping from 77th to 91st percentile came from feature engineering, not hyperparameter tuning
3. **Overfitting risk** - chasing validation improvements doesn't guarantee test improvements

**Takeaway:** 
In production, a 0.0003 AUC improvement requiring 20x more computation is rarely worthwhile. This should be taken as recognition that we reached practical limits.

---

## Conclusion

### 1. Feature Engineering is the Highest Leverage Activity

**Impact comparison:**
- Baseline (no engineered features): 0.747 AUC (77th percentile)
- With 4 engineered features: 0.7499 AUC (91st percentile)
- **Improvement: +0.003 AUC, +14 percentile points**

vs.

- Model tuning (all experiments): +0.0003 AUC gain
- Ensemble methods: +0.0001 AUC gain

**Conclusion:** Investing time in thoughtful feature engineering (based on domain knowledge) provided 10x more value than hyperparameter optimization.

---

### 2. Continuous Features > Binary Thresholds

Binary features like `is_loyal` (AccountAge > 24) are intuitive but less powerful than continuous features that encode the same concept with nuance:
```python
# Less powerful: Binary threshold
is_loyal = (AccountAge > 24)  # Importance: 0.000

# More powerful: Continuous ratio
loyalty_satisfactions = AccountAge / (SupportTickets + 1)  # Importance: 0.133
```

The continuous feature captures:
- How long they've been a customer (AccountAge)
- How satisfied they've been (inverse of SupportTickets)
- Relative magnitude (10 years vs 3 years with same ticket count)

---

### 3. Model Convergence Indicates Feature Completeness

When multiple model types (neural networks, gradient boosting) and architectures converge to the same performance (~0.75), it suggests:
1. The features have captured most available signal
2. Further gains require new data sources or features
3. You've reached the practical ceiling for the problem

**In this project:** 3 submissions with different approaches all scored 0.7499, confirming feature saturation.

---

### 4. Know When to Stop

Recognizing performance ceilings prevents wasted effort:
- Submission 1 (baseline): 0.747
- Submission 2 (with features): 0.7499 ← Big jump!
- Submission 3 (more features): 0.7499 ← Plateau signal

**Stopping criteria:**
- Multiple approaches converge to same score
- Validation improvements don't transfer to test set
- ROI of optimization time diminishes

## How to Run

### Prerequisites
```bash
pip install tensorflow xgboost pandas scikit-learn matplotlib
```

### Training Models Locally
```bash
# Train neural network
python src/churn_train_nn_4_layers.py

# Train XGBoost
python src/churn_train_xgb.py

# Train ensemble
python src/churn_train_ensemble.py
```

### Coursera Submission
Upload `notebooks/coursera_submission.ipynb` to Coursera Jupyter Lab and run all cells.

**Note:** Model training times vary significantly between local and Coursera challenge environments 
(see Challenge 2 for details).

## Author

**Fjord-H** - [LinkedIn](#) | [GitHub](#)

---

*Last updated: November 2025*

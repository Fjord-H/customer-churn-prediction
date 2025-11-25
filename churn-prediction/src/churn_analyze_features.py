import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")

#Check shapes
#print(train.shape,'\n',train.head() )

#check info about std and means
#print(train.describe())

#check missing data
#print(train.isnull())
#print(train.isnull().sum())

# Print all column names
#print(train.columns.tolist())

# Check data types
#print(train.dtypes)

numerical_features = train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = train.select_dtypes(include=['object']).columns.tolist()

numerical_features.remove('Churn')
categorical_features.remove('CustomerID')

#print("Numerical features: ",len(numerical_features),"\n",numerical_features)
#print("Categorical features: ",len(categorical_features),"\n",categorical_features)

# Get (numerical) correlations with Churn using pandas corr() method
correlations = train[numerical_features + ['Churn']].corr()['Churn'].sort_values(ascending=False)

#print("Correlation with Churn:",correlations)

# Get categorical correlation with churn and put it in the csv
categorical_analysis = {}

for feature in categorical_features:
    churn_rate= train.groupby(feature)['Churn'].mean().sort_values(ascending=False)
    #print(churn_rate)
    #print(train[feature].value_counts().to_dict())
    categorical_analysis[feature] = churn_rate

categorical_df = pd.DataFrame(categorical_analysis)
categorical_df.to_csv('categorical_corr.csv')

#Print out the csv for features
numerical_df = correlations.to_frame()
numerical_df.columns = ['Correlation_with_Churn']
numerical_df.to_csv('numerical_corr.csv')






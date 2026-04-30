import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Dataset (Using a standard telecom churn structure)
# df = pd.read_csv('telecom_churn.csv')
# For demonstration, we'll assume a standard structure
def train_model(df):
    # 2. EDA & Feature Engineering
    df = df.drop(['customerID'], axis=1, errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Encoding categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 3. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Modeling
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return accuracy_score(y_test, model.predict(scaler.transform(X_test)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(train_path, test_path):
    """Load training and testing datasets and remove unnamed columns"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Remove any unnamed columns (extra columns from CSV)
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Separate features and labels, encode target"""
    X_train = train_df.drop('prognosis', axis=1)
    y_train = train_df['prognosis']
    X_test = test_df.drop('prognosis', axis=1)
    y_test = test_df['prognosis']
    
    # Encode disease labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return X_train, X_test, y_train_encoded, y_test_encoded, le

def save_model(model, encoder, feature_names, filename='../models/disease_predictor.pkl'):
    """Save trained model and encoder"""
    model_data = {
        'model': model,
        'encoder': encoder,
        'feature_names': feature_names
    }
    joblib.dump(model_data, filename)
    print(f"✅ Model saved to {filename}")

def load_model(filename='../models/disease_predictor.pkl'):
    """Load trained model and encoder"""
    model_data = joblib.load(filename)
    return model_data['model'], model_data['encoder'], model_data['feature_names']
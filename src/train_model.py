import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from utils import load_data, preprocess_data, save_model

print("="*60)
print("🏥 DISEASE PREDICTION MODEL TRAINING")
print("="*60)

# Load data
print("\n📂 Loading data...")
train_df, test_df = load_data('../data/Training.csv', '../data/Testing.csv')
print(f"✅ Training data shape: {train_df.shape}")
print(f"✅ Testing data shape: {test_df.shape}")

# Print column info to debug
print(f"\n📋 Columns in training data: {list(train_df.columns[:5])}... (showing first 5)")
print(f"📋 Last column: {list(train_df.columns)[-1]}")

# Preprocess
print("\n🔄 Preprocessing data...")
X_train, X_test, y_train, y_test, label_encoder = preprocess_data(train_df, test_df)
print(f"✅ Number of diseases: {len(label_encoder.classes_)}")
print(f"✅ Number of features: {X_train.shape[1]}")

# Train Random Forest model
print("\n🌲 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"✅ Random Forest Accuracy: {accuracy_rf:.4f}")

# Try XGBoost (optional - if you have it installed)
try:
    import xgboost as xgb
    print("\n⚡ Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"✅ XGBoost Accuracy: {accuracy_xgb:.4f}")
    
    # Choose best model
    best_model = rf_model if accuracy_rf >= accuracy_xgb else xgb_model
    print(f"\n🏆 Best model accuracy: {max(accuracy_rf, accuracy_xgb):.4f}")
except ImportError:
    print("\n⚠️ XGBoost not installed. Using Random Forest as best model.")
    best_model = rf_model
    print(f"\n🏆 Best model accuracy: {accuracy_rf:.4f}")

# Detailed classification report
print("\n📊 Classification Report (first 10 diseases):")
report = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_, output_dict=True)
for disease in list(label_encoder.classes_)[:10]:
    if disease in report:
        print(f"  {disease}: precision={report[disease]['precision']:.2f}, recall={report[disease]['recall']:.2f}")

# Confusion Matrix Visualization
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Disease Prediction', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('../models/confusion_matrix.png')
print("✅ Confusion matrix saved to models/confusion_matrix.png")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📈 Top 20 Most Important Symptoms:")
for i, row in feature_importance.head(20).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Most Important Symptoms', fontsize=16)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../models/feature_importance.png')
print("✅ Feature importance plot saved to models/feature_importance.png")

# Save the best model
save_model(best_model, label_encoder, X_train.columns.tolist())
print("\n🎉 Model training complete!")
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import subprocess

# === 1. DATA ===
print('Step 1: Loading data...')
df = pd.read_csv('data/reference.csv')
X = df.drop('prediction', axis=1)
y = df['prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. TRAIN + MLFLOW ===
print('Step 2: Training model + logging to MLflow...')
mlflow.set_experiment('mlops-pipeline')

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    mlflow.log_param('n_estimators', 100)
    mlflow.log_metric('accuracy', acc)
    mlflow.sklearn.log_model(model, 'model')
    
    print(f'Accuracy: {acc:.4f}')

# === 3. SAVE MODEL ===
print('Step 3: Saving model with joblib...')
joblib.dump(model, 'model.pkl')

# === 4. MONITOR ===
print('Step 4: Running drift monitoring...')
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
current = pd.read_csv('data/current.csv')
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df, current_data=current)
report.save_html('monitoring_report.html')
print('Drift report updated.')

print('')
print('Pipeline complete.')
print(f'Model accuracy: {acc:.4f}')
print('MLflow UI: run mlflow ui to view experiments')
print('Drift report: open monitoring_report.html')

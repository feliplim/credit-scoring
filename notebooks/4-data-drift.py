# System management 
import os

# Data manipulation
import pandas as pd

# Data drift
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Set main directory
project_path = '/Users/felipelima/Documents/projets/credit-scoring/'
os.chdir(project_path)

# Load data
data = pd.read_csv('data/processed/train_feature_engineering_encoded.csv').drop(columns=['SK_ID_CURR'])
data_test = pd.read_csv('data/processed/test_feature_engineering_encoded.csv').drop(columns=['SK_ID_CURR'])

# Create evidently report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])
data_drift_report.run(reference_data=data.drop(columns='TARGET'),
                      current_data=data_test, column_mapping=None)

# Save report
data_drift_report.save_html('docs/data_drift_report.html')


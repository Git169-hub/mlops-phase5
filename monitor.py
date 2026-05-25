import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

reference = pd.read_csv('data/reference.csv')
current = pd.read_csv('data/current.csv')

report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(reference_data=reference, current_data=current)
report.save_html('monitoring_report.html')
print('Report saved to monitoring_report.html')

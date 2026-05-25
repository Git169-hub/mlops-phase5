import pandas as pd
import numpy as np

np.random.seed(42)

reference = pd.DataFrame({
    'query_length': np.random.normal(50, 10, 200),
    'num_words': np.random.normal(8, 2, 200),
    'response_time_ms': np.random.normal(300, 50, 200),
    'prediction': np.random.choice([0, 1], 200, p=[0.7, 0.3])
})

current = pd.DataFrame({
    'query_length': np.random.normal(80, 15, 100),
    'num_words': np.random.normal(8, 2, 100),
    'response_time_ms': np.random.normal(400, 80, 100),
    'prediction': np.random.choice([0, 1], 100, p=[0.5, 0.5])
})

reference.to_csv('data/reference.csv', index=False)
current.to_csv('data/current.csv', index=False)
print('Data generated.')

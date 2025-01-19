import pandas as pd

# Load the datasets
cardiovascular_health_data_path = 'cardiovascular_health_data.csv'
sleep_quality_data_path = 'sleep_quality_data.csv'

cardiovascular_health_data = pd.read_csv(cardiovascular_health_data_path)
sleep_quality_data = pd.read_csv(sleep_quality_data_path)

# Display the first few rows of each dataset to understand their structure
cardiovascular_health_data.head(), sleep_quality_data.head()

import numpy as np

# Define logic to generate illness/infection based on the deviation and sleep_score
def predict_illness(current_heart_rate_rest, resting_heart_rate, sleep_score):
    deviation = abs(current_heart_rate_rest - resting_heart_rate)
    # Simple rule-based generation for simulation purposes:
    # Higher deviation or poor sleep score (< 5) increases chance of 'yes'
    if deviation > 4 and sleep_score < 5:
        return 'yes'
    return 'no'

# Generate synthetic data
np.random.seed(42)  # Ensure reproducibility
num_rows = 10000

# Randomly sample values based on observed distributions
resting_heart_rate_samples = np.random.uniform(40, 100, size=num_rows)  # Typical RHR range
current_heart_rate_rest_samples = resting_heart_rate_samples + np.random.uniform(-15, 15, size=num_rows)
sleep_score_samples = np.random.uniform(0, 10, size=num_rows)  # Sleep scores range from 0 to 10

# Predict illness/infection based on defined logic
illness_infection_samples = [
    predict_illness(cr, rr, ss)
    for cr, rr, ss in zip(current_heart_rate_rest_samples, resting_heart_rate_samples, sleep_score_samples)
]

# Create the synthesized dataset
synthesized_data = pd.DataFrame({
    'resting_heart_rate': resting_heart_rate_samples,
    'current_heart_rate_rest': current_heart_rate_rest_samples,
    'sleep_score': sleep_score_samples,
    'illness/infection': illness_infection_samples
})

# Display first few rows
synthesized_data.head()

synthesized_data.to_csv("infection_illness.csv", index=False)

import numpy as np
import pandas as pd

# Load previously generated datasets
sleep_data = pd.read_csv("sleep_quality_data.csv")
physical_activity_data = pd.read_csv("physical_activity_data.csv")

# Ensure both datasets have the same number of rows (if necessary)
n_samples = min(len(sleep_data), len(physical_activity_data))
sleep_data = sleep_data.iloc[:n_samples].reset_index(drop=True)
physical_activity_data = physical_activity_data.iloc[:n_samples].reset_index(drop=True)

# Extract relevant scores
sleep_score = sleep_data["sleep_score"]
physical_activity_score = physical_activity_data["physical_activity_score"]

# 33. Generate Resting Heart Rate (40-100, ideal: 60-100)
resting_heart_rate = np.clip(np.random.normal(loc=70, scale=15, size=n_samples), 40, 100)

# 4. Generate Current Heart Rate (Rest)
# Ideal case: within ±4, Non-ideal: within ±10
current_heart_rate_rest = resting_heart_rate + np.where(
    np.random.rand(n_samples) < 0.7,  # 70% ideal
    np.random.uniform(-4, 4, n_samples),  # Ideal range
    np.random.uniform(-10, 10, n_samples)  # Non-ideal range
)

# 5. Air Quality Index (AQI) Data
air_quality = np.clip(np.random.normal(loc=3, scale=1, size=n_samples), 1, 5)

# Calculate Respiratory Health Score (Weighted average)
resp_score = 10 - (
    1 * np.abs(resting_heart_rate - 70) / 70 +  # Penalize deviation from ideal resting HR
    3 * np.abs(current_heart_rate_rest - resting_heart_rate) / 10 +  # Penalize large deviations in current HR
    2 * np.clip(10 - sleep_score, 0, 10) / 10 +  # Penalize deviations in sleep score
    2 * np.clip(10 - physical_activity_score, 0, 10) / 10 +  # Penalize deviations in physical activity score
    1.0 * np.clip(5 - air_quality, 0, 5) / 5  # Account for air quality
)
resp_score = np.clip(resp_score, 1, 10)  # Ensure scores are within 1-10

# Combine into a DataFrame
respiratory_health_data = pd.DataFrame({
    "resting_heart_rate": np.round(resting_heart_rate, 1),
    "current_heart_rate_rest": np.round(current_heart_rate_rest, 1),
    "sleep_score": sleep_score,
    "physical_activity_score": physical_activity_score,
    "air_quality": np.round(air_quality, 1),
    "resp_score": np.round(resp_score, 1)
})

# Save as CSV for download
output_file_cv = "respiratory_health_data.csv"
respiratory_health_data.to_csv(output_file_cv, index=False)
import numpy as np
import pandas as pd

# Generate Cardiovascular Health Data
np.random.seed(42)

# Number of samples
n_samples = 10000

# 1. Generate Heart Rate Condition (Categorical with weighted probabilities)
heart_rate_condition = np.random.choice(
    ["Skipped beats", "Extra beats", "Mixed", "Uneven timing", "Normal"],
    size=n_samples,
    p=[0.1, 0.1, 0.1, 0.1, 0.6]
)

# 2. Generate Resting Heart Rate (40-100, ideal: 60-100)
resting_heart_rate = np.clip(np.random.normal(loc=70, scale=15, size=n_samples), 40, 100)

# 3. Generate Current Heart Rate (Rest)
# Ideal case: within ±4, Non-ideal: within ±10
current_heart_rate_rest = resting_heart_rate + np.where(
    np.random.rand(n_samples) < 0.7,  # 70% ideal
    np.random.uniform(-4, 4, n_samples),  # Ideal range
    np.random.uniform(-10, 10, n_samples)  # Non-ideal range
)

# 4. Generate Increase During Activity (60-200, ideal: 100-170)
increase_during_activity = np.clip(np.random.normal(loc=130, scale=30, size=n_samples), 60, 200)

# 5. Generate Post-Activity Beat Drop (5-45, ideal: >12)
post_activity_beat_drop = np.clip(np.random.normal(loc=20, scale=10, size=n_samples), 5, 45)

# 6. Load Health Management Score (from previous dataset)
health_management_data = pd.read_csv("health_and_weight_management_data.csv")
health_management_score = health_management_data["health_management_score"].iloc[:n_samples]

# 7. Air Quality Index (AQI) Data
air_quality = np.clip(np.random.normal(loc=3, scale=1, size=n_samples), 1, 5)

# 7. Generate Cardiovascular Score (Weighted Calculation)
cardiovascular_score = 10 - (
    2 * np.abs(resting_heart_rate - 70) / 70 +  # Penalize deviation from ideal resting HR
    2 * np.abs(current_heart_rate_rest - resting_heart_rate) / 10 +  # Penalize large deviations in current HR
    1.5 * np.abs(increase_during_activity - 135) / 135 +  # Penalize deviations in activity HR increase
    1.5 * np.clip(12 - post_activity_beat_drop, 0, 12) / 12 +  # Penalize low beat drops
    1.5 * (heart_rate_condition != "Normal").astype(int) +  # Penalize non-normal heart rate conditions
    1.0 * np.clip(10 - health_management_score, 0, 10) / 10  + # Account for health management score
    1.0 * np.clip(5 - air_quality, 0, 5) / 5  # Account for air quality
)
cardiovascular_score = np.clip(cardiovascular_score, 1, 10)  # Ensure scores are within 1-10

# Combine into a DataFrame
cardiovascular_health_data = pd.DataFrame({
    "heart_rate_condition": heart_rate_condition,
    "resting_heart_rate": np.round(resting_heart_rate, 1),
    "current_heart_rate_rest": np.round(current_heart_rate_rest, 1),
    "increase_during_activity": np.round(increase_during_activity, 1),
    "post_activity_beat_drop": np.round(post_activity_beat_drop, 1),
    "health_management_score": np.round(health_management_score, 1),
    "air_quality": np.round(air_quality, 1),
    "cardiovascular_score": np.round(cardiovascular_score, 1)
})

# Save as CSV for download
output_file_cv = "cardiovascular_health_data.csv"
cardiovascular_health_data.to_csv(output_file_cv, index=False)
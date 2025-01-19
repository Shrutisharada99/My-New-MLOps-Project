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

# 1. Generate Heart Rate Condition (Categorical with weighted probabilities)
heart_rate_condition = np.random.choice(
    ["Skipped beats", "Extra beats", "Mixed", "Uneven timing", "Normal"],
    size=n_samples,
    p=[0.1, 0.1, 0.1, 0.1, 0.6]
)

# Calculate Health Management Score (Weighted average)
mental_health_score = 10- (
    3 * (heart_rate_condition != "Normal").astype(int) +
    3.0 * np.clip(10 - sleep_score, 0, 10) / 10 +
    3.0 * np.clip(10 - physical_activity_score, 0, 10) / 10
)

cardiovascular_score = np.clip(mental_health_score, 1, 10)  # Ensure scores are within 1-10

# Combine into a new dataset
mental_health_management_data = pd.DataFrame({
    "heart_rate_condition": heart_rate_condition,
    "sleep_score": sleep_score,
    "physical_activity_score": physical_activity_score,
    "mental_health_score": np.round(mental_health_score, 1)
})

# Save the new dataset
output_file_health = "mental_health_score.csv"
mental_health_management_data.to_csv(output_file_health, index=False)
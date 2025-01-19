import pandas as pd

phy = pd.read_csv('physical_activity_data.csv')
sleep = pd.read_csv('sleep_quality_data.csv')
cardio = pd.read_csv('cardiovascular_health_data.csv')
health = pd.read_csv('health_and_weight_management_data.csv')
mental = pd.read_csv('mental_health_score.csv')
respiratory = pd.read_csv('respiratory_health_data.csv')
infection = pd.read_csv('infection_illness.csv')

# Merge the datasets
overall_wellness_data = pd.concat([phy, sleep, cardio, mental['mental_health_score'], respiratory['resp_score'], infection['illness/infection']], axis=1)

# Save the merged dataset
overall_wellness_data.to_csv('overall_wellness_data.csv', index=False)

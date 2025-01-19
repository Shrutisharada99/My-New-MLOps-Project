import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
wellness = pd.read_csv("../artifacts/overall_wellness_data.csv")
print(wellness.shape)
#print(wellness.info())

def feature_engineering():
    global wellness

    wellness['Deviation_heart_rate_rest'] = np.abs(wellness['resting_heart_rate'] - wellness['current_heart_rate_rest'])
    wellness.drop('current_heart_rate_rest', axis=1, inplace=True)
    
def encode_column():
    global wellness

    feature_engineering()

    # Encoding the categorical variables   

    wellness = pd.get_dummies(wellness, columns= ['heart_rate_condition'], drop_first = True)
    print(wellness.shape)
    wellness[wellness.select_dtypes(include=['bool']).columns] = wellness.select_dtypes(include=['bool']).astype(int)
    
    wellness['illness/infection'] = wellness['illness/infection'].map({'yes': 1, 'no': 0})

    # Encoding the numerical variable
    # Let's normalize the values of the columns, so that the values are in the same range using MinMaxScaler.    

    scaler = MinMaxScaler()
    num_features = wellness.select_dtypes(exclude="object").columns

    columns_to_remove = ['physical_activity_score','sleep_score','health_management_score', 'illness/infection', 'cardiovascular_score', 'mental_health_score', 'resp_score']

    # Remove certain column names
    num_features = [col for col in num_features if col not in columns_to_remove]

    wellness[num_features] = scaler.fit_transform(wellness[num_features])
    wellness['health_management_score_scaled'] = scaler.fit_transform(wellness[['health_management_score']])

    return wellness

if __name__ == "__main__":
    encoded_data = encode_column()

    # Save the encoded data
    encoded_data.to_csv("../artifacts/processed_data.csv", index=False)

    phy_act_data = encoded_data[['steps','distance_walked','very_active_distance','moderately_active_distance','lightly_active_distance','calories_burnt','physical_activity_score']]
    phy_act_data.to_csv("../artifacts/physical_activity_train_data.csv", index=False)

    sleep_data = encoded_data[['total_sleep_duration','light_sleep_stage','deep_sleep_stage','REM_sleep_stage','number_of_awakenings','sleep_score']]
    sleep_data.to_csv("../artifacts/sleep_quality_train_data.csv", index=False)

    cardio_data = encoded_data[['resting_heart_rate','increase_during_activity','post_activity_beat_drop','air_quality','cardiovascular_score','Deviation_heart_rate_rest','heart_rate_condition_Mixed','heart_rate_condition_Normal','heart_rate_condition_Skipped beats','heart_rate_condition_Uneven timing','health_management_score_scaled']]
    cardio_data.to_csv("../artifacts/cardiovascular_health_train_data.csv", index=False)
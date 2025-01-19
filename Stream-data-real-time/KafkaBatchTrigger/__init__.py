import logging
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests

import azure.functions as func
from confluent_kafka import Consumer, KafkaException, KafkaError

logging.basicConfig(level=logging.INFO)

app = func.FunctionApp()

# Kafka configuration
config = {
    'bootstrap.servers': 'pkc-619z3.us-east1.gcp.confluent.cloud:9092',
    'group.id': 'consumer-group-1',
    'auto.offset.reset': 'earliest',
    'sasl.mechanisms': 'PLAIN',
    'security.protocol': 'SASL_SSL',
    'sasl.username': 'VPVOVS6AIQT3V577',
    'sasl.password': 'CscjmODsggXkRt214VwH6G7E8Pv0hpyWTKBwCsS/9zdBf92h3n6ziMffCczZUgAq',
}

# Create a Kafka consumer
consumer = Consumer(config)
topic = "your_kafka_topic"

@app.function_name(name="KafkaConsumerFunction")
@app.route(route="kafka-consume", methods=["POST"])
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Kafka consumer function triggered')
    try:
        consumer.subscribe([topic])

        msg = consumer.poll(timeout=1.0)

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                logging.info('End of partition reached {0}/{1}'.format(msg.topic(), msg.partition()))
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            # Parse input JSON data into a Pandas DataFrame
            data_list = json.loads(msg)
            wellness = pd.DataFrame(data_list)

            # Preprocess the data
            transformed_wellness = encode_column(wellness)

            # Extract specific datasets for different endpoints
            cardio = transformed_wellness[['resting_heart_rate','increase_during_activity','post_activity_beat_drop','air_quality','Deviation_heart_rate_rest','heart_rate_condition_Mixed','heart_rate_condition_Normal','heart_rate_condition_Skipped beats','heart_rate_condition_Uneven timing','health_management_score_scaled']]
            phy = transformed_wellness[['steps','distance_walked','very_active_distance','moderately_active_distance','lightly_active_distance','calories_burnt']]
            sleep = transformed_wellness[['total_sleep_duration','light_sleep_stage','deep_sleep_stage','REM_sleep_stage','number_of_awakenings']]

            # Define the endpoints and headers
            cardio_endpoint = "https://ml-workspace-cardio.eastus2.inference.ml.azure.com/score"
            phy_endpoint = "https://ml-workspace-lrtil.eastus2.inference.ml.azure.com/score"
            sleep_endpoint = "https://sleep-model-deploy.eastus2.inference.ml.azure.com/score"

            # API keys
            cardio_api_key = "re93l0iRsfaIeGrg5xbP5U1UJaKkICKK"
            phy_api_key = "O0o4d9fG2RVx0eraXWuQd6J57IOWArva"
            sleep_api_key = "I1RJkiYPAcSikOfRIqXvU84IuHVYjoxW"

            # Headers for the HTTP requests
            headers = {
                "Content-Type": "application/json"
            }
            cardio_headers = {**headers, "Authorization": f"Bearer {cardio_api_key}"}
            phy_headers = {**headers, "Authorization": f"Bearer {phy_api_key}"}
            sleep_headers = {**headers, "Authorization": f"Bearer {sleep_api_key}"}

            # Convert DataFrames to JSON
            cardio_json = cardio.to_json(orient='records')
            phy_json = phy.to_json(orient='records')
            sleep_json = sleep.to_json(orient='records')

            # Send POST requests to the endpoints
            cardio_response = requests.post(cardio_endpoint, headers=cardio_headers, data=cardio_json)
            phy_response = requests.post(phy_endpoint, headers=phy_headers, data=phy_json)
            sleep_response = requests.post(sleep_endpoint, headers=sleep_headers, data=sleep_json)

            # Log the responses
            logging.info(f'Cardio model response: {cardio_response.json()}')
            logging.info(f'Phy model response: {phy_response.json()}')
            logging.info(f'Sleep model response: {sleep_response.json()}')

    except Exception as e:
        logging.error(f'Error consuming Kafka message: {str(e)}')
    finally:
        consumer.close()

def feature_engineering(wellness):
    """Applies feature engineering logic to the dataframe."""
    wellness['Deviation_heart_rate_rest'] = np.abs(wellness['resting_heart_rate'] - wellness['current_heart_rate_rest'])
    wellness.drop('current_heart_rate_rest', axis=1, inplace=True)
    return wellness

def encode_column(wellness):
    """Encodes and normalizes the dataframe."""
    wellness = feature_engineering(wellness)

    # Encoding categorical variables
    wellness = pd.get_dummies(wellness, columns=['heart_rate_condition'], drop_first=True)
    wellness[wellness.select_dtypes(include=['bool']).columns] = wellness.select_dtypes(include=['bool']).astype(int)

    # Normalizing numerical variables
    scaler = MinMaxScaler()
    num_features = wellness.select_dtypes(exclude="object").columns

    # Scale selected columns
    wellness[num_features] = scaler.fit_transform(wellness[num_features])
    wellness['health_management_score_scaled'] = scaler.fit_transform(wellness[['health_management_score']])

    return wellness

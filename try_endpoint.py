import requests
import pandas as pd

data = pd.DataFrame([0.8207214285714287,0.8307692307692308,0.39999999999999997,0.5,0.6400000000000001,0.14642240203654877,6.5])
phy_endpoint = "https://ml-workspace-lrtil.eastus2.inference.ml.azure.com/score"

phy_api_key = "2dyxVZtkf73fnysvp9sugoV3ty1GkE1K"

phy_headers = {
    "Content-Type": "application/json",  # Specify JSON input
    "Authorization": f"Bearer {phy_api_key}"  # Pass the API key for authentication
}

phy_json = data.to_json(orient='records')
phy_response = requests.post(phy_endpoint, headers=phy_headers, data=phy_json)

print(phy_response)
#print(phy_response.json())






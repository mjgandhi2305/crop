import streamlit as st
import json
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')
import requests
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "771c0299fe390c05d3072068a808c39d"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

# Load JSON data
def load_state_city_mapping(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

df = pd.read_csv('Crop_recommendation.csv')
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

# Splitting into train and test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators = 100, random_state=42)
RF.fit(Xtrain, Ytrain)

predicted_values = RF.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# print("RF's accuracy is: ", x*100)

# print(metrics.classification_report(Ytest,predicted_values))

def main():
    st.title("Crop Predictor")

    # Load state-city mapping from JSON file
    json_file = 'data.json'  # Replace with your file path
    state_city_mapping = load_state_city_mapping(json_file)

    st.header("Enter the required values:")

    # Input fields
    N = st.number_input("Nitrogen (N)", min_value=0, value=0, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, value=0, step=1)
    K = st.number_input("Potassium (K)", min_value=0, value=0, step=1)
    ph = st.number_input("pH", value=0.0, format="%.2f")
    rainfall = st.number_input("Rainfall (mm)", value=0.0, format="%.2f")
    # Select State and City outside the form for dynamic updates
    state = st.selectbox("Select State", options=list(state_city_mapping.keys()))

    if state:
        city = st.selectbox("Select City", options=state_city_mapping[state])

    # Submit button
    if st.button("Predict Crop"):
        if city:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = RF.predict(data)
            st.write(f"Temperature: {temperature}, Humidity: {humidity}")
            st.write(f"Predicted Crop: {prediction[0]}")
        else:
            st.write("Please select a city.")

if __name__ == "__main__":
    main()

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
from datetime import datetime
from flask import Flask, render_template, request, jsonify

import pickle
import joblib


loaded_decision_tree_reg = joblib.load('decision_tree_reg.pkl')

# Define the PCA and StandardScaler objects
n_components = 23  # Adjust the number of components as needed
pca = PCA(n_components=n_components)
scaler = StandardScaler()

# Load the Sentence Transformer model
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')  # Replace with your actual model name or path

# Define the PCA dimension and the trained SentenceTransformer model
n_components = 23  # Adjust the number of components as needed

# Define the numerical feature names used during training
numerical_features = [
    'Package Type_Standard', 'Package Type_Premium', 'Package Type_Luxury',
    'Travel_Month', 'Package Type_Budget', 'Package Type_Deluxe',
    'Hotel Ratings', 'Start City_New Delhi', 'Start City_Mumbai',
    'Travel_DayOfWeek', 'Travel_Year'
]

def predict_price(custom_input):
    text_columns = ['Package Name', 'Destination', 'Itinerary', 'Places Covered', 'Hotel Details', 'Airline', 'Sightseeing Places Covered', 'Cancellation Rules']
    # Apply PCA separately to each text embedding column
    pca = PCA(n_components=1)
    text_embeddings_pca = np.empty((1, n_components * len(text_columns)))

    for i, column in enumerate(text_columns):
        embeddings = [custom_input[column]]  # Assuming custom_input is a dictionary with column names as keys
        embeddings = model.encode(embeddings)
        embeddings_pca = pca.fit_transform(embeddings)
        text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca

    # Combine PCA-transformed text embeddings with numerical features
    X_numerical = np.array([custom_input[feature] for feature in numerical_features]).reshape(1, -1)
    X = np.hstack((text_embeddings_pca, X_numerical))

    # Scale the input data using the same scaler used during training
    X_scaled = scaler.fit_transform(X)

    # Make predictions using the trained Linear Regression model
    y_pred = loaded_decision_tree_reg.predict(X_scaled)

    return y_pred[0]

# Example usage:
custom_input = {
    'Package Name': 'Custom Package',
    'Destination': 'Custom Destination',
    'Itinerary': 'Custom Itinerary',
    'Places Covered': 'Custom Places',
    'Hotel Details': 'Custom Hotel Details',
    'Airline': 'Custom Airline',
    'Sightseeing Places Covered': 'Custom Sightseeing',
    'Cancellation Rules': 'Custom Cancellation Rules',
    'Package Type_Standard': 1,
    'Package Type_Premium': 0,
    'Package Type_Luxury': 0,
    'Travel_Month': 5,
    'Package Type_Budget': 0,
    'Package Type_Deluxe': 0,
    'Hotel Ratings': 4.5,
    'Start City_New Delhi': 0,
    'Start City_Mumbai': 1,
    'Travel_DayOfWeek': 2,
    'Travel_Year': 2023
}

predicted_price = predict_price(custom_input)
print(f"Predicted Price: {predicted_price}")

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def predict():
    return """

    <!DOCTYPE html>
<html>
<head>
    <title>Travel Package Prediction</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h1 {
            color: #007BFF;
            font-size: 36px;
            margin-bottom: 20px;
        }

        form {
            text-align: left;
        }

        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 15px;
            margin: 15px 0;
            border: none;
            border-bottom: 2px solid #007BFF;
            font-size: 18px;
            background-color: transparent;
            color: #333;
            transition: border-bottom 0.3s ease;
        }

        input[type="text"]:focus,
        input[type="number"]:focus {
            border-bottom: 2px solid #0056b3;
            outline: none;
        }

        input[type="checkbox"],
        input[type="radio"] {
            margin-right: 10px;
        }

        input[type="submit"] {
            background-color: #007BFF;
            color: #fff;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 20px;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
        }

        p#prediction {
            margin-top: 20px;
            font-size: 24px;
            color: #007BFF;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Travel Package Prediction</h1>
        <form action="/predict" method="POST">
            <label for="Package Name">Package Name:</label>
            <input type="text" name="Package Name" placeholder="Custom Package" value="A Grand Week in North East - Lachung Special">

            <label for="Destination">Destination:</label>
            <input type="text" name="Destination" placeholder="Custom Destination" value="Gangtok|Lachung|Gangtok|Darjeeling">

            <label for="Itinerary">Itinerary:</label>
            <input type="text" name="Itinerary" placeholder="Custom Itinerary" value="2N Gangtok . 2N Lachung . 1N Gangtok . 2N Darjeeling">

            <label for="Places Covered">Places Covered:</label>
            <input type="text" name="Places Covered" placeholder="Custom Places" value="Gangtok|Lachung|Gangtok|Darjeeling">

            <label for="Hotel Details">Hotel Details:</label>
            <input type="text" name="Hotel Details" placeholder="Custom Hotel Details" value="Lemon Tree Hotel  Gangtok:4.2|Summit Alpine Resort, Lachung- MMT Holidays Special:4.2|Lemon Tree Hotel  Gangtok:4.4|Ramada by Wyndham Darjeeling Gandhi Road:">

            <label for="Airline">Airline:</label>
            <input type="text" name="Airline" placeholder="Custom Airline" value="Air India|IndiGo">

            <label for="Sightseeing Places Covered">Sightseeing Places Covered:</label>
            <input type="text" name="Sightseeing Places Covered" placeholder="Custom Sightseeing" value="Baba Mandir | MG Road - Walk | Visit to Lake Tsomgo with Yak Safari | Snacks at Changu lake (Winter Special) | Yumthang Valley | Hot Spring">

            <label for="Cancellation Rules">Cancellation Rules:</label>
            <input type="text" name="Cancellation Rules" placeholder="Custom Cancellation Rules" value="Not Available">


            <label for="Package Type_Standard">Package Type_Standard:</label>
            <input type="checkbox" name="Package Type_Standard" value="1" checked>

            <label for="Package Type_Premium">Package Type_Premium:</label>
            <input type="checkbox" name="Package Type_Premium" value="1">

            <label for="Package Type_Luxury">Package Type_Luxury:</label>
            <input type="checkbox" name="Package Type_Luxury" value="1">

            <label for="Travel_Month">Travel Month:</label>
            <input type="number" name="Travel_Month" min="1" max="12" placeholder="Travel Month" value="5">

            <label for="Package Type_Budget">Package Type_Budget:</label>
            <input type="checkbox" name="Package Type_Budget" value="1">

            <label for="Package Type_Deluxe">Package Type_Deluxe:</label>
            <input type="checkbox" name="Package Type_Deluxe" value="1">

            <label for="Hotel Ratings">Hotel Ratings:</label>
            <input type="number" name="Hotel Ratings" placeholder="Hotel Ratings" value="4.5">

            <label for="Start City_New Delhi">Start City_New Delhi:</label>
            <input type="checkbox" name="Start City_New Delhi" value="1">

            <label for="Start City_Mumbai">Start City_Mumbai:</label>
            <input type="checkbox" name="Start City_Mumbai" value="1">

            <label for="Travel_DayOfWeek">Travel Day of Week:</label>
            <input type="number" name="Travel_DayOfWeek" min="0" max="6" placeholder="Travel Day of Week" value="2">

            <label for="Travel_Year">Travel Year:</label>
            <input type="number" name="Travel_Year" min="2023" max="2123" placeholder="Travel Year" value="2023">

            <input type="submit" value="Predict">
        </form>
        <p id="prediction"></p>
    </div>
</body>
</html>


    """


@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        package_name = request.form.get('Package Name')
        destination = request.form.get('Destination')
        itinerary = request.form.get('Itinerary')
        places_covered = request.form.get('Places Covered')
        hotel_details = request.form.get('Hotel Details')
        airline = request.form.get('Airline')
        sightseeing_places = request.form.get('Sightseeing Places Covered')
        cancellation_rules = request.form.get('Cancellation Rules')
        package_standard = int(request.form.get('Package Type_Standard', 0))
        package_premium = int(request.form.get('Package Type_Premium', 0))
        package_luxury = int(request.form.get('Package Type_Luxury', 0))
        travel_month = int(request.form.get('Travel_Month'))
        package_budget = int(request.form.get('Package Type_Budget', 0))
        package_deluxe = int(request.form.get('Package Type_Deluxe', 0))
        hotel_ratings = float(request.form.get('Hotel Ratings'))
        start_city_delhi = int(request.form.get('Start City_New Delhi', 0))
        start_city_mumbai = int(request.form.get('Start City_Mumbai', 0))
        travel_day_of_week = int(request.form.get('Travel_DayOfWeek'))
        travel_year = int(request.form.get('Travel_Year'))

        # Create a dictionary to store the input data
        data = {
            'Package Name': package_name,
            'Destination': destination,
            'Itinerary': itinerary,
            'Places Covered': places_covered,
            'Hotel Details': hotel_details,
            'Airline': airline,
            'Sightseeing Places Covered': sightseeing_places,
            'Cancellation Rules': cancellation_rules,
            'Package Type_Standard': package_standard,
            'Package Type_Premium': package_premium,
            'Package Type_Luxury': package_luxury,
            'Travel_Month': travel_month,
            'Package Type_Budget': package_budget,
            'Package Type_Deluxe': package_deluxe,
            'Hotel Ratings': hotel_ratings,
            'Start City_New Delhi': start_city_delhi,
            'Start City_Mumbai': start_city_mumbai,
            'Travel_DayOfWeek': travel_day_of_week,
            'Travel_Year': travel_year
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(data)

        return jsonify({'prediction': round(prediction, 2)})


if __name__ == "__main__":
    app.run()
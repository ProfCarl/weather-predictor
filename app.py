from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np
import joblib
import os

# Load the trained model
# model = joblib.load('wind_model.pkl')  # You will train and save this later

# Initialize Flask app
app = Flask(__name__)
CORS(app)

#initilise models
rain_models = []
humidity_models = []
tmax_models = []
wind_models = []
tmin_models = []

PKL_DIR = os.path.join(os.path.dirname(__file__), 'pkl')

for i in range(1, 32):
    rain_models.append(joblib.load(os.path.join(PKL_DIR,'rain_models',f'rain_model{i}.pkl')))
    humidity_models.append(joblib.load(os.path.join(PKL_DIR,'humidity_models',f'humidity_model{i}.pkl')))
    wind_models.append(joblib.load(os.path.join(PKL_DIR,'wind_models',f'wind_model{i}.pkl')))
    tmax_models.append(joblib.load(os.path.join(PKL_DIR,'tmax_models',f'tmax_model{i}.pkl')))
    tmin_models.append(joblib.load(os.path.join(PKL_DIR,'tmin_models',f'tmin_model{i}.pkl')))

@app.route('/')
def home():
    return 'Wind Speed Prediction API is live!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        required_fields = ['Geogr1', 'Geogr2', 'Year', 'Month', 'Day']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return make_response(jsonify({'error': f'Missing required field: {field}'}), 400)

        try:
            geogr1 = float(data['Geogr1'])
            geogr2 = float(data['Geogr2'])
            year = int(data['Year'])
            month = int(data['Month'])
            day = int(data['Day'])
        except ValueError:
            return make_response(jsonify({'error': 'Invalid data types. Geogr1 and Geogr2 must be floats; Year, Month, Day must be integers.'}), 400)

        if not (1 <= day <= 31):
            return make_response(jsonify({'error': 'Day must be between 1 and 31.'}), 400)

        index = day - 1
        input_features = np.array([[geogr1, geogr2, year, month]])

        try:
            wind_pred = wind_models[index].predict(input_features)[0]
            rain_pred = rain_models[index].predict(input_features)[0]
            humidity_pred = humidity_models[index].predict(input_features)[0]
            tmax_pred = tmax_models[index].predict(input_features)[0]
            tmin_pred = tmin_models[index].predict(input_features)[0]
        except IndexError:
            return make_response(jsonify({'error': f'Model for day {day} not found.'}), 500)
        except Exception as e:
            return make_response(jsonify({'error': f'Prediction error: {str(e)}'}), 500)

        return jsonify({
            'predicted_wind_speed': float(wind_pred),
            'predicted_rain': float(rain_pred),
            'predicted_humidity': float(humidity_pred),
            'predicted_tmax': float(tmax_pred),
            'predicted_tmin': float(tmin_pred)
        })

    except Exception as e:
        return make_response(jsonify({'error': f'Unexpected server error: {str(e)}'}), 500)


if __name__ == '__main__':
    app.run(debug=True)
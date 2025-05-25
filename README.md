# 🌦️ AI-Powered Weather Forecast API for Ghana 🇬🇭

A Flask-based REST API for intelligent weather forecasting using machine learning models trained on historical meteorological data from Ghanaian weather stations.

## 🚀 Features

* 🔍 **Daily Weather Forecasts**
  Predicts rainfall, wind speed, humidity, and temperature (Tmax & Tmin) based on location and date.

* 🤖 **AI-Driven Models**
  Machine learning models trained on localized Ghanaian weather data using `scikit-learn` and serialized with `joblib`.

* 🧠 **Per-Day Model Specialization**
  Separate models trained for each day of the month to improve forecast accuracy.

* 🌐 **API-Ready**
  Clean Flask API interface with support for JSON-based requests and CORS-enabled frontend access.

---

## 📦 Project Structure

```
project/
│
├── app.py               # Flask server with prediction API
├── train_model.py       # Script for training all ML models
├── requirements.txt     # Python dependencies
│
├── csv/                 # Raw weather data
│   ├── Rainfall.csv
│   ├── RH1500.csv
│   ├── RH0600.csv
│   ├── Tmax.csv
│   ├── Tmin.csv
│   └── Wind Speed.csv
│
└── pkl/                 # Trained ML models (joblib format)
    ├── rain_models/
    ├── wind_models/
    ├── humidity_models/
    ├── tmax_models/
    └── tmin_models/
```

---

## 📡 API Usage

### `POST /predict`

**Request Body (JSON):**

```json
{
  "Year": 2025,
  "Month": 5,
  "Day": 21,
  "Geogr1": 6.5244,
  "Geogr2": -0.2261
}
```

**Response (JSON):**

```json
{
  "predicted_rain": 3.21,
  "predicted_wind_speed": 1.45,
  "predicted_humidity": 87.6,
  "predicted_tmax": 32.4,
  "predicted_tmin": 24.1
}
```

---

## 🛠 Setup Instructions

1. **Clone the repo:**

   ```bash
   git clone https://github.com/yourusername/weather-ai-ghana.git
   cd weather-ai-ghana
   ```

2. **Create a virtual environment & install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Train models (optional if you want to regenerate):**

   ```bash
   python train_model.py
   ```

4. **Start the Flask server:**

   ```bash
   flask run
   ```

---

## ⚙️ Requirements

* Python 3.8+
* Flask
* NumPy
* scikit-learn
* joblib
* Flask-CORS
* pandas

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📊 Model Training

The `train_model.py` script automatically loads weather datasets from the `csv/` folder, processes them, and saves per-day models in the `pkl/` directory.

You can retrain models with new or updated weather datasets by running:

```bash
python train_model.py
```

---

## 📬 Contact

Built by Prof0 Technologies
📧 dansocarl8@gmail.com
📍 Ghana

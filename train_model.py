import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os

print("Starting python...")

CSV_DIR = os.path.join(os.path.dirname(__file__), 'csv')
PKL_DIR = os.path.join(os.path.dirname(__file__), 'pkl')

# Load datasets
wind_data = pd.read_csv(os.path.join(CSV_DIR,'csv\Wind Speed.csv'))
rain_data = pd.read_csv(os.path.join(CSV_DIR,'csv\Rainfall.csv'))
humidity_data = pd.read_csv(os.path.join(CSV_DIR,'csv\RH1500.csv'))
tmax_data = pd.read_csv(os.path.join(CSV_DIR,'csv\Tmax.csv'))
tmin_data = pd.read_csv(os.path.join(CSV_DIR,'csv\Tmin.csv'))

#Clean datasets
wind_data = wind_data.dropna()
rain_data = rain_data.dropna()
humidity_data = humidity_data.dropna()
tmax_data = tmax_data.dropna()
tmin_data = tmin_data.dropna()

# Separate features and target
i = 1
my_data_arr = []
my_test_train = []
my_model_arr = []

class MyData:
  def __init__(self, wind_x, wind_y, rain_x, rain_y, humidity_x, humidity_y, tmax_x, tmax_y, tmin_x, tmin_y):
    self.wind_x = wind_x
    self.wind_y = wind_y
    self.rain_x = rain_x
    self.rain_y = rain_y
    self.humidity_x = humidity_x
    self.humidity_y = humidity_y
    self.tmax_x = tmax_x
    self.tmax_y = tmax_y
    self.tmin_x = tmin_x
    self.tmin_y = tmin_y

class MyTestTrain:
  def __init__(self, wind_x_train, wind_x_test, wind_y_train, wind_y_test, rain_x_train, rain_x_test, rain_y_train, rain_y_test, 
               humidity_x_train, humidity_x_test, humidity_y_train, humidity_y_test, tmax_x_train, tmax_x_test, tmax_y_train, tmax_y_test,
               tmin_x_train, tmin_x_test, tmin_y_train, tmin_y_test
               ):
    self.wind_x_train = wind_x_train
    self.wind_x_test = wind_x_test
    self.wind_y_train = wind_y_train
    self.wind_y_test = wind_y_test
    self.rain_x_train = rain_x_train
    self.rain_x_test = rain_x_test
    self.rain_y_train = rain_y_train
    self.rain_y_test = rain_y_test
    self.humidity_x_train = humidity_x_train
    self.humidity_x_test = humidity_x_test
    self.humidity_y_train = humidity_y_train
    self.humidity_y_test = humidity_y_test
    self.tmax_x_train = tmax_x_train
    self.tmax_x_test = tmax_x_test
    self.tmax_y_train = tmax_y_train
    self.tmax_y_test = tmax_y_test
    self.tmin_x_train = tmin_x_train
    self.tmin_x_test = tmin_x_test
    self.tmin_y_train = tmin_y_train
    self.tmin_y_test = tmin_y_test

class MyModelClass:
   def __init__(self, wind_model, rain_model, humidity_model, tmax_model, tmin_model):
      self.wind_model = wind_model
      self.rain_model = rain_model
      self.humidity_model = humidity_model
      self.tmax_model = tmax_model
      self.tmin_model = tmin_model

while i <=31:
    wind_x = wind_data[['Geogr1', 'Geogr2', 'Year', 'Month']]
    wind_y = wind_data[str(i)]
    rain_x = rain_data[['Geogr1', 'Geogr2', 'Year', 'Month']]
    rain_y = rain_data[str(i)]
    humidity_x = humidity_data[['Geogr1', 'Geogr2', 'Year', 'Month']]
    humidity_y = humidity_data[str(i)]
    tmax_x = tmax_data[['Geogr1', 'Geogr2', 'Year', 'Month']]
    tmax_y = tmax_data[str(i)]
    tmin_x = tmax_data[['Geogr1', 'Geogr2', 'Year', 'Month']]
    tmin_y = tmin_data[str(i)]
    my_data_arr.append(MyData(wind_x, wind_y, rain_x, rain_y, humidity_x, humidity_y, tmax_x, tmax_y, tmin_x, tmin_y))
    i += 1

# Split into train and test sets (70% train, 30% test)

i = 0
while i <= 30:
    min_len = min(len(my_data_arr[i].tmax_x), len(my_data_arr[i].tmax_y), len(my_data_arr[i].tmin_x), len(my_data_arr[i].tmin_y),
                  len(my_data_arr[i].rain_x), len(my_data_arr[i].rain_y), len(my_data_arr[i].wind_x), len(my_data_arr[i].wind_y),
                  len(my_data_arr[i].humidity_x), len(my_data_arr[i].humidity_y))
    my_data_arr[i].tmax_x = my_data_arr[i].tmax_x[:min_len]
    my_data_arr[i].tmax_y = my_data_arr[i].tmax_y[:min_len]
    my_data_arr[i].rain_x = my_data_arr[i].rain_x[:min_len]
    my_data_arr[i].rain_y = my_data_arr[i].rain_y[:min_len]
    my_data_arr[i].humidity_x = my_data_arr[i].humidity_x[:min_len]
    my_data_arr[i].humidity_y = my_data_arr[i].humidity_y[:min_len]
    my_data_arr[i].wind_x = my_data_arr[i].wind_x[:min_len]
    my_data_arr[i].wind_y = my_data_arr[i].wind_y[:min_len]
    my_data_arr[i].tmin_x = my_data_arr[i].tmin_x[:min_len]
    my_data_arr[i].tmin_y = my_data_arr[i].tmin_y[:min_len]

    wind_x_train, wind_x_test, wind_y_train, wind_y_test = train_test_split(my_data_arr[i].wind_x, my_data_arr[i].wind_y, test_size=0.3, random_state=42)
    rain_x_train, rain_x_test, rain_y_train, rain_y_test = train_test_split(my_data_arr[i].rain_x, my_data_arr[i].rain_y, test_size=0.3, random_state=42)
    humidity_x_train, humidity_x_test, humidity_y_train, humidity_y_test = train_test_split(my_data_arr[i].humidity_x, my_data_arr[i].humidity_y, test_size=0.3, random_state=42)
    tmax_x_train, tmax_x_test, tmax_y_train, tmax_y_test = train_test_split(my_data_arr[i].tmax_x, my_data_arr[i].tmax_y, test_size=0.3, random_state=42)
    tmin_x_train, tmin_x_test, tmin_y_train, tmin_y_test = train_test_split(my_data_arr[i].tmin_x, my_data_arr[i].tmin_y, test_size=0.3, random_state=42)
    #print("Appending test/train for day index " + str(i))
    my_test_train.append(MyTestTrain(wind_x_train, wind_x_test, wind_y_train, wind_y_test, rain_x_train, rain_x_test, rain_y_train, rain_y_test, 
               humidity_x_train, humidity_x_test, humidity_y_train, humidity_y_test, tmax_x_train, tmax_x_test, tmax_y_train, tmax_y_test,
               tmin_x_train, tmin_x_test, tmin_y_train, tmin_y_test))
    i += 1

print(len(my_test_train))
# Train a regression model
for i in range(len(my_test_train)):
    wind_model = DecisionTreeRegressor()
    wind_model.fit(my_test_train[i].wind_x_train, my_test_train[i].wind_y_train)
    rain_model = DecisionTreeRegressor()
    rain_model.fit(my_test_train[i].rain_x_train, my_test_train[i].rain_y_train)
    humidity_model = DecisionTreeRegressor()
    humidity_model.fit(my_test_train[i].humidity_x_train, my_test_train[i].humidity_y_train)
    tmax_model = DecisionTreeRegressor()
    tmax_model.fit(my_test_train[i].tmax_x_train, my_test_train[i].tmax_y_train)
    tmin_model = DecisionTreeRegressor()
    tmin_model.fit(my_test_train[i].tmin_x_train, my_test_train[i].tmin_y_train)
    my_model_arr.append(MyModelClass(wind_model, rain_model, humidity_model, tmax_model, tmin_model))
  

for i in range(len(my_test_train)):
   # Evaluate model
    y_pred = my_model_arr[i].wind_model.predict(my_test_train[i].wind_x_test)
    rmse = np.sqrt(mean_squared_error(my_test_train[i].wind_y_test, y_pred))
    r2 = r2_score(my_test_train[i].wind_y_test, y_pred)

    print(f'✅ Wind Model for Day 1' + str(i + 1) +' trained!\nRMSE: {rmse:.4f}\nR²: {r2:.4f}')

    # Save model to file
    joblib.dump(my_model_arr[i].wind_model, os.path.join(PKL_DIR,'wind_model' + str(i + 1) + '.pkl'))
    print('🧠 Model saved to ' + 'wind_model' + str(i + 1) + '.pkl')
    i += 1

for i in range(len(my_test_train)):
   # Evaluate model
    y_pred = my_model_arr[i].rain_model.predict(my_test_train[i].rain_x_test)
    rmse = np.sqrt(mean_squared_error(my_test_train[i].rain_y_test, y_pred))
    r2 = r2_score(my_test_train[i].rain_y_test, y_pred)

    print(f'✅ Rain Model for Day' + str(i + 1) +' trained!\nRMSE: {rmse:.4f}\nR²: {r2:.4f}')

    # Save model to file
    joblib.dump(my_model_arr[i].rain_model, os.path.join(PKL_DIR,'rain_model' + str(i + 1) + '.pkl'))
    print('🧠 Model saved to ' + 'rain_model' + str(i + 1) + '.pkl')
    i += 1

for i in range(len(my_test_train)):
   # Evaluate model
    y_pred = my_model_arr[i].humidity_model.predict(my_test_train[i].humidity_x_test)
    rmse = np.sqrt(mean_squared_error(my_test_train[i].humidity_y_test, y_pred))
    r2 = r2_score(my_test_train[i].humidity_y_test, y_pred)

    print(f'✅ Tmax Model for Day 1' + str(i + 1) +' trained!\nRMSE: {rmse:.4f}\nR²: {r2:.4f}')

    # Save model to file
    joblib.dump(my_model_arr[i].humidity_model, os.path.join(PKL_DIR,'humidity_model' + str(i + 1) + '.pkl'))
    print('🧠 Model saved to ' + 'humidity_model' + str(i + 1) + '.pkl')
    i += 1

for i in range(len(my_test_train)):
   # Evaluate model
    y_pred = my_model_arr[i].tmax_model.predict(my_test_train[i].tmax_x_test)
    rmse = np.sqrt(mean_squared_error(my_test_train[i].tmax_y_test, y_pred))
    r2 = r2_score(my_test_train[i].tmax_y_test, y_pred)

    print(f'✅ Tmax Model for Day 1' + str(i + 1) +' trained!\nRMSE: {rmse:.4f}\nR²: {r2:.4f}')

    # Save model to file
    joblib.dump(my_model_arr[i].tmax_model, os.path.join(PKL_DIR,'tmax_model' + str(i + 1) + '.pkl'))
    print('🧠 Model saved to ' + 'tmax_model' + str(i + 1) + '.pkl')
    i += 1

for i in range(len(my_test_train)):
   # Evaluate model
    y_pred = my_model_arr[i].tmin_model.predict(my_test_train[i].tmin_x_test)
    rmse = np.sqrt(mean_squared_error(my_test_train[i].tmin_y_test, y_pred))
    r2 = r2_score(my_test_train[i].tmin_y_test, y_pred)

    print(f'✅ Tmin Model for Day 1' + str(i + 1) +' trained!\nRMSE: {rmse:.4f}\nR²: {r2:.4f}')

    # Save model to file
    joblib.dump(my_model_arr[i].tmin_model, os.path.join(PKL_DIR,'tmin_model' + str(i + 1) + '.pkl'))
    print('🧠 Model saved to ' + 'tmin_model' + str(i + 1) + '.pkl')
    i += 1

print("...Ending python")

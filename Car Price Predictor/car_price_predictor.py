import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and preprocess the data
data = pd.read_csv('raw_car_data.csv')
data = data.dropna()
data = data[data['Price'] != 'Ask For Price']
data['Price'] = data['Price'].str.replace(',', '').astype(int)
data['kms_driven'] = data['kms_driven'].str.replace(',', '').str.replace(' ', '').str.replace('kms', '').astype(int)
data = data[data['fuel_type'] != 'LPG']
label_encoder = preprocessing.LabelEncoder()
data['fuel_type'] = label_encoder.fit_transform(data['fuel_type'])
data['year'] = data['year'].str.replace(',', '').str.replace(' ', '').astype(int)
data_company = pd.get_dummies(data['company'], prefix='company', dtype=int)
data = pd.concat([data, data_company], axis=1)
data.drop(['name','company'], axis=1, inplace=True)
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=900, min_samples_split=10, min_samples_leaf=1, max_features='log2', max_depth=20)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'model/car_price_model.pkl')

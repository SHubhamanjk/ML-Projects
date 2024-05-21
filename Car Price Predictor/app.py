from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/car_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']
        company_name = request.form['company_name']

        # Process input
        columns = ['year', 'kms_driven', 'fuel_type', 'company_Audi', 'company_BMW', 'company_Chevrolet',
                   'company_Datsun', 'company_Fiat', 'company_Force', 'company_Ford', 'company_Hindustan',
                   'company_Honda', 'company_Hyundai', 'company_Jaguar', 'company_Jeep', 'company_Land',
                   'company_Mahindra', 'company_Maruti', 'company_Mercedes', 'company_Mini', 'company_Mitsubishi',
                   'company_Nissan', 'company_Renault', 'company_Skoda', 'company_Tata', 'company_Toyota',
                   'company_Volkswagen', 'company_Volvo']
        company_columns = {col.replace('company_', '').lower(): col for col in columns if col.startswith('company_')}
        input_data = np.zeros(len(columns))
        input_data[0] = year
        input_data[1] = kms_driven
        fuel_type = fuel_type.lower()
        if fuel_type == 'petrol':
            input_data[2] = 1
        elif fuel_type == 'diesel':
            input_data[2] = 0
        else:
            return "Fuel type must be either 'petrol' or 'diesel'"
        company_name = company_name.lower()
        if company_name in company_columns:
            company_col_index = columns.index(company_columns[company_name])
            input_data[company_col_index] = 1
        else:
            return f"Company '{company_name}' not found in the training data."

        # Predict the price
        predicted_price = model.predict([input_data])[0]

        return render_template('index.html', prediction_text=f'{predicted_price}')

if __name__ == "__main__":
    app.run(debug=True)

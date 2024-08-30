from flask import Flask, request, render_template
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

# Load the trained model and preprocessor
with open('gradient_boosting_model.pkl', 'rb') as file:
    gb_model = pickle.load(file)

preprocessor = joblib.load('preprocessor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collect input data
        data = {
            "Date": [request.form['Date']],
            "Area Name": [request.form['Area Name']],
            "Road/Intersection Name": [request.form['Road/Intersection Name']],
            "Traffic Volume": [float(request.form['Traffic Volume'])],
            "Average Speed": [float(request.form['Average Speed'])],
            "Travel Time Index": [float(request.form['Travel Time Index'])],
            "Road Capacity Utilization": [float(request.form['Road Capacity Utilization'])],
            "Incident Reports": [int(request.form['Incident Reports'])],
            "Environmental Impact": [float(request.form['Environmental Impact'])],
            "Public Transport Usage": [float(request.form['Public Transport Usage'])],
            "Traffic Signal Compliance": [float(request.form['Traffic Signal Compliance'])],
            "Parking Usage": [float(request.form['Parking Usage'])],
            "Pedestrian and Cyclist Count": [int(request.form['Pedestrian and Cyclist Count'])],
            "Weather Conditions": [request.form['Weather Conditions']],
            "Roadwork and Construction Activity": [request.form['Roadwork and Construction Activity']]
        }

        # Convert data to DataFrame
        new_data_df = pd.DataFrame(data)

        # Preprocess and predict
        X_new = preprocessor.transform(new_data_df)
        prediction = gb_model.predict(X_new)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

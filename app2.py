from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import plotly.express as px
import pandas as pd
# Load the saved SVM model
model = joblib.load('svm_diabetes_model.pkl')

# Initialize Flask app
app = Flask(__name__)
# Global variable to store user data for visualization
user_data = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global user_data  # Ensure we modify the global user_data list
    try:
        # Get input data from the form
        data = request.form
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetes_pedigree_function']),
            float(data['age'])
        ]

        # Save the user's input data for visualization
        user_data.append(features)

        # Reshape data for prediction
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)

        # Return result
        if prediction[0] == 1:
            result = "The patient is likely to have diabetes."
        else:
            result = "The patient is unlikely to have diabetes."

        return render_template('result.html', result=result, features=features)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/visualizations')
def visualizations():
    global user_data
    if not user_data:
        return render_template('visualizations.html', graph_html="<p>No data available for visualization yet.</p>")

    # Define column names
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Create a DataFrame from user_data
    df = pd.DataFrame(user_data, columns=columns)

    # Generate a bar chart for visualization
    fig = px.bar(
        df.mean().reset_index(), 
        x='index', 
        y=0, 
        labels={'index': 'Features', 0: 'Average Value'},
        title="Average Input Values"
    )
    graph_html = fig.to_html(full_html=False)

    return render_template('visualizations.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)

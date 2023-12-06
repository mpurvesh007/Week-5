# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:12:03 2023

@author: Purvesh
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
from flask import jsonify


app = Flask(__name__)
model = pickle.load(open('svm_model.pkl', 'rb'))

# Define the mapping between numerical labels and species names
species_names = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

@app.route('/')
def home():
    return render_template('iris_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Extracting features from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Making a prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Map the numerical prediction to species name
    predicted_species = species_names[prediction[0]]

    # Display the prediction on the HTML page
    output = f'Predicted Iris Species: {predicted_species}'
    return render_template('iris_index.html', prediction_text=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port)


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:03:13 2023

@author: Purvesh
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))


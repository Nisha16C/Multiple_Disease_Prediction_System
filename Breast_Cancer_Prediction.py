import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets  # Corrected import statement
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load the breast cancer data
def load_data():
    breast_cancer_dataset = datasets.load_breast_cancer()  # Corrected function call
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    data_frame['label'] = breast_cancer_dataset.target
    return data_frame

# Function to train the model
def train_model(data_frame):
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model

# Function to predict breast cancer
def predict_breast_cancer(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit app
def main():
    st.title("Breast Cancer Prediction App")

    # Load data and train the model
    data_frame = load_data()
    model = train_model(data_frame)

    # Show the input form for prediction
    st.subheader("Enter the following features for breast cancer prediction:")
    input_data = []
    for feature_name in data_frame.columns[:-1]:  # Exclude the last column ('label')
        feature_value = st.text_input(feature_name)
        input_data.append(float(feature_value) if feature_value else 0.0)

    # Make prediction
    if st.button("Predict"):
        prediction = predict_breast_cancer(model, input_data)

        if prediction == 0:
            st.write("The Breast cancer is Malignant")
        else:
            st.write("The Breast Cancer is Benign")

if __name__ == "__main__":
    main()

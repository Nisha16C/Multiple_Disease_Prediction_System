import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Function to load the breast cancer data
def load_data():
    breast_cancer_dataset = datasets.load_breast_cancer()
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

# Loading the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Streamlit app
def main():
    st.sidebar.title("✅ Multiple Disease Prediction System")
    selected = st.sidebar.selectbox(
        "",
        ['🩸 Diabetes Prediction', '❤️ Heart Disease Prediction', '🧠 Parkinsons Prediction', '🎗️ Breast Cancer Prediction']
    )

    if selected == '🩸 Diabetes Prediction':
        # Diabetes Prediction Page
        st.title('🩸 Diabetes Prediction using ML')
        # ... (existing code for diabetes prediction)

        # getting the input data from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')

        with col2:
            Glucose = st.text_input('Glucose Level')

        with col3:
            BloodPressure = st.text_input('Blood Pressure value')

        with col1:
            SkinThickness = st.text_input('Skin Thickness value')

        with col2:
            Insulin = st.text_input('Insulin Level')

        with col3:
            BMI = st.text_input('BMI value')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

        with col2:
            Age = st.text_input('Age of the Person')

        # code for Prediction
        diab_diagnosis = ''

        # creating a button for Prediction

        if st.button('Diabetes Test Result'):
            try:
                diab_prediction = diabetes_model.predict(
                    [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'

            except ValueError as e:
                diab_diagnosis = 'Error: Please Fill all blocks for the result.'

        st.success(diab_diagnosis)





    elif selected == '🎗️ Breast Cancer Prediction':
        # Breast Cancer Prediction Page
        st.title('Breast Cancer Prediction App')
        data_frame = load_data()
        model = train_model(data_frame)
        st.subheader("Enter the following features for breast cancer prediction:")
        input_data = []
        for feature_name in data_frame.columns[:-1]:
            feature_value = st.text_input(feature_name)
            input_data.append(float(feature_value) if feature_value else 0.0)

        if st.button("Predict"):
            prediction = predict_breast_cancer(model, input_data)
            if prediction == 0:
                st.write("The Breast cancer is Malignant")
            else:
                st.write("The Breast Cancer is Benign")

if __name__ == "__main__":
    main()

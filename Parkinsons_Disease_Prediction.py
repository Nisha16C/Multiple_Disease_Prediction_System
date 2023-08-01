# Importing the Dependencies
# Import the numpy library for numerical computing
import numpy as np
# Import the panda's library for data manipulation
import pandas as pd
# Import train_test_split function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import StandardScaler for standardizing the data
from sklearn.preprocessing import StandardScaler
# Import svm module for Support Vector Machines (SVM)
from sklearn import svm
# Import accuracy_score for evaluating model performance
from sklearn.metrics import accuracy_score

# This code is importing necessary libraries and modules for implementing and evaluating a Support Vector Machine (SVM)
# model for a given dataset.

# Data Collection & Analysis

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv(r"parkinsons.csv")

# printing the first 5 rows of the dataframe
df = parkinsons_data.head()
print(df)
# number of rows and columns in the dataframe
df = parkinsons_data.shape
print(df)

# getting more information about the dataset
df = parkinsons_data.info()
print(df)

# checking for missing values in each column
df = parkinsons_data.isnull().sum()
print(df)

# getting some statistical measures about the data
df = parkinsons_data.describe()
print(df)

# distribution of target Variable
df = parkinsons_data['status'].value_counts()
# 1 --> Parkinson's Positive
# 0 --> Healthy
print(df)

# Separating the features & Target
# The value of axis = 1 is used for columns, while the value of axis = 0 is used for rows
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']
print(X)
print(Y)

# Splitting the data to training data & Test data

# TEST_SIZE = the proportion of the dataset that should be used for testing.

# X_Train = This is the training set of the input data, which will be used to fit a model
# Y_Train = This is the training set of the target variable, which corresponds to X_train.
# X_Test  = This is the testing set of the input data, which will be used to evaluate the performance of the model.
# Y_Test  = This is also the testing set of the input data, which will be used to evaluate the performance of the model.
# 80% of data X & Y Train
# 20% of data X & Y Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X.shape, X_train.shape, X_test.shape)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

# Model Training
# Support Vector Machine Model


#this code initializes an SVM model with a linear kernel and trains it using the
# training data provided. The resulting trained model can then be used to make predictions on new data
model = svm.SVC(kernel='linear')
# training the SVM model with training data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy Score

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

# Building a Predictive System

input_data = (
    197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680,
    0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have Parkinsons Disease")

else:
    print("The Person has Parkinsons")

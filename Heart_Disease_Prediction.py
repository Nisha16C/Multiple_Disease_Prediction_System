# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart.csv')
# print first 5 rows of the dataset
df = heart_data.head()
print(df)
# print last 5 rows of the dataset
df = heart_data.tail()
print(df)

# number of rows and columns in the dataset
df = heart_data.shape
print(df)

# getting some info about the data
heart_data.info()


# checking for missing values
df = heart_data.isnull().sum()
print(df)

# statistical measures about the data
df = heart_data.describe()
print(df)

# checking the distribution of Target Variable
df = heart_data['target'].value_counts()
# 1 --> Defective Heart
# 0 --> Healthy Heart
print(df)

# Splitting the Features and Target

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)

# Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training
# Logistic Regression

model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)

# Building a Predictive System

input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 1:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')

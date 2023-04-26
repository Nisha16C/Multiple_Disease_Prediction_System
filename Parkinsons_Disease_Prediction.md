# Parkinson's Disease Detection using Support Vector Machines (SVM)

This code uses a Support Vector Machine (SVM) model to detect Parkinson's Disease using a given dataset. The code performs the following tasks:

- Imports necessary libraries and modules for implementing and evaluating a Support Vector Machine (SVM) model.
- Loads the data from a CSV file to a Pandas DataFrame.
- Prints the first 5 rows of the DataFrame, the number of rows and columns in the DataFrame, and some statistical measures about the data.
- Checks for missing values in each column of the DataFrame.
- Separates the features and target from the dataset.
- Splits the dataset into training data and test data.
- Standardizes the training and test data.
- Trains the SVM model using the training data.
- Evaluates the performance of the model using the test data.
- Builds a predictive system that takes in new data and predicts if the person has Parkinson's Disease.

## Dependencies
- numpy
- pandas
- scikit-learn (sklearn)

## Dataset
The dataset used in this code is the "Parkinsons" dataset, which is available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons). This dataset contains biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD).

## Running the code
To run this code, you will need to have the dependencies installed. You can install the dependencies by running the following command:

```
pip install numpy pandas scikit-learn
```

You will also need to have the "parkinsons.csv" file in the same directory as the code. You can download the file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons).

Once you have the dependencies installed and the dataset file in the correct location, you can run the code in your preferred Python environment. You can run the code using the following command:

```
python Parkinsons_Disease_Prediction.py
```
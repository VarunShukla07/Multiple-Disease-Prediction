import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Processing"""

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/heart.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

"""1 --> Defective Heart

0 --> Healthy Heart

Splitting the Features and Target
"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

"""Splitting the Data into Training data & Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training"""

from sklearn import svm
model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

"""Building a Predictive System"""

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the Random Forest model
model.fit(X_train, Y_train)

# Accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data: ', training_data_accuracy)

# Accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data: ', test_data_accuracy)

# Input data for heart disease prediction
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make the prediction
prediction = model.predict(input_data_reshaped)
print(prediction)

# Output the result
if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Create the Linear Regression model
model = LinearRegression()

# Training the Linear Regression model
model.fit(X_train, Y_train)

# Mean Squared Error on training data
X_train_prediction = model.predict(X_train)
training_data_mse = mean_squared_error(Y_train, X_train_prediction)
print('Mean Squared Error of training data: ', training_data_mse)

# Mean Squared Error on test data
X_test_prediction = model.predict(X_test)
test_data_mse = mean_squared_error(Y_test, X_test_prediction)
print('Mean Squared Error of test data: ', test_data_mse)

# Input data for heart disease prediction
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make the prediction
prediction = model.predict(input_data_reshaped)
print('Predicted value:', prediction[0])

# Since Linear Regression outputs continuous values, adjust the threshold (e.g. >= 0.5)
if prediction[0] >= 0.5:
    print('The Person has Heart Disease')
else:
    print('The Person does not have Heart Disease')
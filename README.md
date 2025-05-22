# Titanic Survival Prediction: Neural Network Model
This project contains a machine learning model developed using neural networks to predict the survival probabilities of passengers in the famous Titanic disaster.

## Project Overview
The sinking of the RMS Titanic on April 15, 1912, resulted in the loss of more than 1,500 lives. This project aims to predict which passengers survived using a neural network trained on structured passenger data.

The model is built with Keras (TensorFlow backend) and includes data preprocessing, model training, evaluation, and generating submission-ready predictions.

## Dataset
Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

Main features used:

| Feature       | Description                              |
|---------------|------------------------------------------|
| PassengerId   | Unique passenger ID                      |
| Survived      | Target variable (0 = No, 1 = Yes)        |
| Pclass        | Passenger class (1st, 2nd, 3rd)          |
| Name          | Name of the passenger                    |
| Sex           | Gender                                   |
| Age           | Age in years                             |
| SibSp         | # of siblings/spouses aboard             |
| Parch         | # of parents/children aboard             |
| Ticket        | Ticket number                            |
| Fare          | Ticket fare                              |
| Cabin         | Cabin number                             |
| Embarked      | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Model Features

- Fully connected neural network with:
  - Input: processed numerical & categorical features
  - Hidden layers: ReLU activations
  - Output: sigmoid activation for binary classification
- StandardScaler for feature normalization
- Model evaluation using accuracy and validation split


## Technologies Used
- Python 3.12
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
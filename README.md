# Titanic Survival Prediction: Neural Network Model
This project contains a machine learning model developed using neural networks to predict the survival probabilities of passengers in the famous Titanic disaster.

## Project Overview
The sinking of the RMS Titanic on April 15, 1912, resulted in the loss of more than 1,500 lives. This project aims to predict which passengers survived using a neural network trained on structured passenger data.

The model is built with Keras (TensorFlow backend) and includes data preprocessing, model training, evaluation, and generating submission-ready predictions.

The solution includes:
- Data preprocessing and feature engineering
- Neural network model design with Keras
- Model training and validation
- Prediction and CSV output for submission

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


## Model Architecture
The model is a **fully connected feedforward neural network** implemented using Keras with the TensorFlow backend.

- **Input**: Scaled numeric and encoded categorical features
- **Hidden layers**: ReLU activations, optional dropout
- **Output layer**: Sigmoid activation for binary classification
- **Loss function**: Binary crossentropy
- **Optimizer**: Adam
- **Metric**: Accuracy

## Main Steps
#### Data Loading
- Load the Titanic dataset, explore basic properties, and inspect missing values.

#### Feature Engineering
- FamilySize: Combined SibSp (siblings/spouses) and Parch (parents/children).
- IsAlone: Flag passengers traveling alone.
- Title: Extracted title from names (Mr, Mrs, Miss, etc.) for additional demographic insight.

#### Data Preprocessing
- Fill missing Age and Fare values.
- Apply log1p transformation to reduce skewness in Fare.
- Encode categorical variables.

#### Model Building

Sequential neural network with:
- Input layer (16 neurons, ReLU, dropout 0.3)
- Hidden layer (8 neurons, ReLU, dropout 0.3)
- Output layer (1 neuron, sigmoid)

#### Training and Evaluation
- Train on 70% of data, validate on 30%.
- Monitor loss and accuracy metrics over 60 epochs.

## Technologies Used
- Python 3.12
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Links
- [Kaggle Notebook](https://www.kaggle.com/code/bahadirbor/titanic-survival-prediction-neural-network-model)
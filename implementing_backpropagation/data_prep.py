import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

# Import train_test_split
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

features_final, targets_final = data.drop('admit', axis=1), data['admit']

# Split the 'features' and 'income' data into training and testing sets
features, features_test, targets, targets_test = train_test_split(features_final,
                                                                  targets_final,
                                                                  test_size = 0.1,
                                                                  random_state = 42)

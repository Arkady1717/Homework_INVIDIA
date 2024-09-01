# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:26:11 2024

@author: ארקדי
"""

import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import PercentFormatter
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import randint
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


#building prediction model for the 'TLJYWBE' test

# Threshold
th=1/100000

# Read data
df_dataset = pd.read_feather("home_assignment.feather")  

# Remove rows with nan values in 'TLJYWBE' column
df_target=df_dataset[['TLJYWBE']]

ind=np.where(df_target['TLJYWBE'].notnull())[0]

df_dataset=df_dataset.iloc[ind]

# Count number of nan values per column
z_count=df_dataset.isnull().sum(axis = 0)

z_count=z_count/df_dataset.shape[0]

# Adding labels and title
fig = plt.subplots(1,1,figsize=(12,5))

plt.hist(z_count, bins=10, color='skyblue', edgecolor='black')

plt.xlabel('Number of nan values per column')

plt.ylabel('Frequency')

plt.title('Histogram, Number of NaN values per column (in percents)')

plt.grid()

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

# Labeling pass/fail
df_dataset['TLJYWBE']=np.where(df_dataset['TLJYWBE'] < th, 0, 1)

# Remove columns with nan values
df_dataset=df_dataset.dropna(axis=1)

df_dataset=df_dataset.drop(columns=df_dataset.columns[(df_dataset == 'nan').any()])

minority_class = df_dataset[df_dataset['TLJYWBE']== 1]

majority_class = df_dataset[df_dataset['TLJYWBE'] == 0]

# Downsample the majority class
majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)

# Combine the downsampled majority class with the minority class
balanced_data = pd.concat([minority_class, majority_downsampled])

# shuffle the DataFrame rows
balanced_data = balanced_data.sample(frac = 1).reset_index(drop=True)

# Send to csv file
balanced_data.to_csv('Downsample_Balanced_Data.csv')

# Data slitting
X_resampled = balanced_data.drop('TLJYWBE', axis=1)

y_resampled = balanced_data['TLJYWBE']

# Reset index
X_resampled = X_resampled.reset_index(drop=True)

y_resampled = y_resampled.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)

clf = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# Score
print(clf.score(X_test, y_test))

# Confusion matrix
_ = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)

# Feature importance
feature_importance=pd.DataFrame(clf.feature_importances_, index=X_train.columns).sort_values(by=0, ascending=False)

fig = plt.subplots(1,1,figsize=(12,5))

feature_importance[:10].plot.bar(legend=False)

plt.grid()

plt.title('Feature Importance')

plt.xlabel('Features')

plt.ylabel('Weight')

# Check consistency
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_idx, test_idx in cv.split(X_resampled, y_resampled):
    
    X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
    
    y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]


    clf = RandomForestClassifier(random_state=42)
    
    clf.fit(X_train, y_train)

# Evaluate the model
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred))

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)

# Hyperparameters
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)    

# Create the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

print(classification_report(y_test, y_pred))

# Get predicted class probabilities for the test set 
y_pred_prob = best_rf.predict_proba(X_test)[:, 1] 

# Compute the false positive rate (FPR)  
# and true positive rate (TPR) for different classification thresholds 
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)

# Compute the ROC AUC score 
roc_auc = roc_auc_score(y_test, y_pred_prob) 

# Plot the ROC curve 
fig = plt.subplots(1,1,figsize=(12,5))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') 

plt.xlabel('False Positive Rate') 

plt.ylabel('True Positive Rate') 

plt.title('ROC Curve') 

plt.grid()

plt.legend(loc="lower right") 

plt.show()

#####################################
# Features
feature_importance=feature_importance.reset_index().rename(columns={"index": "Features", 0: "Score"})

X_resampled = X_resampled[[feature_importance['Features'].loc[0],feature_importance['Features'].loc[1]]]

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)

# Hyperparameters
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)    

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

print(classification_report(y_test, y_pred))

############################ Random Oversampling #############################

# Upsample the minority class 
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

# Combine the upsampled minority class with the majority class
balanced_data = pd.concat([majority_class, minority_upsampled])

# Shuffle data
SampleSize=10000

ind=np.random.choice(balanced_data.shape[0], SampleSize, replace=False) 

balanced_data=balanced_data.iloc[ind]

balanced_data=balanced_data.reset_index(drop=True)

# Send to csv file
balanced_data.to_csv('Upsample_Balanced_Data.csv')

# Data slitting
X_resampled = balanced_data.drop('TLJYWBE', axis=1)

y_resampled = balanced_data['TLJYWBE']

# Reset index
X_resampled = X_resampled.reset_index(drop=True)

y_resampled = y_resampled.reset_index(drop=True)

# Important features
X_resampled=X_resampled[[feature_importance['Features'].loc[0],feature_importance['Features'].loc[1],feature_importance['Features'].loc[2]]]

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

####################
# Create a random forest classifier
rf = RandomForestClassifier()

# Hyperparameters
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,30),
              'max_features': ['auto', 'sqrt'],
              'min_samples_split' : [1, 2, 5, 10, 15, 20, 30],
              'bootstrap': [True, False]}


# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)    

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

print(classification_report(y_test, y_pred))




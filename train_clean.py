import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from clearml import Task, Dataset

# Connecting ClearML
task = Task.init(project_name="assignment1", task_name="training_clean")

# get dataset with split/test
dataset = Dataset.get(dataset_project='assignment1', dataset_name='clean_data_split')

# get a read only version of the data
dataset_folder = dataset.get_local_copy()

# df = pd.read_csv(dataset_folder + '/dirty_data.csv')

# open the dataset pickle file
with open(dataset_folder + '/clean_train.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# train the model
rf = RandomForestRegressor(max_depth=2, random_state=0)
rf.fit(X_train,y_train)

# store the trained model
joblib.dump(rf, 'rf_clean.pkl', compress=True)

# print model predication results
result = rf.score(X_test, y_test)

y_pred = rf.predict(X_test)

# Evaluating the Algorithm
from sklearn import metrics
print('Clean Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Clean Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Clean Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# plot test results
plt.scatter(y_pred, y_test)
plt.title('Clean Predicted vs Actual Results')
plt.show()

# plot train results
plt.scatter(rf.predict(X_train), y_train)
plt.title('Clean Training Results')
plt.show()

# show feature importances
(pd.Series(rf.feature_importances_, index=X_train.columns)
   .plot(kind='barh'))
plt.title('Clean Feature Importance')
plt.show()

 
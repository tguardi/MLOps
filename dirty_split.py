import os
import csv
import pickle
import pandas as pd
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split


# Connecting ClearML
task = Task.init(project_name="assignment1", task_name="split_dirty")

# get the original dataset
dataset = Dataset.get(dataset_project='assignment1', dataset_name='data_dirty')

# create a copy that we can change,
dataset_folder = dataset.get_mutable_local_copy(target_folder='/Users/guardi/MSCA/MLOps/ClearML/working_dataset', overwrite=True)
print(f"dataset_folder: {dataset_folder}")

dfDirty = pd.read_csv(dataset_folder + '/dirty_data.csv')

X = dfDirty[['GDP per capita',
             'Social support',
             'Freedom to make life choices', 
             'Generosity', 
             'Perceptions of corruption']]
# target
y = dfDirty['Healthy life expectancy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# store the dataset split into a pickle file
with open(dataset_folder + '/dirty_train.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)

# create a new version of the dataset with the pickle file
new_dataset = Dataset.create(
    dataset_project='assignment1', dataset_name='dirty_data_split', parent_datasets=[dataset])
new_dataset.sync_folder(local_path=dataset_folder)
new_dataset.upload()
new_dataset.finalize()

print('we are done')
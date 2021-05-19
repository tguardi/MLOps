import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Flexible integration for any Python script
import wandb
# 1. Start a W&B run
wandb.init(project='happiness', entity='tguardi')

directory = '/Users/guardi/MSCA/MLOps/ClearML/Assignment1_scripts/'
# open the dataset pickle file
with open(directory+'clean_train.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# 2. Save model inputs and hyperparameters
# Start a new run, tracking hyperparameters in config
config={
    "generation": 5,
    "population_size": 100,
    "verbosity": 2,
    "random_state": 42,
}

config = wandb.config
config.generation = 4
config.population_size = 25
config.verbosity = 2
config.random_state = 42

# train the model
tpot = TPOTRegressor()

# Fit the TPOT object
tpot.fit(X_train, y_train)
# Print score
# print(tpot.score(X_test, y_test))

# Export the pipeline settings to a file
pipeline_file = tpot.export('tpot_digits_pipeline.py')
artifact = wandb.Artifact(pipeline_file, type='pipeline_file')
artifact.add_file('pipeline_file.txt')
run.log_artifact(artifact)


# Model Evaluation
# Build evaluation function
def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
# Predict the values
predicted_qualities = tpot.predict(X_test)
# get rmse, mae, and r2
(rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

wandb.log({"rmse":rmse, "mae":mae, "r2":r2})

run.finish()

# Print out metrics
# print("  RMSE: %s" % rmse)
# print("  MAE: %s" % mae)
# print("  R2: %s" % r2)

 

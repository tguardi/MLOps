#!/bin/bash
clearml-data create --project task1 --name clean1
# clearml-data sync --folder ./data/
clearml-data sync --folder /MSCA/MLOps/ClearML/data

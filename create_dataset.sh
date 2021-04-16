#!/bin/bash
clearml-data create --project assignment1 --name data_dirty
# sync to local folder:
# clearml-data sync --folder {insert path to local folder with your data}
clearml-data sync --folder /Users/guardi/MSCA/MLOps/ClearML/Assignment1/data

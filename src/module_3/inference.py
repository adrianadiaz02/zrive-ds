
from pathlib import Path

import pandas as pd
import numpy as np
import os
from joblib import load

from sklearn.linear_model import LogisticRegression

from general_functions import prepare_data,train_val_split, feature_target_split, standardise_data, evaluate_model

OUTPUT_PATH = Path("/mnt/c/Users/Adriana/Desktop/ZRIVE/zrive-ds/src/module_3/out")
DATA_PATH = Path("/mnt/c/Users/Adriana/Desktop/ZRIVE/data/groceries/box_builder_dataset/feature_frame.csv")
FEATURE_COLS = ['ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price']


def main():
    model_name = "best_model.pkl"
    model = load(os.path.join(OUTPUT_PATH, model_name))

    data = prepare_data(DATA_PATH)
    _, _, _, _, X_test , Y_test = train_val_split(data)
    X_test = standardise_data(X_test)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    pr_auc = evaluate_model(Y_test, y_pred)
    print(pr_auc)

if __name__ == "__main__":
    main()
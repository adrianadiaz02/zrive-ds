from pathlib import Path

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression


from general_functions import prepare_data, train_val_split, standardise_data, evaluate_model

# We only consider the features which we have seen previously that are relevant
FEATURE_COLS = ['ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price']
TARGET_COL = 'outcome'
Cs = [1e-8, 1e-4, 1e-2]

OUTPUT_PATH = Path("/mnt/c/Users/Adriana/Desktop/ZRIVE/zrive-ds/src/module_3/out")
DATA_PATH = Path("/mnt/c/Users/Adriana/Desktop/ZRIVE/data/groceries/box_builder_dataset/feature_frame.csv")


def ridge_model_selection(data):
    X_train, Y_train, X_val, Y_val, _ , _ = train_val_split(data)

    # First, standardize the data
    X_train_scaled = standardise_data(X_train)
    X_val_scaled = standardise_data(X_val)

    best_pr_auc = 0

    for c in Cs:
        lr = LogisticRegression(penalty='l2', C=c)
        
        # Fit the model on scaled training data
        lr.fit(X_train_scaled, Y_train)

        # Predict probabilities for the positive class on the training set
        train_proba = lr.predict_proba(X_train_scaled)[:, 1]
        pr_auc_train = evaluate_model(Y_train, train_proba)

        # Predict probabilities for the positive class on the validation set
        val_proba = lr.predict_proba(X_val_scaled)[:, 1]
        pr_auc_val = evaluate_model(Y_val, val_proba)

        # Check if the current model is better than the best so far
        if pr_auc_val > best_pr_auc:
            best_pr_auc = pr_auc_val
            best_c = c

    # Train the model with c that has provided the best auc
    best_model = LogisticRegression(penalty='l2', C=best_c)
    best_model.fit(X_train_scaled, Y_train)

    return best_model


def save_model(model, filename):
    joblib.dump(model, filename)


def main():
    data = prepare_data(DATA_PATH)
    best_model = ridge_model_selection(data)
    save_model(best_model, os.path.join(OUTPUT_PATH, "best_model.pkl"))

if __name__ == "__main__":
    main()
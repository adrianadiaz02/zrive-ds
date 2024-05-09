from pathlib import Path

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler

# We only consider the features which we have seen previously that are relevant
FEATURE_COLS = ['ordered_before', 'global_popularity', 'abandoned_before', 'normalised_price']
TARGET_COL = 'outcome'

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def get_relevant_orders(df):
    """"We are only interested in orders with at least 5 products"""
    # Group data by order_id and sum the outcome (A product is in the order if its outcome is 1)
    order_size = df.groupby('order_id').outcome.sum()

    # Identify orders with size at least 5 and extract the order_id of these big orders
    big_orders = order_size[order_size >= 5].index 

    # Filter data to only include rows where order_id is in the list of big_orders
    filtered_data = df.loc[lambda x: x.order_id.isin(big_orders)]
    return filtered_data


def transform_data(data):
    """Transform data to the adequate types"""
    data['variant_id'] = data['variant_id'].astype('str')
    data['order_id'] = data['order_id'].astype('str')
    data['user_id'] = data['user_id'].astype('str')

    data['created_at'] = data['created_at'].astype('datetime64[us]')
    data['order_date'] = data['order_date'].astype('datetime64[us]')

    data['outcome'] = data['outcome'].astype(int)

    binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
    for col in binary_cols:
        data[col] = data[col].astype(int)

    # Iterate over each column and check if it starts with 'count_'
    for col in data.columns:
        if col.startswith('count_'):
            # Convert the column to integer type
            data[col] = data[col].astype(int)

    data['people_ex_baby'] = data['people_ex_baby'].astype('int64')

    data['days_since_purchase_variant_id'] = data['days_since_purchase_variant_id'].astype('int64')
    data['days_since_purchase_product_type'] = data['days_since_purchase_product_type'].astype('int64')

    return data

def prepare_data(file_path):
    data = load_data(file_path)
    transformed_data = transform_data(data)
    filtered_data = get_relevant_orders(transformed_data)


    return filtered_data


def feature_target_split(df, target):
    X = df.drop(columns=[target])  # Drop the target column to create the features DataFrame
    Y = df[target]   # Target variable we want to predict
    return X, Y


def train_val_split(data):
    daily_orders = data.groupby('order_date').order_id.nunique()
    cum_sum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
    train_val_cut = cum_sum_daily_orders[cum_sum_daily_orders <= 0.7].idxmax()
    val_test_cut = cum_sum_daily_orders[cum_sum_daily_orders <= 0.9].idxmax()

    train_data = data[data.order_date <= train_val_cut]
    val_data = data[(data.order_date > train_val_cut) & (data.order_date <= val_test_cut) ]
    test_data = data[data.order_date > val_test_cut]

    X_train, Y_train = feature_target_split(train_data, TARGET_COL)
    X_val, Y_val = feature_target_split(val_data, TARGET_COL)
    X_test, Y_test = feature_target_split(test_data, TARGET_COL)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def standardise_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[FEATURE_COLS])
    return scaled_data

def evaluate_model(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc
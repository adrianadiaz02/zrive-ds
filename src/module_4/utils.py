from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.level = logging.INFO

def load_raw_dataset():
    file_path = Path("/mnt/c/Users/Adriana/Desktop/ZRIVE/data/groceries/box_builder_dataset/feature_frame.csv")
    data = pd.read_csv(file_path)
    logger.info(f"Dataset loaded from {file_path}")
    return data

def correct_data_types(data):
    data['variant_id'] = data['variant_id'].astype('str')
    data['order_id'] = data['order_id'].astype('str')
    data['user_id'] = data['user_id'].astype('str')

    data['created_at'] = data['created_at'].astype('datetime64[us]')
    data['order_date'] = data['order_date'].astype('datetime64[us]')

    data['outcome'] = data['outcome'].astype(int)

    binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
    for col in binary_cols:
        data[col] = data[col].astype(int)

    for col in data.columns:
        if col.startswith('count_'):
            data[col] = data[col].astype(int)

    data['people_ex_baby'] = data['people_ex_baby'].astype('int64')

    data['days_since_purchase_variant_id'] = data['days_since_purchase_variant_id'].astype('int64')
    data['days_since_purchase_product_type'] = data['days_since_purchase_product_type'].astype('int64')

def get_relevant_orders(data, min_products=5):
    order_size = data.groupby('order_id').outcome.sum()
    big_orders = order_size[order_size >= min_products].index 
    filtered_data = data.loc[lambda x: x.order_id.isin(big_orders)]
    return filtered_data

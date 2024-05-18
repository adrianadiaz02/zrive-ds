import json
import pandas as pd
import joblib

class PushModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, data):
        return self.model.predict_proba(data)[:, 1]


def handler_predict(event, _):
    """
    Receives a json with a field "users" that contains:
    {
    "user_id": {"feature 1": feature value, "feature 2": feature value, ...},
    "user_id2": {"feature 1": feature value, "feature 2": feature value, ...}.
    }
    Output is a json with a field "body" with fields "prediction":
    {"user_id": prediction, "user_id2": prediction ...}

    """

    model_path = event.get("model_path", "push_latest.pkl")
    users_data = json.loads(event["users"])
    
    data_to_predict = pd.DataFrame.from_dict(users_data, orient='index')
    
    model = PushModel(model_path)
    predictions = model.predict(data_to_predict)
    
    predictions_dict = dict(zip(data_to_predict.index, predictions))
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": predictions_dict
        })
    }


# Example event
event = {
    "model_path": "push_2024_05_18.pkl",
    "users": json.dumps({
        "user_id1": {"user_order_seq": 3, "ordered_before": 0.0, "abandoned_before": 1.0,
        "active_snoozed":0.0, "set_as_regular":1.0, "normalised_price": 0.081052, "discount_pct":0.053512, 
        "global_popularity": 0.1, "count_adults": 2, "count_children": 0, "count_babies": 0, "count_pets": 0, 
        "people_ex_baby": 0, "days_since_purchase_variant_id":33.0, "avg_days_to_buy_variant_id": 42, 
        "std_days_to_buy_variant_id":31.134053, "days_since_purchase_product_type": 35, 
        "avg_days_to_buy_product_type": 33, "std_days_to_buy_product_type": 24.27618
        },
    })
}

if __name__ == "__main__":
    handler_predict(event, None)

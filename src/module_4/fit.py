import json
import pandas as pd
import datetime
import joblib
import logging
from pathlib import Path

# Import utility functions
from utils import load_raw_dataset, correct_data_types, get_relevant_orders
from push_model import PushModel


logger = logging.getLogger(__name__)
logger.level = logging.INFO

def load_data():
    data = load_raw_dataset()
    correct_data_types(data)
    data = get_relevant_orders(data)
    return data


def handler_fit(event, _):
    """
    Receives the model parametrisation, loads the data, trains the model and saves it to disk.
    The output of the function is a dictionary containing the model name and model path.
    """
    model_parametrisation = event.get("model_parametrisation", {})
    
    data = load_data()
    
    model = PushModel(model_parametrisation)
    trained_model = model.fit_model(data)
    
    model_name = f"push_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    model_path = f"{model_name}.pkl"
    joblib.dump(trained_model, model_path)
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "model_path": model_path
        })
    }


# Example event
event = {
    "model_parametrisation": {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 5
    }
}

if __name__ == "__main__":
    print(handler_fit(event, None))
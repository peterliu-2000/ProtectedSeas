import json
import joblib
from utils.constants import *

def load_model(model_path = MODEL_FILE_NAME, model_metadata_path = MODEL_METADATA_FILE_NAME):
    model = joblib.load(model_path)
    with open(model_metadata_path, "r") as f:
        model_info = json.load(f)
    return {
        "model" : model,
        "features" : model_info['features'],
        "label_map" : {int(k): v for k, v in model_info['label_mapping'].items()}
    }
    
def model_predict(model, X, label = True):
    if label:
        numeric = list(model["model"].predict(X[model["features"]]))
        if len(numeric) == 1:
            return model["label_map"][numeric[0]]
        else:
            return [model["label_map"][item] for item in numeric]
    else:
        return model["model"].predict_proba(X[model["features"]])
        
def get_label_encodings(model):
    return model["label_map"]
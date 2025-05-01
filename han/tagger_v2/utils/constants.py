"""
This file contains all necessary constants and lookup tables.
"""

MODEL_FILE_NAME = "support/models/xgb_activity_model.pkl"
MODEL_METADATA_FILE_NAME = "support/models/xgb_activity_model_description.json"

ACT_CODE = [
    "transit", 
    "drifting", 
    "fishing", 
    "lobster",
    "hook", 
    "seine", 
    "trawl", 
    "stopped", 
    ""
]

ACT_NAMES = [
    "Transit",
    "Drifting / Low Speed",
    "Fishing (Unspecified)",
    "Fishing (Lobster)",
    "Fishing (Hook & Line)",
    "Fishing (Purse Seine)",
    "Fishing (Trawl)",
    "Stopped",
    "Activity Unspecified"
]


ACT_CODE_2_NAME = {c : n for (c, n) in zip(ACT_CODE, ACT_NAMES)}
ACT_NAME_2_CODE = {n : c for (c, n) in zip(ACT_CODE, ACT_NAMES)}

# Lookup Tables for vessel types:
TYPE_NAMES = [
    "",
    "Class B Vessels",
    "Cargo / Tanker Ship" ,
    "Fishing Boat",
    "Military Ship" ,
    "Passenger Ship" ,
    "Pleasure Craft" ,
    "Tug / Tow Boat",
    "Others / Unspecified"
]

TYPE_CODE = [
    None,
    "class_b",
    "cargo/tanker",
    "fishing_boat",
    "military_ship",
    "passenger_ship",
    "pleasure_craft",
    "tug/tow",
    "other"
]


TYPE_CODE_2_NAME = {c : n for (c, n) in zip(TYPE_CODE, TYPE_NAMES)}
TYPE_NAME_2_CODE = {n : c for (c, n) in zip(TYPE_CODE, TYPE_NAMES)}

PROGRAM_CACHE_PATH = "support/cache/"



# Default Filter Parameters
FILTER_DEFAULT = {'tag': None, 'type': None, "pred" : None, 'confidence_low': 0.0, 'confidence_high': 1.0}
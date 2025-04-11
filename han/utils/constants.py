"""
This file contains all necessary constants and lookup tables.
"""

# Vessel Activity Tags
ACTIVITY_TAGS = ["transit", "overnight", "loiter", "cleanup", "fishing_c", "fishing_r",
            "research", "diving", "repairs", "distress", "other"]
# N/A for vessel type values are accepted as well.
VESSEL_TYPES = ["cargo/tanker", "class_b", "passenger_ship", "other",
                "tug/tow", "military_ship", "fishing_boat", "pleasure_craft", "sail"]


# Lookup Tables for activity tags:
ACT_NAMES = [
    "All",
    "Transit",
    "Loiter",
    "Overnight Loiter",
    "Cleanup" ,
    "Fishing Comm." ,
    "Fishing Rec." ,
    "Research" ,
    "Diving" ,
    "Repairs" ,
    "Distress" ,
    "Other" 
]

LOOKUP_ACT_name_to_code = {
            "All" : None,
            "Transit" : "transit",
            "Loiter" : "loiter",
            "Overnight Loiter" : "overnight",
            "Cleanup" : "cleanup",
            "Fishing Comm." : "fishing_c",
            "Fishing Rec." : "fishing_r",
            "Research" : "research",
            "Diving" : "diving",
            "Repairs" : "repairs",
            "Distress" : "distress",
            "Other" : "other"
        }

LOOKUP_ACT_code_to_name = {"transit" : "Transit",
                     "loiter" : "Loiter",
                     "overnight" : "Overnight Loiter",
                     "cleanup" : "Cleanup", 
                     "fishing_c" : "Fishing Comm.",
                     "fishing_r" : "Fishing Rec.",
                     "research" : "Research",
                     "diving" : "Diving",
                     "repairs" : "Repairs",
                     "distress" : "Distress",
                     "other" : "Other",
                     None : "All"}

# Lookup Tables for vessel types:
TYPE_NAMES = [
    "All",
    "Class B Vessels",
    "Cargo & Tanker Ship" ,
    "Fishing Boat",
    "Military Ship" ,
    "Passenger Ship" ,
    "Pleasure Craft" ,
    "Sail Boat",
    "Tug & Tow Boat",
    "Others / Unspecified"
]

LOOKUP_TYPE_name_to_code = {
    "All" : None,
    "Class B Vessels" : "class_b",
    "Cargo & Tanker Ship" : "cargo/tanker",
    "Fishing Boat" : "fishing_boat",
    "Military Ship" : "military_ship",
    "Passenger Ship" : "passenger_ship",
    "Pleasure Craft" : "pleasure_craft",
    "Sail Boat" : "sail",
    "Tug & Tow Boat" : "tug/tow",
    "Others / Unspecified" : "other"   
}

LOOKUP_TYPE_code_to_name = {
    None: "All" ,
    "class_b": "Class B Vessels" ,
    "cargo/tanker": "Cargo & Tanker Ship" ,
    "fishing_boat": "Fishing Boat" ,
    "military_ship": "Military Ship" ,
    "passenger_ship": "Passenger Ship" ,
    "pleasure_craft": "Pleasure Craft" ,
    "tug/tow": "Tug & Tow Boat" ,
    "sail" : "Sail Boat",
    "other": "Others / Unspecified"
}

# In app display
LOOKUP_TYPE_name_to_code_app = {
    "Select Vessel Type" : None,
    "Class B Vessels" : "class_b",
    "Cargo & Tanker Ship" : "cargo/tanker",
    "Fishing Boat" : "fishing_boat",
    "Military Ship" : "military_ship",
    "Passenger Ship" : "passenger_ship",
    "Pleasure Craft" : "pleasure_craft",
    "Sail Boat" : "sail",
    "Tug & Tow Boat" : "tug/tow",
    "Others / Unspecified" : "other"   
}

LOOKUP_TYPE_code_to_name_app = {
    None: "Select Vessel Type" ,
    "class_b": "Class B Vessels" ,
    "cargo/tanker": "Cargo & Tanker Ship" ,
    "fishing_boat": "Fishing Boat" ,
    "military_ship": "Military Ship" ,
    "passenger_ship": "Passenger Ship" ,
    "pleasure_craft": "Pleasure Craft" ,
    "tug/tow": "Tug & Tow Boat" ,
    "sail" : "Sail Boat",
    "other": "Others / Unspecified"
}


# Default Filter Parameters
FILTER_DEFAULT = {'tag': None, 'type': None, 'has_notes': False, 'no_tags': False, 'duplicate_tags': False, 'valid_only': False, 'confidence_low': 0.0, 'confidence_high': 1.0}
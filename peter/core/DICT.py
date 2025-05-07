ACTIVITIES = ["stopped", "transit", "drifting", "lobster", "hook", "seine", "trawl", "fishing"]
ACT_to_LABEL = {k:v for k,v in zip(ACTIVITIES, range(len(ACTIVITIES)))}
N_ACT = len(ACTIVITIES)


TYPES = ["class_b", "cargo/tanker", "passenger_ship", "tug/tow", 
         "military_ship", "fishing_boat", "other"]
TYPE_to_LABEL = {k:v for k,v in zip(TYPES, range(len(TYPES)))}
N_TYPE = len(TYPES)

TYPES_TO_AGG = {
            'tanker_ship': 'cargo/tanker',
            'cargo_ship': 'cargo/tanker',
            'tug': 'tug/tow',
            'towing_ship': 'tug/tow',
            'fishing_boat': 'fishing_boat',
            'commercial_fishing_boat': 'fishing_boat',
            'military_ship': 'military_ship',
            'class_b':'class_b',
            'passenger_ship': 'passenger_ship',
            'pleasure_craft': 'class_b',
            'sailboat': 'class_b',
            'search_and_rescue_boat': 'other',
            'pilot_boat': 'other',
            'high_speed_craft': 'other',
            'law_enforcement_boat': 'other',
            'other': 'other',
            'unknown': 'other'
        }
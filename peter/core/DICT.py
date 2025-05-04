ACTIVITIES = ["stopped", "transit", "drifting", "lobster", "hook", "seine", "trawl", "fishing"]
ACT_to_LABEL = {k:v for k,v in zip(ACTIVITIES, range(len(ACTIVITIES)))}
N_ACT = len(ACTIVITIES)


TYPES = ["class_b", "cargo/tanker", "passenger_ship", "tug/tow", 
         "military_ship", "pleasure_craft", "fishing_boat", "other"]
TYPE_to_LABEL = {k:v for k,v in zip(TYPES, range(len(TYPES)))}
N_TYPE = len(TYPES)
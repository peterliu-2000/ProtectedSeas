#Activity Labels
ACTIVITIES = ["stopped", "transit", "drifting", "lobster", "hook", "seine", "trawl", "fishing"]
NUM2ACTIVITY = {
        0: "drifting",
        1: "transit",
        2: "hook",
        3: "lobster",
        4: "stopped",
        5: "seine",
        6: "trawl"
}
ACITIVTY2NUM = {v: k for k, v in NUM2ACTIVITY.items()}
N_ACT = len(ACTIVITIES)

#Type Labels
TYPES = ["class_b", "cargo/tanker", "passenger_ship", "tug/tow", 
         "military_ship", "fishing_boat", "other"]
NUM2TYPE = {
    0: "class_b",
    1: "cargo/tanker",
    2: "passenger_ship",
    3: "tug/tow",
    4: "military_ship",
    5: "fishing_boat",
    6: "other"
}
TYPE2NUM = {v: k for k, v in NUM2TYPE.items()}
N_TYPE = len(TYPES)

#Type m2 into type_m2_agg
TYPES2AGG = {
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

#AIS Type Code to type m2
CODE2TYPE = {
    **{i: 'other' for i in range(0, 20)},
    **{i: 'wing_in_ground_effect' for i in range(20, 30)},
    30: 'fishing_boat',
    **{i: 'towing_ship' for i in range(31, 33)},
    **{i: 'other' for i in range(33, 35)},
    35: 'military_ship',
    36: 'sailboat',
    37: 'pleasure_craft',
    **{i: 'other' for i in range(38, 40)},
    **{i: 'high_speed_craft' for i in range(40, 50)},
    50: 'pilot_boat',
    51: 'search_and_rescue_boat',
    52: 'tug',
    53: 'port_tender',
    54: 'other',
    55: 'law_enforcement_boat',
    **{i: 'other' for i in range(56, 60)},
    **{i: 'passenger_ship' for i in range(60, 70)},
    **{i: 'cargo_ship' for i in range(70, 80)},
    **{i: 'tanker_ship' for i in range(80, 90)},
    **{i: 'other' for i in range(90, 100)}
}
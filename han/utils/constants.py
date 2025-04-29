# Label Definitions

ACTIVITIES = ["stopped", "transit", "drifting", "lobster", "hook", "seine", "trawl", "fishing"]
ACT_to_LABEL = {k:v for k,v in zip(ACTIVITIES, range(len(ACTIVITIES)))}
ACT_N_CLASSES = 8
## April 5, W1

[Trajectory Viewer](peter/tracks_viewer.ipynb): different vessel type track viz

[Baseline Vessel Type](peter/type_classfication_baseline.ipynb): idea is to build a baseline type classifier (XGBoost) using raw track-level features (0.84 overall accuracy) to which future models can be compared. Currently have 9 types: ['class_b' 'passenger_ship' 'other' 'fishing_boat' 'tug/tow'
 'military_ship' 'search_and_rescue_boat' 'pleasure_craft' 'cargo/tanker']

#### Questions

* Prioritize certain vessel types?
* Birds/Aves in labels ?
* How are summary statistics of tracks calculated? provide functions ?
* Original binary classifier: is it dynamic or static (after gathering enough datapoints of a tracks to obtain summary statistics). Can we look at the code? Should our type prediction follow its paradigm?

#### Features

Current features
* Min, max, avg of speed over ground
* Curviness ??
* Mean, sd of heading (degrees)
* Mean, sd of turning (the absolute change in heading between consec point)
* Total Distance Travelled
* Max Distance target travelled from origin
* Duration in designated area of interest

New features that might be promising:
* Max lat/lon span (coverage)

#### Next Steps

* Data requires cleaning ... need to talk abt methods
* Extract more info from notes?
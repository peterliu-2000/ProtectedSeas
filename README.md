## April 8, W2

#### Objectives
1. Perform some EDAs on vessel size
2. Data Cleaning: Dig deeper into dataset relationship, and try using Sam's method on dataset cleaning: i.e. only find ais-radar track pairs with the most number of detection points
3. Look into vessel activity (distinguish from type), currently three main types of interests: _in transit, fishing, leisure_
4. Work on the tagger application; perhaps establish a model to predict transit vs not transit? Consider some extent of hand-labelling thru the tagger?
5. (If we get our hands on sum_stats functions): Look into it and try it on tracks

#### Discovery 
* Radar tracks correspond to 16591 ais tracks, but only 15345 are present in `ais_tracks/radar_tracks` provided. Examples of some missing ['id_track', 'assoc_id']: (36979855 , 36979840), (32063462 , 32065809), (31251315, 31251147). A total of 1246 missing
* Size seems 

#### Qs
* More on background algorithm: my understanding is, radar detecting a signal, keeps a log of its tracks, backend algorithm calculates the track summary statistics and feed it to XGBoost model to decide whether the signal is true/false positive

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

#### Labeling Application

* I wrote an application (via Python's Tkinter GUI) for visualizing vessel trajectories and modifying the tags in our tagged data.
* Upon investigating the tags, some activity tags appear to be incorrect. (A fishing boat in transit is tagged fishing_c rather than transit.)
* Labels that reflects vessel activities may be more informative for avtivity type classification.
* This application can be later expanded for labeling vessel types, providing additional data for training / validating the vessel type classifier.


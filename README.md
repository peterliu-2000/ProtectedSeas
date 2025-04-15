## April 15, W3

#### Meeting Notes

* Sailboats: two distinctive behaviors: sails down using motors vs sails up (quite some in the dataset), useful to think about for _activity prediction_
* Site 45: bay area where pleasure crafts (like sailboats putting down anchor), hence lot of detection points w/ few movements
* _activity tag_: achor/stop (might be different from loitering)
* transit vs. travelling in a straight line w/o changing speed 

#### Objectives

* Think about sensible _activity_ categories
* Look into sum stats calculation file

## April 8, W2

#### Discovery 
* Dataset understanding: current working on cleaning and labelling (vessel type, activity) rows with notes the `labelled_tracks` to produce a better-quality detection datasets (~4000 tracks). Combine w/ `radar_detections` (~15000 tracks that have info in AIS) to generate around 20K tracks as complete set. 
* Radar tracks correspond to 16591 ais tracks, but only 15345 are present in `ais_tracks/radar_tracks` provided. Examples of some missing ['id_track', 'assoc_id']: (36979855 , 36979840), (32063462 , 32065809), (31251315, 31251147). A total of 1246 missing
* Size EDAs: less informative than EDAs on vessel type, less clear on which direction to focus on. Also size is very dependent on type of vessel as well. One idea will be to come up with a few different size-bucket (small, medium, large) but this is just kinda like categorizing vessels by type
* Follow-up on summary statisitcs calculation functions: tried self-implementation but does have (slight) mismatches w. provided radar_tracks 
* Current vessel type aggregation scheme:

| Original Category              | Aggregated Category   |
|-------------------------------|------------------------|
| `tanker_ship`                 | `cargo/tanker`        |
| `cargo_ship`                  | `cargo/tanker`        |
| `tug`                         | `tug/tow`             |
| `towing_ship`                | `tug/tow`             |
| `fishing_boat`               | `fishing_boat`        |
| `commercial_fishing_boat`    | `fishing_boat`        |
| `military_ship`              | `military_ship`       |
| `class_b`                    | `class_b`             |
| `passenger_ship`             | `passenger_ship`      |
| `pleasure_craft`             | `pleasure_craft`      |
| `sailboat`                   | `pleasure_craft`      |
| `search_and_rescue_boat`     | `other`               |
| `pilot_boat`                 | `other`               |
| `high_speed_craft`           | `other`               |
| `law_enforcement_boat`       | `other`               |
| `other`                      | `other`               |
| `unknown`                    | `other`               |


#### Data Cleaning
* For tagged data: Removed most unwanted tracks (eg. tracks with notes identifying as false targets, disrupted tracks, and tracks I have significant evidence for being false targets, such as those going inland / thru land.)
* Tagged most observations with a straight trajectory as transit. (Added about 1k transit tags during data cleaning.) Training a model for transit tags was not necessary since I removed about 4k unwanted observations during preliminary filtering.
* Tagged many noted tracks with vessel type information to be incorporated into Peter's dataset
* Tagged some tracks with patterns that I believe could be fishing as the unused tag "repairs." ~100 of such tags.

#### Qs
* More on background algorithm: my understanding is, radar detecting a signal, keeps a log of its tracks, backend algorithm calculates the track summary statistics and feed it to XGBoost model to decide whether the signal is true/false positive (we will need to implement new-feature-functions, need to think about how can it integrate w/ the existing system)
* There are a lot of trajectories in site 45 (I believe thousands) where I see a tracked object lingering / not moving for a very long period of time. (Sometimes lasting for almost a day.) Should I tag these activities as loitering or ignore these observations all together? (Since I cant verify if these are false targets or not.)
* Site 45 also has many kayaking / sailing activities going on. I think it is most appropirate to tag kayaking activities as loitering.
* Songyu: The tagged data has a lot of observations tagged as "kayaks" or "sailboat". Maybe incorporate these observations to Peter's vessel type dataset? For now, I have tagged kayaks as sailboats, but let me know a more approporate type is required.

#### Next Steps
* For tagged vessel data: I plan to remove all tracks with less than 50 detection points. Such tracks are not very informative on activities and some are likely to be false targets. (I have seen many of such tracks going through land, and yes I removed all the AVES)
* For longer tracks, it may be helpful to break it down into mutliple segments. Most statsitical / ML models expects inputs of fixed dimensions, and the great variation we have in track lengths may need to be considered when we eventually start modeling on the detection data. Splitting detections into segments could also help with potential class imbalance issues later down the line.
* Possibly investigate the transit tags a bit more using the summary statistics. My current vessel tracker app does not display the summary statistics: I could have falsely tagged some fishing / loitering as transit just because they appear to be a straight line on the map. Asking Sam about the typical vessel transit speed could help as well.



...

#### Objectives
1. Perform some EDAs on vessel size
2. Data Cleaning: Dig deeper into dataset relationship, and try using Sam's method on dataset cleaning: i.e. only find ais-radar track pairs with the most number of detection points
3. Look into vessel activity (distinguish from type), currently three main types of interests: _in transit, fishing, leisure_
4. Work on the tagger application; perhaps establish a model to predict transit vs not transit? Consider some extent of hand-labelling thru the tagger?
5. (If we get our hands on sum_stats functions): Look into it and try it on tracks

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


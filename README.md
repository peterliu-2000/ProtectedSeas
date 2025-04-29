## April 22, W4

Experimentation with computer vision based models
* Rasterized vessel trajectory according to methods outlined in this paper: https://arxiv.org/html/2401.01676v1
  * Red channel: Number of detection points within each pixel (capped at 255)
  * Green channel: Average speed of detection points within each pixel (clipped to range (0, 25.5))
  * Blue channel: Average turning of detection points within each pixel (clipped to range (0, 180) unit: degree)
* ResNet-18 Model vastly overfits to the training data even with moderate dropout (p = 0.2, at the end of each resnet block)
  * Training Accuracy: 0.99, Validation Accuracy: 0.89
  * Model is really struggling with Hook and Line: very likely to label it as transit, drifting and lobster.
  * Generalization gap presumably caused by the lack of observations in certain activity classes (eg. purse seine, trawl, lobster, hook and line)
  * Highlights the need to perform some form of data augmentation
  * "fishing" can be misleading to the model -> It contains some tracks that should belong to seine, trawl, lobster, or hook, but specific label could not be determined due to a lack of annotation.
  * ResNet is not as effective at vessel type prediction compared to activity: 90% Train Accuracy, 75% Validation Accuracy. Still overfitting. (Too much variance of track pattern within each vessel type?)
 
Next Steps:
* Incorporate XGBoost activity prediction model to assist the labeling of more activity data points (especially for fishing activities)
* Merge observations in the "fishing" class to one of the four predefined fishing activities
* Selectively perform data augmentation (eg. only perform data augmentation for the fishing activity classes)


Preprocessing pipeline:
* Songyu: push ais <-> radar detections matched data online
* Remove disrupted (turns out to be 8%) by finding any group with observation of >= 150 knots instantaneous speed


## April 15, W3

#### Insights from Summary Statistics Calc
* Some tracks, with various issues, disrupt summary statistics calculations (ref PPT)

#### Activity Data Progress:
* Preliminary screening: Removed all invalid tracks as well as tracks with fewer than 50 detections. Currently have 4366 Tracks.
* Tentatively defined the following labels: Transit, Low Speed, Stopped / Anchored, Work, Other / Unspecified, Fishing - Hook and Line, Fishing - Seine, Fishing - Trawl, and Fishing - Others
* For vessels traveling in a straight line (previosuly all manually tagged as transit), categoriezed avg speed >= 5kts as transit, and <5kts as low speed
* Categorized all kayaking activities as low speed
* Categorized fishing activities based on user submitted notes.
* Categorized vessles with an average speed of <= 1km/hr (0.54 kts) as stopped. Manually removed some mistakes the heuristics made.data
* May consider separating lobster fishing into its own category.

#### Questions and Notes:
* `radar_detection` file has attribute like speed, course etc. How does radar pick up these attributes? They seem to be really robust to coordinate disruptions
* Radar data seems to be quite noisy on some tracks, denoising techniques are applied to the radar detection time series data (basically a convolution with a gaussian kernel).

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


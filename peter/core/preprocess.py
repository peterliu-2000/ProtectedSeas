import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from utilities import DisruptionFilter, remove_low_count_tracks

"""
Directly Run this file to preprocess the radar detections 
Output:
    - peter/data/cleaned_data/preprocessed_radar_detections.csv

"""

def preprocess_data(ais_track_path, radar_detection_path, print_mode = False):
    """"
    Main preprocess pipeline on radar_detections:
        1. One-to-one matching between ais_tracks and radar_detections (results match with Songyu's)
        2. Remove tracks with less than 50 detections counts
        3. Remove tracks with max instant speed >= 150 knots

    Inputs: original ais_track and radar_detection data path
    Returns: preprocessed radar_detections
    """
    ais_tracks = pd.read_csv(ais_track_path)
    radar_detections = pd.read_csv(radar_detection_path)
    print(' ---- Read radar and AIS data ----')

    #add datetime
    radar_detections['datetime'] = pd.to_datetime(radar_detections['cdate'] + ' ' + radar_detections['ctime'])

    #only keep one-to-one matched radar/ais detections
    matched_detections = one_to_one_matching(ais_tracks, radar_detections)
    print(' ---- One to one matching complete ----')

    ##Remove tracks with less than 50 detections counts
    valid_detections = remove_low_count_tracks(matched_detections)
    print(' ---- Remove tracks with less than 50 detections counts ----')

    ##Remove disrupted tracks
    filterer = DisruptionFilter(valid_detections)
    non_disrupted_detections = filterer()
    print(' ---- Remove disrupted tracks ----')

    #grab type_m2 from ais_tracks & aggregate into type_m2_agg
    #drop unnecessary columns
    non_disrupted_detections.drop(columns=['latitude_prev', 'longitude_prev', 'time_prev', 'distance_diff', 'time_diff', 'instant_speed','disrupted', 'cdate', 'ctime'], inplace=True)
    print(' ---- Aggregate vessel type & unnecessary columns dropped ----')

    if print_mode:
        print('\n')
        print(f'Original radar detections: {len(radar_detections)} observations and {len(radar_detections["id_track"].unique())} unique tracks')
        print(f'After 1-1 matching towards ais tracks: {len(matched_detections)} observations and {len(matched_detections["id_track"].unique())} unique tracks')
        print(f'After removing tracks <50 observations: {len(valid_detections)} observations and {len(valid_detections["id_track"].unique())} unique tracks')
        print(f'After removing disrupted tracks: {len(non_disrupted_detections)} observations and {len(non_disrupted_detections["id_track"].unique())} unique tracks')

    return non_disrupted_detections

def one_to_one_matching(ais_tracks, radar_detections):

    radars = []
    radar_2_ais = {}
    
    #create a dictionary of radar ('assoc_id') -> (ais, strength) from ais_tracks
    for _, row in ais_tracks.iterrows():
        ais = row['id_track']
        assoc_radar = row['assoc_id']
        strength = row['assoc_str']  
        if assoc_radar not in radar_2_ais:
            radar_2_ais[assoc_radar] = set()
        radar_2_ais[assoc_radar].add((ais, strength))

    for _, row in tqdm(radar_detections.iterrows(), total=len(radar_detections), desc='ais/radar one-to-one matching'):
        radar = row['id_track']
        ais_in_radar = row['assoc_id']
        #only investigate radar ids that also appear in radar_detections
        if radar in radar_2_ais.keys():
            ais_strengths_pairs = radar_2_ais[radar]
            ais_candidates = [i[0] for i in ais_strengths_pairs]

            if len(ais_candidates) == 0:
                continue
            elif len(ais_candidates) == 1:
                ais_candidate = ais_candidates[0]
            #if multiple ais_candidates, choose the one with highest strength
            else:
                ais_strengths = list(ais_strengths_pairs)
                ais_strengths.sort(key=lambda x: x[1], reverse=True)
                ais_candidate = ais_strengths[0][0]
            
            #finally check ais_id in ais_tracks is the same as assoc_id in radar_detections for the same radar track
            if ais_candidate == ais_in_radar:
                radars.append(radar)
             
    matched_radar_detections = radar_detections[radar_detections['id_track'].isin(radars)]

    assert len(matched_radar_detections['id_track'].unique()) == len(matched_radar_detections['assoc_id'].unique()), 'one to one matching failed'
    assert np.isin(matched_radar_detections['assoc_id'].unique(), ais_tracks['id_track'].unique()).all(), 'some assoc_id is not in ais_tracks'
    assert np.isin(matched_radar_detections['id_track'].unique(), radar_detections['id_track'].unique()).all(), 'some id_track is not in radar_detections'

    return matched_radar_detections
    
def extract_type_label(ais_track_path):
    ais_tracks = pd.read_csv(ais_track_path)
    type_df = ais_tracks[['assoc_id', 'type_m2']]
    type_df.rename(columns={'assoc_id': 'id_track'}, inplace=True)
    return type_df


# Uncomment to preprocess radar detections
# if __name__ == '__main__':
#     print(' ---- Preprocessing radar detections ----')
#     ais_track_path = '../../data/tracks_ais.csv'
#     radar_detection_path = '../../data/detections_radar.csv'
#     processed_radar_data = preprocess_data(ais_track_path, radar_detection_path, print_mode=True)
#     save_path = '../../data/cleaned_data/preprocessed_radar_detections.csv'
#     processed_radar_data.to_csv(save_path, index=False)

#     ais_label_path = '../../data/ais_type_labels.csv'
#     ais_labels = extract_type_label(ais_track_path)
#     ais_labels.to_csv(ais_label_path, index=False)

if __name__ == "__main__":
    print(' ---- Preprocessing tagged detections ----')
    tagged_detections = pd.read_csv('../../data/raw_data/detections_tagged.csv')

    tagged_detections['datetime'] = pd.to_datetime(tagged_detections['datetime'])

    valid_detections = remove_low_count_tracks(tagged_detections)
    print(' ---- Remove tracks with less than 50 detections counts ----')

    non_disrupted_detections = DisruptionFilter(valid_detections)()
    print(f' ---- Remove disrupted tracks ----')

    non_disrupted_detections.drop(columns=['latitude_prev', 'longitude_prev', 'time_prev', 'distance_diff', 'time_diff', 'instant_speed','disrupted'], inplace=True)

    save_path = '../../data/cleaned_data/preprocessed_tagged_detections.csv'
    non_disrupted_detections.to_csv(save_path, index=False)

    print(f"Original tagged detections tracks: {tagged_detections['id_track'].nunique()}")
    print(f"After preprocessing: {non_disrupted_detections['id_track'].nunique()}")





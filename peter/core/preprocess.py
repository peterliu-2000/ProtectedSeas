import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

def preprocess_data(ais_track_path, radar_detection_path, print_mode = False):
    """"
    Main preprocess pipeline on radar_detections:
        1. One-to-one matching between ais_tracks and radar_detections (results match with Songyu's)
        2. Remove tracks with less than 50 detections counts
        3. Remove disrupted tracks if max instant speed >= 150 

    Inputs: original ais_track and radar_detection data path
    Returns: preprocessed radar_detections
    """
    ais_tracks = pd.read_csv(ais_track_path)
    radar_detections = pd.read_csv(radar_detection_path)

    #add datetime
    radar_detections['datetime'] = pd.to_datetime(radar_detections['cdate'] + ' ' + radar_detections['ctime'])

    #only keep one-to-one matched radar/ais detections
    matched_detections = one_to_one_matching(ais_tracks, radar_detections)

    ##Remove tracks with less than 50 detections counts
    counts = matched_detections.groupby('id_track').count()['assoc_id'].rename('detection_count').reset_index()
    valid_ids = counts[counts['detection_count'] >= 50]['id_track']
    valid_detections = matched_detections[matched_detections['id_track'].isin(valid_ids)]

    ##add datetime & remove disrupted tracks
    filterer = DisruptionFilter(valid_detections)
    non_disrupted_detections = filterer()

    #drop columns that are not needed
    final_detections = non_disrupted_detections.drop(columns=['latitude_prev', 'longitude_prev', 'time_prev', 'distance_diff', 'time_diff', 'instant_speed','disrupted'])

    if print_mode:
        print(f'Original radar detections: {len(radar_detections)} observations and {len(radar_detections["id_track"].unique())} unique tracks')
        print(f'After 1-1 matching towards ais tracks: {len(matched_detections)} observations and {len(matched_detections["id_track"].unique())} unique tracks')
        print(f'After removing tracks <50 observations: {len(valid_detections)} observations and {len(valid_detections["id_track"].unique())} unique tracks')
        print(f'After removing disrupted tracks: {len(non_disrupted_detections)} observations and {len(non_disrupted_detections["id_track"].unique())} unique tracks')

    return final_detections

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

import pandas as pd
import numpy as np
import warnings

class DisruptionFilter:
    KMPS_TO_KNOTS = 1943

    def __init__(self, valid_detections):
        self.valid_detections = valid_detections

    def __call__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.valid_detections = self.valid_detections.copy()

        valid_detections = self.valid_detections.groupby('id_track', group_keys=False).apply(self.filter_disrupted_tracks)
        return valid_detections[valid_detections['disrupted'] == False]

    def filter_disrupted_tracks(self, group):

        group = group.sort_values(by='datetime').copy()
        group['latitude_prev'] = group['latitude'].shift(1)
        group['longitude_prev'] = group['longitude'].shift(1)
        group['time_prev'] = group['datetime'].shift(1)
        group['distance_diff'] = self._haversine_distance(
            group['latitude_prev'], group['longitude_prev'],group['latitude'], group['longitude']
        )
        group['time_diff'] = (group['datetime'] - group['time_prev']).dt.total_seconds()
        group['instant_speed'] = group['distance_diff'] / group['time_diff'] * self.KMPS_TO_KNOTS
        
        if group['instant_speed'].max() >= 150:
            group['disrupted'] = True
        else:
            group['disrupted'] = False

        return group
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2, r=6371):
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return r * c
    
if __name__ == '__main__':
    ais_track_path = '../../data/tracks_ais.csv'
    radar_detection_path = '../../data/detections_radar.csv'
    processed_radar_data = preprocess_data(ais_track_path, radar_detection_path, print_mode=True)
    save_path = '../../data/cleaned_data/processed_radar_detections.csv'
    processed_radar_data.to_csv(save_path, index=False)

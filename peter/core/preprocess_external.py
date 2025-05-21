import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from DICT import CODE2TYPE
import pandas as pd
import numpy as np
import warnings
from utilities import DisruptionFilter, remove_low_count_tracks

"""
Directly Run this file to preprocess the AIS_2024_01_01.csv file.
Output stored in peter/data/external_ais/cleaned_AIS_2024_01_01.csv
"""

def preprocess_data(detections_path, print_mode = False):
    """"
    Main preprocess pipeline on radar_detections:
        1. One-to-one matching between ais_tracks and radar_detections (results match with Songyu's)
        2. Remove tracks with less than 50 detections counts
        3. Remove tracks with max instant speed >= 150 knots

    Inputs: original ais_track and radar_detection data path
    Returns: preprocessed radar_detections
    """
    detections = pd.read_csv(detections_path)
    detections['datetime'] = pd.to_datetime(detections['BaseDateTime'])

    #renaming columns
    detections.rename(columns={
        'MMSI':'id_track', 
        'LAT':'latitude', 
        'LON':'longitude',
        'SOG': 'speed',
        'COG': 'course'
    }, inplace=True)

    #convert numerical code to type_m2
    detections['type_m2'] = detections['VesselType'].map(CODE2TYPE)

    ##Remove tracks with less than 50 detections counts
    valid_detections = remove_low_count_tracks(detections)

    ##Remove disrupted tracks
    non_disrupted_detections = DisruptionFilter(valid_detections)()

    #drop unnecessary columns
    non_disrupted_detections.drop(columns = ['BaseDateTime', 'VesselType', 'IMO', 'CallSign', 'Status', 'Length', 'Width', 'Draft', 'Cargo'], inplace=True)
    non_disrupted_detections.drop(columns=['latitude_prev', 'longitude_prev', 'time_prev', 'distance_diff', 'time_diff', 'instant_speed','disrupted'], inplace=True)

    if print_mode:
        print('\n')
        print(f'Original radar detections: {len(detections)} observations and {len(detections["id_track"].unique())} unique tracks')
        print(f'After removing tracks <50 observations: {len(valid_detections)} observations and {len(valid_detections["id_track"].unique())} unique tracks')
        print(f'After removing disrupted tracks: {len(non_disrupted_detections)} observations and {len(non_disrupted_detections["id_track"].unique())} unique tracks')

    return non_disrupted_detections

if __name__ == '__main__':
    detections_path = '../../data/external_ais/AIS_2024_01_02.csv'
    preprocessed_detections = preprocess_data(detections_path, print_mode=True)
    save_path = '../../data/external_ais/cleaned_AIS_2024_01_02.csv'
    preprocessed_detections.to_csv(save_path, index=False)







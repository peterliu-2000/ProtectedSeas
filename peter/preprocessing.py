import pandas as pd
import numpy as np
import os

def preprocess_data(ais_track_path, radar_detection_path):
    ais_tracks = pd.read_csv(ais_track_path)
    radar_detections = pd.read_csv(radar_detection_path)


    
        

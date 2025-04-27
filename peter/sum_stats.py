import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import os

class SumStats:
    """
    Summary Statistics for Track Detections, revised implementation from ProtectedSeas
    Inputs: preprocessed radar detections
    Returns: summary df: each row is an id_track with summary statistics
    """
    def __init__(self, preprocessed_radar_detections):
        self.df = preprocessed_radar_detections
        
    def __call__(self):   
        return self.df.groupby('id_track', group_keys=False).apply(self.compute_sum_stats)
    
    def compute_track_stats(self, group):

        n_detect = len(group)
        group = group.sort_values(by='datetime').copy()

        start_lat = group['latitude'].iloc[0]
        start_lon = group['longitude'].iloc[0]
        group['latitude_prev'] = group['latitude'].shift(1)
        group['longitude_prev'] = group['longitude'].shift(1)
        group['distance_diff'] = SumStats._haversine_distance(
            group['latitude_prev'], group['longitude_prev'],
            group['latitude'], group['longitude']
        )
        group['time_diff'] = (group['datetime'] - group['time_prev']).dt.total_seconds()
        group['speed_diff'] = abs(group['speed'] - group['speed_prev'])
        
        # Course & Turning calculation
        course = group['course'].apply(lambda x: x - 360 if x > 180 else x) #restrict course to [-180, 180]
        turning = np.abs(course - course.shift(1))[1:]
        turning = turning.apply(lambda x: x - 180 if x > 180 else x) #restrict turning to [0, 180]
        
        #Distance Calculation
        group['distance_from_origin'] = self._haversine_distance(start_lat, start_lon, group['latitude'], group['longitude'])

        # Speed is in kts, Distance is in km, Heading / Turning in Deg
        track_summary = {
            "duration"       : group["datetime"].iloc[-1] - group["datetime"].iloc[0],
            "p95_speed"      : group["speed"].quantile(0.95),
            "p5_speed"      : group["speed"].quantile(0.05),
            "med_speed"      : group["speed"].median(),
            "curviness"      : np.sum(group['distance_diff']) / group['distance_from_origin'].quantile(0.95),
            "heading_mean"   : self._circular_mean(group['course']),
            "heading_std"    : self._circular_std(group['course']),
            "turning_mean"   : self._circular_mean(turning),
            "turning_std"    : self._circular_std(turning),
            "distance_total" : np.sum(group['distance_diff']),
            "distance_o"     : group['distance_from_origin'].quantile(0.95),
            "detections"     : n_detect
        }
        return pd.Series(track_summary)
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2, r = 6371):
        """
        Calculate the distance between two points on the Earth's surface
        using the Haversine formula.

        Args:
            lat1 (float): Latitude of the first point
            lon1 (float): Longitude of the first point
            lat2 (float): Latitude of the second point
            lon2 (float): Longitude of the second point
            r (float): Radius of the Earth in kilometers
            
        Returns:
            Distance in km
        """
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return r * c
    
    @staticmethod
    def _circular_mean(angles):
        """
        Calculate the circular mean of a list of angles.
        Inputs: angles(pd series): angles in degress
        Returns: circular mean of the angles in degrees
        """
        angles = angles * np.pi / 180
        x = np.cos(angles)
        y = np.sin(angles)
        return np.arctan2(np.sum(y), np.sum(x)) * 180 / np.pi

    @staticmethod
    def _circular_std(angles):
        """
        Calculate the circular standard deviation of a list of angles
        Args:
            angles(pd series): angles in degress
        Returns:
            float: circular std of the angles in degrees
        """
        angles = angles * np.pi / 180
        x = np.cos(angles).sum()
        y = np.sin(angles).sum()
        x /= len(angles)
        y /= len(angles)
        R = np.sqrt(x**2 + y**2)
        std = np.sqrt(-2 * np.log(R)) * 180 / np.pi
        return std


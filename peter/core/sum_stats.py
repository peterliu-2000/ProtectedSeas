import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

"""
SumStatsBaseline follows ProtectedSeas original implementation
SumStats2 is has revised implementations, also with more features

Usage:
    summary_df = SumStatsBaseline(preprocessed_radar_detections)()

"""

class SumStatsBaseline:
    """
    Summary Statistics for Track Detections, revised implementation from ProtectedSeas
    Inputs: preprocessed radar detections, preferably with type/activity columns
    Returns: summary df: each row is an id_track with summary statistics
    """

    FEATURE_NAMES = [
        "p95_speed",
        "p5_speed",
        "median_speed",
        "heading_mean",
        "heading_std",
        "turning_mean",
        "turning_std",
        "curviness",
        "distance_o"
    ]

    def __init__(self, preprocessed_radar_detections):
        self.df = preprocessed_radar_detections
        
    def __call__(self): 
        summary_df = self.df.groupby('id_track', group_keys=False).apply(self.compute_track_stats)
        return summary_df.reset_index()
    
    def compute_track_stats(self, group):
        n_detect = len(group)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            group['datetime'] = pd.to_datetime(group['datetime'], errors='coerce')
        group = group.sort_values(by='datetime').copy()

        start_lat = group['latitude'].iloc[0]
        start_lon = group['longitude'].iloc[0]
        group['latitude_prev'] = group['latitude'].shift(1)
        group['longitude_prev'] = group['longitude'].shift(1)
        group['distance_diff'] = self._haversine_distance(
            group['latitude_prev'], group['longitude_prev'],
            group['latitude'], group['longitude']
        )
        group['time_prev'] = group['datetime'].shift(1)
        group['time_diff'] = (group['datetime'] - group['time_prev']).dt.total_seconds()
        
        # Course & Turning calculation
        course = group['course'].apply(lambda x: x - 360 if x > 180 else x) #restrict course to [-180, 180]
        headingCos = np.cos(course * np.pi / 180)
        headingSin = np.sin(course * np.pi / 180)
        meanCos = np.mean(headingCos)
        meanSin = np.mean(headingSin)
        avg_heading = np.arctan2(meanSin, meanCos) * 180 / np.pi
        R_squared = np.clip(meanCos**2 + meanSin**2, 1e-10, 1)
        std_heading = np.sqrt(-np.log(R_squared))
        if std_heading == np.nan: std_heading = 0
        
        # Compute the mean and std of these trig vectors
        turning = np.abs(course - course.shift(1))[1:]
        
        #Distance Calculation
        group['distance_from_origin'] = self._haversine_distance(start_lat, start_lon, group['latitude'], group['longitude'])
        total_distance = np.sum(group['distance_diff'])
        distance_o = group['distance_from_origin'].max()
        curviness = total_distance / distance_o if distance_o != 0 else 0

        # Speed is in kts, Distance is in km, Heading / Turning in Deg
        track_summary = {
            "duration"       : group["datetime"].iloc[-1] - group["datetime"].iloc[0],
            "distance_total" : np.sum(group['distance_diff']),
            "detections"     : n_detect,
            "p95_speed"      : np.percentile(group["speed"], 95),
            "p5_speed"       : np.percentile(group["speed"], 5),
            "median_speed"   : group["speed"].median(),
            "heading_mean"   : avg_heading,
            "heading_std"    : std_heading,
            "turning_mean"   : np.mean(turning),
            "turning_std"    : np.std(turning),
            "distance_o"     : distance_o,
            "curviness"      : curviness
        }

        return pd.Series(track_summary)
    
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


class SumStats2:
    """
    Summary Statistics for Track Detections, revised implementation from ProtectedSeas
    Inputs: preprocessed radar detections
    Returns: summary df: each row is an id_track with summary statistics
    """
    # Class attribute to store feature names
    FEATURE_NAMES = [
        "max_speed",
        "min_speed",
        "avg_speed",
        "curviness",
        "heading_mean",
        "heading_std",
        "turning_mean",
        "turning_std",
        "distance_total",
        "distance_o"
    ]

    def __init__(self, preprocessed_radar_detections):
        self.df = preprocessed_radar_detections
        
    def __call__(self): 
        summary_df = self.df.groupby('id_track', group_keys=False).apply(self.compute_track_stats)
        return summary_df.reset_index()
    
    def compute_track_stats(self, group):
        n_detect = len(group)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            group['datetime'] = pd.to_datetime(group['datetime'], errors='coerce')
        group = group.sort_values(by='datetime').copy()

        start_lat = group['latitude'].iloc[0]
        start_lon = group['longitude'].iloc[0]
        group['latitude_prev'] = group['latitude'].shift(1)
        group['longitude_prev'] = group['longitude'].shift(1)
        group['distance_diff'] = self._haversine_distance(
            group['latitude_prev'], group['longitude_prev'],
            group['latitude'], group['longitude']
        )
        group['time_prev'] = group['datetime'].shift(1)
        group['time_diff'] = (group['datetime'] - group['time_prev']).dt.total_seconds()
        
        # Course & Turning calculation
        course = group['course'].apply(lambda x: x - 360 if x > 180 else x) #restrict course to [-180, 180]
        turning = np.abs(course - course.shift(1))[1:]
        
        #Distance Calculation
        group['distance_from_origin'] = self._haversine_distance(start_lat, start_lon, group['latitude'], group['longitude'])
        total_distance = np.sum(group['distance_diff'])
        distance_o = group['distance_from_origin'].max()
        curviness = total_distance / distance_o

        #Create various speed features; 0-2, 2-6, 6-12, 12-20
        speed_0_to_2, speed_2_to_6, speed_6_to_12, speed_12_to_20 = [], [], [], []
        for speed in group['speed']:
            if speed >= 0 and speed <= 2:
                speed_0_to_2.append(speed)
            elif speed > 2 and speed <= 6:
                speed_2_to_6.append(speed)
            elif speed > 6 and speed <= 12:
                speed_6_to_12.append(speed)
            else:
                speed_12_to_20.append(speed)

        f1, f2, f3, f4 = self._extract_speed_subfeatures(speed_0_to_2)
        f5, f6, f7, f8 = self._extract_speed_subfeatures(speed_2_to_6)
        f9, f10, f11, f12 = self._extract_speed_subfeatures(speed_6_to_12)
        f13, f14, f15, f16 = self._extract_speed_subfeatures(speed_12_to_20)

        # Speed is in kts, Distance is in km, Heading / Turning in Deg
        track_summary = {
            "duration"       : group["datetime"].iloc[-1] - group["datetime"].iloc[0],
            "detections"     : n_detect,
            "distance_total" : np.sum(group['distance_diff']),
            "max_speed"      : group["speed"].max(),
            "min_speed"      : group["speed"].min(),
            "avg_speed"      : group["speed"].mean(),
            "curviness"      : curviness,
            "heading_mean"   : self._circular_mean(group['course']),
            "heading_std"    : self._circular_std(group['course']),
            "turning_mean"   : self._circular_mean(turning),
            "turning_std"    : self._circular_std(turning),
            "distance_o"     : distance_o,
            'f1'             : f1,
            'f2'             : f2,
            'f3'             : f3,
            'f4'             : f4,
            'f5'             : f5,
            'f6'             : f6,
            'f7'             : f7,  
            'f8'             : f8,
            'f9'             : f9,
            'f10'            : f10,
            'f11'            : f11,
            'f12'            : f12,
            'f13'            : f13,
            'f14'            : f14,
            'f15'            : f15, 
            'f16'            : f16
        }

        if 'type_m2' in group.columns:
            track_summary['type_m2'] = group['type_m2'].iloc[0]
        if 'activity' in group.columns:
            track_summary['activity'] = group['activity'].iloc[0]
        return pd.Series(track_summary)
    
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
    
    def _extract_speed_subfeatures(self, speeds):
        if len(speeds) == 0:
            return 0, 0, 0, 0
        f1 = np.median(speeds)
        f2 = np.quantile(speeds, 0.25)
        f3 = np.quantile(speeds, 0.75)
        f4 = np.std(speeds)
        return f1, f2, f3, f4
    

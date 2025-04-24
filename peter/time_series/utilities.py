import numpy as np
import warnings
import pandas as pd

class VesselTypeAggregator:
    """
    Aggregate vessel type based on defined categories from insights
    """
    
    def __init__(self):
        self.type_mapping = {
            'tanker_ship': 'cargo/tanker',
            'cargo_ship': 'cargo/tanker',
            'tug': 'tug/tow',
            'towing_ship': 'tug/tow',
            'fishing_boat': 'fishing_boat',
            'commercial_fishing_boat': 'fishing_boat',
            'military_ship': 'military_ship',
            'class_b':'class_b',
            'passenger_ship': 'passenger_ship',
            'pleasure_craft': 'pleasure_craft',
            'sailboat': 'pleasure_craft',
            'search_and_rescue_boat': 'other',
            'pilot_boat': 'other',
            'high_speed_craft': 'other',
            'law_enforcement_boat': 'other',
            'other': 'other',
            'unknown': 'other'
        }

    def aggregate_vessel_type(self, df):
        """
        Args:
            df: dataframe with 'type_m2' column
        """

        df['type_m2_agg'] = df['type_m2'].map(self.type_mapping).fillna('other')

class SumStats:
    columns = ['id_detect', 'id_track', 'id_site', 'id_m2', 'source', 'speed','course', 'assoc_str', 'assoc_id', 'confidence', 'cdate', 'ctime',
        'longitude', 'latitude']    

    def __init__(self):
        pass

    @staticmethod
    def compute_track_features(group, KMPS_TO_KNOTS=1943.84449):
        """Helper function to compute track features for a single group"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            group['datetime'] = pd.to_datetime(group['cdate'] + ' ' + group['ctime'])

        group = group.sort_values(by='datetime').copy()
        group['latitude_prev'] = group['latitude'].shift(1)
        group['longitude_prev'] = group['longitude'].shift(1)
        group['time_prev'] = group['datetime'].shift(1)
        group['speed_prev'] = group['speed'].shift(1)
        group['course_prev'] = group['course'].shift(1)
        
        # Calculate differences between consecutive detections
        group['distance_diff'] = SumStats._haversine_distance(
            group['latitude_prev'], group['longitude_prev'],
            group['latitude'], group['longitude']
        )
        group['time_diff'] = (group['datetime'] - group['time_prev']).dt.total_seconds()
        group['speed_diff'] = abs(group['speed'] - group['speed_prev'])
        group['heading_diff'] = abs(group['course'] - group['course_prev'])
        
        # Calculate instantaneous speed
        group['instant_speed'] = group['distance_diff'] / group['time_diff'] * KMPS_TO_KNOTS
        return group

    @classmethod
    def compute_track_stats(cls, group, KMPS_TO_KNOTS = 1943.84449):
        """
        Takes df of detections for a single track and computes summary statistics.
        For a full detection dataset, use df.groupby('id_track').apply(SumStats.compute_track_stats).reset_index()
        """
        
        # Use compute_track_features to calculate necessary features
        group = cls.compute_track_features(group, KMPS_TO_KNOTS)

        # Calculate the summary statistics based on the features
        start_lat = group['latitude'].iloc[0]
        start_lon = group['longitude'].iloc[0]
        end_lat = group['latitude'].iloc[-1]
        end_lon = group['longitude'].iloc[-1]

        total_distance = group['distance_diff'].sum()
        total_time = (group['datetime'].iloc[-1] - group['datetime'].iloc[0]).total_seconds()

        if total_time <= 0:
            print(f'Problematic track with 0 total time: id_track ={group["id_track"].iloc[0]}')

        avg_speed = group['speed'].mean()
        max_speed = group['speed'].max()
        min_speed = group['speed'].min()
        
        curviness = total_distance / cls._haversine_distance(start_lat, start_lon, end_lat, end_lon)
        heading_std = cls._circular_std(group['course'])
        turning_mean = cls._circular_mean(group['heading_diff'])
        turning_std = cls._circular_std(group['heading_diff'])
        
        return pd.Series({
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'total_distance': total_distance,
            'curviness': curviness,
            'heading_std': heading_std,
            'turning_mean': turning_mean,
            'turning_std': turning_std
        })

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
        Args:
            angles(pd series): angles in degress
        Returns:
            float: circular mean of the angles in degrees
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



def haversine_distance(lat1, lon1, lat2, lon2, r = 6371):
    """
    Calculate the distance between two points on the Earth's surface
    using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point
        lon1 (float): Longitude of the first point
        lat2 (float): Latitude of the second point
        lon2 (float): Longitude of the second point
        r (float): Radius of the Earth in kilometers
    """
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c
    
   
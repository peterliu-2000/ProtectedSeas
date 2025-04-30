import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

class SumStats:
    """
    Summary Statistics for Track Detections. 
    Initialize by s = SumStats(detections_dataframe)
    Then obtain track features by s(track_id)
    """
    # Data Frame Functions
    def __init__(self, detections_df:pd.DataFrame):
        # Must be a pre-processed dataframe with the time column
        self.df = detections_df
        
    def __call__(self, track_id):
        # A wrapper function for get_trajectory and compute_track_features
        detections = self.get_trajectory(self.df, track_id)
        return self.compute_track_features(detections)
        
    @staticmethod
    def get_trajectory(trajectory_df, track_id):
        """
        Obtain the trajectory list of a given track id

        Args:
            trajectory_df: dataframe for trajectory
            track_id: track id
        
        Returns:
        A list of [[latitude, longitude]] in sorted time order.
        """
        detections = trajectory_df[trajectory_df["id_track"] == track_id]
        detections_sorted = detections.sort_values(by = "time", ascending=True)
        return detections_sorted
    
    # Additional Helper Mathematical Functions:
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

    def compute_track_features(self, detections:pd.DataFrame):
        """
        Computes the track features from a detection dataframe

        Args:
            detections: Detections of a single track

        Returns:
            A pandas Series for track features
        """
        n_detect = len(detections)
        
        # Correct the course vector to be in range [-180, 180]
        course = detections["course"].apply(lambda x: x - 360 if x > 180 else x)
        headingCos = np.cos(detections["course"] * np.pi / 180)
        headingSin = np.sin(detections["course"] * np.pi / 180)
        
        # Compute the mean and std of these trig vectors
        meanCos = np.mean(headingCos)
        meanSin = np.mean(headingSin)
        # Compute the average heading and heading standard deviation
        avg_heading = np.arctan2(meanSin, meanCos) * 180 / np.pi
        std_heading = np.sqrt(-np.log(meanCos*meanCos + meanSin*meanSin))
        if std_heading == np.nan: std_heading = 0
        
        # Compute some deltas (length: n_detect - 1)
        delta_course1 = np.abs(course - course.shift(1))[1:]
        delta_course = np.where(delta_course1 < 180, delta_course1, delta_course1 - 180)
        
        # Compute pointwise distances
        def dist_helper_1(i):
            return self._haversine_distance(
                detections["latitude"].iloc[i], detections["longitude"].iloc[i],
                detections["latitude"].iloc[i+1], detections["longitude"].iloc[i+1]
            )
        
        
        distances = [dist_helper_1(i) for i in range(n_detect - 1)]
        
        # Compute the distances from the origin:
        def dist_helper_2(i):
            return self._haversine_distance(
                detections["latitude"].iloc[0], detections["longitude"].iloc[0],
                detections["latitude"].iloc[i+1], detections["longitude"].iloc[i+1]
            )
        distances_from_origin = [dist_helper_2(i) for i in range(n_detect - 1)]
        
        # Compute the straight line distance between start and end points
        straight_distance = self._haversine_distance(
                detections["latitude"].iloc[0], detections["longitude"].iloc[0],
                detections["latitude"].iloc[-1], detections["longitude"].iloc[-1]
            )

        # Speed is in kts, Distance is in km, Heading / Turning in Deg
        track_summary = {
            "duration"      : detections["time"].iloc[-1] - detections["time"].iloc[0],
            "max_speed"     : detections["speed"].max(),
            "min_speed"     : detections["speed"].min(),
            "avg_speed"     : detections["speed"].mean(),
            "curviness"     : np.sum(distances) / straight_distance,
            "heading_mean"  : avg_heading,
            "heading_std"   : std_heading,
            "turning_mean"  : np.mean(delta_course),
            "turning_std"   : np.std(delta_course),
            "distance"      : np.sum(distances),
            "distance_o"    : np.max(distances_from_origin),
            "detections"    : n_detect
        }
        return pd.Series(track_summary)
             

if __name__ == "__main__":
    from data_ops import *
    try:
        detections_file = "../../data/detections_radar_cached.csv"
        df = init_detections_df(detections_file)
    except Exception:
        detections_file = "../../data/detections_radar.csv"
        df = init_detections_df(detections_file)
    s = SumStats(df)
    ids = df["id_track"].unique()
    id = np.random.choice(ids)
    print(s(id))
        

import pandas as pd
import numpy as np

class SumStats:
    """
    Summary Statistics for Track Detections. 
    Initialize by s = SumStats(detections_dataframe)
    Then obtain track features by s(track_id)
    """
    # Data Frame Functions
    def __init__(self):
        pass
        
    def __call__(self, detections):
        return self.compute_track_features(detections)
    
    def init_summary_dataframe(self, length : int):
        """
        Returns an empty summary statistic dataframe
        """
        return pd.DataFrame(columns = [
            "duration","max_speed","min_speed","avg_speed","curviness","heading_mean",
            "heading_std","turning_mean","turning_std","distance_total",
            "distance_o","detections","p95_speed","p5_speed","med_speed",
            "start_time", "end_time"], index = range(length))
        
        
    def compute_all_summaries(self, detections:pd.DataFrame, track_ids = None):
        if track_ids is not None:
            summary_df = detections[detections["id_track"].isin(track_ids)]
        summary_df = detections.groupby("id_track").apply(self.compute_track_features)
        summary_df = summary_df.sort_values(by = "id_track", ascending=True) # Sort by track id so it matches with the program implementation
        return summary_df
        
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
        course = detections["course"].apply(lambda x: x - 380 if x > 180 else x)
        headingCos = np.cos(course * np.pi / 180)
        headingSin = np.sin(course * np.pi / 180)
        
        # Compute the mean and std of these trig vectors
        meanCos = np.mean(headingCos)
        meanSin = np.mean(headingSin)
        # Compute the average heading and heading standard deviation
        avg_heading = np.arctan2(meanSin, meanCos) * 180 / np.pi
        std_heading = np.sqrt(-np.log(meanCos*meanCos + meanSin*meanSin))
        if std_heading == np.nan: std_heading = 0
        
        # Compute some deltas (length: n_detect - 1)
        delta_course = np.abs(course - course.shift(1))[1:]
        delta_course = np.where(delta_course < 180, delta_course, delta_course - 180)
        
        # Compute pointwise distances
        
        latitude_prev = detections["latitude"].shift(1)[1:]
        longitude_prev = detections["longitude"].shift(1)[1:]
        distances = self._haversine_distance(latitude_prev, longitude_prev,
                                             detections["latitude"].iloc[1:],
                                             detections["longitude"].iloc[1:])
        
        # Compute the distances from the origin:
        distances_from_origin = self._haversine_distance(
            detections["latitude"].iloc[0], detections["longitude"].iloc[0],
            detections["latitude"], detections["longitude"]
        )
        
        # Compute the straight line distance between start and end points
        straight_distance = self._haversine_distance(
                detections["latitude"].iloc[0], detections["longitude"].iloc[0],
                detections["latitude"].iloc[-1], detections["longitude"].iloc[-1]
        )
        
        if np.allclose(straight_distance, 0): # Correct for division by 0
            straight_distance = straight_distance + 1e-8

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
            "distance_total": np.sum(distances),
            "distance_o"    : np.max(distances_from_origin),
            "detections"    : n_detect,
            "p95_speed"     : detections["speed"].quantile(0.95),
            "p5_speed"      : detections["speed"].quantile(0.05),
            "med_speed"     : detections["speed"].median(),
            "start_time"    : detections["time"].iloc[0],
            "end_time"      : detections["time"].iloc[-1]
        }
        return pd.Series(track_summary, dtype = pd.Float64Dtype())
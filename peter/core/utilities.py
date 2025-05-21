import warnings
import numpy as np

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
    
def remove_low_count_tracks(detections):
    counts = detections.groupby('id_track').count()['latitude'].rename('detection_count').reset_index()
    valid_ids = counts[counts['detection_count'] >= 50]['id_track']
    valid_detections = detections[detections['id_track'].isin(valid_ids)]
    return valid_detections
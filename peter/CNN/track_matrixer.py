import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
"""
Changes from track_rasterizer.py:
1. add more input channels: speed std, heading std. Output is 5, H, W
2. Remove speed_ceil; default to no bias
"""

# prevent image from being zoomed in too mcuh
MIN_PIXEL_SPAN = 2.5e-5 

class TrajectoryMatrixer():
    def __init__(self, matrix_width, matrix_height):
        """
        Initialize the TrackRasterizer with specified dimensions.
        
        Args:
            matrix_width (int): Width of the output matrix
            matrix_height (int): Height of the output matrix
        """
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height
        
    def standardize_coordinates(self, lats, longs, min_lat_span = None,
                                min_long_span = None):
        """
        Standardize the latitude and longitude coordinates to range (0,1)
        This implementation preserves aspect ratio of the trajectory

        Args:
            lats: A vector of latitude coordinates
            longs: A vector of longitude coordinates
        """
        img_aspect_ratio = self.matrix_width / self.matrix_height
        
        lat_max, lat_min = np.max(lats), np.min(lats)
        long_max, long_min = np.max(longs), np.min(longs)
        lat_span, long_span = lat_max - lat_min, long_max - long_min
        
        if min_lat_span is None:
            min_lat_span = self.matrix_height * MIN_PIXEL_SPAN
        if min_long_span is None:
            min_long_span = self.matrix_width * MIN_PIXEL_SPAN
            
        # Now compute the correct latitude and longitude spans,
        # after accounting for the image aspect ratio and minimum span
        corrected_lat_span = np.max([min_lat_span, lat_span, long_span/img_aspect_ratio])
        corrected_long_span = np.max([min_long_span, long_span, lat_span*img_aspect_ratio])
        
        # Now correct the minimum value of lat long
        lat_mid = (lat_max + lat_min) / 2
        lat_start = lat_mid - 0.5 * corrected_lat_span
        long_mid = (long_max + long_min) / 2
        long_start = long_mid - 0.5 * corrected_long_span
        
        # Now standardize the corrdinates to be in range [0, 1]
        lat_standardized = (lats - lat_start) / corrected_lat_span
        long_standardized = (longs - long_start) / corrected_long_span
        return lat_standardized, long_standardized
    
    def assign_pixel_position(self, lat_std, long_std):
        """
        Assign the pixel position within the rasterized image for each detection point

        Args:
            lat_std: standardized latitude
            long_std: standardized longitude
        
        Returns:
            An numpy array of shape (N, 2) with the row and column indices
        """
        # We require the row / column indices to be of int type
        row_idxs = np.minimum(np.floor(lat_std * self.matrix_height), self.matrix_height - 1).astype(np.int64)
        col_idxs = np.minimum(np.floor(long_std * self.matrix_width), self.matrix_width - 1).astype(np.int64)
        return np.vstack((row_idxs, col_idxs)).T # -> N, 2
        
    def aggregate_pixels(self, speed, heading, turning, indices):
        """
        Aggregate the pixel level information

        Args:
            speed (np.array): Trajectory Speed data
            heading (np.array): Trajectory Heading data
            turning (np.array): The turning (change of course) vector, value between 0 and 180
            indices (np.array): row, col indices returned by assign_pixel_position
        
        Returns:
            A 3D tensor with 5 channels (count, average_speed, max_speed, average_heading, heading_std)
        """
        # Uses PyTorch indexing convension (C, H, W)
        result = np.zeros((5, self.matrix_height, self.matrix_width), dtype = np.float32)
        dict = np.empty((self.matrix_height, self.matrix_width), dtype = object)

        # C = 0 for count
        # C = 1 for average Speed
        # C = 2 for average turning
        # C = 3 for speed std
        # C = 4 for heading std
        
        for i in range(len(indices)):
            r, c = indices[i, 0], indices[i, 1]
            if dict[r, c] is None:
                dict[r, c] = {
                    'count': 1,
                    'speed': np.array([speed[i]]),
                    'heading': np.array([heading[i]]),
                    'turning': np.array([turning[i]])
                }
            else:
                dict[r, c]['count'] += 1
                dict[r, c]['speed'] = np.append(dict[r, c]['speed'], speed[i])
                dict[r, c]['heading'] = np.append(dict[r, c]['heading'], heading[i])
                dict[r, c]['turning'] = np.append(dict[r, c]['turning'], turning[i])
        
        for r in range(dict.shape[0]):
            for c in range(dict.shape[1]):
                if dict[r, c] is not None:
                    result[0, r, c] = dict[r, c]['count']
                    result[1, r, c] = np.mean(dict[r, c]['speed'])
                    result[2, r, c] = self._circular_mean(dict[r, c]['turning'])
                    result[3, r, c] = np.std(dict[r, c]['speed'])
                    result[4, r, c] = self._circular_std(dict[r, c]['heading'])

        return result # -> 5, H, W
    
    def _circular_mean(self, angles):
        """
        Inputs:
            angles(pd series): of angles in degrees
        Returns:
            float: circular mean of the angles in degrees
        """
        angles = angles * np.pi / 180
        x = np.cos(angles)
        y = np.sin(angles)
        return np.arctan2(np.sum(y), np.sum(x)) * 180 / np.pi

    def _circular_std(self, angles):
        """
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
    
    def normalize_matrix(self, agg, speed_ceil = 22.5, bias = True):
        """
        Inputs:
            agg (np.array): 5, H, W; returned by aggregate_pixels
            speed_ceil (float): maximum speed value for normalization
            bias (bool): whether to apply bias
        Returns:
            np.ndarray: 5, H, W; normalized matrix
        """
 
        # First computes the dynamic range of the image (Bias takes RGB value of 30)
        dynamic_range = 225 if bias else 255
        
        agg[0, :, :] = np.minimum(agg[0, :, :], dynamic_range)
        # speed mean
        agg[1, :, :] = agg[1, :, :] * dynamic_range / speed_ceil
        agg[1, :, :] = np.minimum(agg[1, :, :], dynamic_range)
        # turning mean
        agg[2, :, :] *= dynamic_range / 180
        agg[2, :, :] = np.minimum(agg[2, :, :], dynamic_range)
        # speed std
        agg[3, :, :] = np.minimum(agg[3, :, :], dynamic_range)
        agg[3, :, :] = agg[3, :, :] * dynamic_range / speed_ceil
        # heading std
        agg[4, :, :] = agg[4, :, :] * dynamic_range / 180

        # Add a bias term to all detection points in the image
        if bias:
            mask = agg[0, :, :] > 0
            agg += mask * 30

        return agg # -> 5, H, W
    
    def rasterize(self, track_data, speed_ceil=None, bias=False):
        """
        Rasterize track data into a matrix representation.
        
        Args:
            track_data (pd.DataFrame): Track data with columns ['longitude', 'latitude', 'speed', 'course']
            speed_ceil (float): Maximum speed value for normalization
            bias (bool): Whether to apply bias in the rasterization
            
        Returns:
            np.ndarray: Rasterized track matrix
        """
        # Create empty matrix
        matrix = np.zeros((self.matrix_height, self.matrix_width))
        
        # Normalize coordinates to matrix dimensions
        lons = track_data['longitude'].values
        lats = track_data['latitude'].values
        
        # Scale coordinates to matrix dimensions
        x_coords = ((lons - lons.min()) / (lons.max() - lons.min()) * (self.matrix_width - 1)).astype(int)
        y_coords = ((lats - lats.min()) / (lats.max() - lats.min()) * (self.matrix_height - 1)).astype(int)
        
        # Fill matrix with speed values
        speeds = track_data['speed'].values
        if speed_ceil is not None:
            speeds = np.clip(speeds, 0, speed_ceil)
            speeds = speeds / speed_ceil
        
        for x, y, speed in zip(x_coords, y_coords, speeds):
            if 0 <= x < self.matrix_width and 0 <= y < self.matrix_height:
                matrix[y, x] = speed
                
        if bias:
            matrix = self._apply_bias(matrix)
            
        return matrix

class VesselTrajectoryMatrixer(TrajectoryMatrixer):
    def __init__(self, matrix_width, matrix_height, trajectory_data: pd.DataFrame):
        super().__init__(matrix_width, matrix_height)
        self.data = trajectory_data
    
    def get_track(self, track_id):
        detections = self.data[self.data["id_track"] == track_id]
        detections = detections.sort_values(by = "datetime", ascending=True)
        n = len(detections)
        if n == 0: raise RuntimeError(f"Track id {track_id} has empty record")
        
        # Compute the turning vector, the turning at the initial point is always
        turning = np.nan_to_num(np.abs(detections["course"] - detections["course"].shift(1)), nan = 0.0)
        turning = np.where(turning < 180, turning, 360 - turning)
        
        # We only need to keep track of these:
        return {
            "lats" : detections["latitude"].to_numpy(),
            "longs" : detections["longitude"].to_numpy(),
            "speed" : detections["speed"].to_numpy(),
            "heading" : detections["course"].to_numpy(),
            "turning" : turning
        }
        
    def __call__(self, track_id, speed_ceil=22.5, bias=True):
        """
        Perform an entire sequence for vessel trajectory rasterization

        Args:
            track_id: 
            speed_ceil: Clamp value for speed channels. Defaults to 22.5.
            bias: Add a bias to each pixel with detection point. Default to True.
        """
        data = self.get_track(track_id)
        lat, long = self.standardize_coordinates(data["lats"], data["longs"])
        pixel_idx = self.assign_pixel_position(lat, long) #N,2
        agg_np = self.aggregate_pixels(data["speed"], data["heading"], data["turning"], pixel_idx)
        agg_np = self.normalize_matrix(agg_np, speed_ceil, bias)

        return agg_np # -> 5, H, W

if __name__ == "__main__":
    radar_detections_path = '../../data/cleaned_data/preprocessed_radar_detections.csv'
    radar_detections = pd.read_csv(radar_detections_path)

    MATRIX_WIDTH = 224
    MATRIX_HEIGHT = 224
    matrixer = VesselTrajectoryMatrixer(MATRIX_WIDTH, MATRIX_HEIGHT, radar_detections)
    save_path = 'track_matrices/'
    track_ids = list(radar_detections["id_track"].unique())

    #Save matrix for ALL tracks
    for id in tqdm(track_ids):
        matrix = matrixer(id, speed_ceil=22.5, bias=True)
        torch.save(torch.from_numpy(matrix), save_path + f"{id}.pt")

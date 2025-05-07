import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

"""
Directly Run this file to rasterize the vessel trajectories
"""

# prevent image from being zoomed in too mcuh
MIN_PIXEL_SPAN = 2.5e-5 

class TrajectoryRasterize():
    def __init__(self, image_width, image_height):
        self.width = image_width
        self.height = image_height
        
    def standardize_coordinates(self, lats, longs, min_lat_span = None,
                                min_long_span = None):
        """
        Standardize the latitude and longitude coordinates to range (0,1)
        This implementation preserves aspect ratio of the trajectory

        Args:
            lats: A vector of latitude coordinates
            longs: A vector of longitude coordinates
        """
        img_aspect_ratio = self.width / self.height
        
        lat_max, lat_min = np.max(lats), np.min(lats)
        long_max, long_min = np.max(longs), np.min(longs)
        lat_span, long_span = lat_max - lat_min, long_max - long_min
        
        if min_lat_span is None:
            min_lat_span = self.height * MIN_PIXEL_SPAN
        if min_long_span is None:
            min_long_span = self.width * MIN_PIXEL_SPAN
            
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
        row_idxs = np.minimum(np.floor(lat_std * self.height), self.height - 1).astype(np.int64)
        col_idxs = np.minimum(np.floor(long_std * self.width), self.width - 1).astype(np.int64)
        return np.vstack((row_idxs, col_idxs)).T # -> N, 2
        
    def aggregate_pixels(self, speed, turning, indices):
        """
        Aggregate the pixel level information

        Args:
            speed: Trajectory Speed data
            turning: The turning (change of course) vector, value between 0 and 180
            indices: row, col indices returned by assign_pixel_position
        
        Returns:
            A 3D tensor with 3 channels (count, average_speed, max_speed)
        """
        # Uses PyTorch indexing convension (C, H, W)
        result = np.zeros((3, self.height, self.width), dtype = np.float32)
        # C = 0 for count
        # C = 1 for average Speed
        # C = 2 for average turning
        
        for i in range(len(indices)):
            r, c = indices[i, 0], indices[i, 1]
            prev_count = result[0, r, c]
            # Increment count, aggregate average and update max
            result[0, r, c] += 1
            result[1, r, c] = (prev_count * result[1, r, c] + speed[i])/(prev_count + 1)
            result[2, r, c] = (prev_count * result[2, r, c] + turning[i])/(prev_count + 1) 
        return result
    
    def to_image(self, agg, speed_ceil = 22.5, bias = True):
        """
        Standardize the tensor values given by aggregate_pixels to be integer
        values between 0 and 255.

        Args:
            agg: object returned by aggregate_pixels
            speed_ceil: The ceiling value for speed. Defaults to 22.5
            bias: Add a bias to add pixels with detection point. Defaults to True
            
        Returns:
            A numpy array with (H, W, C) layout with an int dtype.
        """
        
        # First computes the dynamic range of the image (Bias takes RGB value of 30)
        dynamic_range = 225 if bias else 255
        
        # Simply clip all count values higher than 255 to 255
        agg[0, :, :] = np.minimum(agg[0, :, :], dynamic_range)
        # Computes the step for speed values
        agg[1, :, :] = agg[1, :, :] * dynamic_range / speed_ceil
        # Clip the speed values to 255
        agg[1, :, :] = np.minimum(agg[1, :, :], dynamic_range)
        # Normalize the turning values to range (0,255)
        agg[2, :, :] *= dynamic_range / 180
        
        # Add a bias term to all detection points in the image
        if bias:
            mask = agg[0, :, :] > 0
            agg += mask * 30

        return np.moveaxis(agg.astype(np.uint8), 0, -1)
    
class VesselTrajectoryRasterize(TrajectoryRasterize):
    def __init__(self, image_width, image_height, trajectory_data: pd.DataFrame):
        super().__init__(image_width, image_height)
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
            "turning" : turning
        }
        
    def __call__(self, track_id, speed_ceil = 22.5, bias = True):
        """
        Perform an entire sequence for vessel trajectory rasterization

        Args:
            track_id: 
            speed_ceil: Clamp value for speed channels. Defaults to 25.5.
            bias: Add a bias to each pixel with detection point. Default to True.
        """
        data = self.get_track(track_id)
        lat, long = self.standardize_coordinates(data["lats"], data["longs"])
        pixel_idx = self.assign_pixel_position(lat, long) #N,2
        agg_np = self.aggregate_pixels(data["speed"], data["turning"], pixel_idx)
        return self.to_image(agg_np, speed_ceil, bias)

if __name__ == "__main__":
    radar_detections_path = '../../data/cleaned_data/preprocessed_radar_detections.csv'
    radar_detections = pd.read_csv(radar_detections_path)
    
    rasterizer = VesselTrajectoryRasterize(image_width=224, image_height=224, trajectory_data=radar_detections)

    save_path = '../track_images/'
    track_ids = list(radar_detections["id_track"].unique())

    for id in tqdm(track_ids):
        img = Image.fromarray(rasterizer(id))
        img.save(save_path + f"{id}.jpg", quality=95)

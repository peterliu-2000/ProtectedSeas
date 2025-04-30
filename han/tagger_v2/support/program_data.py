"""
This file contains all the structs for the program data
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.data_primitives import *
from utils.model_primitives import *
from utils.constants import *
from utils.scripts.sum_stats import SumStats


# Summary generator
sum_gen = SumStats()


class ProgramData():
    def __init__(self, track_filename, trajectory_filename) -> None:
        
        # Import and initialize required data objects:
        self.tracks = read_and_init_track_df(track_filename)
        self.detections = read_and_init_detections_df(trajectory_filename)
        
        # Data size statistics
        self.num_tracks = len(self.tracks)
        self.num_filtered_tracks = self.num_tracks
        
        # A boolean vector corresponding to filtered observations
        # None -> No filter set.
        self.filter = None
        
        # Lazily initialize the summary statistics
        self.summaries = sum_gen.init_summary_dataframe(self.num_tracks)
        # Lazily initialize the prediction matrix for the model
        # The untagged class is not included in the model.
        self.predict_probs = -np.ones((self.num_tracks, len(ACT_CODE) - 1))
        self.predict_label = [None] * self.num_tracks
        
        # Initialize the xgboost model:
        self.model = load_model()
        
    def __len__(self):
        """
        Only returns the number of tracks present. Not the number of detection points
        """
        return self.num_tracks
    
    def get_filtered_count(self):
        return self.num_filtered_tracks
        
    def get_track(self, index):
        """
        Assumes pandas "copy on write" is enabled. Returns a copy of the track row.
        """
        return self.tracks.iloc[index]
    
    def get_summary(self, index):
        # First try to obtain summary from cache
        summ = self.summaries.iloc[index]
        if summ.isnull().any():
            # Need to compute new summary statistics:
            id = self.tracks.iloc[index]["id_track"]
            trajectory = get_trajectory(self.detections, id)
            summ = sum_gen.compute_track_features(trajectory)
            # Writes back to cache
            self.summaries.iloc[index] = summ
        return summ
    
    def get_model_prediction(self, index):
        """
        Obtains the model prediction

        Args:
            index: observation index

        Returns:
            predicted label and predicted probabilities
        """
        # First try to obtain predictions from cache:
        pred_label, pred_probs = self.predict_label[index], self.predict_probs[index]
        if pred_label is None:
            # Need to compute
            features = pd.DataFrame(self.get_summary(index), dtype = pd.Float64Dtype()).transpose()
            pred_label = model_predict(self.model, features)
            pred_probs = model_predict(self.model, features, label=False)
            # Writes back to cache
            self.predict_label[index] = pred_label
            self.predict_probs[index, :] = pred_probs
        return pred_label, pred_probs
            
    def fill_all_summaries(self):
        """
        Eagerly fill all summary statistics values.
        """
        self.summaries = sum_gen.compute_all_summaries(self.detections, self.tracks["id_track"])
        # Also fills predictions
        self.predict_label = np.array(model_predict(self.model, self.summaries))
        self.predict_probs = np.array(model_predict(self.model, self.summaries, False))
    
    def save_track(self, row:pd.Series, index):
        """
        Assumes pandas "copy on write" is enabled. Saves a row to track data.
        """
        self.tracks.iloc[index] = row
        
    def save_to_file(self, filename):
        """
        Save the modified track data to csv file

        Args:
            filename:
        """
        self.tracks.to_csv(filename, index = False)
        
    def get_trajectory(self, track_id):
        """
        A wrapper for "get_trajectory" in the data_ops file. Only pass the relevant stuff to the application level.
        Args:
            track_id:
        Returns:
            A lat long np array
        """
        traj = get_trajectory(self.detections, track_id)
        if len(traj) == 0:
            raise RuntimeWarning(f"Track id {track_id} returned an empty trajectory record.")
    
        return traj[["latitude", "longitude"]].to_numpy()
    
    # Data Seek Function:
    
    def seek_next(self, idx):
        """
        Get the row index of the next track.

        Args:
            idx: row index of the current track
        """
        # If no filter is set, simply return the next observation
        if self.filter is None:
            return (idx + 1) % self.num_tracks
        
        # If filter is set: Find the next observation with a True Filter.
        curr = (idx + 1) % self.num_tracks
        while(curr != idx):
            if(self.filter[curr]): return curr
            curr = (curr + 1) % self.num_tracks
        return idx
    
    def seek_prev(self, idx):
        """
        Get the row index of the previous track.

        Args:
            idx: row index of the current track
        """
        # If no filter is set, simply return the previous observation
        if self.filter is None:
            return (idx - 1) % self.num_tracks
        
        # If filter is set: Find the previous observation with a True Filter.
        curr = (idx - 1) % self.num_tracks
        while(curr != idx):
            if(self.filter[curr]): return curr
            curr = (curr - 1) % self.num_tracks
        return idx
    
    # def set_filter(self, tag = None, type = None,
    #                has_notes = False, no_tags = False,
    #                duplicate_tags = False, valid_only = False,
    #                confidence_low = -np.inf, confidence_high = np.inf):
    #     """
    #     Filter the observations according to the following options:

    #     Args:
    #         tag: The activity tag for the vessel. Defaults to None (All).
    #         type: The type for the vessel. Defaults to None (All).
    #         has_notes: Only show observations with a note. Defaults to False.
    #         no_tags: Only show observations with at no activity tag. Defaults to False.
    #         duplicate_tags: Only show observations with more than one activity tag checked. Defaults to False.
    #         valid_only: Only show valid observations. Defaults to False.
    #         confidence_low: Confidence Filter Low threshold. Defaults to -np.inf.
    #         confidence_high: Confidence Filter High threshold. Defaults to np.inf.
            
    #     Returns:
    #         True if filter is set successfully, False otherwise.
    #     """
    #     # Initial data_filter
    #     data_filter = [True] * self.num_tracks
    #     # Filter according to activity tags
    #     if tag is not None:
    #         tag_filter = univariate_filter(self.tracks, "activity", lambda x: x == tag)
    #         data_filter = np.logical_and(data_filter, tag_filter)
    #     # Filter according to vessel type tags
    #     if type is not None:
    #         type_filter = univariate_filter(self.tracks, "type_m2_agg", lambda x: x == type)
    #         data_filter = np.logical_and(data_filter, type_filter)
    #     # Filter according to tags:
    #     if duplicate_tags and no_tags:
    #         return False # Two flags cannot be set simultaneously
    #     if duplicate_tags:
    #         data_filter = np.logical_and(filter_duplicate_tags(self.tracks), data_filter)
    #     if no_tags:
    #         data_filter = np.logical_and(filter_no_tags(self.tracks), data_filter)
    #     # Filter according to other attributes:
    #     if has_notes:
    #         note_filter = univariate_filter(self.tracks, "notes", lambda x: not pd.isna(x) and len(x.strip()) > 0)
    #         data_filter = np.logical_and(data_filter, note_filter)
    #     if valid_only:
    #         valid_filter = univariate_filter(self.tracks, "valid", lambda x: x)
    #         data_filter = np.logical_and(data_filter, valid_filter)
    #     # Filter according to confidence
    #     conf_filter = univariate_filter(self.tracks, "confidence",
    #                                     lambda x: x >= confidence_low and x < confidence_high)
    #     data_filter = np.logical_and(data_filter, conf_filter)
        
    #     # determine if the filter is valid:
    #     filtered_obs = np.sum(data_filter)
    #     if filtered_obs > 0:
    #         self.filter = data_filter
    #         self.num_filtered_tracks = filtered_obs
    #         return True
    #     else:
    #         return False # No observations match.
            
    def unset_filter(self):
        """
        Remove the tag filter
        """
        self.filter = None
        self.num_filtered_tracks = self.num_tracks
        
    def next(self, idx):
        """
        API function for seek_next

        Args:
            idx: row index of the current track
        """
        return self.seek_next(idx)
            
    def prev(self, idx):
        """
        API function for seek_prev

        Args:
            idx: row index of the current track
        """
        return self.seek_prev(idx)
            

if __name__ == "__main__":
    path1 = '../data/tracks_tagged.csv'
    path2 = '../data/detections_tagged_cached.csv'
    data = ProgramData(path1, path2)
    id = data.tracks.iloc[1]["id_track"]
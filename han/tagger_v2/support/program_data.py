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




class ProgramData():
    def __init__(self, track_filename, trajectory_filename) -> None:
        
        # Import and initialize required data objects:
        self.tracks = read_and_init_track_df(track_filename)
        print(f"Loaded track data @ {track_filename}")
        self.detections = read_and_init_detections_df(trajectory_filename)
        print(f"loaded detections data @ {trajectory_filename}")
        
        # Check if all tracks in the track dataframe has detections data
        tracks_in_detections = set(self.detections["id_track"].unique())
        self.tracks = self.tracks[self.tracks["id_track"].isin(tracks_in_detections)]
        self.num_tracks = len(self.tracks)
        self.num_filtered_tracks = self.num_tracks
        
        if self.num_tracks == 0:
            raise RuntimeError(f"Provided Track file {track_filename} and Detections file {trajectory_filename} does not match.")
        else:
            print(f"Found {self.num_tracks} tracks with detections data.")
        
        # Assign a cache file for this track file
        self.cache_name = track_filename.split("/")[-1].replace(".csv", ".cache")
        self.cache_name = PROGRAM_CACHE_PATH + self.cache_name
        self.summaries = self.init_summ_stats(set(self.tracks["id_track"]))
        
        # Sort both the summaries and the track dataframe so the order is consistent
        self.summaries = self.summaries.sort_values(by = "id_track", ascending=True)
        self.tracks = self.tracks.sort_values(by = "id_track", ascending=True)
        
        # Initialize the xgboost model:
        print(f"Initializing prediction models...")
        self.model = load_model()      
        self.preds = self.init_model_predictions()  
        
        # Initialize filter
        self.filter = None
        
        
    def init_summ_stats(self, track_ids):
        """
        Initialize the summary statistics dataframe.
        """
        # Summary generator
        sum_gen = SumStats()
        # First try to read from file. 
        try:
            df = pd.read_parquet(self.cache_name)
            # Do some checks with the loaded summary file:
            if len(set(df["id_track"]) & set(track_ids)) != len(track_ids) or df.isnull().sum().sum() > 0:
                raise RuntimeError()
        except Exception:
            print("Track Summary Cache doesn't exist or is corrupted.\n"
                  "Calculating all summary statistics (This may take a while...)")
            df = sum_gen.generate_summary_data(self.detections, track_ids)
            df = df.reset_index()
            # Save the generated summary statistics to file
            df.to_parquet(self.cache_name)
            print(f"Successfully cached track_summaries.")
        
        return df
            
    def init_model_predictions(self):
        pred_label = model_predict(self.model, self.summaries)
        pred_probs = model_predict(self.model, self.summaries, label=False)
        self.tracks["predict_activity"] = pred_label
        self.tracks["predict_score"] = np.max(pred_probs, axis = 1)
        return pred_probs
        
            
            
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
        return self.summaries.iloc[index]
    
    def get_model_prediction(self, index):
        """
        Obtains the model prediction

        Args:
            index: observation index

        Returns:
            predicted label and predicted probabilities
        """
        pred_label = self.tracks.iloc[index]["predict_activity"]
        pred_probs = self.preds[index, :] #type:ignore
        return pred_label, pred_probs
            
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
    
    def set_filter(self, tag = None, type = None, pred = None,
                   confidence_low = -np.inf, confidence_high = np.inf):
        """
        Filter the observations according to the following options:

        Args:
            tag: The activity tag for the vessel. Defaults to None (All).
            type: The type for the vessel. Defaults to None (All).
            pred: The activity tag predicted by the builtin model. Defaults to None (All)
            confidence_low: Confidence Filter Low threshold. Defaults to -np.inf.
            confidence_high: Confidence Filter High threshold. Defaults to np.inf.
            
        Returns:
            True if filter is set successfully, False otherwise.
        """
        # Initial data_filter
        data_filter = [True] * self.num_tracks
        # Filter according to activity tags
        if tag is not None:
            tag_filter = univariate_filter(self.tracks, "activity", lambda x: x == tag)
            data_filter = np.logical_and(data_filter, tag_filter)
        # Filter according to vessel type tags
        if type is not None:
            type_filter = univariate_filter(self.tracks, "type_m2_agg", lambda x: x == type)
            data_filter = np.logical_and(data_filter, type_filter)
        # Filter according to model prediction:
        if pred is not None:
            model_filter = univariate_filter(self.tracks, "predict_activity", lambda x: x == pred)
            data_filter = np.logical_and(data_filter, model_filter)
        # Filter according to confidence
        conf_filter = univariate_filter(self.tracks, "predict_score",
                                        lambda x: x >= confidence_low and x < confidence_high)
        data_filter = np.logical_and(data_filter, conf_filter)
        
        # determine if the filter is valid:
        filtered_obs = np.sum(data_filter)
        if filtered_obs > 0:
            self.filter = data_filter
            self.num_filtered_tracks = filtered_obs
            return True
        else:
            return False # No observations match.
            
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
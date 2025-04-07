"""
This file contains all the structs for the program data
"""

import pandas as pd
import numpy as np
from utils.data_ops import *

class ProgramData():
    def __init__(self, track_filename, trajectory_filename) -> None:
        # Call data import / processing function
        self.trajectory = read_and_init_trajectory(trajectory_filename)
        self.tracks, self.has_tag = read_and_init_label(track_filename)
        
        # Data size statistics
        self.num_tracks = len(self.tracks)
        
        # track row indecies for filtered observations. Initially set to none.
        self.filter_idx = None
        
        
    def get_track(self, index):
        return self.tracks.iloc[index]
        
    def get_trajectory(self, track_id):
        """
        A wrapper for "get_trajectory" in the data_ops file
        Args:
            track_id
        """
        traj = get_trajectory(self.trajectory, track_id)
        if len(traj) == 0:
            raise RuntimeWarning(f"Track id {track_id} returned an empty trajectory record.")
        return traj
    
    def set_filter(self, tag):
        """
        Filter the dataset according to a specific tag.

        Args:
            tag
        """
        try:
            filter_idxs = filter_tracks(self.tracks, tag)
        except RuntimeError:
            raise RuntimeWarning(f"Filter not set as {tag} is not a valid tag.")
        else:
            if len(filter_idxs) == 0:
                raise RuntimeWarning(f"Filter not set as no tracks has {tag} set to True.")
            self.filter_idx = filter_idxs
            
    def unset_filter(self):
        """
        Remove the tag filter
        """
        self.filter_idx = None
        
    def next(self, idx):
        """
        Get the row index of the next track, subject to the tag filter

        Args:
            idx: row index of the current track
        """
        if self.filter_idx is None:
            return (idx + 1) % self.num_tracks
        else:
            tmp_idxs = [i for i in self.filter_idx if i > idx]
            # Wrap arround
            if len(tmp_idxs) == 0: return self.filter_idx[0]
            else: return min(tmp_idxs)
            
    def prev(self, idx):
        """
        Get the row index of the previous track, subject to the tag filter

        Args:
            idx: row index of the current track
        """
        if self.filter_idx is None:
            return (idx - 1) % self.num_tracks
        else:
            tmp_idxs = [i for i in self.filter_idx if i < idx]
            # Wrap arround
            if len(tmp_idxs) == 0: return self.filter_idx[-1]
            else: return max(tmp_idxs)
            
    def next_untagged(self, idx):
        """
        Get the row index of the next untagged track.

        Args:
            idx: row index of the current track
        """
        if self.filter_idx is not None:
            print("A tag filter has set. Using next instead.")
            return self.next(idx)
        
        curr = (idx + 1) % self.num_tracks
        while(curr != idx):
            if(not self.has_tag[curr]): return curr
            curr = (curr + 1) % self.num_tracks
        print("All tracks have been completely tagged.")
        return idx
    
    def prev_untagged(self, idx):
        """
        Get the row index of the previous untagged track.

        Args:
            idx: row index of the current track
        """
        if self.filter_idx is not None:
            print("A tag filter has set. Using prev instead.")
            return self.prev(idx)
        
        curr = (idx - 1) % self.num_tracks
        while(curr != idx):
            if(not self.has_tag[curr]): return curr
            curr = (curr - 1) % self.num_tracks
        print("All tracks have been completely tagged.")
        return idx
    
    
if __name__ == "__main__":
    path1 = '../data/tracks_tagged.csv'
    path2 = '../data/detections_tagged_cached.csv'
    data = ProgramData(path1, path2)
    id = data.tracks.iloc[1]["id_track"]
    
    print(data.get_trajectory(data.tracks.iloc[1]["id_track"]))
    print(data.trajectory[data.trajectory["id_track"] == id])
"""
This file provides methods and dependencies for data frame operations
"""

import numpy as np
import pandas as pd
from utils.time_conversion import *
from tqdm import tqdm

def read_and_init_label(filename):
    """
    Read and initialize the data frame for labels. The label file should contain
    all the tagged information.

    Args:
        filename
        
    Returns:
        (df, complete), where df is the label dataframe, and complete is a 
        boolean vector corresponding to tag completeness
    """
    df = pd.read_csv(filename)
    # A complete tag means that the track is identified as at least one of
    # the following:
    tags = ["transit", "overnight", "loiter", "cleanup", "fishing_c", "fishing_r",
            "research", "diving", "repairs", "distress", "other"]
    # Y should give us a vector of complete tags
    complete = np.sum(df[tags].to_numpy(), axis = 1) > 0
    return df, complete

def read_and_init_trajectory(filename):
    """
    Read and initialize the data frame for trajectories. The trajectory file
    should contain all detection points corresponding to each track in the label
    file.
    
    This function also automatically identifies if the read dataframe is a cached
    trajectory file or the original input file. If the read dataframe requires
    processing for time information, expect ~3 minutes for this function to return.
    Args:
        filename
    """
    df = pd.read_csv(filename)
    # Determine if the read dataframe is already preprocessed:
    if "time" in df.columns:
        return df    
    # Otherwise, perform preprocessing on time.
    std_time = np.zeros(len(df), dtype=np.int64)
    str_date = df["cdate"]
    str_time = df["ctime"]
    for i in tqdm(range(len(df)), desc = "Preparing Trajectory Dataframe"):
        std_time[i] = parse_time(str_date[i], str_time[i])
    df["time"] = std_time
    # Only keep relevant columns.
    relevant_cols = ["id_detect", "id_track", "id_site", "time",
                     "speed", "course", "latitude", "longitude"]
    return df[relevant_cols]

def cache_trajectory_data(trajectory_df, save_filename):
    """
    Save the processed trajectory data to a cache file.
    Args:
        trajectory_df: _description_
    """
    trajectory_df.to_csv(save_filename)

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
    return detections_sorted[["latitude", "longitude"]].to_numpy()


def filter_tracks(label_df, filter_by):
    """
    Filter the tracks based on a particular tag

    Args:
        label_df: label dataframe with tags
        filter_by: tag name to filter (eg. 'transit', 'loiter')
        
    Returns: 
        A list of track df indices (not track_id) with the filtered tracks.
    """
    if filter_by not in label_df.columns:
        raise RuntimeError("Tag {filter_by} does not exist.")
    return np.nonzero(label_df[filter_by] == 1)[0]

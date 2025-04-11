"""
This file provides methods and dependencies for data frame operations
"""

import numpy as np
import pandas as pd
from utils.time_conversion import *
from utils.constants import ACTIVITY_TAGS
from tqdm import tqdm

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

        df['type_m2_agg'] = df['type_m2']
        
        # Apply the mapping only for keys that exist in the mapping
        for original, aggregated in self.type_mapping.items():
            df.loc[df['type_m2'] == original, 'type_m2_agg'] = aggregated
            
        return df


def read_and_init_track_df(filename):
    """
    Read the dataframe for the track data (including tags.) Initialize any remaining
    fields to make data compliant.
    """
    df = pd.read_csv(filename)
    df_length = len(df)
    # Ensures that the data frame contains all the following columns
    # Mandatory Fields, Raise error if missing
    mandatory = ["id_track", 'id_site', 'duration', "detections", 'confidence']
    # Tags: Optional Fields, Initialize if necessary
    tags = ["transit", "overnight", "loiter", "cleanup", "fishing_c", "fishing_r",
            "research", "diving", "repairs", "distress", "other", "valid"]
    # Other Optional Labels, Initialize if Necessary:
    optional = ["type_m2_agg", "sdate", "stime", "ldate", "ltime", "notes"]
    
    # Check dataframe compliance:
    columns = list(df.columns)
    for c in mandatory:
        if c not in columns:
            raise RuntimeError(f"Imported track {filename} has invalid formatting.")
    
    for c in tags:
        if c not in columns:
            df[c] = np.zeros(df_length)
            
    # Use Peter's aggregator for type_m2_agg.
    if "type_m2" in columns:
        df = VesselTypeAggregator().aggregate_vessel_type(df)
        columns = list(df.columns)
            
    for c in optional:
        if c not in columns:
            if c == "type_m2_agg":
                df[c] = [None] * df_length
            elif c == "notes":
                df[c] = [""] * df_length
            else:
                df[c] = ["N/A"] * df_length

    return df

def read_and_init_detections_df(filename):
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
    df_length = len(df)
    columns = list(df.columns)
    
    # Target data frame columns
    relevant_cols = ["id_detect", "id_track", "id_site", "time",
                     "speed", "course", "latitude", "longitude"]
    
    # Check for data frame compliance
    mandatory = ["id_detect", "id_track", "id_site",
                 "speed", 'course', 'latitude', 'longitude']
    for c in mandatory:
        if c not in columns:
            raise RuntimeError(f"Imported track {filename} has invalid formatting.")
        
    # If the dataframe imported is already processed:
    if "time" in columns:
        return df[relevant_cols]
    
    # Otherwise, requires "cdate" and "ctime"
    if "cdate" not in columns or "ctiem" not in columns:
        raise RuntimeError(f"Imported track {filename} has invalid formatting.")
    
    print(f"The imported detections file {filename} requires preprocessing.\
          Please expect ~2 minutes to finish.")
    
    std_time = np.zeros(len(df), dtype=np.int64)
    str_date = df["cdate"]
    str_time = df["ctime"]
    for i in tqdm(range(df_length), desc = "Preparing Trajectory Dataframe"):
        std_time[i] = parse_time(str_date[i], str_time[i])
    df["time"] = std_time
    
    cached_filename = "cached_" + filename
    cache_trajectory_data(df[relevant_cols], cached_filename)
    print(f"Pre_processing complete. Saved cached data to {cached_filename}")
    
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

def univariate_filter(track_df, var, condition):
    """
    Performs univariate filtering on the track data

    Args:
        track_df: track dataframe
        var: variable name to perform filter on
        condition: Filter Condition
    
    Returns:
        A boolean vector corresponding to the conditions 
    """
    return np.array(list(map(condition, track_df[var])))

def filter_no_tags(track_df):
    """
    Filters the data for entries with no activity tags

    Args:
        track_df: track dataframe
    """
    return np.sum(track_df[ACTIVITY_TAGS].to_numpy(), axis = 1) == 0 

def filter_duplicate_tags(track_df):
    """
    Filters the data for entries with duplicate activity tags

    Args:
        track_df: track dataframe
    """
    return np.sum(track_df[ACTIVITY_TAGS].to_numpy(), axis = 1) > 1
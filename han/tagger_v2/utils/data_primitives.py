"""
This file provides methods and dependencies for data frame operations
"""

import numpy as np
import pandas as pd
from utils.scripts.time_conversion import *
from utils.scripts.vessel_agg import VesselTypeAggregator
from utils.constants import ACT_CODE
from tqdm import tqdm

def read_and_init_track_df(filename):
    """
    Read the dataframe for the track data (including tags.) Initialize any remaining
    fields to make data compliant.
    """
    df = pd.read_csv(filename)
    df_length = len(df)
    # Ensures that the data frame contains all the following columns
    # id_track is required
    columns = list(df.columns)
    if "id_track" not in columns:
        raise RuntimeError(f"Track file {filename} is not supported.")
    
    if np.sum(df["id_track"].duplicated().to_list()):
        raise RuntimeWarning(f"Track file {filename} contains duplicated ids.")
    
    # Tags: Optional Fields, Initialize if necessary
    legacy_tags = ["transit", "overnight", "loiter", "cleanup", "fishing_c", "fishing_r",
            "research", "diving", "repairs", "distress", "other"]
    
    for c in legacy_tags:
        if c in columns:
            print("Warning: Legacy M2 Datasets are no longer supported.")
            break
            
    # Initialize other fields for the track dataframe
    if "activity" in columns:
        df["activity"] = df["activity"].fillna("")
    else:
        df["activity"] = [""] * df_length
        
    if "type_m2_agg" in columns:
        df["type_m2_agg"] = df["type_m2_agg"].fillna("")
    else:
        # Fill in the aggregated result by inference.
        if "type_m2" in columns:
            VesselTypeAggregator().aggregate_vessel_type(df)
        else:
            df["type_m2_agg"] = [""] * df_length
            
    if "valid" not in columns:
        df["valid"] = np.ones(df_length)
        
    # Sort the data frame by id_track and only return relevant columns
    df = df.sort_values(by = "id_track", ascending=True)
    # [["id_track", "activity", "type_m2_agg", "valid"]]
    return df

def read_and_init_detections_df(filename, save = True):
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
    
    # Helper routines
    def process_time():
        std_time = np.zeros(len(df), dtype=np.int64)
        str_date = df["cdate"]
        str_time = df["ctime"]
        for i in tqdm(range(df_length), desc = "Preprocessing time information"):
            std_time[i] = parse_time(str_date[i], str_time[i])
        df["time"] = std_time

    def cache_trajectory_data(trajectory_df, save_filename):
        """
        Save the processed trajectory data to a cache file.
        Args:
            trajectory_df: _description_
        """
        trajectory_df.to_csv(save_filename)
        
    # Target data frame columns
    relevant_cols = ["id_detect", "id_track", "time",
                     "speed", "course", "latitude", "longitude"]

    for c in relevant_cols:
        if c not in columns:
            if c != "time":
                raise RuntimeError(f"Detection file {filename} is not supported.")
            elif "cdate" not in columns or "ctime" not in columns:
                raise RuntimeError(f"Detection file {filename} is not supported.")
            else:
                print(f"The imported detections file {filename} requires preprocessing.\
                Please expect ~2 minutes to finish.")
                process_time()
                if save:
                    cached_filename = filename.replace(".csv", "_cached.csv")
                    cache_trajectory_data(df[relevant_cols], cached_filename)
                    print(f"Pre_processing complete. Saved cached data to {cached_filename}")
            
    return df[relevant_cols]

def get_trajectory(trajectory_df, track_id):
    """
    Obtain the trajectory list of a given track id

    Args:
        trajectory_df: dataframe for trajectory
        track_id: track id
    
    Returns:
    sorted trajectory dataframe
    """
    detections = trajectory_df[trajectory_df["id_track"] == track_id]
    return detections.sort_values(by = "time", ascending=True)

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
    return np.logical_or(pd.isna(track_df["activity"]),track_df["activity"] == "")
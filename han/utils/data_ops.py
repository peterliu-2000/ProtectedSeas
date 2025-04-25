import numpy as np
import pandas as pd
from tqdm import tqdm
import time, calendar

def parse_time(d:str, t:str) -> int:
    """
    Convert a string representation of date and time to the unix time encoding
    Remark: This function assumes that the date/time is recorded in UTC.
    However, the functionality of this function shall not be affected if
    the timezone offset is consistent
    
    Args:
        d: date string, format "yyyy-mm-dd"
        t: time string, format "hh:mm:ss"
    """
    time_struct = time.strptime(d+t, "%Y-%m-%d%H:%M:%S")
    return calendar.timegm(time_struct)

def str_time(t:int) -> tuple:
    """
    The inverse of parse_time. Written for debugging purposes
    """
    s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    return tuple(s.split(" "))

def init_detections_df(filename, save = True):
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
            raise RuntimeError(f"Imported detection {filename} has invalid formatting.")
        
    # If the dataframe imported is already processed:
    if "time" in columns:
        return df[relevant_cols]
    
    # Otherwise, requires "cdate" and "ctime"
    if "cdate" not in columns or "ctime" not in columns:
        raise RuntimeError(f"Imported detections {filename} has invalid formatting.")
    
    print(f"The imported detections file {filename} requires preprocessing.\n\
          Please expect ~2 minutes to finish.")
    
    std_time = np.zeros(len(df), dtype=np.int64)
    str_date = df["cdate"]
    str_time = df["ctime"]
    for i in tqdm(range(df_length), desc = "Preparing Trajectory Dataframe"):
        std_time[i] = parse_time(str_date[i], str_time[i])
    df["time"] = std_time
    
    if save:
        cached_filename = filename.replace(".csv", "_cached.csv")
        df[relevant_cols].to_csv(cached_filename)
        print(f"Pre_processing complete. Saved cached data to {cached_filename}")
    
    return df[relevant_cols]
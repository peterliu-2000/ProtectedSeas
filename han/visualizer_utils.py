import numpy as np
import pandas as pd
import plotly.graph_objects as go
        
# Utility function for track filtering

def get_trajectory(trajectory_df, track_id, lat_long_only = False):
    """
    Obtain the trajectory list of a given track id

    Args:
        trajectory_df: dataframe for trajectory
        track_id: track id
    
    Returns:
    sorted trajectory dataframe
    """
    detections = trajectory_df[trajectory_df["id_track"] == track_id]
    detections = detections.sort_values(by = "time", ascending=True)
    if lat_long_only: 
        return np.array(detections[["latitude", "longitude"]])
    return detections

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

# Utility function for plotting

def calc_map_center_zoom(trajectory):
    max_latlong = np.max(np.array(trajectory), axis=0)
    min_latlong = np.min(np.array(trajectory), axis=0)
    span = np.max(max_latlong - min_latlong)
    return 0.5 * (max_latlong + min_latlong), span
    
def plot_lat_long_tracks(lat_long: np.ndarray, title = ""):
    """
    Generates a trajectory map object from the trajectory lat long data
    """
    # computes the center and span
    center, span = calc_map_center_zoom(lat_long)
    (lat_center, long_center) = center
    margin = 1e-6
    span = span + margin
    
    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            showlegend=False,
            mode = "markers+lines",
            lat = list(lat_long[:, 0]),
            lon = list(lat_long[:, 1]),
            marker=go.scattermap.Marker(
                size= 4,
                color = "Blue"
            ),
            line = go.scattermap.Line(
                width = 2,
                color ="Blue"
            )
        )
    )
    fig.add_trace(
        go.Scattermap(
            name = "Start",
            mode = "markers",
            lat = [lat_long[0, 0]],
            lon = [lat_long[0, 1]],
            marker=go.scattermap.Marker(
                size=8,
                color = "Green"
            ),
        )
    )
    fig.add_trace(
        go.Scattermap(
            name = "End",
            mode = "markers",
            lat = [lat_long[-1, 0]],
            lon = [lat_long[-1, 1]],
            marker=go.scattermap.Marker(
                size=8,
                color = "Red"
            ),
        )
    )
    # Compute the title margin
    if len(title) > 0: 
        paper_margin = 30
    else:
        paper_margin = 0
    
    fig.update_layout(
        margin = {'l':0,'t':0,'b':paper_margin,'r':0},
        map = {
            "bounds" : dict(east = long_center + span, west = long_center - span,
                            north = lat_center + span, south = lat_center - span),
            "style" : "open-street-map"
        },
        legend = dict(x = 0, y = 1,
                      bgcolor = "rgba(255,255,255,0.8)", bordercolor = "Black", borderwidth = 1),
        title=dict(text = title, font=dict(size=20), y = 0.022, x = 0.5,
                   xanchor = "center", yanchor = "bottom")
    )
    return fig

def save_fig(fig, path, width = 400, height = 400, scale = 4):
    fig.write_image(path, width = width, height = height, scale = scale)
    
    
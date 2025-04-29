import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_trajectory(mode, type, radar_detections, num_samples = 6):
    """
    Args: 
        type: vessel type of interest
        num_samples: number of random trajectories to be plotted
        merged_radar_detections: default is to use merged_radar_detections

    Plot trajectories of vessel of type based on radar detections
    """
    assert mode in ['type', 'activity'], "mode must be either 'type' or 'activity"
    if mode == 'type':
        df = radar_detections[radar_detections['type_m2_agg'] == type]
    elif mode == 'activity':
        df = radar_detections[radar_detections['activity'] == type]

    df['datetime'] = pd.to_datetime(df['datetime'])
    track_ids = df['id_track'].unique()
    sampled_tracks = np.random.choice(track_ids, size = num_samples, replace = False)

    # Create subplot grid
    n_rows = num_samples // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6*n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Add main title
    fig.suptitle(f"{type}", fontsize=40, y=1.02)

    # Plot each trajectory
    for idx, track_id in enumerate(sampled_tracks):
        track_data = df[df['id_track'] == track_id].sort_values('datetime')
        
        # Calculate duration
        start_time = pd.to_datetime(track_data['ctime'].iloc[0], format='%H:%M:%S')
        end_time = pd.to_datetime(track_data['ctime'].iloc[-1], format='%H:%M:%S')
        duration = end_time - start_time
        hours = duration.components.hours
        minutes = duration.components.minutes
        seconds = duration.components.seconds
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Plot trajectory
        axes[idx].plot(track_data['longitude'], track_data['latitude'], 
                      marker='o', markersize=2, linewidth=1, alpha=0.6)
        
        # Add start and end points
        axes[idx].scatter(track_data['longitude'].iloc[0], track_data['latitude'].iloc[0], 
                         color='green', marker='^', s=100, label='Start')
        axes[idx].scatter(track_data['longitude'].iloc[-1], track_data['latitude'].iloc[-1], 
                         color='red', marker='v', s=100, label='End')
        
        # Customize subplot
        axes[idx].set_title(f'Track {track_id}\nDuration: {duration_str}')
        axes[idx].set_xlabel('Longitude')
        axes[idx].set_ylabel('Latitude')
        axes[idx].grid(True)
        axes[idx].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
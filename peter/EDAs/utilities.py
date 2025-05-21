import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

class SingleTrackPlotter:
    """
    Plot trajectory of a vessel
    """
    
    def __init__(self):
        pass

    def plot_trajectory(self, track_data):
        """
        Args:
            track_data: pd dataframe; detections associated with 1 track
        """
        assert len(track_data['id_track'].unique()) == 1, "track_data must be associated with 1 track"

        track_id = track_data['id_track'].iloc[0]  # Ensure we grab the first element of the track id

        # Sort data by datetime
        track_data = track_data.sort_values('datetime')

        # Calculate duration
        start_time = track_data['datetime'].iloc[0]
        end_time = track_data['datetime'].iloc[-1]
        duration = end_time - start_time
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create figure and axis for a single plot
        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot trajectory
        ax.plot(track_data['longitude'], track_data['latitude'], 
                marker='o', markersize=2, linewidth=1, alpha=0.6)

        # Add start and end points
        ax.scatter(track_data['longitude'].iloc[0], track_data['latitude'].iloc[0], 
                   color='green', marker='^', s=100, label='Start')
        ax.scatter(track_data['longitude'].iloc[-1], track_data['latitude'].iloc[-1], 
                   color='red', marker='v', s=100, label='End')

        # Customize plot
        ax.set_title(f'Track {track_id}\nDuration: {duration_str}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
        ax.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

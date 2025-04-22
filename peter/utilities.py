import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

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
            'military_ship': 'military_ship',
            'class_b':'class_b',
            'passenger_ship': 'passenger_ship',
            'pleasure_craft': 'pleasure_craft',
            'sailboat': 'pleasure_craft',
            'search_and_rescue_boat': 'other',
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

        df['type_m2_agg'] = df['type_m2'].map(self.type_mapping).fillna('other')


class TrajectoryPlotter:
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

        track_data['datetime'] = track_data['cdate'] + ' ' + track_data['ctime']
        track_id = track_data['id_track'].iloc[0]  # Ensure we grab the first element of the track id

        # Sort data by datetime
        track_data = track_data.sort_values('datetime')

        # Calculate duration
        start_time = pd.to_datetime(track_data['ctime'].iloc[0], format='%H:%M:%S')
        end_time = pd.to_datetime(track_data['ctime'].iloc[-1], format='%H:%M:%S')
        duration = end_time - start_time
        hours = duration.components.hours
        minutes = duration.components.minutes
        seconds = duration.components.seconds
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create figure and axis for a single plot
        fig, ax = plt.subplots(figsize=(12, 8))

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

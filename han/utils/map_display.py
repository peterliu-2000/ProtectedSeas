"""
This file contains utility functions and package dependencies for mapping and
trajectory display
"""

import folium

locations = {
    10 : (32.815495,	-117.272235),
    19 : (34.407361,	-119.878556),
    22 : (32.867021,	-117.257220),
    23 : (26.015045,	-111.340189),
    26 : (48.558441,	-123.173264),
    28 : (33.047259,	-117.298026),
    42 : (25.566275,	-111.149116),
    43 : (34.015887,	-119.359494),
    45 : (26.359411,	-111.429349)
}

def get_site_coordinates(site_id:int):
    """
    Obtain the site lat-long coordinates.

    Args:
        site_id
    """
    if site_id not in locations:
        raise RuntimeError(f"Fatal Error: Site ID {site_id} is not valid.")
    return locations[site_id]

def init_map(site_location: tuple[float, float]):
    """
    Initialize a folium map object, given a location lat-long coordinate

    Args:
        site_location
    """
    return folium.Map(location = site_location, zoom_start=9, min_zoom=5, max_zoom=12)

def plot_trajectory(map, trajectory):
    """
    Add a trajectory line to the map

    Args:
        map: map object given by init_map
        trajectory: a list of lat-long coordinates
    """
    # First add the trajectory line
    folium.PolyLine(locations = trajectory, weight = 1, 
                    smooth_factor = 0, color = "cornflowerblue").add_to(map)
    # Add the trajectory points
    def get_color(index):
        if index == 0: return "green"
        elif index == len(trajectory) - 1: return "red"
        else: return "blue"
    
    for i, point in enumerate(trajectory):
        folium.CircleMarker(
            location = point,
            stroke = False,
            fill = True,
            fill_color = get_color(i),
            fill_opacity = 1.0,
            opacity = 1,
            radius = 2
        ).add_to(map)
    
    return map
    

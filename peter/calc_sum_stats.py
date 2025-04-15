import numpy as np

def distance_between_points(lat1, lon1, lat2, lon2, r = 6371):
    """
    Calculate the distance between two points on the Earth's surface
    using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point
        lon1 (float): Longitude of the first point
        lat2 (float): Latitude of the second point
        lon2 (float): Longitude of the second point
        r (float): Radius of the Earth in kilometers
    """
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c






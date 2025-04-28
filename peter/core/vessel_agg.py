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
        
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


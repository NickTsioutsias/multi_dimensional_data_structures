"""
Data loader for coffee reviews dataset
"""
import pandas as pd
import numpy as np

class CoffeeDataLoader:
    def __init__(self, filepath):
        """Initialize with path to CSV file"""
        self.filepath = filepath
        self.df = None
        self.processed_df = None
        
    def load_data(self):
        """Load the CSV file"""
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def preprocess(self):
        """Prepare data for 4D indexing"""
        if self.df is None:
            self.load_data()
            
        df = self.df.copy()
        
        # Remove rows with missing values in key columns
        df = df.dropna(subset=['rating', '100g_USD'])
        
        # Convert review_date to year
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        df['year'] = df['review_date'].dt.year
        df = df[df['year'].notna()]
        df['year'] = df['year'].astype(int)
        
        # Convert country to numerical ID
        df['country_id'], country_mapping = pd.factorize(df['loc_country'])
        self.country_mapping = {i: country for i, country in enumerate(country_mapping)}
        
        # Combine review texts (if they exist)
        if 'desc_1' in df.columns:
            df['full_review'] = (
                df['desc_1'].fillna('') + ' ' +
                df['desc_2'].fillna('') + ' ' +
                df['desc_3'].fillna('')
            ).str.strip()
        else:
            df['full_review'] = df['name'].fillna('')
        
        self.processed_df = df
        
        print(f"Preprocessed {len(df)} records")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
        print(f"Price range: ${df['100g_USD'].min():.2f} - ${df['100g_USD'].max():.2f}")
        
        return self.processed_df
    
    def get_points(self):
        """Get 4D points for indexing [year, rating, price, country_id]"""
        if self.processed_df is None:
            self.preprocess()
        
        points = self.processed_df[['year', 'rating', '100g_USD', 'country_id']].values
        return points
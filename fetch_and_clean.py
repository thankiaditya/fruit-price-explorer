#!/usr/bin/env python3
"""
Fruit Price Data Pipeline
Fetches daily fruit price data from data.gov.in, cleans it, and appends to CSV with El Niño fields.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import requests
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fruit_price_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MANDI_API_URL = "https://mandi-api.vercel.app/v1/prices"
MANDI_HISTORY_URL = "https://mandi-api.vercel.app/v1/prices/history"
OUTPUT_CSV = "fruit_prices_clean.csv"
ARCHIVE_CSV = "fruit_prices_raw.csv"

# Fruit mapping for standardization
FRUIT_MAPPING = {
    'Banana': ['banana', 'banana - green', 'banana - ripe'],
    'Mango': ['mango', 'mango (raw-ripe)', 'mango - raw', 'mango - ripe'],
    'Apple': ['apple'],
    'Orange': ['orange', 'mosambi'],
    'Tomato': ['tomato'],
    'Onion': ['onion'],
    'Potato': ['potato'],
    'Grape': ['grape', 'grapes']
}

# Top 8 states to keep
TOP_STATES = [
    'Maharashtra', 'Uttar Pradesh', 'West Bengal', 'Karnataka',
    'Madhya Pradesh', 'Rajasthan', 'Punjab', 'Tamil Nadu'
]

# El Niño thresholds (ONI index)
ONI_WEAK_EL_NINO = 0.5
ONI_MODERATE_EL_NINO = 1.0
ONI_STRONG_EL_NINO = 1.5


class ONILookup:
    """Handles El Niño/La Niña classification via ONI index"""
    
    def __init__(self):
        self.oni_data = self._fetch_oni_data()
    
    def _fetch_oni_data(self) -> Dict[Tuple[int, int], float]:
        """
        Fetch ONI (Oceanic Niño Index) data from NOAA
        Returns dict of {(year, month): oni_value}
        """
        try:
            logger.info("Fetching ONI data from NOAA...")
            url = "https://ggweather.com/oni/oni.txt"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            oni_dict = {}
            for line in response.text.split('\n'):
                if line.strip() and not line.startswith('Year'):
                    parts = line.split()
                    if len(parts) >= 13:
                        try:
                            year = int(parts[0])
                            for month in range(1, 13):
                                oni_value = float(parts[month])
                                if not np.isnan(oni_value):
                                    oni_dict[(year, month)] = oni_value
                        except (ValueError, IndexError):
                            continue
            
            logger.info(f"Loaded ONI data with {len(oni_dict)} month-year entries")
            return oni_dict
        except Exception as e:
            logger.warning(f"Failed to fetch ONI data: {e}. Will proceed without El Niño classification.")
            return {}
    
    def classify(self, date: pd.Timestamp) -> Tuple[bool, int]:
        """
        Classify date as El Niño condition
        Returns (is_el_nino: bool, oni_strength: int)
        oni_strength: -2 (strong La Niña), -1 (weak La Niña), 0 (neutral), 1 (weak El Niño), 2 (moderate+)
        """
        oni_value = self.oni_data.get((date.year, date.month), None)
        
        if oni_value is None:
            return False, 0
        
        if oni_value >= ONI_MODERATE_EL_NINO:
            return True, 2
        elif oni_value >= ONI_WEAK_EL_NINO:
            return True, 1
        elif oni_value <= -ONI_MODERATE_EL_NINO:
            return False, -2
        elif oni_value <= -ONI_WEAK_EL_NINO:
            return False, -1
        else:
            return False, 0


class FruitPriceProcessor:
    """Handles data fetching, cleaning, and aggregation"""
    
    def __init__(self, api_key: str = None):
        # api_key kept for compatibility but not required — Mandi API is keyless
        self.oni = ONILookup()
        self.fruit_standardizer = self._create_fruit_mapping()
    
    def _create_fruit_mapping(self) -> Dict[str, str]:
        """Create lowercase mapping for fruit standardization"""
        mapping = {}
        for standard_name, variants in FRUIT_MAPPING.items():
            for variant in variants:
                mapping[variant.lower()] = standard_name
        return mapping
    
    def fetch_from_api(self, date: datetime = None) -> pd.DataFrame:
        """
        Fetch data from Mandi Price API (mandi-api.vercel.app)
        Free, no API key required, daily synced from data.gov.in
        If date is None, fetches latest available data
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)

        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"Fetching mandi price data for {date_str}...")

        fruits_to_fetch = list(FRUIT_MAPPING.keys())
        states_to_fetch = TOP_STATES
        all_records = []

        # Mandi API supports filtering by state and commodity
        for state in states_to_fetch:
            try:
                params = {'state': state}
                response = requests.get(MANDI_API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Mandi API returns list directly or under a key
                records = data if isinstance(data, list) else data.get('data', data.get('records', []))

                if records:
                    for r in records:
                        r['state'] = r.get('state', state)
                    all_records.extend(records)
                    logger.info(f"  {state}: {len(records)} records")

            except Exception as e:
                logger.warning(f"  {state}: fetch failed — {e}")
                continue

        if not all_records:
            logger.warning("No records returned from Mandi API")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info(f"Fetched {len(df)} total raw records")
        return df
    
    def standardize_fruit(self, fruit_name: str) -> str:
        """Map fruit name to standardized name"""
        if pd.isna(fruit_name):
            return None
        return self.fruit_standardizer.get(str(fruit_name).lower(), None)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data:
        - Standardize fruit names
        - Parse dates
        - Convert prices to numeric
        - Remove invalid rows
        """
        if df.empty:
            return df
        
        logger.info("Cleaning data...")
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Mandi API field aliases → normalize to standard names
        col_aliases = {
            'arrival_date': 'date', 'arrivaldate': 'date', 'price_date': 'date',
            'commodity': 'commodity', 'commodity_name': 'commodity', 'crop': 'commodity',
            'modal_price': 'modal_price', 'modalprice': 'modal_price',
            'modal price': 'modal_price', 'price': 'modal_price',
            'state': 'state', 'state_name': 'state',
        }
        df.rename(columns=col_aliases, inplace=True)

        # Keep relevant columns
        required_cols = ['commodity', 'state', 'modal_price']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}. Got: {list(df.columns)}")
            return pd.DataFrame()

        # Parse dates — use today if no date column present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = pd.Timestamp.now().normalize()
        df = df[df['date'].notna()]
        
        # Standardize fruit names
        df['fruit'] = df['commodity'].apply(self.standardize_fruit)
        df = df[df['fruit'].notna()]  # Remove unmapped fruits
        
        # Keep only top fruits
        top_fruits = df['fruit'].value_counts().head(8).index
        df = df[df['fruit'].isin(top_fruits)]
        
        # Keep only top states
        df = df[df['state'].isin(TOP_STATES)]
        
        # Convert price to numeric
        df['price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df = df[df['price'].notna() & (df['price'] > 0)]
        
        # Remove outliers: drop if price outside min-max range for that fruit
        for fruit in df['fruit'].unique():
            fruit_data = df[df['fruit'] == fruit]['price']
            if len(fruit_data) > 0:
                q1 = fruit_data.quantile(0.01)
                q99 = fruit_data.quantile(0.99)
                df.loc[df['fruit'] == fruit, 'price'] = df.loc[
                    df['fruit'] == fruit, 'price'
                ].clip(lower=q1, upper=q99)
        
        # Add El Niño fields
        df['is_el_nino'] = df['date'].apply(lambda x: self.oni.classify(x)[0])
        df['oni_strength'] = df['date'].apply(lambda x: self.oni.classify(x)[1])
        
        # Final dataframe
        result = df[[
            'date', 'fruit', 'state', 'price', 'is_el_nino', 'oni_strength'
        ]].copy()
        result.columns = ['date', 'fruit', 'state', 'modal_price', 'is_el_nino', 'oni_strength']
        
        # Remove duplicates (same fruit, state, date)
        result = result.drop_duplicates(subset=['date', 'fruit', 'state'], keep='first')
        
        logger.info(f"Cleaned data: {len(result)} rows remaining")
        return result
    
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to median price per date, state, fruit
        This matches the cleaned dataset structure
        """
        if df.empty:
            return df
        
        logger.info("Aggregating to daily medians...")
        
        aggregated = df.groupby(['date', 'fruit', 'state']).agg({
            'modal_price': 'median',
            'is_el_nino': 'first',  # Same for all records on same date
            'oni_strength': 'first'
        }).reset_index()
        
        logger.info(f"Aggregated data: {len(aggregated)} rows")
        return aggregated
    
    def load_existing_csv(self) -> pd.DataFrame:
        """Load existing clean CSV"""
        if os.path.exists(OUTPUT_CSV):
            try:
                df = pd.read_csv(OUTPUT_CSV)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Loaded {len(df)} existing rows from {OUTPUT_CSV}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing CSV: {e}")
        return pd.DataFrame()
    
    def append_new_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Append new data, avoiding duplicates"""
        if new_df.empty:
            logger.warning("No new data to append")
            return existing_df
        
        if existing_df.empty:
            combined = new_df.copy()
        else:
            # Find dates already in existing data
            existing_dates = set(existing_df['date'].dt.date)
            new_dates = new_df['date'].dt.date.unique()
            
            dates_to_add = [d for d in new_dates if d not in existing_dates]
            
            if not dates_to_add:
                logger.info("No new dates to add")
                return existing_df
            
            new_data_to_add = new_df[new_df['date'].dt.date.isin(dates_to_add)]
            logger.info(f"Adding {len(new_data_to_add)} rows for {len(dates_to_add)} new dates")
            
            combined = pd.concat([existing_df, new_data_to_add], ignore_index=True)
        
        # Sort by date
        combined = combined.sort_values('date').reset_index(drop=True)
        return combined
    
    def process_daily(self, date: datetime = None) -> bool:
        """
        Main pipeline: fetch, clean, and append data
        Returns True if data was added
        """
        logger.info("="*60)
        logger.info("Starting daily fruit price pipeline")
        logger.info("="*60)
        
        # Fetch
        raw_df = self.fetch_from_api(date)
        if raw_df.empty:
            logger.warning("No data fetched")
            return False
        
        # Clean
        cleaned_df = self.clean_data(raw_df)
        if cleaned_df.empty:
            logger.warning("No data after cleaning")
            return False
        
        # Aggregate
        agg_df = self.aggregate_daily(cleaned_df)
        if agg_df.empty:
            logger.warning("No data after aggregation")
            return False
        
        # Load existing
        existing_df = self.load_existing_csv()
        
        # Append
        combined_df = self.append_new_data(existing_df, agg_df)
        
        # Save
        try:
            combined_df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Saved {len(combined_df)} rows to {OUTPUT_CSV}")
            
            # Also save raw for archive
            if not existing_df.empty:
                raw_df.to_csv(ARCHIVE_CSV, index=False, mode='a', header=False)
            else:
                raw_df.to_csv(ARCHIVE_CSV, index=False)
            
            logger.info("Pipeline completed successfully")
            return len(combined_df) > len(existing_df)
        
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return False


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fruit Price Data Pipeline')
    parser.add_argument('--date', type=str, help='Date to fetch (YYYY-MM-DD), defaults to yesterday')
    parser.add_argument('--api-key', type=str, help='data.gov.in API key')
    args = parser.parse_args()
    
    date = None
    if args.date:
        try:
            date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    
    processor = FruitPriceProcessor(api_key=args.api_key)
    success = processor.process_daily(date)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

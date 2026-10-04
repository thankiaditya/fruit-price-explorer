#!/usr/bin/env python3
"""
Fruit Price Data Pipeline
Fetches daily mandi price data, cleans it, and appends to CSV with El Niño fields.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple
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
MANDI_API_URL = "https://mandi-api.onrender.com/v1/prices"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
OUTPUT_CSV = "fruit_prices_clean.csv"

# Fruit mapping for standardization
FRUIT_MAPPING = {
    'Banana': ['banana', 'banana - green', 'banana - ripe'],
    'Mango':  ['mango', 'mango (raw-ripe)', 'mango - raw', 'mango - ripe'],
    'Apple':  ['apple'],
    'Orange': ['orange', 'mosambi'],
    'Tomato': ['tomato'],
    'Onion':  ['onion'],
    'Potato': ['potato'],
    'Grape':  ['grape', 'grapes'],
}

# States covered by Mandi API
MANDI_STATES = ['Maharashtra', 'Uttar Pradesh', 'Punjab', 'Madhya Pradesh', 'Karnataka']

# El Niño thresholds
ONI_EL_NINO  =  0.5
ONI_LA_NINA  = -0.5


class ONILookup:
    """Fetch and classify ONI (El Niño) index from NOAA."""

    def __init__(self):
        self.oni_data = self._fetch()

    def _fetch(self) -> Dict[Tuple[int, int], float]:
        try:
            logger.info("Fetching ONI data from NOAA...")
            r = requests.get(ONI_URL, timeout=15)
            r.raise_for_status()
            oni = {}
            for line in r.text.splitlines():
                parts = line.split()
                # Format: YYYY MM  anom  ...  (first 3 columns we need)
                if len(parts) >= 3:
                    try:
                        year  = int(parts[0])
                        month = int(parts[1])
                        val   = float(parts[2])
                        oni[(year, month)] = val
                    except ValueError:
                        continue
            logger.info(f"ONI loaded: {len(oni)} months")
            return oni
        except Exception as e:
            logger.warning(f"ONI fetch failed: {e} — El Niño fields will be 0")
            return {}

    def classify(self, ts: pd.Timestamp) -> Tuple[bool, int]:
        val = self.oni_data.get((ts.year, ts.month))
        if val is None:
            return False, 0
        if val >= 1.5:  return True,  2
        if val >= ONI_EL_NINO: return True,  1
        if val <= -1.5: return False, -2
        if val <= ONI_LA_NINA: return False, -1
        return False, 0


class FruitPriceProcessor:
    """Fetch, clean, aggregate and store daily fruit prices."""

    def __init__(self):
        self.oni = ONILookup()
        self.fruit_map = {v.lower(): k
                          for k, variants in FRUIT_MAPPING.items()
                          for v in variants}

    def fetch(self) -> pd.DataFrame:
        all_rows = []
        for state in MANDI_STATES:
            for fruit_std, variants in FRUIT_MAPPING.items():
                # Try each variant name the API might know
                for commodity in variants:
                    try:
                        params = {'state': state, 'commodity': commodity}
                        r = requests.get(MANDI_API_URL, params=params, timeout=20)
                        if r.status_code == 404:
                            continue
                        r.raise_for_status()
                        data = r.json()
                        records = data if isinstance(data, list) else \
                                  data.get('data', data.get('records', []))
                        if records:
                            for rec in records:
                                rec['_state_std']  = state
                                rec['_fruit_std']  = fruit_std
                            all_rows.extend(records)
                            logger.info(f"  {state} / {commodity}: {len(records)} rows")
                            break   # got data for this fruit — skip other variants
                    except Exception as e:
                        logger.warning(f"  {state}/{commodity}: {e}")
                        continue

        if not all_rows:
            logger.warning("No records from Mandi API")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        logger.info(f"Total fetched: {len(df)} rows across {len(MANDI_STATES)} states")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        logger.info("Cleaning...")
        df.columns = df.columns.str.lower().str.strip()

        # Normalise column names from API variance
        aliases = {
            'arrival_date': 'date', 'arrivaldate': 'date', 'price_date': 'date',
            'modal_price': 'modal_price', 'modalprice': 'modal_price', 'price': 'modal_price',
            'commodity': 'commodity', 'commodity_name': 'commodity',
            'state': 'state', 'state_name': 'state',
            '_state_std': 'state_std', '_fruit_std': 'fruit_std',
        }
        df.rename(columns=aliases, inplace=True)

        # Date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = pd.Timestamp.now().normalize()
        df = df[df['date'].notna()]

        # Use pre-mapped fruit/state from fetch loop when available
        if 'fruit_std' not in df.columns:
            df['fruit_std'] = df['commodity'].str.lower().map(self.fruit_map)
        if 'state_std' not in df.columns:
            df['state_std'] = df.get('state', pd.Series(dtype=str))

        df = df[df['fruit_std'].notna() & df['state_std'].notna()]

        # Price
        df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df = df[df['modal_price'].notna() & (df['modal_price'] > 0)]

        # Trim outliers per fruit (1–99 %)
        for fruit in df['fruit_std'].unique():
            mask = df['fruit_std'] == fruit
            q1, q99 = df.loc[mask, 'modal_price'].quantile([0.01, 0.99])
            df.loc[mask, 'modal_price'] = df.loc[mask, 'modal_price'].clip(q1, q99)

        # El Niño
        df['is_el_nino']  = df['date'].apply(lambda x: self.oni.classify(x)[0])
        df['oni_strength'] = df['date'].apply(lambda x: self.oni.classify(x)[1])

        result = df[['date','fruit_std','state_std','modal_price','is_el_nino','oni_strength']].copy()
        result.columns = ['date','fruit','state','modal_price','is_el_nino','oni_strength']
        result = result.drop_duplicates(subset=['date','fruit','state'])
        logger.info(f"After cleaning: {len(result)} rows")
        return result

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        agg = df.groupby(['date','fruit','state']).agg(
            modal_price=('modal_price','median'),
            is_el_nino=('is_el_nino','first'),
            oni_strength=('oni_strength','first'),
        ).reset_index()
        logger.info(f"After aggregation: {len(agg)} rows")
        return agg

    def load_existing(self) -> pd.DataFrame:
        if os.path.exists(OUTPUT_CSV):
            try:
                df = pd.read_csv(OUTPUT_CSV)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Loaded existing CSV: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Could not load CSV: {e}")
        return pd.DataFrame()

    def append(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if new.empty:
            return existing
        if existing.empty:
            return new.sort_values('date').reset_index(drop=True)

        existing_dates = set(existing['date'].dt.date)
        new_only = new[~new['date'].dt.date.isin(existing_dates)]
        if new_only.empty:
            logger.info("No new dates to add")
            return existing

        logger.info(f"Adding {len(new_only)} new rows")
        combined = pd.concat([existing, new_only], ignore_index=True)
        return combined.sort_values('date').reset_index(drop=True)

    def run(self) -> bool:
        logger.info("=" * 60)
        logger.info("Fruit Price Pipeline — starting")
        logger.info("=" * 60)

        raw  = self.fetch()
        if raw.empty:  return False

        clean = self.clean(raw)
        if clean.empty: return False

        agg   = self.aggregate(clean)
        if agg.empty:  return False

        existing = self.load_existing()
        combined = self.append(existing, agg)

        try:
            combined.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Saved {len(combined)} rows → {OUTPUT_CSV}")
            return len(combined) > len(existing)
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='YYYY-MM-DD (unused — API returns latest)')
    args = parser.parse_args()
    success = FruitPriceProcessor().run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

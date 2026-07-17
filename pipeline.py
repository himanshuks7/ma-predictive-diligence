import pandas as pd
import numpy as np
import os

class MAProductDataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_df = None
        self.cleaned_df = None

    def load_data(self):
        """Safely ingest the raw corporate M&A dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please download it from Kaggle.")
        self.raw_df = pd.read_csv(self.data_path)
        print(f"Successfully ingested raw dataset. Total records: {len(self.raw_df)}")
        return self.raw_df

    def process_and_clean(self):
        """Clean messy real-world text entries and transform them into quantitative product metrics."""
        df = self.raw_df.copy()

        # 1. Standardize text features
        df['Parent Company'] = df['Parent Company'].str.strip().str.upper()
        df['Category'] = df['Category'].fillna('Uncategorized').str.strip().str.title()

        # 2. Financial Engineering: Parse string values to numeric representations
        df['Acquisition Price'] = pd.to_numeric(df['Value (USD)'], errors='coerce') if 'Value (USD)' in df.columns else pd.to_numeric(df['Acquisition Price'], errors='coerce')
        
        # Fill missing deal sizes with the median value to avoid dropping data
        median_price = df['Acquisition Price'].median()
        df['Acquisition Price'] = df['Acquisition Price'].fillna(median_price)

        # 3. Feature Derivation: Add metrics to assess M&A risk profiles
       # 3. Feature Derivation: Add metrics to assess M&A risk profiles
        df['Deal_Scale_Tier'] = pd.qcut(df['Acquisition Price'].rank(method='first'), q=3, labels=['Small-Cap', 'Mid-Market', 'Mega-Deal'])
        # Regulatory Exposure Score (Based on Sector Congestion)
        category_counts = df['Category'].value_counts().to_dict()
        df['Sector_Congestion_Index'] = df['Category'].map(category_counts)
        max_congestion = df['Sector_Congestion_Index'].max()
        df['Sector_Congestion_Index'] = df['Sector_Congestion_Index'] / (max_congestion if max_congestion > 0 else 1)

        # 4. Generate Target Labels for the ML Model (Simulating Deal Outcome Statuses)
        np.random.seed(42)
        df['Deal_Status'] = 'COMPLETED'
        
        # Inject systematic regulatory blocks (High value + Crowded tech sector = high chance of block)
        block_condition = (df['Acquisition Price'] > df['Acquisition Price'].quantile(0.75)) & (df['Sector_Congestion_Index'] > 0.5)
        df.loc[block_condition, 'Deal_Status'] = np.random.choice(
            ['LAPSED_REGULATORY_BLOCK', 'COMPLETED'], 
            size=block_condition.sum(), 
            p=[0.65, 0.35]
        )
        
        self.cleaned_df = df
        print("Data cleaning and feature engineering complete.")
        return self.cleaned_df

    def save_clean_dataset(self, output_path):
        """Persist the structured data layer for downstream ML modeling and dashboard consumption."""
        if self.cleaned_df is not None:
            self.cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned analytical asset saved successfully to: {output_path}")
        else:
            print("No cleaned data available to save. Run process_and_clean() first.")

if __name__ == "__main__":
    # Define local data paths
    INPUT_FILE = "data/acquisitions_update_2021.csv"
    OUTPUT_FILE = "data/cleaned_ma_data.csv"

    # Execute the data pipeline end-to-end
    pipeline = MAProductDataPipeline(INPUT_FILE)
    pipeline.load_data()
    pipeline.process_and_clean()
    pipeline.save_clean_dataset(OUTPUT_FILE)
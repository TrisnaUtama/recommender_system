import pandas as pd
from collaborative_filtering.cf import CollaborativeFiltering
from utils.db import fetch_ratings_from_db
from config import MODEL_PATH
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
retrain_logger = logging.getLogger(__name__)

def retrain_and_save_model():
    retrain_logger.info("Starting model retraining process...")

    db_df = fetch_ratings_from_db()
    if db_df.empty:
        retrain_logger.warning("No new rating data found in the database. Skipping model retraining.")
        return

    try:
        csv_df = pd.read_csv('./dataset/data_rating_clean.csv')
        csv_df.columns = csv_df.columns.str.replace("'", "").str.strip()
        retrain_logger.info(f"Loaded {len(csv_df)} records from CSV.")
    except FileNotFoundError:
        retrain_logger.error("CSV file './dataset/data_rating_clean.csv' not found. Cannot combine data.")
        return
    except Exception as e:
        retrain_logger.error(f"Error loading CSV data: {e}")
        return

    retrain_logger.info(f"Fetched {len(db_df)} records from the database.")

    combined_df = pd.concat([csv_df, db_df], ignore_index=True).drop_duplicates()
    retrain_logger.info(f"Combined data has {len(combined_df)} unique records after deduplication.")

    model = CollaborativeFiltering()
    model.df = combined_df
    model.all_places = combined_df['destination_id'].unique().tolist()
    
    retrain_logger.info("Fitting the Collaborative Filtering model...")
    model.fit() 

    retrain_logger.info(f"Saving the retrained model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH) 

    retrain_logger.info("Model retraining and saving process completed successfully.")


if __name__ == "__main__":
    retrain_and_save_model()
import pandas as pd
from collaborative_filtering.cf import CollaborativeFiltering
from utils.db import fetch_ratings_from_db
from config import MODEL_PATH

def retrain_and_save_model():
    print("Retraining model...")

    db_df = fetch_ratings_from_db()
    if db_df.empty:
        print("No rating data in database. Skipping retrain.")
        return

    csv_df = pd.read_csv('./dataset/data_rating_clean.csv')
    csv_df.columns = csv_df.columns.str.replace("'", "").str.strip()

    combined_df = pd.concat([csv_df, db_df], ignore_index=True).drop_duplicates()

    model = CollaborativeFiltering()
    model.df = combined_df
    model.all_places = combined_df['destination_id'].unique().tolist()
    model.fit()
    model.save_model(MODEL_PATH)

    print("Model retrained and saved with DB + CSV.")


if __name__ == "__main__":
    retrain_and_save_model()

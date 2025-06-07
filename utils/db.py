import psycopg2
import pandas as pd
from config import DB_CONFIG

def fetch_ratings_from_db():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT "userId" AS customer_id,
       "targetId" AS destination_id,
       "ratingValue" AS customer_rating
        FROM "Rating"
        WHERE "ratingValue" IS NOT NULL
        AND "ratedType" = 'DESTINATION'
        AND "status" = TRUE
        AND "deleted_at" IS NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

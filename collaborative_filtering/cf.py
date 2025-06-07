import pandas as pd
from surprise import Dataset, Reader, KNNBasic
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = None
        self.trainset = None
        self.all_places = []
        logger.info("CollaborativeFiltering instance initialized.")

    def fit(self):
        if self.df.empty:
            logger.warning("DataFrame is empty. Cannot fit the model without data.")
            return

        self.all_places = self.df['destination_id'].unique().tolist()
        logger.info(f"Found {len(self.all_places)} unique destinations.")

        reader = Reader(rating_scale=(
            self.df['customer_rating'].min(),
            self.df['customer_rating'].max()
        ))
        data = Dataset.load_from_df(
            self.df[['customer_id', 'destination_id', 'customer_rating']], reader
        )
        self.trainset = data.build_full_trainset()
        logger.info(f"Trainset built with {self.trainset.n_users} users and {self.trainset.n_items} items.")

        sim_options = {
            'name': 'cosine',
            'user_based': True
        }

        self.model = KNNBasic(sim_options=sim_options)
        self.model.fit(self.trainset)
        logger.info("KNNBasic model fitted successfully.")

    def recommend_places(self, user_id, k=5):
        if self.model is None:
            logger.error("Model has not been fitted yet. Call .fit() before recommending.")
            return pd.DataFrame(columns=['destination_id', 'predicted_score'])

        rated = self.df[self.df['customer_id'] == user_id]['destination_id'].tolist()
        not_rated = [p for p in self.all_places if p not in rated]

        logger.debug(f"User {user_id} has rated: {rated}")
        logger.debug(f"Candidates for recommendation (not yet rated by {user_id}): {not_rated}")

        predictions = []
        for place in not_rated:
            try:
                # Use model.predict's optional `r_ui` and `details` arguments for more info if needed
                pred = self.model.predict(user_id, place)
                logger.debug(f"Prediction for user {user_id} and place {place}: {pred.est:.2f}")
                predictions.append((place, pred.est))
            except Exception as e:
                logger.error(f"Failed predicting for user {user_id} and place {place}: {e}")

        recommendations = pd.DataFrame(predictions, columns=['destination_id', 'predicted_score'])
        recommendations.sort_values(by='predicted_score', ascending=False, inplace=True)
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}.")
        return recommendations.head(k)

    def save_model(self, filepath):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump((self.model, self.trainset, self.df, self.all_places), f)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")

    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, tuple):
                    if len(data) == 4:
                        self.model, self.trainset, self.df, self.all_places = data
                        logger.info(f"Model loaded successfully from {filepath} (full data).")
                    elif len(data) == 2:
                        self.model, self.trainset = data
                        self.df = pd.DataFrame() # Initialize empty as they were not in the pickle
                        self.all_places = []    # Initialize empty as they were not in the pickle
                        logger.warning(f"Model loaded successfully from {filepath} (partial data - df and all_places are empty).")
                    else:
                        raise ValueError(f"Unexpected number of objects in pickle: {len(data)}")
                else:
                    raise TypeError("Pickle content is not a tuple")
        except FileNotFoundError:
            logger.error(f"Model file not found at {filepath}")
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Failed to unpickle model from {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading model from {filepath}: {e}")
            raise
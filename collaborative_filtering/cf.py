import pandas as pd
from surprise import Dataset, Reader, KNNBasic
import pickle

class CollaborativeFiltering:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = None
        self.trainset = None
        self.all_places = []

    def fit(self):
        self.all_places = self.df['destination_id'].unique().tolist()
        reader = Reader(rating_scale=(
            self.df['customer_rating'].min(),
            self.df['customer_rating'].max()
        ))
        data = Dataset.load_from_df(
            self.df[['customer_id', 'destination_id', 'customer_rating']], reader
        )
        self.trainset = data.build_full_trainset()

        sim_options = {
            'name': 'cosine',
            'user_based': True
        }

        self.model = KNNBasic(sim_options=sim_options)
        self.model.fit(self.trainset)

    def recommend_places(self, user_id, k=5):
        rated = self.df[self.df['customer_id'] == user_id]['destination_id'].tolist()
        not_rated = [p for p in self.all_places if p not in rated]

        print(f"[DEBUG] Rated by {user_id}: {rated}")
        print(f"[DEBUG] Not rated candidates: {not_rated}")

        predictions = []
        for place in not_rated:
            try:
                pred = self.model.predict(user_id, place)
                print(f"[DEBUG] Prediction for {place}: {pred.est}")
                predictions.append((place, pred.est))
            except Exception as e:
                print(f"[ERROR] Failed predicting for {place}: {e}")

        recommendations = pd.DataFrame(predictions, columns=['destination_id', 'predicted_score'])
        print(f"[DEBUG] Total predictions: {len(recommendations)}")
        recommendations.sort_values(by='predicted_score', ascending=False, inplace=True)
        return recommendations.head(k)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.model, self.trainset, self.df, self.all_places), f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple):
                if len(data) == 4:
                    self.model, self.trainset, self.df, self.all_places = data
                elif len(data) == 2:
                    self.model, self.trainset = data
                    self.df = pd.DataFrame()
                    self.all_places = []
                else:
                    raise ValueError(f"Unexpected number of objects in pickle: {len(data)}")
            else:
                raise TypeError("Pickle content is not a tuple")

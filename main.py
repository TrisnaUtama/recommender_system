from fastapi import FastAPI, HTTPException
from collaborative_filtering.cf import CollaborativeFiltering
from models.request import RequestBody
import threading
from config import MODEL_PATH
from retrain_model import retrain_and_save_model
import threading


app = FastAPI()
cf_model = CollaborativeFiltering()
model_lock = threading.Lock()

@app.on_event("startup")
def load_model_on_startup():
    try:
        cf_model.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/recommend")
def recommend_places(body: RequestBody):
    print("BODY:", body)
    print("userId:", body.userId)
    user_id = body.userId
    with model_lock:
        if cf_model.df.empty or user_id not in cf_model.df['customer_id'].unique():
            raise HTTPException(status_code=404, detail="User ID not found in model data")
        recommendations = cf_model.recommend_places(user_id, k=5)
    return recommendations.to_dict(orient='records')

@app.post("/retrain-model")
def retrain_model():
    def background_retrain():
        with model_lock:
            retrain_and_save_model()
            cf_model.load_model(MODEL_PATH)
    threading.Thread(target=background_retrain).start()
    return {"message": "Retrain and reload triggered in background."}

@app.post("/reload-model")
def reload_model():
    def background_reload():
        try:
            cf_model.load_model(MODEL_PATH)
            print("Model reloaded successfully.")
        except Exception as e:
            print(f"Error reloading model: {e}")

    threading.Thread(target=background_reload).start()
    return {"message": "Model reload triggered in background."}


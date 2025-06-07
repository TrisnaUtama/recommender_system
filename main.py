from fastapi import FastAPI, HTTPException
from collaborative_filtering.cf import CollaborativeFiltering
from models.request import RequestBody
import threading
from config import MODEL_PATH
from retrain_model import retrain_and_save_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__) 

app = FastAPI()
cf_model = CollaborativeFiltering() 
model_lock = threading.Lock()

@app.on_event("startup")
def load_model_on_startup():
    try:
        cf_model.load_model(MODEL_PATH)
        app_logger.info("Model loaded successfully on startup.")
    except Exception as e:
        app_logger.error(f"Failed to load model on startup: {e}")

@app.post("/recommend")
def recommend_places(body: RequestBody):
    user_id = body.userId
    with model_lock:
        if cf_model.df.empty:
            app_logger.warning(f"Attempted recommendation for user {user_id} but model data is empty.")
            raise HTTPException(status_code=500, detail="Model data is not available. Please ensure the model is loaded/trained.")
        if user_id not in cf_model.df['customer_id'].unique():
            app_logger.warning(f"User ID {user_id} not found in model data for recommendation.")
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in model data.")

        app_logger.info(f"Requesting recommendations for user ID: {user_id}")
        recommendations = cf_model.recommend_places(user_id, k=5)

    if recommendations.empty:
        app_logger.info(f"No recommendations found for user ID: {user_id}")
        return [] 
    
    app_logger.info(f"Successfully generated recommendations for user ID: {user_id}")
    return recommendations.to_dict(orient='records')

@app.post("/retrain-model")
def retrain_model():
    app_logger.info("Retrain model endpoint called.")
    def background_retrain():
        app_logger.info("Starting background model retraining and reloading.")
        with model_lock:
            try:
                retrain_and_save_model() 
                cf_model.load_model(MODEL_PATH)
                app_logger.info("Model retraining and reloading completed successfully in background.")
            except Exception as e:
                app_logger.error(f"Error during background model retraining and reloading: {e}")
    
    threading.Thread(target=background_retrain).start()
    return {"message": "Retrain and reload triggered in background. Check logs for progress."}

@app.post("/reload-model")
def reload_model():
    app_logger.info("Reload model endpoint called.")
    def background_reload():
        try:
            with model_lock: 
                cf_model.load_model(MODEL_PATH)
            app_logger.info("Model reloaded successfully in background.")
        except Exception as e:
            app_logger.error(f"Error reloading model in background: {e}")

    threading.Thread(target=background_reload).start()
    return {"message": "Model reload triggered in background. Check logs for progress."}
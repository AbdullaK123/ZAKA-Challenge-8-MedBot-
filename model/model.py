from pathlib import Path
import keras
import joblib
import pandas as pd
import numpy as np
import logging
from schemas import ModelInput, PredictionResponse

MODEL_DIR = Path(__file__).parent

class StrokePredictor:
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    model = keras.saving.load_model(MODEL_DIR / "model.keras")
    logger = logging.getLogger(__name__)

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.logger.info(f"StrokePredictor initialized | threshold={self.threshold}")

    def _preprocess(self, data: dict) -> np.ndarray:
        try:
            validated = ModelInput(**data)
            df = pd.DataFrame([validated.model_dump()])
            transformed = self.preprocessor.transform(df)
            self.logger.debug(f"Input validation passed | fields={list(data.keys())}")
            return transformed
        except Exception as e:
            self.logger.error(f"Preprocessing failed | error={e}")
            raise

    def predict(self, data: dict) -> dict:
        self.logger.info(f"Prediction requested | input={data}")
        try:
            transformed = self._preprocess(data)
            predicted_probability = float(self.model.predict(transformed)[0][0])
            predicted_class = "Stroke" if predicted_probability >= self.threshold else "No Stroke"
            self.logger.info(f"Prediction complete | class={predicted_class} | p={predicted_probability:.4f}")
            response = PredictionResponse(
                predicted_probability=predicted_probability,
                predicted_class=predicted_class
            )
            return response.model_dump()
        except Exception as e:
            self.logger.error(f"Prediction failed | error={e}")
            raise
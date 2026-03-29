# MedBot — Stroke Risk Predictor

A web app that predicts a patient's stroke risk based on a few health inputs. It uses a Keras neural network trained on the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and serves predictions through a Flask API with a simple browser-based UI.

## What the Model Does

The model takes five inputs — age, marital status, smoking status, heart disease, and hypertension — and outputs a stroke probability along with a "Stroke" or "No Stroke" classification (using a 0.5 threshold).

Under the hood it's a small feedforward neural network (3 dense layers with dropout) trained with binary crossentropy. Input features are preprocessed with robust scaling (for age) and one-hot encoding (for categorical fields). The training data was balanced using random undersampling.

## Setup — Running Locally with Docker

Make sure you have [Docker](https://docs.docker.com/get-docker/) installed.

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd ZAKA-Challenge-8-MedBot
   ```

2. **Build the image**
   ```bash
   docker build -t medbot .
   ```

3. **Run the container**
   ```bash
   docker run -p 5000:5000 medbot
   ```

4. **Open the app**
   Go to [http://localhost:5000](http://localhost:5000) in your browser.

## How to Use the Interface

1. Open the app in your browser.
2. Fill in the form:
   - **Age** — patient's age (0–120)
   - **Ever Married** — check if the patient has been married
   - **Smoking Status** — select from the dropdown (Unknown, Never Smoked, Formerly Smoked, Smokes)
   - **Heart Disease** — check if the patient has heart disease
   - **Hypertension** — check if the patient has hypertension
3. Click **Predict**.
4. The result will show the predicted class ("Stroke" or "No Stroke") and the probability score.

You can also call the API directly:

```bash
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "ever_married": "Yes", "smoking_status": "formerly smoked", "heart_disease": 1, "hypertension": 1}'
```

## Known Issues and Limitations

- **Limited input features.** The model only uses 5 of the 11 features available in the original dataset (e.g., BMI, glucose level, gender, and work type are not used). Adding more features would likely improve accuracy.
- **Small and imbalanced dataset.** The original dataset has ~5,000 rows with only ~5% positive (stroke) cases. Undersampling balances the classes but reduces overall training data.
- **Fixed threshold.** The classification threshold is hardcoded at 0.5. A tuned threshold (e.g., based on ROC analysis) could improve recall for the stroke class.
- **Not for medical use.** This is a learning project. Do not use it for real clinical decisions.

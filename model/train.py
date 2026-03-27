import keras
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = "data/healthcare-dataset-stroke-data.csv"


def make_model(n_features: int, hidden_units: int = 128, learning_rate: float = 0.001, dropout_rate: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(hidden_units, activation='relu')(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(hidden_units // 2, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(hidden_units // 4, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'f1_score']
    )
    return model


def train(
    df: pd.DataFrame,
    num_features: list[str],
    cat_ohe_features: list[str],
    cat_passthrough_features: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    **model_params,
):
    all_features = num_features + cat_ohe_features + cat_passthrough_features
    X = df[all_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 1: Resample
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # Step 2: Preprocess
    preprocessor = ColumnTransformer(transformers=[
        ("scale", RobustScaler(quantile_range=(0.01, 0.99)), num_features),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_ohe_features),
        ("keep", "passthrough", cat_passthrough_features),
    ])

    X_processed = preprocessor.fit_transform(X_resampled)

    # Step 3: Train
    n_features = X_processed.shape[1]
    model = make_model(n_features, **model_params)
    model.fit(X_processed, y_resampled, epochs=10, batch_size=32)

    # Save separately
    joblib.dump(preprocessor, "model/preprocessor.joblib")
    model.save("model/model.keras")

    return preprocessor, model, X_test, y_test


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR)

    preprocessor, model, X_test, y_test = train(
        df=df,
        num_features=['age'],
        cat_ohe_features=['ever_married', 'smoking_status'],
        cat_passthrough_features=['hypertension', 'heart_disease'],
        target='stroke',
        hidden_units=64,
        learning_rate=0.001,
        dropout_rate=0.3,
    )
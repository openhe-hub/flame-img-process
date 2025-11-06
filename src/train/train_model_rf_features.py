"""
Train a Random Forest model with additional engineered features.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_CSV_PATH = Path("data/dataset/total_dataset.csv")
TARGET_VARIABLE = "tip_distance"

LABEL_COLUMNS = [
    "area",
    "arc_length",
    "area_vec",
    "regression_circle_center",
    "regression_circle_radius",
    "expand_dist",
    "expand_vec",
    "tip_distance",
]

FEATURES_TO_DROP = ["experiment_name", "frame_id"] + LABEL_COLUMNS
GAS_DENSITIES: Dict[str, float] = {
    "Ar": 1.784,   # kg/m^3
    "He": 0.1786,
    "N2": 1.2506,
}
BACKGROUND_DENSITY = 1.225  # Air at STP (kg/m^3)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pressure ratio (safe divide)
    pressure2_safe = df["pressure2"].replace(0, pd.NA)
    df["pressure_ratio"] = df["pressure1"] / pressure2_safe
    df["pressure_ratio"] = df["pressure_ratio"].replace([np.inf, -np.inf], pd.NA)

    # Mixture density based on gas type and oxygen fraction
    gas_density = df["gas_type"].map(GAS_DENSITIES).fillna(BACKGROUND_DENSITY)
    fraction = df["gas_percent"] / 100.0
    df["mixture_density"] = fraction * gas_density + (1.0 - fraction) * BACKGROUND_DENSITY

    return df


def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        logger.success(f"Loaded dataset from '{DATA_CSV_PATH}' with shape {df.shape}")
    except FileNotFoundError as exc:
        logger.error(
            f"Dataset not found at '{DATA_CSV_PATH}'. Please generate it before training."
        )
        raise exc

    logger.info("Removing duplicate frames based on ['experiment_name', 'frame_id']...")
    before = len(df)
    df.drop_duplicates(subset=["experiment_name", "frame_id"], keep="first", inplace=True)
    removed = before - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} duplicate rows. New shape: {df.shape}")
    else:
        logger.info("No duplicates detected.")

    df = add_engineered_features(df)
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df


def train_model() -> None:
    logger.info("Starting Random Forest training with engineered features...")

    df = load_dataset()
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=FEATURES_TO_DROP, errors="ignore")

    logger.info(f"Feature columns: {list(X.columns)}")
    logger.info(f"Using {X.shape[1]} features.")

    X = pd.get_dummies(X, columns=["gas_type"], drop_first=True)
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Split dataset into {len(X_train)} training samples and {len(X_test)} testing samples."
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    logger.info("Training RandomForestRegressor with engineered features...")
    model.fit(X_train_scaled, y_train)
    logger.success("Model training completed.")

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error (MSE): {mse:.6f}")
    logger.info(f"R-squared (R²) Score: {r2:.6f}")

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_importances = importances.nlargest(min(10, len(importances)))
    logger.info("Top feature importances:")
    print(top_importances)


if __name__ == "__main__":
    train_model()

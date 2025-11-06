from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_flame_model_autogluon():
    """
    Train Tabular AutoGluon models on the flame dataset and report metrics.
    """
    DATA_CSV_PATH = Path('data/dataset/total_dataset.csv')
    TARGET_VARIABLE = 'tip_distance'
    LABELS = ['area', 'arc_length', 'area_vec', 'regression_circle_center', 'regression_circle_radius', 'expand_dist', 'expand_vec', 'tip_distance']
    MODEL_OUTPUT_DIR = Path('artifacts/models/autogluon_flame')

    logger.info("Starting AutoGluon training pipeline...")

    if not DATA_CSV_PATH.exists():
        logger.error(f"Data file not found at {DATA_CSV_PATH}. Please run the data processing script first.")
        return

    df = pd.read_csv(DATA_CSV_PATH)
    logger.success(f"Successfully loaded data from {DATA_CSV_PATH}. Shape: {df.shape}")

    initial_rows = len(df)
    logger.info("Checking for duplicates based on ['experiment_name', 'frame_id']...")
    df.drop_duplicates(subset=['experiment_name', 'frame_id'], keep='first', inplace=True)
    final_rows = len(df)

    if initial_rows > final_rows:
        logger.warning(f"Removed {initial_rows - final_rows} duplicate rows. New shape: {df.shape}")
    else:
        logger.info("No duplicate rows found.")

    df.fillna(df.mean(numeric_only=True), inplace=True)

    labels_to_exclude = [label for label in LABELS if label != TARGET_VARIABLE]
    columns_to_exclude = ['experiment_name', 'frame_id'] + labels_to_exclude

    feature_columns = [col for col in df.columns if col not in columns_to_exclude + [TARGET_VARIABLE]]
    logger.info(f"Using {len(feature_columns)} features for training: {feature_columns}")

    model_df = df[feature_columns + [TARGET_VARIABLE]]

    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)
    logger.info(f"Data split into training ({train_df.shape[0]} samples) and testing ({test_df.shape[0]} samples).")

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training artifacts will be stored in {MODEL_OUTPUT_DIR.resolve()}")

    predictor = TabularPredictor(
        label=TARGET_VARIABLE,
        path=str(MODEL_OUTPUT_DIR),
        eval_metric='root_mean_squared_error'
    )

    presets = 'medium_quality_faster_train'
    logger.info(f"Starting AutoGluon fit with presets='{presets}'...")
    predictor.fit(train_data=train_df, presets=presets)
    logger.success("AutoGluon training completed.")

    X_test = test_df.drop(columns=[TARGET_VARIABLE])
    y_test = test_df[TARGET_VARIABLE]
    y_pred = predictor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"R-squared (R²) Score: {r2:.4f}")

    leaderboard_df = predictor.leaderboard(test_df, silent=True)
    logger.info("AutoGluon leaderboard (top models):")
    print(leaderboard_df.head())

    importance_df = predictor.feature_importance(test_df)
    logger.info("Top 10 Feature Importances:")
    print(importance_df['importance'].nlargest(10))

    y_test = y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    results_df = pd.DataFrame({
        'Ground Truth': y_test,
        'Predicted Value': y_pred
    })
    results_df['Absolute Error'] = (results_df['Ground Truth'] - results_df['Predicted Value']).abs()

    sample_count = min(10, len(results_df))
    if sample_count > 0:
        logger.info(f"Displaying {sample_count} examples of predictions vs. ground truth...")
        print("\n--- Prediction Examples ---")
        print(results_df.sample(sample_count, random_state=42))
        print("---------------------------")
    else:
        logger.warning("No samples available to display from the test set.")


if __name__ == '__main__':
    train_flame_model_autogluon()

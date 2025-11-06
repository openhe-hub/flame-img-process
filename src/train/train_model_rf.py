import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger

def train_flame_model():
    """
    A script to train a machine learning model on the aggregated flame dataset.
    This version includes a deduplication step.
    """
    # --- 1. Configuration ---
    DATA_CSV_PATH = 'data/dataset/total_dataset.csv'
    
    # Define the target variable we want to predict.
    # Common choices could be 'area', 'tip_distance', 'arc_length', etc.
    TARGET_VARIABLE = 'tip_distance' 

    LABELS = ['area', 'arc_length', 'area_vec', 'regression_circle_center', 'regression_circle_radius', 'expand_dist', 'expand_vec', 'tip_distance']
    
    # Define features to be used for training.
    # We exclude identifiers and the target variable itself.
    # 'time' is also excluded as it's highly correlated with frame_id.
    FEATURES_TO_DROP = ['experiment_name', 'frame_id'] + LABELS

    logger.info("Starting model training process...")

    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        logger.success(f"Successfully loaded data from {DATA_CSV_PATH}. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {DATA_CSV_PATH}. Please run the data processing script first.")
        return

    # --- 新增：根据 experiment_name 和 frame_id 去重 ---
    initial_rows = len(df)
    logger.info(f"Checking for duplicates based on ['experiment_name', 'frame_id']...")
    df.drop_duplicates(subset=['experiment_name', 'frame_id'], keep='first', inplace=True)
    final_rows = len(df)
    
    if initial_rows > final_rows:
        logger.warning(f"Removed {initial_rows - final_rows} duplicate rows. New shape: {df.shape}")
    else:
        logger.info("No duplicate rows found.")
    # --- 去重结束 ---

    # Handle potential missing values, e.g., by filling with the mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # --- 3. Feature Engineering ---
    logger.info(f"Target Variable: '{TARGET_VARIABLE}'")
    
    # Separate target variable (y) from features (X)
    y = df[TARGET_VARIABLE]
    X = df.drop(columns=FEATURES_TO_DROP)

    logger.info(f"Feature Variable: {X.columns}")

    # One-Hot Encode categorical features like 'gas_type'
    X = pd.get_dummies(X, columns=['gas_type'], drop_first=True)
    
    # Save feature names for later
    feature_names = X.columns
    logger.info(f"Using {len(feature_names)} features for training.")

    # --- 4. Data Splitting ---
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

    # --- 5. Feature Scaling ---
    # Scale numerical features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 6. Model Training ---
    logger.info("Training RandomForestRegressor model...")
    # Initialize the model with some default parameters
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    logger.success("Model training completed.")

    # --- 7. Model Evaluation ---
    logger.info("Evaluating model performance on the test set...")
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"R-squared (R²) Score: {r2:.4f}")

    # --- 8. Feature Importance ---
    logger.info("Top 10 Feature Importances:")
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_10 = importances.nlargest(10)
    print(top_10)

    # --- 9. Show Prediction Examples ---
    logger.info("Displaying some examples of predictions vs. ground truth...")
    
    # Create a DataFrame for comparison
    results_df = pd.DataFrame({
        'Ground Truth': y_test,
        'Predicted Value': y_pred
    })
    # Add a column for the absolute difference (error)
    results_df['Absolute Error'] = abs(results_df['Ground Truth'] - results_df['Predicted Value'])
    
    # Reset index to easily sample and display
    results_df.reset_index(drop=True, inplace=True)
    
    # Display 10 random examples from the test set
    print("\n--- Prediction Examples (10 random samples) ---")
    print(results_df.sample(10, random_state=42))
    print("---------------------------------------------")


if __name__ == '__main__':
    train_flame_model()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    """
    This function trains a wine quality prediction model.
    It loads the data, preprocesses it, trains a Random Forest Classifier,
    and saves the model and the scaler.
    """
    print("Starting model training process...")

    # Define file paths
    data_path = 'winequality.csv'
    model_path = 'wine_model.joblib'
    scaler_path = 'scaler.joblib'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    # 1. Load Data
    try:
        wine_df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Preprocessing
    # Binarize the 'quality' column. 1 for good (7 or higher), 0 for not good.
    wine_df['quality'] = [1 if x >= 7 else 0 for x in wine_df['quality']]
    
    # Separate features (X) and target (y)
    X = wine_df.drop('quality', axis=1)
    y = wine_df['quality']
    print("Data preprocessing completed.")

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # 4. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Features scaled using StandardScaler.")

    # 5. Model Training
    # Using RandomForestClassifier as it performed well in the notebook
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 6. Evaluation
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=['Not Good', 'Good']))
    print("------------------------\n")

    # 7. Save the model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("\nTraining process finished successfully!")

if __name__ == '__main__':
    train_model()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_rbf():
    print("Loading PCA preprocessed data...")
    # 1. Read processed data from the first stage (PCA)
    try:
        df = pd.read_csv('processed_data/data_pca.csv')
    except FileNotFoundError:
        print("Error: PCA data not found. Please ensure pca_module.py ran successfully and generated the data.")
        return

    # Separate features from target
    X = df.drop('target', axis=1)
    y = df['target']

    # 2. Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # 3. Build RBF (Radial Basis Function) network for prediction
    # Using SVC with RBF Kernel is mathematically the most robust way to build an RBF network in Python for classification
    print("\nTraining RBF prediction model...")
    rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    # Training process
    rbf_model.fit(X_train, y_train)

    # 4. Evaluate model performance
    y_pred = rbf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print("      RBF Model Results")
    print("="*30)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*30)

    # 5. Save the ready model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rbf_model, 'models/rbf_model.pkl')
    print("\n✅ RBF Model trained and saved successfully to 'models/rbf_model.pkl'.")

if __name__ == '__main__':
    train_rbf()

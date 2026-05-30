import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

def run_pca_analysis():
    print("--- Starting Advanced PCA Feature Extraction ---")
    
    # Load data
    file_path = 'heart data -2.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1, errors='ignore')
    feature_names = X.columns.tolist()

    # 1. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA - Select the number of components that cover 95% of variance
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    n_components = pca.n_components_
    print(f"\nDimensionality Reduced from {len(feature_names)} to {n_components} components.")
    print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # 3. Feature Importance / Loadings Analysis
    # We will see which original features affect PC1 and PC2 the most
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(n_components)], 
        index=feature_names
    )
    
    print("\n--- Key Feature Contributions (Top 5 per Component) ---")
    for i in range(min(2, n_components)):
        top_features = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
        print(f"\nPrincipal Component {i+1} is mainly driven by:")
        for feat, val in top_features.items():
            print(f" - {feat}: {val:.3f}")

    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca.pkl')
    
    # Save transformed data for subsequent training
    os.makedirs('processed_data', exist_ok=True)
    pd.DataFrame(X_pca).to_csv('processed_data/X_pca.csv', index=False)
    df['target'].to_csv('processed_data/y.csv', index=False)
    
    print("\nPCA Models and Processed Data saved successfully.")

if __name__ == '__main__':
    run_pca_analysis()

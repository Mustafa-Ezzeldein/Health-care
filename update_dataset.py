import pandas as pd
import numpy as np

def update_dataset():
    file_path = 'heart data -2.csv'
    print(f"Enhancing dataset {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Add Oxygen (SpO2) and Temperature based on target
    # Target 0: Healthy, Target 1: Sick
    
    np.random.seed(42)
    
    def generate_oxygen(target):
        if target == 0:
            return np.random.uniform(95.0, 100.0)
        else:
            return np.random.uniform(85.0, 94.0)

    def generate_temp(target):
        if target == 0:
            return np.random.uniform(36.4, 37.2)
        else:
            return np.random.uniform(37.5, 39.5)

    df['oxygen'] = df['target'].apply(generate_oxygen)
    df['temp'] = df['target'].apply(generate_temp)

    # Save to a new version or overwrite? To be safe, let's keep original and save to enhanced
    enhanced_path = 'heart data -2.csv' # Overwriting as requested to keep things simple for models
    df.to_csv(enhanced_path, index=False)
    
    print(f"✅ Dataset enhanced with 'oxygen' and 'temp' columns! Path: {enhanced_path}")
    print(df[['oxygen', 'temp', 'target']].head())

if __name__ == '__main__':
    update_dataset()

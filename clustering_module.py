import pandas as pd
import numpy as np
import os
import joblib

# 1. Self-Organizing Map (SOM)
class SimpleSOM:
    """
    The SOM algorithm builds a classification map that spreads patients on a grid
    where patients with similar symptoms will cluster at the same point without human intervention.
    """
    def __init__(self, x_size, y_size, input_len, learning_rate=0.5, epochs=100):
        self.x_size = x_size
        self.y_size = y_size
        self.input_len = input_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.radius = max(x_size, y_size) / 2.0
            
        np.random.seed(42)
        # Initialize weights
        self.weights = np.random.rand(x_size, y_size, input_len)

    def _euclidean_dist(self, vector, weights):
        return np.linalg.norm(vector - weights, axis=-1)

    def _find_bmu(self, vector):
        # Search for the Best Matching Unit (BMU)
        distances = self._euclidean_dist(vector, self.weights)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def train(self, data):
        print(f"Training SOM for {self.epochs} epochs...")
        initial_lr = self.learning_rate
        initial_radius = self.radius
        time_constant = self.epochs / np.log(self.radius) if self.radius > 1 else 1

        for epoch in range(self.epochs):
            decay = np.exp(-epoch / time_constant)
            curr_radius = initial_radius * decay
            curr_lr = initial_lr * decay
            
            for vector in data:
                bmu = self._find_bmu(vector)
                
                # Update surrounding weights to form a cluster
                for x in range(self.x_size):
                    for y in range(self.y_size):
                        dist_to_bmu = np.linalg.norm(np.array([x, y]) - np.array(bmu))
                        if dist_to_bmu <= curr_radius:
                            influence = np.exp(-(dist_to_bmu**2) / (2 * (curr_radius**2)))
                            self.weights[x, y] += curr_lr * influence * (vector - self.weights[x, y])

    def get_clusters(self, data):
        clusters = []
        for vector in data:
            bmu = self._find_bmu(vector)
            # Every patient gets (x,y) coordinates representing their cluster
            clusters.append(bmu)
        return np.array(clusters)

# 2. Adaptive Clustering (ART)
class ContinuousART:
    """
    Simulates the ART2 algorithm to discover unexpected new patterns (Novelty Detection).
    """
    def __init__(self, vigilance=0.9):
        self.vigilance = vigilance # Vigilance parameter (higher means stricter separation)
        self.clusters = []
        
    def train(self, data):
        print(f"Training ART with vigilance {self.vigilance}...")
        for x in data:
            matched = False
            for i, c in enumerate(self.clusters):
                # Calculate similarity
                similarity = np.dot(x, c) / (np.linalg.norm(x) * np.linalg.norm(c) + 1e-8)
                if similarity >= self.vigilance:
                    self.clusters[i] = (self.clusters[i] + x) / 2
                    matched = True
                    break
            if not matched:
                # If no similar pattern is found, discover an entirely new group (disease)
                self.clusters.append(x)
        print(f"ART discovered {len(self.clusters)} unique physical patterns (clusters).")

def run_clustering():
    try:
        # Read data reduced in the first stage (PCA)
        df = pd.read_csv('processed_data/data_pca.csv')
    except OSError:
        print("processed_data/data_pca.csv not found.")
        return
        
    # As requested, we remove the Label completely to test algorithm intelligence (Unsupervised Learning)
    X = df.drop('target', axis=1).values
    
    # 1. Train SOM
    print("\n--- Starting SOM Clustering (Zero Labels) ---")
    som = SimpleSOM(x_size=3, y_size=3, input_len=X.shape[1], epochs=50) # 3x3 grid = max 9 clusters
    som.train(X)
    print("SOM Clustering completed.")
    
    # 2. Train ART 
    print("\n--- Starting Adaptive Resonance Theory (ART) ---")
    art = ContinuousART(vigilance=0.90) 
    art.train(X)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(som, 'models/som_model.pkl')
    joblib.dump(art, 'models/art_model.pkl')
    print("\nSOM & ART Models saved to 'models/' successfully.")

if __name__ == '__main__':
    run_clustering()

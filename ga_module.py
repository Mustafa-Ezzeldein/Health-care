import numpy as np
import pandas as pd
from clustering_module import ContinuousART
from sklearn.metrics import silhouette_score
import joblib
import os

# Data Preparation
def get_data():
    if os.path.exists('heart data -2.csv'):
        df = pd.read_csv('heart data -2.csv')
    else:
        df = pd.read_csv('heart.csv')
    
    X = df.drop('target', axis=1, errors='ignore').values
    # Normalization
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    return X

X_global = get_data()

def fitness_function(vigilance):
    """
    Evaluates the quality of the selected vigilance.
    Goal: Get the highest Silhouette Score with the least number of clusters (balance).
    """
    if vigilance <= 0 or vigilance >= 1: return 0
    
    art = ContinuousART(vigilance=vigilance)
    clusters = art.train(X_global)
    
    unique_clusters = len(np.unique(clusters))
    
    # Avoid extreme cases (one cluster or as many clusters as data points)
    if unique_clusters < 2 or unique_clusters > len(X_global) * 0.5:
        return 0
        
    try:
        score = silhouette_score(X_global, clusters)
        # Encourage the system not to create unnecessary clusters
        penalty = 0.1 * (unique_clusters / 50) 
        return max(0, score - penalty)
    except:
        return 0

def run_ga_optimization():
    print("\n--- Starting Genetic Algorithm Optimization for ART Vigilance ---")
    
    # Genetic Algorithm Parameters
    pop_size = 10
    generations = 5
    mutation_rate = 0.1
    
    # Initial Population (Random vigilance values between 0.7 and 0.95)
    population = np.random.uniform(0.7, 0.95, pop_size)
    
    for gen in range(generations):
        fitness_scores = np.array([fitness_function(v) for v in population])
        best_idx = np.argmax(fitness_scores)
        best_v = population[best_idx]
        best_s = fitness_scores[best_idx]
        
        print(f"Generation {gen+1}: Best Vigilance = {best_v:.4f}, Fitness = {best_s:.4f}")
        
        # Selection (Simple Elitism + Random)
        new_population = [best_v] # Keep the best
        while len(new_population) < pop_size:
            # Crossover & Mutation
            parent = population[np.random.randint(0, pop_size)]
            child = parent + np.random.normal(0, 0.05)
            child = np.clip(child, 0.5, 0.98)
            new_population.append(child)
            
        population = np.array(new_population)

    final_best_vigilance = population[np.argmax([fitness_function(v) for v in population])]
    print(f"\nOptimization Finished! Optimal Vigilance: {final_best_vigilance:.4f}")
    
    # Apply the result and save the final model
    print("Retraining final ART model with optimized parameters...")
    final_art = ContinuousART(vigilance=final_best_vigilance)
    final_art.train(X_global)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_art, 'models/art_model.pkl')
    print("Optimized ART model saved to 'models/art_model.pkl'.")

if __name__ == '__main__':
    run_ga_optimization()

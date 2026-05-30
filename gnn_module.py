import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def build_normalized_adj(df_features, threshold=0.15):
    """
    This function builds a Graph connecting sensors based on the strength of Correlation.
    """
    corr = df_features.corr().values
    
    # Adjacency Matrix. If correlation exceeds the threshold, connect them (1).
    adj = (np.abs(corr) > threshold).astype(float)
    np.fill_diagonal(adj, 1.0) # Every sensor is connected to itself
    
    # Normalization D^-0.5 A D^-0.5 to ensure neural network stability
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized

def train_pure_gnn():
    print("Loading and scaling data for GNN...")
    try:
        df = pd.read_csv('heart data -2.csv')
    except OSError:
        print("Error: heart data -2.csv not found.")
        return

    X = df.drop('target', axis=1)
    y = df['target'].values
    
    # Local scaling to ensure graph accuracy
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 1. Build Graph Structure and relations
    print("Building Data Graph (Adjacency Matrix)...")
    A_hat = build_normalized_adj(X_df, threshold=0.1)
    
    # 2. Graph Convolution process (Spatial integration of data)
    # Multiply original patient data with the graph structure to merge related signals
    X_numpy = X_df.values  
    X_graph_embedded = X_numpy.dot(A_hat) # Basic GCN equation (A * X)
    
    # 3. Prepare data for the neural network
    X_train, X_test, y_train, y_test = train_test_split(X_graph_embedded, y, test_size=0.2, random_state=42)
    
    print("Training GNN classifier using Multilayer Perceptron on Graph Embeddings...")
    # Neural Network reading data after merging their relationships into the graph
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=800, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print("       GNN Model Results")
    print("="*30)
    print(f"GNN Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    print("="*30)

    # Saving
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/gnn_model.pkl')
    np.save('models/gnn_adj.npy', A_hat)
    print("\n✅ GNN Model trained successfully and saved to 'models/gnn_model.pkl'.")

if __name__ == '__main__':
    train_pure_gnn()

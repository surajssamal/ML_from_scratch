#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        centered_x = x - self.mean
        cov_matrix = (1 / x.shape[0]) * np.dot(centered_x.T, centered_x)
        
        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and corresponding eigenvectors in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]
        
        # Select the top n_components eigenvectors as principal components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, x):
        centered_x = x - self.mean
        return np.dot(centered_x, self.components)

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Load Iris dataset
    data = load_iris()
    X = data.data  # Features (150 samples, 4 features)
    Y = data.target  # Target (150 samples)

    # Project the data onto the 2 primary principal components
    pca = PCA(n_components=2)
    pca.fit(X)
    x_projection = pca.transform(X)

    # Scatter plot of PCA components
    plt.scatter(x_projection[:, 0], x_projection[:, 1], c=Y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Species")
    plt.title("PCA of Iris Dataset")
    plt.show()

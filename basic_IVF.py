##
# Implement an IVF (Inverted File) based search engine using K-Means for space partitioning
##

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans

class VectorIndex(ABC):
    """Abstract Base Class for Vector Search Indices."""
    
    @abstractmethod
    def fit(self, data: np.ndarray):
        """Train the index with provided vector data."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors."""
        pass

class IVFIndex(VectorIndex):
    """
    Inverted File Index (IVF) implementation for Approximate Nearest Neighbor search.
    
    This implementation partitions the vector space into Voronoi cells using K-Means.
    During search, only the most relevant cells (clusters) are probed, 
    significantly reducing the search space.
    """

    def __init__(self, n_clusters: int = 10, n_probe: int = 2):
        """
        Args:
            n_clusters (int): Number of clusters (centroids) to partition the space.
            n_probe (int): Number of nearby clusters to check during search.
        """
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.kmeans: Optional[KMeans] = None
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: List[List[int]] = []
        self.raw_data: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        """
        Clusters the data and builds the inverted file structure.
        
        Args:
            data (np.ndarray): Training vectors of shape (N, Dimensions).
        """
        self.raw_data = data
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = self.kmeans.fit_predict(data)
        self.centroids = self.kmeans.cluster_centers_
        
        # Initialize inverted lists (one per cluster)
        self.inverted_lists = [[] for _ in range(self.n_clusters)]
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(idx)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs ANN search by probing the nearest 'n_probe' clusters.
        
        Args:
            query (np.ndarray): Query vector.
            k (int): Number of neighbors to return.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Distances, Indices of the neighbors).
        """
        if self.kmeans is None or self.raw_data is None:
            raise ValueError("Index must be fitted before searching.")

        # Step 1: Find the nearest centroids (coarse quantizer)
        query_reshaped = query.reshape(1, -1)
        centroid_distances = np.linalg.norm(self.centroids - query_reshaped, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[:self.n_probe]

        # Step 2: Aggregate candidates from the selected clusters
        candidate_indices = []
        for cluster_id in nearest_clusters:
            candidate_indices.extend(self.inverted_lists[cluster_id])

        if not candidate_indices:
            return np.array([]), np.array([])

        # Step 3: Precise search within the candidates (refinement)
        candidates_data = self.raw_data[candidate_indices]
        distances = np.linalg.norm(candidates_data - query_reshaped, axis=1)
        
        # Get top-k nearest among candidates
        top_k_local_indices = np.argsort(distances)[:k]
        
        final_indices = np.array(candidate_indices)[top_k_local_indices]
        final_distances = distances[top_k_local_indices]

        return final_distances, final_indices

# --- Usage ---

if __name__ == "__main__":
    # Simulate 1000 high-dimensional vectors (e.g., 128 dimensions)
    np.random.seed(42)
    dataset = np.random.random((1000, 128)).astype('float32')
    query_vec = np.random.random((128,)).astype('float32')

    # Initialize and train the IVF ANN Index
    index = IVFIndex(n_clusters=20, n_probe=3)
    index.fit(dataset)

    # Execute Search
    dist, ids = index.search(query_vec, k=5)

    print(f"Top 5 Approximate Neighbors Indices: {ids}")
    print(f"Distances: {dist}")

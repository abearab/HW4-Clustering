import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        Two main ideas to quantify “goodness” of clusters. For a given point, compute:
        a: How far is that point from other points in the same cluster (on average)?
        b: How far is the smallest mean distance to a different cluster?

        The silhouette score for an observation is calculated using the following formula:

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

        where:
        - a(i) is the average distance between the i-th observation and all other points in the same cluster.
        - b(i) is the minimum average distance between the i-th observation and all points in the nearest cluster.

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Check if the input is a 2D array
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        
        # Check if the input is not empty
        if X.shape[0] == 0:
            raise ValueError("Input data must not be empty.")
        
        # Check if the labels are provided
        if y is None:
            raise ValueError("Labels must be provided.")
        
        # Check if the number of labels matches the number of samples
        if len(y) != X.shape[0]:
            raise ValueError("Number of labels must match number of samples.")
        
        # Check if the labels are valid
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Labels must be integers.")
        
        # Implementation of silhouette score
        unique_labels = np.unique(y)
        n_clusters = unique_labels.shape[0]
        n_samples = X.shape[0]
        
        distances = cdist(X, X)
        a = np.zeros(n_samples)
        b = np.zeros(n_samples)

        for i in range(n_clusters):
            mask = (y == unique_labels[i])
            a[mask] = np.mean(distances[mask][:, mask], axis=1)
            for j in range(n_clusters):
                if i != j:
                    other_mask = (y == unique_labels[j])
                    b[mask] = np.minimum(b[mask], np.mean(distances[mask][:, other_mask], axis=1))

        s = (b - a) / np.maximum(a, b)
        s[np.isnan(s)] = 0  # Handle the case where a and b are both 0

        return s

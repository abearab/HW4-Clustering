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
        assert X.shape[0] == y.shape[0]

        # calculate the distance matrix
        dist_mat = cdist(X, X)

        # calculate a(i) for each observation
        a = np.array([np.mean(dist_mat[i][y == y[i]]) for i in range(X.shape[0])])
        # calculate b(i) for each observation
        b = np.array([np.min([np.mean(dist_mat[i][y == j]) for j in np.unique(y) if j != y[i]]) for i in range(X.shape[0])])
        # calculate the silhouette score for each observation
        s = (b - a) / np.maximum(a, b)
        return s

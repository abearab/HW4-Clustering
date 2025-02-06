import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self._fitted = False
    
    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Randomly initialize centroids
        self._centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False)]

        for _ in range(self.max_iter):
            # Compute distances from points to centroids
            distances = cdist(mat, self._centroids)
            # Assign clusters based on closest centroid
            self.labels = np.argmin(distances, axis=1)
            # Compute new centroids
            new_centroids = np.array([mat[self.labels == i].mean(axis=0) for i in range(self.k)])
            # Check for convergence and otherwise update centroids
            if np.linalg.norm(new_centroids - self._centroids) < self.tol:
                break
            self._centroids = new_centroids

        self._fitted = True

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Check if the model has been fitted
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet.")
        
        # Check if the number of features matches
        if mat.shape[1] != self._centroids.shape[1]:
            raise ValueError("Input data must have the same number of features as the training data.")
        if type(mat) != np.ndarray:
            raise TypeError("Input data must be a numpy array.")
        if mat.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        if mat.shape[0] == 0:
            raise ValueError("Input data must not be empty.")
        
        # Compute distances from points to centroids
        distances = cdist(mat, self._centroids)
        # Assign clusters based on closest centroid
        return np.argmin(distances, axis=1)


    def get_error(self, mat: np.ndarray) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        The "squared error" for a point P with respect to its cluster center C is 
        the distance between P and C squared; that is, (Px - Cx)^2 + (Py - Cy)^2.
        source: https://stackoverflow.com/questions/34710589/k-means-algorithm-working-out-squared-error

        outputs:
            float
                the squared-mean error of the fit model
        """
        # Check if the model has been fitted
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet.")

        distances = cdist(mat, self._centroids)
        closest_centroids = np.argmin(distances, axis=1)
        squared_errors = np.sum((mat - self._centroids[closest_centroids]) ** 2, axis=1)
        return np.mean(squared_errors)


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # Check if the model has been fitted
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet.")
        return self._centroids

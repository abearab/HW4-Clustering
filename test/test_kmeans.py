# k-means unit tests

import numpy as np
from scipy.spatial.distance import cdist

from cluster.kmeans import KMeans


def test_kmeans():
    """
    - should your model still run if the provided k=0?
    - should your model still run if the number of observations < the provided k?
    - can your model handle very high k?
    - can your model handle very high dimensionality? what about only a single dimension?

    """
    # Test with valid input
    kmeans = KMeans(k=3, tol=1e-4, max_iter=200)
    assert kmeans.k == 3
    assert kmeans.tol == 1e-4
    assert kmeans.max_iter == 200

    # Test with invalid k
    try:
        KMeans(k=-1)
    except ValueError as e:
        assert str(e) == "k must be a positive integer."

    try:
        KMeans(k=0)
    except ValueError as e:
        assert str(e) == "k must be a positive integer."

    try:
        KMeans(k="a")
    except ValueError as e:
        assert str(e) == "k must be a positive integer."

    # Test with a simple dataset
    data = np.array([[1, 2], [1, 4], [1, 0],
                     [4, 2], [4, 4], [4, 0]])
    kmeans = KMeans(k=2)
    kmeans.fit(data)
    assert kmeans._fitted == True
    assert kmeans.predict(data).shape == (6,)

    # Test with high k value, k > number of data points
    kmeans = KMeans(k=10)
    try: 
        kmeans.fit(data)
    except ValueError as e:
        assert str(e) == "Number of clusters cannot be greater than number of data points."

    # Test with high dimensionality and high k
    high_dim_data = np.random.rand(10, 100)
    kmeans = KMeans(k=10)
    kmeans.fit(high_dim_data)
    assert kmeans._fitted == True
    assert kmeans.predict(high_dim_data).shape == (10,)

    # Test higher max_iter
    kmeans = KMeans(k=3, max_iter=10000)
    kmeans.fit(data)
    assert kmeans._fitted == True

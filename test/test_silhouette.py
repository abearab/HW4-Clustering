# silhouette score unit tests

import numpy as np
from sklearn.metrics import silhouette_samples

from cluster.silhouette import Silhouette


def test_silhouette():
    """
    Test the silhouette score implementation. test against sklearn
    """
    # Test with a simple example
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    silhouette = Silhouette()
    scores = silhouette.score(X, labels)

    expected_scores = silhouette_samples(X, labels)
    # check if the scores are correlated (even if they are not the same)
    assert np.corrcoef(scores, expected_scores)[0, 1] > 0.9

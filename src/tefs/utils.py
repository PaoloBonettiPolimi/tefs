from typing import Dict, List, Union

import pandas as pd

IterationResult = Dict[str, Union[Dict[int, float], float]]


def select_features(
        results: List[IterationResult],
        threshold: float,
        direction: str,
        verbose: int = 0,
        ) -> List[int]:
    """
    Selects the features based on the transfer entropy scores and a threshold.

    :param results: A list of IterationResult objects containing the feature scores.
    :type results: List[IterationResult]
    :param threshold: The threshold to use for feature selection.
    :type threshold: float
    :param direction: The direction of selection. Can be either "forward" or "backward".
    :type direction: str
    :param verbose: Verbosity level, defaults to 0.
    :type verbose: int, optional
    :return: A list of feature indexes representing the selected features.
    """

    assert direction in ["forward", "backward"], "Invalid direction"

    if verbose > 0:
        print(f"Selecting features with direction {direction} and threshold {threshold}")
    stop = False
    initial_features = None

    for iteration in results:
        assert isinstance(iteration["TE"], (float, int)), "TE score must be a number"
        assert isinstance(iteration["feature_scores"], dict), "feature_scores must be a dictionary"

        if initial_features is None:
            initial_features = iteration["feature_scores"].keys()

        # First check if the threshold is reached
        if iteration["TE"] > threshold:
            if verbose > 0:
                print(f"Stopping condition reached: TE score {iteration['TE']} > threshold {threshold}")
            stop = True

        # Then check if the feature scores are all positive or negative
        if direction == "backward" and min(iteration["feature_scores"].values()) > 0:
            if verbose > 0:
                print("Stopping condition reached: all feature scores are positive")
            stop = True
        elif direction == "forward" and max(iteration["feature_scores"].values()) < 0:
            if verbose > 0:
                print("Stopping condition reached: all feature scores are negative")
            stop = True

        # If the stop condition is reached, return the list of selected features indexes
        if stop:
            if direction == "backward":
                # In the backward case the remaining features are the ones that were not removed
                return list(iteration["feature_scores"].keys())
            elif direction == "forward":
                # In the forward case the remaining features are the ones that were not added yet
                # so I need to take the initial set of features and remove the remaining ones
                return list(set(initial_features) - set(iteration["feature_scores"].keys()))

    # If the loop reaches the end, no stop condition was reached
    if verbose > 0:
        print("Stopping condition not reached")
    if direction == "backward":
        # Didn't reach the stop condition, I should return no features, but I decide to return the last feature left
        return list(results[-1]["feature_scores"].keys())
    elif direction == "forward":
        # Didn't reach the stop condition, I return all the features
        return list(initial_features)

def select_n_features(
        results: List[IterationResult],
        n: int,
        direction: str,
        verbose: int = 0,
        ) -> List[int]:
    """
    Selects the `n` features with the highest or lowest score, regardless of the threshold and the transfer entropy scores.

    :param results: A list of IterationResult objects containing the feature scores.
    :type results: List[IterationResult]
    :param n: The number of features to select.
    :type n: int
    :param direction: The direction of selection. Can be either "forward" or "backward".
    :type direction: str
    :param verbose: Verbosity level, defaults to 0.
    :type verbose: int, optional
    :return: A list of feature indexes representing the selected features.
    """

    assert direction in ["forward", "backward"], "Invalid direction"
    num_total_features = len(results[0]["feature_scores"])
    assert n <= num_total_features, f"n must be <= {num_total_features}"
    assert n > 0, "n must be > 0"

    if verbose > 0:
        print(f"Selecting {n} features with direction {direction}")
    if direction == "backward":

        for iteration in results:
            features_indexes = iteration["feature_scores"].keys()
            if len(features_indexes) == n:
                return list(features_indexes)
        
    elif direction == "forward":

        initial_features = None

        for iteration in results:

            features_indexes = iteration["feature_scores"].keys()

            if initial_features is None:
                initial_features = features_indexes

            if len(features_indexes) == num_total_features - n:
                return list(set(initial_features) - set(features_indexes))
        
        # If the loop reaches the end, it means that n = num_total_features, so I return all the features
        return list(initial_features)

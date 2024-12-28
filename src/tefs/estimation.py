import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma


# TODO might be worth checking
# - https://github.com/syanga/pycit/blob/master/pycit/estimators/mixed_cmi.py
# - https://github.com/wgao9/knnie/blob/master/knnie.py
def estimate_mi(
        X: np.ndarray,
        Y: np.ndarray,
        k: int = 5,
        estimation_method: str = "digamma",
        ) -> float:
    """
    Estimate the Mutual Information (MI) between :math:`X` and :math:`Y`, i.e. :math:`I(X;Y)`, based on *Mixed Random Variable Mutual Information Estimator - Gao et al.*.

    :param X: The first input array.
    :type X: numpy.ndarray
    :param Y: The second input array.
    :type Y: numpy.ndarray
    :param k: The number of nearest neighbors to consider, defaults to 5.
    :type k: int, optional
    :param estimation_method: The estimation method to use, can be either 'digamma' or 'log', defaults to 'digamma'.
    :type estimation_method: str, optional
    :return: The estimated mutual information.
    :rtype: float
    """

    assert k > 0, "k must be greater than 0"
    assert k % 1 == 0, "k must be an integer"
    assert estimation_method in ["digamma", "log"], "Invalid estimation method"

    num_samples = len(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("Input arrays must have the same number of samples")
    
    if X.shape[0] == 0:
        raise ValueError("Input arrays must not be empty")

    dataset = np.concatenate((X, Y), axis=1)

    tree_xy = cKDTree(dataset)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    # rho
    knn_distances = [tree_xy.query(sample, k + 1, p=float("inf"))[0][k] for sample in dataset]

    res = 0

    for i in range(num_samples):
        
        k_hat, n_xi, n_yi = k, k, k
        
        if knn_distances[i] <= 1e-15:
            # Points at a distance less than or equal to (nearly) 0
            k_hat = len(tree_xy.query_ball_point(dataset[i], 1e-15, p=float("inf")))
            n_xi = len(tree_x.query_ball_point(X[i], 1e-15, p=float("inf")))
            n_yi = len(tree_y.query_ball_point(Y[i], 1e-15, p=float("inf")))
        else:
            # Points at distances less than or equal to rho
            k_hat = k
            n_xi = len(tree_x.query_ball_point(X[i], knn_distances[i] - 1e-15, p=float("inf")))
            n_yi = len(tree_y.query_ball_point(Y[i], knn_distances[i] - 1e-15, p=float("inf")))
        
        if estimation_method == "digamma":
            res += (digamma(k_hat) + np.log(num_samples) - digamma(n_xi) - digamma(n_yi)) / num_samples
        elif estimation_method == "log":
            res += (digamma(k_hat) + np.log(num_samples) - np.log(n_xi + 1) - np.log(n_yi + 1)) / num_samples
    
    return res


def estimate_cmi(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        k: int = 5,
        estimation_method: str = "digamma",
        ) -> float:
    """
    Estimate the Conditional Mutual Information (CMI) between :math:`X` and :math:`Y` given :math:`Z`, i.e. :math:`I(X;Y \mid Z)`, using the equivalance

    .. math::
        I(X;Y \mid Z) = I(X,Z;Y) - I(Z;Y)

    Note that :math:`I(X;Y \mid Z) = I(Y;X \mid Z)`.

    :param X: The input variable X.
    :type X: numpy.ndarray
    :param Y: The input variable Y.
    :type Y: numpy.ndarray
    :param Z: The input variable Z.
    :type Z: numpy.ndarray
    :param k: The number of nearest neighbors for k-nearest neighbor estimation (default is 5).
    :type k: int
    :param estimation_method: The estimation method to use (default is "digamma").
    :type estimation_method: str
    :return: The estimated CMI between X and Y given Z.
    :rtype: float
    """

    assert estimation_method in ["digamma", "log"], "Invalid estimation method"

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    XZ = np.hstack((X, Z))

    return estimate_mi(XZ, Y, k, estimation_method) - estimate_mi(Z, Y, k, estimation_method)
    



def estimate_conditional_mutual_information_for_TEFS(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        k: int,
        cycle_len: int,
        lag_features: list[int] = [1],
        lag_target: list[int] = [1],
        lag_conditioning: list[int] = None,
        estimation_method: str = "digamma",
        ) -> float:
    """
    Computes the CMI from X to Y given Z, using the specified lags. 
    !!THIS IS SPECIFIC FOR THE TEFS ALGORITHM, if you are looking for a generic CMI estimation use the function estimate_cmi!!

    :param X: Sample of a (multivariate) random variable representing the input
    :type X: np.ndarray of shape (n_samples, n_features)
    :param Y: Sample of a (multivariate) random variable representing the target
    :type Y: np.ndarray of shape (n_samples, n_targets)
    :param Z: Sample of a (multivariate) random variable representing the conditioning
    :type Z: np.ndarray of shape (n_samples, n_conditioning)
    :param lag_features: the lag applied on X
    :type lag_features: List[int]
    :param lag_target: the lag applied on Y
    :type lag_target: List[int]
    :param lag_conditioning: the lag applied on Z, if None it is set to lag_features
    :type lag_conditioning: List[int]
    :return: a scalar of the value of the CMI
    """
    if lag_conditioning is None:
        lag_conditioning = lag_features
    if 0 not in lag_target: 
        # this is to ensure that the target is always included in the conditioning, 
        # for consistency with the TE algorithm
        lag_target = [0] + lag_target

    member1= np.array([])
    member2 = np.array([])
    member3 = np.array([])
    max_lag = max(max(lag_features), max(lag_target), max(lag_conditioning))
        
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Base version
    if cycle_len is None:

        # Filling member1
        member1 = np.hstack([X[max_lag - lag : X.shape[0]-lag, :] for lag in lag_features])

        # Filling member2
        member2 = np.hstack([Y[max_lag - lag : Y.shape[0]-lag, :] for lag in lag_target])

        # Filling member3
        member3 = np.hstack([Z[max_lag - lag : Z.shape[0]-lag, :] for lag in lag_conditioning])

        # Cycle version
    if cycle_len is not None:
        n_cycle = int(X.shape[0]/cycle_len)

        # Filling members
        member1 = np.array([])
        member2 = np.array([])
        member3 = np.array([])
        for lag in range(max_lag+1):
            # computing indeces
            aa = np.array(range(max_lag - lag ,cycle_len - lag))
            index = []
            for i in range(n_cycle):
                index.extend(aa+cycle_len*i) 

            # filling memmbers
            if lag in lag_features:
                member1 = np.hstack([member1, X[index,:]]) if member1.size else X[index,:]
            if lag in lag_target:
                member2 = np.hstack([member2, Y[index,:]]) if member2.size else Y[index,:]
            if lag in lag_conditioning:
                member3 = np.hstack([member3, Z[index,:]]) if member3.size else Z[index,:]


        
        # print("\n member1")
        # print(member1[0:10,:])
        # print("\n member2")
        # print(member2[0:10,:])
        # print("\n member3")
        # print(member3[0:10,:])

    #raise ValueError(member1.shape,member2.shape,member3.shape)
    if member3.shape[1] == 0:  # if no conditioning => MI
        valid_idx = ~np.any(pd.isna(member1), axis=1) & ~np.any(pd.isna(member2), axis=1)
        return estimate_mi(member1[valid_idx], member2[valid_idx], k,estimation_method=estimation_method)
    else:   # if conditioning => CMI
        valid_idx = ~np.any(pd.isna(member1), axis=1) & ~np.any(pd.isna(member2), axis=1) & ~np.any(pd.isna(member3), axis=1)
        return estimate_cmi(member1[valid_idx], member2[valid_idx], member3[valid_idx], k,estimation_method=estimation_method)


def estimate_conditional_transfer_entropy(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        k: int,
        cycle_len: int,
        lag_features: list[int] = [1],
        lag_target: list[int] = [1],
        lag_conditioning: list[int] = None,
        estimation_method: str = "digamma",
    
        ) -> float:
    """
    Computes the conditional transfer entropy from X to Y given Z, using the specified lags.

    :param X: Sample of a (multivariate) random variable representing the input
    :type X: np.ndarray of shape (n_samples, n_features)
    :param Y: Sample of a (multivariate) random variable representing the target
    :type Y: np.ndarray of shape (n_samples, n_targets)
    :param Z: Sample of a (multivariate) random variable representing the conditioning
    :type Z: np.ndarray of shape (n_samples, n_conditioning)
    :param lag_features: the lag applied on X
    :type lag_features: List[int]
    :param lag_target: the lag applied on Y
    :type lag_target: List[int]
    :param lag_conditioning: the lag applied on Z, if None it is set to lag_features
    :type lag_conditioning: List[int]
    :return: a scalar of the value of the transfer entropy
    """

    if lag_conditioning is None:
        lag_conditioning = lag_features
    if 0 in lag_target:
        lag_target.remove(0)

    member1 = np.array([])
    member2 = np.array([])
    member3 = np.array([])
    max_lag = max(max(lag_features), max(lag_target), max(lag_conditioning))
        
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Base version
    if cycle_len is None:

        # Filling member1
        member1 = np.hstack([X[max_lag - lag : X.shape[0]-lag, :] for lag in lag_features])

        # Filling member2
        member2 = Y[max_lag:, :]
    
        # Filling member3
        member3 = np.hstack([
            # Filling the part relative the past of the target
            *[Y[max_lag - lag : Y.shape[0]-lag, :] for lag in lag_target],
            # Filling the part relative the past of the conditioning features
            *[Z[max_lag - lag : Z.shape[0]-lag, :] for lag in lag_conditioning],
        ])

        # Cycle version
    if cycle_len is not None:
        n_cycle = int(X.shape[0]/cycle_len)

        # Filling members
        member1 = np.array([])
        member2 = np.array([])
        member3Y = np.array([])
        member3Z = np.array([])
        for lag in range(max_lag+1):
            # computing indeces
            aa = np.array(range(max_lag - lag ,cycle_len - lag))
            index = []
            for i in range(n_cycle):
                index.extend(aa+cycle_len*i) 

            # filling memmbers
            if lag in lag_features:
                member1 = np.hstack([member1, X[index,:]]) if member1.size else X[index,:]
            if lag == 0:
                member2 = Y[index,:]
            if lag in lag_target:
                member3Y = np.hstack([member3Y, Y[index,:]]) if member3Y.size else Y[index,:]
            if lag in lag_conditioning:
                member3Z = np.hstack([member3Z, Z[index,:]]) if member3Z.size else Z[index,:]
        member3 = np.hstack([member3Y, member3Z])

        
        # print("\n member1")
        # print(member1[0:10,:])
        # print("\n member2")
        # print(member2[0:10,:])
        # print("\n member3")
        # print(member3[0:10,:])
    
    # drop rows containing nan   (ie  set  t,t-1,...,t-lag not fully available)    
    valid_idx = ~np.any(pd.isna(member1), axis=1) & ~np.any(pd.isna(member2), axis=1) & ~np.any(pd.isna(member3), axis=1)
    return estimate_cmi(member1[valid_idx], member2[valid_idx], member3[valid_idx], k,estimation_method=estimation_method)











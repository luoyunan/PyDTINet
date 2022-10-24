import numpy as np

def RWR(A, rsp=0.5, max_iter=50, epsilon=1e-6):
    """
    Random Walk with Restart algorithm.
    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix.
    rsp : float
        Restart probability.
    max_iter : int
        Maximum number of iterations.
    epsilon : float
        Convergence threshold.
    Returns
    -------
    Q : numpy.ndarray
        Diffusion state vector.
    """

    # Add self loops to isolated nodes
    A = A + np.diag((np.sum(A, axis=1)) == 0)
    # Normalize the adjacency matrix
    P = A / np.sum(A, axis=1)[:, None]

    N = A.shape[0]
    restart = np.eye(N)
    Q = np.eye(N)
    for _ in range(max_iter):
        Q_new = (1 - rsp) * np.matmul(P, Q) + rsp * restart
        diff = np.linalg.norm(Q - Q_new, 'fro')
        Q = Q_new.copy()
        if diff < epsilon:
            break
    return Q

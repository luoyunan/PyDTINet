import numpy as np
from RWR import RWR

def DCA(networks, dim, rsp=0.5, max_iter=50, epsilon=1e-6):
    """
    Diffusion Component Analysis (a.k.a Mashup)
    Parameters
    ----------
    network : numpy.ndarray of shape (N, N), or list of numpy.ndarray of shape (N, N)
        Network(s) in adjacency matrix format.
    """

    # run RWR on each network
    _networks = [networks] if isinstance(networks, np.ndarray) else networks
    Q = None
    for net in _networks:
        curQ = RWR(net, rsp, max_iter, epsilon)
        Q = curQ if Q is None else np.concatenate((Q, curQ), axis=1)

    # add small constant to avoid log of zero
    nnode = Q.shape[0]
    alpha = 1. / nnode
    Q = np.log(Q + alpha) - np.log(alpha)

    # compute embeddings
    Q = np.matmul(Q, Q.T)
    U, S, _ = np.linalg.svd(Q)
    U = U[:, :dim]
    S = S[:dim]
    x = np.matmul(U, np.diag(np.sqrt(np.sqrt(S))))
    return x

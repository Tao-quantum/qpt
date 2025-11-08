import numpy as np
from qutip.random_objects import rand_unitary
import jax.numpy as jnp


def random_kraus_op(N, weights, rank, density=0.5):
    """Generates a sum of random unitaries to form a process given a set of
    weights and the rank of the process.

    Args:
        N (int): Hilbert space dimension.
        weights (array): Random weights that will be normalized.
        rank (int): The rank of the process.
        density (float): A number between 0, 1 to specifying the density of the
                         random unitaries.

    Returns:
        kraus_ops (array): A (k x N x N) complex-valued array of Kraus operators.
    """
    weights_unnormalized = weights**np.arange(rank)
    weights = weights_unnormalized / (weights_unnormalized).sum()
    kraus_ops = np.array([np.sqrt(w)*rand_unitary(N, density=density) for w in weights])
    return kraus_ops


# # number of qubits
# n = 2 
# # Rank of process and number of Kraus operators
# rank = 3
# # Hilbert space dimension
# N = 2**n

# kraus_true = random_kraus(N, np.random.uniform(0.1, 1.), rank)
# print(len(kraus_true))
# print(kraus_true[0].full())

# kraus_true = random_kraus(N, np.random.uniform(0.1, 1.), rank)
# Kr = jnp.array([item for lst in kraus_true for item in lst]).reshape((2**n * rank, 2**n))
# # print(Kr)
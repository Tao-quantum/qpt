import numpy as np

def kraus_operators_random_2_nqubit(nqubit):
    kraus_operators = []
    for _ in range(2**nqubit):
        op = np.random.rand(2**nqubit, 2**nqubit) + 1j * np.random.rand(2**nqubit, 2**nqubit)
        op /= np.sqrt(np.trace(np.dot(op.conj().T, op)))  # 单位迹约束
        op = (op + op.conj().T) / 2  # 半正定约束
        kraus_operators.append(op)
    return kraus_operators

def generate_stiefel_matrix(n, k):
    A = np.random.rand(n, k) + 1j * np.random.rand(n, k)
    Q, R = np.linalg.qr(A) # 由于Q是正交的，Q^H @ Q = I，因此Q是Stiefel矩阵
    return Q


def generate_unitary_matrix(dim):
    # 使用QR分解生成一个随机酉矩阵
    A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    Q, R = np.linalg.qr(A)
    return Q



def amp_damp_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return K0, K1

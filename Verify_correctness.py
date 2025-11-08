from qutip import *
import numpy as np

def test_kraus(nqubit, r, Kr):
    sum_kraus = np.zeros((2**nqubit, 2**nqubit), dtype=np.complex128)
    K_list = [Kr[i*(2**nqubit):(i+1)*(2**nqubit), :] for i in range(r)]
    for k in range(len(K_list)):
        sum_kraus += K_list[k].conj().T @ K_list[k]
    return sum_kraus 

def rho_kraus(nqubit, r, rho0, Kr):
    """
    Apply Kraus operators to a density matrix.
    
    Args:
        nqubit (int): Number of qubits
        r (int): Rank of the Kraus representation (number of Kraus operators)
        rho0 (np.ndarray): Input density matrix (2^nqubit x 2^nqubit)
        Kr (np.ndarray): Kraus operator matrix (r*2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Output density matrix after applying Kraus operators
    """
    dim = 2**nqubit
    sum_rho = np.zeros((dim, dim), dtype=np.complex128)
    
    # Extract individual Kraus operators and apply them
    for i in range(r):
        start_idx = i * dim
        end_idx = (i + 1) * dim
        kraus = Kr[start_idx:end_idx, :]
        sum_rho += kraus @ rho0 @ kraus.conj().T
    
    return sum_rho

def fidelity(rho1, rho2):
    rho_tp = np.sqrt(np.sqrt(rho1) @ rho2 @ np.sqrt(rho1))
    fid_tp = np.trace(rho_tp)
    return fid_tp

def random_density_matrix(nqubit):
    dim = 2**nqubit
    matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    mat_ = (matrix + matrix.conj().T) / 2
    mat = mat_/np.linalg.norm(mat_)
    return mat

def testFid(nqubit, times, gate, r, Kr):
    fid = []
    for _ in range(times):
        rho0 = random_density_matrix(nqubit)
        rho_T = gate @ rho0 @ gate.conj().T
        rho_k = rho_kraus(nqubit, r, rho0, Kr)
        fid_tp = np.linalg.norm(rho_T-rho_k)
        fid.append(fid_tp)
    mean_fid = np.mean(fid)
    return mean_fid 
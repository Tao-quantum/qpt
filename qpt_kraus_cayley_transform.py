"""
Kraus Operator Representation: Gradient and Cayley Transform Functions

This module contains functions for computing gradients and performing
Cayley transform updates for quantum process tomography with Kraus operators.
"""

import numpy as np


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


def trace_rho_M(nqubit, r, rho, M, Kr):
    """
    Compute trace of (Kraus-transformed density matrix) @ measurement operator.
    
    Args:
        nqubit (int): Number of qubits
        r (int): Rank of the Kraus representation
        rho (np.ndarray): Input density matrix
        M (np.ndarray): Measurement operator
        Kr (np.ndarray): Kraus operator matrix
    
    Returns:
        float: Trace value
    """
    sum_rho = rho_kraus(nqubit, r, rho, Kr)
    return np.trace(sum_rho @ M)


# ============================================================================
# Gradient Computation Functions
# ============================================================================

def G_func_term(nqubit, r, rho, M, Kr):
    """
    Compute gradient term for a single (state, measurement) pair.
    
    For each Kraus operator K_k, computes M @ K_k @ rho and stacks them
    vertically to form a gradient matrix of shape (r*2^nqubit x 2^nqubit).
    
    Args:
        nqubit (int): Number of qubits
        r (int): Rank of the Kraus representation
        rho (np.ndarray): Input density matrix (2^nqubit x 2^nqubit)
        M (np.ndarray): Measurement operator (2^nqubit x 2^nqubit)
        Kr (np.ndarray): Kraus operator matrix (r*2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Gradient term matrix (r*2^nqubit x 2^nqubit)
    """
    dim = 2**nqubit
    grad_terms = []
    
    # Compute M @ K_k @ rho for each Kraus operator
    for k in range(r):
        start_idx = k * dim
        end_idx = (k + 1) * dim
        kraus_k = Kr[start_idx:end_idx, :]
        grad_term = M @ kraus_k @ rho
        grad_terms.append(grad_term)
    
    # Stack all gradient terms vertically
    grad_matrix = np.vstack(grad_terms)
    return grad_matrix


def G_func(nqubit, r, p_exp_list, rho_list, M_list, Kr):
    """
    Compute the normalized gradient function G for Kraus QPT.
    
    G = sum_{i,j} -2 * (p_exp - p_theory) * (M @ K_k @ rho) for all k
    The result is normalized by its Frobenius norm.
    
    Args:
        nqubit (int): Number of qubits
        r (int): Rank of the Kraus representation
        p_exp_list (np.ndarray): Experimental probabilities (len(M_list) x len(rho_list))
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        Kr (np.ndarray): Kraus operator matrix (r*2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Normalized gradient matrix G (r*2^nqubit x 2^nqubit)
    """
    dim = 2**nqubit
    grad_sum = np.zeros((r * dim, dim), dtype=np.complex128)
    
    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            p_exp = p_exp_list[j][i]
            p_theory = trace_rho_M(nqubit, r, rho, M, Kr)
            G_term = G_func_term(nqubit, r, rho, M, Kr)
            grad_sum += -2 * (p_exp - p_theory) * G_term
    
    # Normalize by Frobenius norm
    norm = np.linalg.norm(grad_sum)
    if norm > 0:
        grad_sum = grad_sum / norm
    
    return grad_sum


# ============================================================================
# Cayley Transform Update
# ============================================================================

def Cayley_Transform(nqubit, Grad_E, step, Kr):
    """
    Perform Cayley transform update on the Kraus operator matrix Kr.
    
    The Cayley transform preserves the Stiefel manifold constraint while
    updating Kr along the gradient direction.
    
    Update formula:
        Kr_new = Kr - step * A @ C @ B^dagger @ Kr
    where:
        A = [Grad_E, Kr]
        B^dagger = [Kr^dagger, -Grad_E^dagger]
        C = (I + (step/2) * B^dagger @ A)^(-1)
    
    Args:
        nqubit (int): Number of qubits
        Grad_E (np.ndarray): Gradient matrix (r*2^nqubit x 2^nqubit)
        step (float): Step size for the update
        Kr (np.ndarray): Current Kraus operator matrix (r*2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Updated Kraus operator matrix (r*2^nqubit x 2^nqubit)
    """
    dim = 2**nqubit
    
    # Construct matrices for Cayley transform
    A = np.concatenate((Grad_E, Kr), axis=1)
    B_dag = np.concatenate((Kr.conj().T, -Grad_E.conj().T), axis=0)
    
    # Compute Cayley transform matrix
    Id = np.eye(2 * dim)
    C = np.linalg.inv(Id + 0.5 * step * B_dag @ A)
    
    # Update Kraus operator matrix
    Kr_new = Kr - step * A @ C @ B_dag @ Kr
    
    return Kr_new

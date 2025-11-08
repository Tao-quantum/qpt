"""
Quantum Process Tomography Loss Functions

This module contains loss functions for both Kraus operator representation
and unitary operator representation of quantum channels.
"""

import numpy as np


# ============================================================================
# Kraus Operator Representation Functions
# ============================================================================

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
    # Direct slicing without creating intermediate list
    for i in range(r):
        start_idx = i * dim
        end_idx = (i + 1) * dim
        kraus = Kr[start_idx:end_idx, :]
        sum_rho += kraus @ rho0 @ kraus.conj().T
    return sum_rho


def trace_rho_M_kraus(nqubit, r, rho, M, Kr):
    """
    Compute trace of (Kraus-transformed density matrix) @ measurement operator.
    
    Optimized version that computes trace directly without constructing full density matrix.
    
    Args:
        nqubit (int): Number of qubits
        r (int): Rank of the Kraus representation
        rho (np.ndarray): Input density matrix
        M (np.ndarray): Measurement operator
        Kr (np.ndarray): Kraus operator matrix
    
    Returns:
        float: Trace value
    """
    dim = 2**nqubit
    trace_sum = 0.0
    # Compute trace directly: Tr(M @ K_i @ rho @ K_i^dagger) = Tr(K_i^dagger @ M @ K_i @ rho)
    for i in range(r):
        start_idx = i * dim
        end_idx = (i + 1) * dim
        kraus = Kr[start_idx:end_idx, :]
        # More efficient: Tr(K^dagger @ M @ K @ rho) using cyclic property
        trace_sum += np.trace(kraus.conj().T @ M @ kraus @ rho)
    return trace_sum


def loss_qpt_kraus(nqubit, r, p_exp_list, rho_list, M_list, Kr):
    """
    Loss function for Quantum Process Tomography using Kraus representation.
    
    Computes the mean squared error between experimental probabilities and
    theoretical predictions, with L1 regularization.
    
    Args:
        nqubit (int): Number of qubits
        r (int): Rank of the Kraus representation
        p_exp_list (np.ndarray): Experimental probabilities (len(M_list) x len(rho_list))
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        Kr (np.ndarray): Kraus operator matrix (r*2^nqubit x 2^nqubit)
    
    Returns:
        float: Loss value (MSE + regularization)
    """
    dim = 2**nqubit
    norm_factor = 1.0 / dim
    loss_sum = 0.0
    
    # Pre-compute regularization term once
    regularization = 0.001 * np.linalg.norm(Kr, 1)
    
    # Optimized loop: compute trace directly without intermediate matrices
    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            # Compute trace directly using optimized method
            tp = trace_rho_M_kraus(nqubit, r, rho, M, Kr)
            p = p_exp_list[j][i]
            residual = p - tp
            loss_sum += np.real(residual * residual) * norm_factor
    
    return loss_sum + regularization


# ============================================================================
# Unitary Operator Representation Functions
# ============================================================================

def loss_qpt_unitary(nqubit, p_exp_list, rho_list, M_list, U):
    """
    Loss function for Quantum Process Tomography using unitary representation.
    
    Computes the mean squared error between experimental probabilities and
    theoretical predictions, with L1 regularization.
    
    Args:
        nqubit (int): Number of qubits
        p_exp_list (np.ndarray): Experimental probabilities (len(M_list) x len(rho_list))
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Unitary operator matrix (2^nqubit x 2^nqubit)
    
    Returns:
        float: Loss value (MSE + regularization)
    """
    dim = 2**nqubit
    norm_factor = 1.0 / dim
    loss_sum = 0.0

    regularization = 0.001 * np.linalg.norm(U, 1)

    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            # Reuse Kraus trace helper with r=1 to ensure consistency
            rho_transformed = U @ rho @ U.conj().T
            tp = np.trace(rho_transformed @ M)
            
            p = p_exp_list[j][i]
            residual = p - tp
            loss_sum += np.real(residual * residual) * norm_factor

    return loss_sum + regularization

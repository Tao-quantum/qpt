"""
Unitary Operator Representation: Gradient and Cayley Transform Functions

This module contains functions for computing gradients and performing
Cayley transform updates for quantum process tomography with unitary operators.
"""

import numpy as np


# ============================================================================
# Helper Functions
# ============================================================================

def trace_rhoM_term(rho_list, M_list, U, i, j):
    """
    Compute trace of (unitary-transformed density matrix) @ measurement operator.
    
    Args:
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Unitary operator matrix (2^nqubit x 2^nqubit)
        i (int): Index for density matrix
        j (int): Index for measurement operator
    
    Returns:
        float: Trace value Tr((U @ rho @ U^dagger) @ M)
    """
    M = M_list[j]
    rho = rho_list[i]
    rho_transformed = U @ rho @ U.conj().T
    return np.trace(rho_transformed @ M)


# ============================================================================
# Gradient Computation Functions
# ============================================================================

def G_func_term(rho_list, M_list, U, i, j):
    """
    Compute gradient term for a single (state, measurement) pair.
    
    This computes M @ U @ rho, which is used in the gradient computation.
    
    Args:
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Unitary operator matrix
        i (int): Index for density matrix
        j (int): Index for measurement operator
    
    Returns:
        np.ndarray: Gradient term matrix
    """
    M = M_list[j]
    rho = rho_list[i]
    return M @ U @ rho


def G_func(nqubit, p_exp_list, rho_list, M_list, U):
    """
    Compute the (normalized) gradient function G for unitary QPT.
    
    G = sum_{i,j} -2 * (p_exp - p_theory) * (M @ U @ rho)
    The result is normalized by its Frobenius norm to align with the
    Kraus-operator-based formulation when the Kraus rank is 1.
    
    Args:
        nqubit (int): Number of qubits
        p_exp_list (np.ndarray): Experimental probabilities (len(M_list) x len(rho_list))
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Unitary operator matrix (2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Gradient matrix G (2^nqubit x 2^nqubit)
    """
    dim = 2**nqubit
    grad_sum = np.zeros((dim, dim), dtype=np.complex128)
    
    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            p_exp = p_exp_list[j][i]
            p_theory = trace_rhoM_term(rho_list, M_list, U, i, j)
            G_term = G_func_term(rho_list, M_list, U, i, j)
            grad_sum += -2 * (p_exp - p_theory) * G_term
    
    # Normalize by Frobenius norm to match Kraus rank-1 behavior
    norm = np.linalg.norm(grad_sum)
    if norm > 0:
        grad_sum = grad_sum / norm

    return grad_sum


def grad_func_term(rho_list, M_list, U, i, j):
    """
    Compute gradient term for gradient function computation.
    
    Computes: M @ U @ rho @ U^dagger - U @ rho @ U^dagger @ M
    This represents the gradient with respect to U.
    
    Args:
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Unitary operator matrix
        i (int): Index for density matrix
        j (int): Index for measurement operator
    
    Returns:
        np.ndarray: Gradient term matrix
    """
    M = M_list[j]
    rho = rho_list[i]
    rho_transformed = U @ rho @ U.conj().T
    return M @ rho_transformed - rho_transformed @ M


def grad_func(nqubit, p_exp_list, rho_list, M_list, U):
    """
    Compute the gradient of the loss function with respect to U.
    
    grad = sum_{i,j} -2 * (p_exp - p_theory) * (M @ U @ rho @ U^dagger - U @ rho @ U^dagger @ M)
    
    Args:
        nqubit (int): Number of qubits
        p_exp_list (np.ndarray): Experimental probabilities (len(M_list) x len(rho_list))
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Unitary operator matrix (2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Gradient matrix (2^nqubit x 2^nqubit)
    """
    dim = 2**nqubit
    grad_sum = np.zeros((dim, dim), dtype=np.complex128)
    
    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            p_exp = p_exp_list[j][i]
            p_theory = trace_rhoM_term(rho_list, M_list, U, i, j)
            grad_term = grad_func_term(rho_list, M_list, U, i, j)
            grad_sum += -2 * (p_exp - p_theory) * grad_term
    
    return grad_sum


# ============================================================================
# Cayley Transform Update
# ============================================================================

def Cayley_Transform(nqubit, step, p_exp_list, rho_list, M_list, U):
    """
    Perform Cayley transform update on the unitary matrix U.
    
    The Cayley transform preserves the unitary constraint while updating U
    along the gradient direction on the Stiefel manifold.
    
    Update formula:
        U_new = U - step * A @ C @ B^dagger @ U
    where:
        A = [G, U]
        B^dagger = [U^dagger, -G^dagger]
        C = (I + (step/2) * B^dagger @ A)^(-1)
    
    Args:
        nqubit (int): Number of qubits
        step (float): Step size for the update
        p_exp_list (np.ndarray): Experimental probabilities
        rho_list (list): List of input density matrices
        M_list (list): List of measurement operators
        U (np.ndarray): Current unitary operator matrix (2^nqubit x 2^nqubit)
    
    Returns:
        np.ndarray: Updated unitary matrix (2^nqubit x 2^nqubit)
    """
    dim = 2**nqubit
    
    # Compute gradient
    G = G_func(nqubit, p_exp_list, rho_list, M_list, U)
    
    # Construct matrices for Cayley transform
    A = np.concatenate((G, U), axis=1)
    B_dag = np.concatenate((U.conj().T, -G.conj().T), axis=0)
    
    # Compute Cayley transform matrix
    Id = np.eye(2 * dim)
    C = np.linalg.inv(Id + 0.5 * step * B_dag @ A)
    
    # Update unitary matrix
    U_new = U - step * A @ C @ B_dag @ U
    
    return U_new

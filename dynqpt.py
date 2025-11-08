r"""
Dynamic Kraus Rank Optimization for Quantum Process Tomography

This script implements the three-step procedure described in the provided
figures:

1. **Step 1**: Optimize a single Kraus operator that is constrained to be
   unitary (rank-1 channel).
2. **Step 2**: Given an existing rank-`R` channel, alternately optimize a
   mixing parameter `p` and a new unitary Kraus operator `K_new` by solving
   the sub-problem

       min_{p \in [0,1], K_new}  \mathcal{L}([\sqrt{p}K_1; \ldots ; \sqrt{p}K_R;
                                              \sqrt{1-p}K_new])

   subject to `K_new` being unitary.
3. **Step 3**: Use the optimized `(p*, K_new*)` as an initialization for a full
   Riemannian optimization over the larger Kraus manifold with rank `R+1`.

The implementation reuses the existing loss and Cayley-transform utilities in
the repository and provides high-level functions that mirror the mathematical
formulation in the images.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from qpt_data import ideal_data
from Measurement import Bases_measure
from State import n_qubit_4base_states
from Stiefel_matrix import generate_unitary_matrix
from qpt_kraus_cayley_transform import G_func as kraus_gradient_function
from qpt_kraus_cayley_transform import Cayley_Transform as kraus_cayley_update
from qpt_loss_fn import loss_qpt_kraus, loss_qpt_unitary
from qpt_unitary_exp_and_cayley import Cayley_Transform as unitary_cayley_update


# =============================================================================
# Utility helpers
# =============================================================================


def stack_kraus_blocks(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a list of Kraus operators vertically."""

    return np.vstack(blocks)


def unstack_kraus_blocks(stack: np.ndarray, dim: int) -> List[np.ndarray]:
    """Convert a stacked Kraus matrix back into a list of individual operators."""

    block_count = stack.shape[0] // dim
    return [stack[i * dim : (i + 1) * dim, :] for i in range(block_count)]


def project_to_unitary(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix onto the unitary group via SVD-based polar decomposition."""

    u, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def kraus_forward_probability(
    kraus_blocks: Sequence[np.ndarray],
    rho: np.ndarray,
    M: np.ndarray,
) -> complex:
    """Compute Tr(M @ sum_k K_k rho K_k^dagger)."""

    acc = np.zeros_like(rho, dtype=np.complex128)
    for K in kraus_blocks:
        acc += K @ rho @ K.conj().T
    return np.trace(acc @ M)


def kraus_gradient_blocks(
    nqubit: int,
    kraus_blocks: Sequence[np.ndarray],
    p_exp_list: Sequence[Sequence[complex]],
    rho_list: Sequence[np.ndarray],
    M_list: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute the Euclidean gradients for each Kraus block without normalization.

    This mirrors the derivative expression in the sub-problem described in the
    images: each block contributes `-2 (p_exp - p_theory) M @ K @ rho`.
    """

    dim = 2**nqubit
    grads = [np.zeros((dim, dim), dtype=np.complex128) for _ in kraus_blocks]

    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            residual = (
                p_exp_list[j][i]
                - kraus_forward_probability(kraus_blocks, rho, M)
            )
            for idx, K in enumerate(kraus_blocks):
                grads[idx] += -2.0 * residual * (M @ K @ rho)

    return grads


def loss_for_blocks(
    nqubit: int,
    kraus_blocks: Sequence[np.ndarray],
    p_exp_list: Sequence[Sequence[complex]],
    rho_list: Sequence[np.ndarray],
    M_list: Sequence[np.ndarray],
) -> float:
    """Wrapper around `loss_qpt_kraus` for convenience."""

    stacked = stack_kraus_blocks(kraus_blocks)
    rank = len(kraus_blocks)
    return float(
        loss_qpt_kraus(nqubit, rank, p_exp_list, rho_list, M_list, stacked)
    )


# =============================================================================
# Step 1: Optimize a single unitary Kraus operator
# =============================================================================


@dataclass
class StepOneResult:
    unitary: np.ndarray
    losses: List[float]


def optimize_single_unitary(
    nqubit: int,
    p_exp_list: Sequence[Sequence[complex]],
    rho_list: Sequence[np.ndarray],
    M_list: Sequence[np.ndarray],
    epoch: int = 500,
    step: float = 0.05,
    tol: float = 1e-8,
    unitary_init: np.ndarray | None = None,
) -> StepOneResult:
    """
    Solve `min_{K in U(d)} L([K])` using the Cayley-transform based Riemannian
    optimization available in `qpt_unitary_exp_and_cayley`.
    """

    dim = 2**nqubit
    U = unitary_init if unitary_init is not None else generate_unitary_matrix(dim)
    losses: List[float] = []

    for k in range(epoch):
        loss = loss_qpt_unitary(nqubit, p_exp_list, rho_list, M_list, U)
        losses.append(loss)
        if loss < tol:
            break
        U = unitary_cayley_update(nqubit, step, p_exp_list, rho_list, M_list, U)

    return StepOneResult(unitary=U, losses=losses)


# =============================================================================
# Step 2: Alternating optimization for the new Kraus operator and mixing p
# =============================================================================


@dataclass
class StepTwoResult:
    p_opt: float
    K_new: np.ndarray
    losses: List[float]
    p_history: List[float]


def alternating_optimize_new_kraus(
    nqubit: int,
    base_blocks: Sequence[np.ndarray],
    p_exp_list: Sequence[Sequence[complex]],
    rho_list: Sequence[np.ndarray],
    M_list: Sequence[np.ndarray],
    max_outer_iters: int = 10,
    unitary_step: float = 0.05,
    unitary_inner_iters: int = 20,
    p_grid_size: int = 51,
    tol: float = 1e-6,
) -> StepTwoResult:
    """
    Alternately optimize the scalar `p` and the new unitary Kraus operator
    `K_new` to solve the Step-2 sub-problem.
    """

    dim = 2**nqubit
    K_new = generate_unitary_matrix(dim)
    p_value = 0.5
    losses: List[float] = []
    p_history: List[float] = []

    for outer in range(max_outer_iters):
        # ------------------------------------------------------------------
        # Optimize p on a 1-D grid while holding K_new fixed
        # ------------------------------------------------------------------
        p_candidates = np.linspace(1e-6, 1.0 - 1e-6, p_grid_size)
        candidate_losses: List[float] = []
        for p_cand in p_candidates:
            scaled_blocks = [np.sqrt(p_cand) * K for K in base_blocks] + [
                np.sqrt(1.0 - p_cand) * K_new
            ]
            L_val = loss_for_blocks(
                nqubit, scaled_blocks, p_exp_list, rho_list, M_list
            )
            candidate_losses.append(L_val)

        best_idx = int(np.argmin(candidate_losses))
        p_new = float(p_candidates[best_idx])
        current_loss = candidate_losses[best_idx]
        losses.append(current_loss)
        p_history.append(p_new)

        # Check convergence on p
        if abs(p_new - p_value) < tol:
            p_value = p_new
            break

        p_value = p_new

        # ------------------------------------------------------------------
        # Optimize K_new with fixed p using Riemannian gradient steps
        # ------------------------------------------------------------------
        for _ in range(unitary_inner_iters):
            scaled_blocks = [np.sqrt(p_value) * K for K in base_blocks] + [
                np.sqrt(1.0 - p_value) * K_new
            ]
            grads = kraus_gradient_blocks(
                nqubit, scaled_blocks, p_exp_list, rho_list, M_list
            )

            # Extract gradient wrt the scaled block and apply chain rule
            grad_block = grads[-1]
            grad_K_new = np.sqrt(1.0 - p_value) * grad_block

            # Project gradient onto the tangent space of the unitary manifold
            sym_part = 0.5 * (
                K_new.conj().T @ grad_K_new + grad_K_new.conj().T @ K_new
            )
            grad_riemann = grad_K_new - K_new @ sym_part

            K_temp = K_new - unitary_step * grad_riemann
            K_new = project_to_unitary(K_temp)

    # Final loss with optimized parameters
    final_blocks = [np.sqrt(p_value) * K for K in base_blocks] + [
        np.sqrt(1.0 - p_value) * K_new
    ]
    final_loss = loss_for_blocks(
        nqubit, final_blocks, p_exp_list, rho_list, M_list
    )
    losses.append(final_loss)

    return StepTwoResult(
        p_opt=p_value,
        K_new=K_new,
        losses=losses,
        p_history=p_history,
    )


# =============================================================================
# Step 3: Full Riemannian optimization on the Kraus manifold (rank R+1)
# =============================================================================


@dataclass
class StepThreeResult:
    kraus_blocks: List[np.ndarray]
    losses: List[float]


@dataclass
class DynamicRankGrowthResult:
    """Aggregate result for the adaptive rank-growth procedure."""

    step1: StepOneResult
    step2_results: List[StepTwoResult]
    step3_results: List[StepThreeResult]
    rank_losses: List[float]
    kraus_blocks: List[np.ndarray]
    final_rank: int
    final_loss: float


def riemannian_optimize_kraus(
    nqubit: int,
    initial_blocks: Sequence[np.ndarray],
    p_exp_list: Sequence[Sequence[complex]],
    rho_list: Sequence[np.ndarray],
    M_list: Sequence[np.ndarray],
    epoch: int = 500,
    step: float = 0.02,
    tol: float = 1e-8,
) -> StepThreeResult:
    """
    Perform Riemannian optimization on the Stiefel manifold of Kraus
    operators using the Cayley-transform update from the existing toolkit.
    """

    dim = 2**nqubit
    rank = len(initial_blocks)
    Kr = stack_kraus_blocks(initial_blocks)
    losses: List[float] = []

    for _ in range(epoch):
        loss = loss_qpt_kraus(nqubit, rank, p_exp_list, rho_list, M_list, Kr)
        losses.append(float(loss))
        if loss < tol:
            break
        Grad_E = kraus_gradient_function(
            nqubit, rank, p_exp_list, rho_list, M_list, Kr
        )
        Kr = kraus_cayley_update(nqubit, Grad_E, step, Kr)

    final_blocks = unstack_kraus_blocks(Kr, dim)
    return StepThreeResult(kraus_blocks=final_blocks, losses=losses)


def dynamic_rank_growth(
    nqubit: int,
    p_exp_list: Sequence[Sequence[complex]],
    rho_list: Sequence[np.ndarray],
    M_list: Sequence[np.ndarray],
    final_loss_tol: float = 1e-6,
    improvement_tol: float | None = None,
    max_rank: int | None = None,
    max_iterations: int | None = None,
    step1_kwargs: Dict | None = None,
    step2_kwargs: Dict | None = None,
    step3_kwargs: Dict | None = None,
) -> DynamicRankGrowthResult:
    """Adaptively grow the Kraus rank until the loss converges."""

    dim = 2**nqubit
    if max_rank is None:
        max_rank = dim**2

    step1_kwargs = step1_kwargs or {}
    step2_kwargs = step2_kwargs or {}
    step3_kwargs = step3_kwargs or {}

    step1 = optimize_single_unitary(
        nqubit,
        p_exp_list,
        rho_list,
        M_list,
        **step1_kwargs,
    )

    base_blocks = [step1.unitary]
    current_rank = 1
    best_loss = step1.losses[-1] if step1.losses else float("inf")

    rank_losses = [best_loss]
    step2_results: List[StepTwoResult] = []
    step3_results: List[StepThreeResult] = []

    if best_loss <= final_loss_tol:
        return DynamicRankGrowthResult(
            step1=step1,
            step2_results=step2_results,
            step3_results=step3_results,
            rank_losses=rank_losses,
            kraus_blocks=base_blocks,
            final_rank=current_rank,
            final_loss=best_loss,
        )

    iteration = 0
    while current_rank < max_rank:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            break

        step2 = alternating_optimize_new_kraus(
            nqubit,
            base_blocks,
            p_exp_list,
            rho_list,
            M_list,
            **step2_kwargs,
        )
        step2_results.append(step2)

        init_blocks = [np.sqrt(step2.p_opt) * K for K in base_blocks]
        init_blocks.append(np.sqrt(1.0 - step2.p_opt) * step2.K_new)

        step3 = riemannian_optimize_kraus(
            nqubit,
            init_blocks,
            p_exp_list,
            rho_list,
            M_list,
            **step3_kwargs,
        )
        step3_results.append(step3)

        base_blocks = step3.kraus_blocks
        current_rank = len(base_blocks)
        current_loss = step3.losses[-1]
        rank_losses.append(current_loss)

        improvement = best_loss - current_loss
        if improvement > 0:
            best_loss = current_loss

        if current_loss <= final_loss_tol:
            break

        if improvement_tol is not None and improvement < improvement_tol:
            break

        if current_rank >= max_rank:
            break

    return DynamicRankGrowthResult(
        step1=step1,
        step2_results=step2_results,
        step3_results=step3_results,
        rank_losses=rank_losses,
        kraus_blocks=base_blocks,
        final_rank=current_rank,
        final_loss=best_loss,
    )


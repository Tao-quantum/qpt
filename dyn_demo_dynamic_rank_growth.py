"""
Comprehensive demo of the adaptive `dynamic_rank_growth` workflow.

This script showcases two scenarios:

1. Unitary target channel (2-qubit CNOT gate)
2. Non-unitary target channel (two-qubit amplitude damping)

For each scenario, we:
    - Generate synthetic tomography data (states + POVMs + expected probabilities)
    - Run the adaptive rank-growth routine until convergence
    - Report intermediate loss history and final Kraus rank
    - Optionally visualize the loss trajectories (disabled by default)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dynqpt import dynamic_rank_growth
from Measurement import Bases_measure
from State import n_qubit_4base_states
from qpt_data import ideal_data


@dataclass
class ScenarioConfig:
    name: str
    nqubit: int
    p_exp: np.ndarray
    rho_list: list[np.ndarray]
    M_list: list[np.ndarray]
    description: str


def _build_cnot_scenario() -> ScenarioConfig:
    nqubit = 2
    rho_list = n_qubit_4base_states(nqubit)
    M_list = Bases_measure(nqubit)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    p_exp = ideal_data(rho_list, M_list, cnot)
    return ScenarioConfig(
        name="cnot_unitary",
        nqubit=nqubit,
        p_exp=p_exp,
        rho_list=rho_list,
        M_list=M_list,
        description="2-qubit CNOT gate (unitary)",
    )


def _build_amplitude_damping_scenario() -> ScenarioConfig:
    nqubit = 2
    rho_list = n_qubit_4base_states(nqubit)
    M_list = Bases_measure(nqubit)

    gamma_a = 0.18
    gamma_b = 0.32

    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma_a)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma_a)], [0, 0]], dtype=np.complex128)
    L0 = np.array([[1, 0], [0, np.sqrt(1 - gamma_b)]], dtype=np.complex128)
    L1 = np.array([[0, np.sqrt(gamma_b)], [0, 0]], dtype=np.complex128)

    kraus_ops = [
        np.kron(K0, L0),
        np.kron(K0, L1),
        np.kron(K1, L0),
        np.kron(K1, L1),
    ]

    def apply_kraus(rho: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(rho)
        for K in kraus_ops:
            acc += K @ rho @ K.conj().T
        return acc

    p_exp = np.zeros((len(M_list), len(rho_list)), dtype=np.float64)
    for j, M in enumerate(M_list):
        for i, rho in enumerate(rho_list):
            p_exp[j, i] = np.trace(M @ apply_kraus(rho)).real

    return ScenarioConfig(
        name="two_qubit_amplitude_damping",
        nqubit=nqubit,
        p_exp=p_exp,
        rho_list=rho_list,
        M_list=M_list,
        description="Two-qubit amplitude damping channel (Kraus rank 4)",
    )


def run_scenario(
    scenario: ScenarioConfig,
    output_dir: str = "Img",
    show_plots: bool = False,
) -> None:
    print("=" * 80)
    print(f"Scenario: {scenario.description}")
    print("=" * 80)

    result = dynamic_rank_growth(
        scenario.nqubit,
        scenario.p_exp,
        scenario.rho_list,
        scenario.M_list,
        final_loss_tol=1e-3,
        step1_kwargs=dict(epoch=400, step=0.05, tol=1e-8),
        step2_kwargs=dict(max_outer_iters=8, unitary_step=0.03, unitary_inner_iters=30, p_grid_size=61),
        step3_kwargs=dict(epoch=600, step=0.01, tol=1e-8),
    )

    print("Final rank:", result.final_rank)
    for r, loss_value in enumerate(result.rank_losses, start=1):
        print(f"  Rank {r}: loss = {loss_value:.6e}")

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, f"demo_{scenario.name}_dynamic_rank.npz"),
        step1_losses=np.array(result.step1.losses),
        step2_losses=np.array([res.losses for res in result.step2_results], dtype=object),
        step2_p_history=np.array([res.p_history for res in result.step2_results], dtype=object),
        step3_losses=np.array([res.losses for res in result.step3_results], dtype=object),
        kraus_rank=result.final_rank,
        kraus_blocks=np.stack(result.kraus_blocks),
    )

    # Optional visualization
    if show_plots:
        plt.style.use("seaborn-v0_8-paper")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(result.step1.losses, color="tab:blue", linewidth=2, label="Step 1")
        for idx, step2_res in enumerate(result.step2_results, start=2):
            axes[0].plot(step2_res.losses, linestyle="-.", linewidth=2, label=f"Step2 R={idx}")
        for idx, step3_res in enumerate(result.step3_results, start=2):
            axes[0].plot(step3_res.losses, linestyle="--", linewidth=2, label=f"Step3 R={idx}")

        axes[0].set_title("Loss history (linear scale)")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(frameon=False)

        axes[1].semilogy(result.step1.losses, color="tab:blue", linewidth=2)
        for step2_res in result.step2_results:
            axes[1].semilogy(step2_res.losses, linestyle="-.", linewidth=2)
        for step3_res in result.step3_results:
            axes[1].semilogy(step3_res.losses, linestyle="--", linewidth=2)

        axes[1].set_title("Loss history (log scale)")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"demo_{scenario.name}_loss_curves.png"), dpi=200)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def main() -> None:
    scenarios = [
        _build_cnot_scenario(),
        _build_amplitude_damping_scenario(),
    ]

    for scenario in scenarios:
        run_scenario(scenario, show_plots=False)

if __name__ == "__main__":
    main()



import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from State import n_qubit_4base_states
from Measurement import Bases_measure
from Stiefel_matrix import  generate_stiefel_matrix
import matplotlib.pyplot as plt
from qpt_data import ideal_data
from qpt_kraus_cayley_transform import G_func as G_func_kraus, Cayley_Transform as Cayley_Transform_kraus
from qpt_unitary_exp_and_cayley import (
    G_func as G_func_unitary,
    Cayley_Transform as Cayley_Transform_unitary,
)
from qpt_loss_fn import loss_qpt_kraus, loss_qpt_unitary 

nqubit =2

H = np.array([[1,1],[1,-1]])/np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])
gate = CNOT

rank = 1
rho0_list = n_qubit_4base_states(nqubit)
M_list = Bases_measure(nqubit)

dim = 2**nqubit
Kr = generate_stiefel_matrix(dim * rank, dim)
# Use the same initialization for the unitary case when rank = 1
U = Kr.copy()
p_exp_list = ideal_data(rho0_list, M_list, gate)

epoch = 100
step = 0.08

def RGD_Cayley_kraus(nqubit, rank, p_exp_list, rho_list, M_list, Kr, epoch, step, tol=1e-6):
    losses = []
    kr_res = []
    for k in range(epoch):
        loss = loss_qpt_kraus(nqubit, rank, p_exp_list, rho_list, M_list, Kr)
        Grad_E = G_func_kraus(nqubit, rank, p_exp_list, rho_list, M_list, Kr)
        Kr = Cayley_Transform_kraus(nqubit, Grad_E, step, Kr)
        losses.append(loss)
        kr_res.append(Kr)
        if k % 2 == 0:
            print(f"Epoch: {k}, Loss: {loss}")

        if loss < tol:
            break
    return losses, kr_res


def RGD_Cayley_unitary(nqubit, p_exp_list, rho_list, M_list, U, epoch, step, tol=1e-6):
    losses = []
    unitary_res = []
    for k in range(epoch):
        loss = loss_qpt_unitary(nqubit, p_exp_list, rho_list, M_list, U)
        U = Cayley_Transform_unitary(nqubit, step, p_exp_list, rho_list, M_list, U)
        losses.append(loss)
        unitary_res.append(U)
        if k % 2 == 0:
            print(f"[Unitary] Epoch: {k}, Loss: {loss}")

        if loss < tol:
            break
    return losses, unitary_res


loss_kraus, kr_res = RGD_Cayley_kraus(nqubit, rank, p_exp_list, rho0_list, M_list, Kr, epoch, step)
loss_unitary, unitary_res = RGD_Cayley_unitary(nqubit, p_exp_list, rho0_list, M_list, U, epoch, step)


os.makedirs('Img', exist_ok=True)

epochs_kraus = range(1, len(loss_kraus) + 1)
epochs_unitary = range(1, len(loss_unitary) + 1)

plt.plot(epochs_kraus, loss_kraus, marker='o', linestyle='-', color='r', label='Kraus rank=1')
plt.plot(epochs_unitary, loss_unitary, marker='s', linestyle='--', color='b', label='Unitary')
plt.title('Loss vs. Epoch (Kraus rank=1 vs Unitary)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('Img/cnot_rank1_vs_unitary.png', dpi=150, bbox_inches='tight')
print("Plot saved to Img/cnot_rank1_vs_unitary.png")
plt.show()

# 验证CNOT的正确性，随机生成大量rho，计算层析门和目标门对应每个rho，得到量子态的保真度的均值

from Verify_correctness import testFid
print()
print("=" * 60)
print("Cell 5: Test with random density matrices")
print("=" * 60)

times = 10000
print(f"Testing with {times} random density matrices...")
fid_mean_kraus = testFid(nqubit, times, gate, rank, kr_res[-1])
fid_mean_unitary = testFid(nqubit, times, gate, 1, unitary_res[-1])
print(f"Fidelity (Kraus rank=1): {fid_mean_kraus}")
print(f"Fidelity (Unitary): {fid_mean_unitary}")
print()


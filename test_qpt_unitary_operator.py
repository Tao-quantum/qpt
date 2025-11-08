
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from State import n_qubit_4base_states, n_qubit_allstates
from Measurement import Bases_measure
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt

print("=" * 60)
print("Cell 0: 初始化和理论数据")
print("=" * 60)

H = np.array([[1,1],[1,-1]])/np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])
gate = H

def theory_data(rho_list, M_list):
    theory_values = np.zeros((len(M_list), len(rho_list)))
    for j in range(len(M_list)):
        M = M_list[j]
        for i in range(len(rho_list)):
            rho = gate @ rho_list[i] @ gate.conj().T
            theory_value = np.real(np.trace(rho @ M))
            theory_values[j, i] = theory_value
    return theory_values

nqubit = 1
rho_list = n_qubit_4base_states(nqubit)
print(f"Trace of rho_list[0]: {np.trace(rho_list[0])}")
M_list = Bases_measure(nqubit)  
print(f"Pauli strings: {M_list}")
theory_array = theory_data(rho_list, M_list)
print(f"Shape of theory_array: {np.shape(theory_array)}")
print(f"theory_array[0][0]: {theory_array[0][0]}")
print()

print("=" * 60)
print("Cell 1: RGD with exponential map")
print("=" * 60)

from qpt_unitary_exp_and_cayley import grad_func, G_func, Cayley_Transform
from qpt_loss_fn import loss_qpt_unitary as loss_qpt

def RGD_exp(nqubit, p_exp_list, rho_list, M_list, U, epoch, step, tol=1e-6):
    losses = []
    U_res = []
    grad_res = []
    for k in range(epoch):
        loss = loss_qpt(nqubit,  p_exp_list, rho_list, M_list, U)
        Grad = grad_func(nqubit, p_exp_list, rho_list, M_list, U)
        U = expm(-step * Grad) @ U
        losses.append(loss)
        U_res.append(U)
        grad_res.append(np.linalg.norm(Grad))
        if k % 20 == 0:
            print(f"Epoch: {k}, Loss: {loss}")
    return losses , U_res ,grad_res

step = 0.1
epoch  = 100
# U should be 2**nqubit x 2**nqubit for unitary operator
U = np.eye(2**nqubit)
p_exp_list = theory_data(rho_list, M_list)

print("Running RGD with exponential map...")
loss_s, U_res, grad_res = RGD_exp(nqubit, p_exp_list, rho_list, M_list, U, epoch, step, tol=1e-6)

plt.figure(figsize=(10, 6))
plt.plot(range(1, epoch + 1), loss_s, marker='o', linestyle='-', color='b', label='Loss')
plt.plot(range(1, epoch + 1), grad_res, marker='o', linestyle='-', color='r', label='Gradient Norm')
plt.title('Loss vs. Epoch (Exponential Map)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('Img/rgd_exp_result.png', dpi=150, bbox_inches='tight')
print("Plot saved to rgd_exp_result.png")
print()

print("=" * 60)
print("Cell 2: RGD with Cayley Transform")
print("=" * 60)

def RGD_Cayley(nqubit, p_exp_list, rho_list, M_list, U, epoch, step, tol=1e-6):
    losses = []
    U_res = []
    for k in range(epoch):
        loss = loss_qpt(nqubit,  p_exp_list, rho_list, M_list, U)
        U = Cayley_Transform(nqubit, step, p_exp_list, rho_list, M_list, U)
        losses.append(loss)
        U_res.append(U)
        if k % 5 == 0:
            print(f"Epoch: {k}, Loss: {loss}")
    return losses, U_res

epoch_cayley, step_cayley = 200, 0.1
# Reset U for Cayley transform
U_cayley = np.eye(2**nqubit)
print("Running RGD with Cayley Transform...")
loss_cayley, U_res_cayley = RGD_Cayley(nqubit, p_exp_list, rho_list, M_list, U_cayley, epoch_cayley, step_cayley, tol=1e-3)

plt.figure(figsize=(10, 6))
plt.plot(range(1, epoch_cayley + 1), loss_cayley, marker='o', linestyle='-', color='b')
plt.title('Loss vs. Epoch (Cayley Transform)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('Img/rgd_cayley_result.png', dpi=150, bbox_inches='tight')
plt.show()


def fidelity(rho1, rho2):
    rho_tp = np.sqrt(np.sqrt(rho1) @ rho2 @ np.sqrt(rho1))
    fid_tp = np.trace(rho_tp).real
    return fid_tp

rho0 = rho_list[0]
rho1 = gate @ rho0 @ gate.conj().T
rho2 = U @ rho0 @ U.conj().T
print(f"Fidelity: {fidelity(rho1, rho2)}")
print(f"Norm difference: {np.linalg.norm(rho1-rho2)}")
print()

print("=" * 60)
print("Cell 4: Test with random density matrices")
print("=" * 60)

def random_density_matrix(nqubit):
    dim = 2**nqubit
    matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    mat_ = (matrix + matrix.conj().T) / 2
    mat = mat_/np.trace(mat_)
    return mat

def testFid(nqubit, times):
    fid = []
    for _ in range(times):
        rho0 = random_density_matrix(nqubit)
        rho_T = gate @ rho0 @ gate.conj().T
        rho_k = U @ rho0 @ U.conj().T
        fid_tp = np.linalg.norm(rho_T-rho_k)
        fid.append(fid_tp)
    mean_fid = np.mean(fid)
    return mean_fid 

times = 10000
print(f"Testing with {times} random density matrices...")
fid_mean = testFid(nqubit, times)
print(f"Mean fidelity (norm difference): {fid_mean}")
print()

print("=" * 60)
print("All cells executed successfully!")
print("=" * 60)


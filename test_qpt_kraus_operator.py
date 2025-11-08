"""
Test QPT Kraus Operator
从 Test_qpt_kraus_operator.ipynb 转换而来
"""

import numpy as np
import matplotlib.pyplot as plt
from State import n_qubit_4base_states
from Measurement import Bases_measure
from Stiefel_matrix import generate_stiefel_matrix
from qpt_kraus_cayley_transform import G_func, Cayley_Transform, rho_kraus
from qpt_loss_fn import loss_qpt_kraus as loss_qpt

# Toffoli门的矩阵表示
Toffoli = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
])

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


def RGD_Cayley(nqubit, r, p_exp_list, rho_list, M_list, Kr,  epoch, step, tol=1e-6):
    losses = []
    kr_res = []
    for k in range(epoch):
        loss = loss_qpt(nqubit, r, p_exp_list, rho_list, M_list, Kr)
        Grad_E = G_func(nqubit, r, p_exp_list, rho_list, M_list, Kr)
        Kr = Cayley_Transform(nqubit, Grad_E, step, Kr)
        losses.append(loss)
        kr_res.append(Kr)
        if k % 2 == 0:
            print(f"Epoch: {k}, Loss: {loss}")
    return losses, kr_res


if __name__ == "__main__":
    print("=" * 60)
    print("Test QPT Kraus Operator")
    print("=" * 60)
    
    epoch, step = 50, 0.1
    r = 4
    nqubit = 1

    print("\n[1] 生成状态和测量基...")
    rho_list = n_qubit_4base_states(nqubit)
    M_list = Bases_measure(nqubit)  
    print(f"Pauli strings: {M_list}")

    print("\n[2] 初始化 Kraus 算子和理论数据...")
    Kr = generate_stiefel_matrix(r*2**nqubit, 2**nqubit)
    p_exp_list = theory_data(rho_list, M_list)
    print(f"Theory data shape: {np.shape(p_exp_list)}")

    print("\n[3] 运行 RGD with Cayley Transform...")
    loss, kr_res = RGD_Cayley(nqubit, r, p_exp_list, rho_list, M_list, Kr, epoch, step, tol=1e-6)
    print(f"\nFinal Kr shape: {np.shape(kr_res[-1])}")
    print(f"Final loss: {loss[-1]}")

    print("\n[4] 绘制损失曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), loss, marker='o', linestyle='-', color='b')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('Img/test_kraus_loss.png', dpi=150, bbox_inches='tight')
    print("Plot saved to test_kraus_loss.png")

    print("\n[5] 测试 Kraus 算子的完备性...")
    Kr = kr_res[-1]

    def test_kraus(nqubit, r, Kr):
        sum_kraus = np.zeros((2**nqubit, 2**nqubit), dtype=np.complex128)
        K_list = [Kr[i*(2**nqubit):(i+1)*(2**nqubit), :] for i in range(r)]
        for k in range(len(K_list)):
            sum_kraus += K_list[k].conj().T @ K_list[k]
        return sum_kraus 

    completeness = test_kraus(nqubit, r, Kr)
    print("Kraus completeness (should be close to identity):")
    print(completeness)

    print("\n[6] 计算保真度...")
    def fidelity(rho1, rho2):
        rho_tp = np.sqrt(np.sqrt(rho1) @ rho2 @ np.sqrt(rho1))
        fid_tp = np.trace(rho_tp)
        return fid_tp

    rho0 = rho_list[2]
    rho1 = gate @ rho0 @ gate.conj().T
    rho2 = rho_kraus(nqubit, r, rho0, Kr)
    print(f"Fidelity: {fidelity(rho1, rho2)}")
    print(f"Norm difference: {np.linalg.norm(rho1 - rho2)}")

    print("\n[7] 测试随机密度矩阵...")
    def random_density_matrix(nqubit):
        dim = 2**nqubit
        matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        mat_ = (matrix + matrix.conj().T) / 2
        mat = mat_/np.linalg.norm(mat_)
        return mat

    def testFid(nqubit, times):
        fid = []
        for _ in range(times):
            rho0 = random_density_matrix(nqubit)
            rho_T = gate @ rho0 @ gate.conj().T
            rho_k = rho_kraus(nqubit, r, rho0, Kr)
            fid_tp = fidelity(rho_T, rho_k)
            fid.append(fid_tp)
        mean_fid = np.mean(fid)
        return mean_fid 

    times = 10000
    print(f"Testing with {times} random density matrices...")
    fid_mean = testFid(nqubit, times)
    print(f"Mean fidelity (norm difference): {fid_mean}")

    print("\n[8] 使用 qutip 测试...")
    try:
        from qutip.random_objects import rand_dm

        def testFid_qutip(nqubit, times):
            fid = []
            for _ in range(times):
                rho0 = rand_dm(2**nqubit, density=0.5).full()
                rho_T = gate @ rho0 @ gate.conj().T
                rho_k = rho_kraus(nqubit, r, rho0, Kr)
                fid_tp = np.linalg.norm(rho_T-rho_k)
                fid.append(fid_tp)
            mean_fid = np.mean(fid)
            return mean_fid 

        times = 10000
        print(f"Testing with {times} qutip random density matrices...")
        fid_mean_qutip = testFid_qutip(nqubit, times)
        print(f"Mean fidelity (norm difference): {fid_mean_qutip}")
    except ImportError:
        print("qutip not available, skipping qutip test")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

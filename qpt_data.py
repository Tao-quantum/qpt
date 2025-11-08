import numpy as np
import matplotlib.pyplot as plt
from State import n_qubit_4base_states
from Measurement import Bases_measure
import os


def ideal_data(rho0_list, M_list, gate):
    ideal_values = np.zeros((len(M_list), len(rho0_list)))
    for j in range(len(M_list)):
        M = M_list[j]
        for i in range(len(rho0_list)):
            rho = gate @ rho0_list[i] @ gate.conj().T  
            ideal_value = np.real(np.trace(rho @ M))
            ideal_values[j, i] = ideal_value
    return ideal_values


def noisy_data(rho0_list, M_list, gate, noise):
    ideal_values = ideal_data(rho0_list, M_list, gate)
    data = ideal_values  + np.random.normal(0, noise, size = ideal_values.shape)
    return data


# # 测试
# if __name__ == "__main__":
#     # 创建 Img 目录（如果不存在）
#     os.makedirs('qpt_code/Img', exist_ok=True)
    
#     rho0_list = n_qubit_4base_states(1)
#     M_list = Bases_measure(1)
#     gate = np.eye(2)
#     order = 100  # noise is 1/order
#     ideal_values = ideal_data(rho0_list, M_list, gate)

#     data = noisy_data(rho0_list, M_list, gate, 1/order)

#     plt.hist(data.ravel() - ideal_values.ravel(), bins=100)
#     plt.xlabel("Error")
#     plt.ylabel("Number of probabilities")
#     plt.title("Noise Distribution")
#     plt.grid(True, alpha=0.3)
#     plt.savefig('qpt_code/Img/noisy_data.png', dpi=150, bbox_inches='tight')
#     print("Plot saved to qpt_code/Img/noisy_data.png")
#     plt.show()



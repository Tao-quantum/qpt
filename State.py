
import numpy as np
from itertools import product

# 定义X、Y、Z基下的正负态
z_plus = np.array([[1], [0]])
z_minus = np.array([[0], [1]])
x_plus = 1 / np.sqrt(2) * np.array([[1], [1]])
x_minus = 1 / np.sqrt(2) * np.array([[1], [-1]])
y_plus = 1 / np.sqrt(2) * np.array([[1], [1j]])
y_minus = 1 / np.sqrt(2) * np.array([[1], [-1j]])

phi0 = {
    'Z+': z_plus,
    'Z-': z_minus,
    'X+': x_plus,
    'Y+': y_plus,
}


def n_qubit_4base_states(n):
    # 获取所有可能的状态标签
    state_labels = list(phi0.keys())
    
    # 使用itertools.product生成所有可能的状态组合
    all_states = []
    for combination in product(state_labels, repeat=n):
        # 对于每个组合，计算张量积
        state = np.array([1])  # 初始状态为1（归一化因子）
        for state_label in combination:
            state = np.kron(state, phi0[state_label])
            density = np.outer(state, state.conj().T)
        all_states.append(density)
    
    all_states = np.array(all_states)
    
    return all_states

psi0 = {
    'Z+': z_plus,
    'Z-': z_minus,
    'X+': x_plus,
    'X-': x_minus,
    'Y+': y_plus,
    'Y-': y_minus,
}

# 函数：生成所有可能的两量子比特状态
def n_qubit_allstates(n):
    state_labels = list(psi0.keys())
    
    # 使用itertools.product生成所有可能的状态组合
    all_states = []
    for combination in product(state_labels, repeat=n):
        # 对于每个组合，计算张量积
        state = np.array([1])  
        for state_label in combination:
            state = np.kron(state, psi0[state_label])
            density = np.outer(state, state.conj().T)
        all_states.append(density)
    
    all_states = np.array(all_states)
    
    return all_states


def bloch_state(theta, phi):
    cos_half_theta = np.cos(theta / 2)
    sin_half_theta = np.sin(theta / 2)
    
    # 创建密度矩阵
    rho = np.array([
        [cos_half_theta**2, np.exp(-1j * phi) * sin_half_theta * cos_half_theta],
        [np.exp(1j * phi) * sin_half_theta * cos_half_theta, sin_half_theta**2]
    ])
    
    return rho


def bell_state(state_type):
    """
    生成Bell态的密度矩阵。
    
    参数:
    state_type -- Bell态的类型，可以是'Phi+', 'Phi-', 'Psi+', 'Psi-'
    
    返回:
    一个NumPy数组，表示Bell态的密度矩阵
    """
    state_dict = {
        'Phi+': [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)],
        'Phi-': [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)],
        'Psi+': [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
        'Psi-': [0, 1/np.sqrt(2), -1/np.sqrt(2), 0]
    }
    
    if state_type not in state_dict:
        raise ValueError("Invalid state type. Choose from 'Phi+', 'Phi-', 'Psi+', 'Psi-'.")
    
    # 生成Bell态的态矢量
    state_vector = np.array(state_dict[state_type], dtype=complex)
    
    # 计算密度矩阵，即态矢量的外积
    density_matrix = np.outer(state_vector, np.conj(state_vector))
    
    return density_matrix / np.trace(density_matrix)  # 归一化密度矩阵

def multiqubit_state(num_qubits, coefficients):
    """
    生成任意多量子比特的叠加态的密度矩阵。
    
    参数:
    num_qubits -- 量子比特的数量
    coefficients -- 一个包含复数系数的列表，长度为2^num_qubits
    
    返回:
    一个NumPy数组，表示多量子比特叠加态的密度矩阵
    """
    if len(coefficients) != 2**num_qubits:
        raise ValueError("The number of coefficients must be 2^num_qubits.")
    
    # 检查归一化条件
    if not np.isclose(np.sum(np.abs(coefficients)**2), 1):
        raise ValueError("The coefficients do not satisfy the normalization condition.")

    state_vector = np.array(coefficients, dtype=complex)
    
    density_matrix = np.outer(state_vector, np.conj(state_vector))
    
    # 确保密度矩阵是 Hermitian 矩阵，即 A[i,j] = A[j,i]*
    
    density_matrix = (density_matrix + density_matrix.conj().T) / 2
    
    return density_matrix


def GHZ_state(N, delta=0):
    """
    生成N量子比特的GHZ态。

    参数:
    N -- 量子比特的数量
    delta -- 全局相位因子，默认为0

    返回:
    一个NumPy数组，表示N量子比特的GHZ态
    """
    # 计算所有可能状态的二进制表示
    states = np.arange(2**N)
    
    # 初始化GHZ态的系数数组
    ghz_coefficients = np.zeros(2**N, dtype=complex)
    
    # 应用GHZ态的定义
    ghz_coefficients[0] = 1/np.sqrt(2)  # |0...0>
    ghz_coefficients[-1] = (np.exp(1j*delta) / np.sqrt(2))  # |1...1>
    
    # 将系数数组转换为态矢量
    ghz_state_vector = ghz_coefficients
    ghz_density_matrix = np.outer(ghz_state_vector, np.conj(ghz_state_vector))

    return ghz_density_matrix


def random_density_matrix(n):
    dim = 2**n
    # real_part = 2 * np.random.rand(dim, 1) - 1
    # imag_part = 2 * np.random.rand(dim, 1) - 1
    # a = real_part + 1j * imag_part
    # a = a / np.linalg.norm(a)
    # mat = np.outer(a, a.conj().T)
    matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    mat_ = (matrix + matrix.conj().T) / 2
    mat = mat_/np.trace(mat_)
    return mat
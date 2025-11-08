import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from State import n_qubit_4base_states
from Measurement import Bases_measure
from Stiefel_matrix import generate_stiefel_matrix
import matplotlib.pyplot as plt
from qpt_data import ideal_data
from qpt_kraus_cayley_transform import G_func, Cayley_Transform as Cayley_Transform_kraus
from qpt_loss_fn import loss_qpt_kraus
from Verify_correctness import testFid

nqubit = 2

# CNOT gate
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
gate = CNOT

# 定义要测试的 rank 值
rank_list = [1, 2, 3, 4]

# 准备初始数据
rho0_list = n_qubit_4base_states(nqubit)
M_list = Bases_measure(nqubit)
p_exp_list = ideal_data(rho0_list, M_list, gate)

# 优化参数
epoch = 1000
step = 0.005
tol = 1e-6

# 存储所有 rank 的结果
all_losses = {}
all_fidelities = {}

def RGD_Cayley_kraus(nqubit, rank, p_exp_list, rho_list, M_list, Kr, epoch, step, tol=1e-6):
    losses = []
    kr_res = []
    for k in range(epoch):
        loss = loss_qpt_kraus(nqubit, rank, p_exp_list, rho_list, M_list, Kr)
        Grad_E = G_func(nqubit, rank, p_exp_list, rho_list, M_list, Kr)
        Kr = Cayley_Transform_kraus(nqubit, Grad_E, step, Kr)
        losses.append(loss)
        kr_res.append(Kr)
        if k % 10 == 0:
            print(f"Rank {rank}, Epoch: {k}, Loss: {loss}")

        if loss < tol:
            break
    return losses, kr_res

# 对每个 rank 运行优化
print("=" * 60)
print("开始对不同 rank 进行优化")
print("=" * 60)

for rank in rank_list:
    print(f"\n正在处理 rank = {rank}...")
    
    # 初始化 Stiefel 矩阵
    Kr = generate_stiefel_matrix(2**nqubit * rank, 2**nqubit)
    
    # 运行优化
    loss_kraus, kr_res = RGD_Cayley_kraus(
        nqubit, rank, p_exp_list, rho0_list, M_list, Kr, epoch, step, tol
    )
    
    # 存储损失函数
    all_losses[rank] = loss_kraus
    
    # 计算平均保真度
    print(f"计算 rank = {rank} 的平均保真度...")
    times = 10000
    fid_mean = testFid(nqubit, times, gate, rank, kr_res[-1])
    all_fidelities[rank] = fid_mean
    print(f"Rank {rank} 的平均保真度: {fid_mean}")

# 创建图像保存目录
os.makedirs('Img', exist_ok=True)

# 绘制不同 rank 的损失函数曲线
plt.figure(figsize=(10, 6))
colors = ['r', 'b', 'g', 'm', 'c', 'y']
for i, rank in enumerate(rank_list):
    losses = all_losses[rank]
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, marker='o', linestyle='-', 
             color=colors[i % len(colors)], label=f'rank={rank}', markersize=3)

plt.title('Loss vs. Epoch for Different Ranks (CNOT)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')  # 使用对数刻度以便更好地观察
plt.savefig('Img/cnot_rank_comparison_loss.png', dpi=150, bbox_inches='tight')
print("\n损失函数图像已保存到 Img/cnot_rank_comparison_loss.png")
plt.show()

# 绘制不同 rank 对应的平均保真度
plt.figure(figsize=(10, 6))
ranks = list(all_fidelities.keys())
fidelities = list(all_fidelities.values())

plt.plot(ranks, fidelities, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title('Average Fidelity vs. Rank (CNOT)')
plt.xlabel('Rank')
plt.ylabel('Average Fidelity')
plt.grid(True)
plt.xticks(ranks)  # 设置 x 轴刻度为 rank 值

# 在点上标注数值
for rank, fid in zip(ranks, fidelities):
    plt.annotate(f'{fid:.4f}', (rank, fid), textcoords="offset points", 
                 xytext=(0,10), ha='center')

plt.savefig('Img/cnot_rank_comparison_fidelity.png', dpi=150, bbox_inches='tight')
print("平均保真度图像已保存到 Img/cnot_rank_comparison_fidelity.png")
plt.show()

# 打印总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
for rank in rank_list:
    print(f"Rank {rank}: 平均保真度 = {all_fidelities[rank]:.6f}")
print("=" * 60)
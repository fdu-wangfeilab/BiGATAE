import numpy as np
from scipy.sparse import coo_matrix
import torch

def get_edge_index(M):
    # 将稠密矩阵M转换为稀疏矩阵，并获得边的索引edge_index 
    M_ = coo_matrix(M) # 稠密矩阵--->稀疏矩阵
    values = M_.data
    indices = np.vstack((M_.row, M_.col))  # 我们真正需要的coo形式
    adj = torch.LongTensor(indices)
    return adj

def get_edge_index_sparse(M):
    # 将稠密矩阵M转换为稀疏矩阵，并获得边的索引edge_index 
    M_ = coo_matrix(M) # 稠密矩阵--->稀疏矩阵
    return M_

# edge_index = get_edge_index(S_)
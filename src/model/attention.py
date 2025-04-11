import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import numpy as np  # 导入NumPy库，用于数学计算
from torch import Tensor  # 导入Tensor类型
from typing import Optional  # 导入Optional类型，用于类型提示

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dK: int, dV: int, dModel: int, numHeads: int, p: float = 0.):
        """
        初始化多头注意力层
        参数:
        - dK: 键向量的维度
        - dV: 值向量的维度
        - dModel: 模型的维度
        - numHeads: 注意力头的数量
        - p: dropout比率，默认为0
        """
        super(MultiHeadAttention, self).__init__()
        self.dModel = dModel  # 存储模型维度
        self.dK = dK  # 存储键向量维度
        self.dV = dV  # 存储值向量维度
        self.numHeads = numHeads  # 存储头数量
        self.dropout = nn.Dropout(p)  # 创建dropout层，用于防止过拟合
        
        # 线性投影层
        self.wQ = nn.Linear(dModel, dK * numHeads)  # 查询向量的线性投影
        self.wK = nn.Linear(dModel, dK * numHeads)  # 键向量的线性投影
        self.wV = nn.Linear(dModel, dV * numHeads)  # 值向量的线性投影
        self.wOut = nn.Linear(dV * numHeads, dModel)  # 输出的线性投影

        # 初始化参数
        # 使用正态分布初始化权重，均值为0，标准差基于维度计算
        nn.init.normal_(self.wQ.weight, mean=0, std=np.sqrt(2.0 / (dModel + dK)))
        nn.init.normal_(self.wK.weight, mean=0, std=np.sqrt(2.0 / (dModel + dK)))
        nn.init.normal_(self.wV.weight, mean=0, std=np.sqrt(2.0 / (dModel + dV)))
        nn.init.normal_(self.wOut.weight, mean=0, std=np.sqrt(2.0 / (dModel + dV)))

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, attnMask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播计算
        参数:
        - Q: 查询张量
        - K: 键张量 
        - V: 值张量
        - attnMask: 注意力掩码，可选
        返回:
        - 多头注意力的输出张量
        """
        N = Q.size(0)  # 获取批次大小
        
        # 多头分割
        # 将查询、键、值投影并重塑为多头形式 [batch_size, num_heads, seq_len, head_dim]
        Q = self.wQ(Q).view(N, -1, self.numHeads, self.dK).transpose(1, 2)  # [N, num_heads, seq_len_q, dK]
        K = self.wK(K).view(N, -1, self.numHeads, self.dK).transpose(1, 2)  # [N, num_heads, seq_len_k, dK]
        V = self.wV(V).view(N, -1, self.numHeads, self.dV).transpose(1, 2)  # [N, num_heads, seq_len_v, dV]
        
        # 掩码预处理
        if attnMask is not None:
            # 将掩码扩展到所有注意力头
            attnMask = attnMask.unsqueeze(1).repeat(1, self.numHeads, 1, 1)  # [N, num_heads, seq_len_q, seq_len_k]
            attnMask = attnMask.bool()  # 转换为布尔型掩码

        # 计算注意力权重
        # 执行点积注意力计算: Q·K^T/sqrt(dK)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dK)  # [N, num_heads, seq_len_q, seq_len_k]
        if attnMask is not None:
            # 将掩码位置的注意力分数设为一个很小的负数，使softmax后接近0
            scores.masked_fill_(attnMask, -1e4)
        # 应用softmax得到注意力权重
        attns = torch.softmax(scores, dim=-1)  # [N, num_heads, seq_len_q, seq_len_k]
        attns = self.dropout(attns)  # 应用dropout

        # 计算输出
        # 将注意力权重与值相乘，得到加权求和的结果
        output = torch.matmul(attns, V)  # [N, num_heads, seq_len_q, dV]
        # 转置并重塑以合并所有头
        output = output.transpose(1, 2).contiguous().view(N, -1, self.dV * self.numHeads)  # [N, seq_len_q, num_heads*dV]
        # 通过输出线性层
        output = self.wOut(output)  # [N, seq_len_q, dModel]

        return output  # 返回注意力机制的输出 
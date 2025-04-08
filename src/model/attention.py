import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dK: int, dV: int, dModel: int, numHeads: int, p: float = 0.):
        super(MultiHeadAttention, self).__init__()
        self.dModel = dModel
        self.dK = dK
        self.dV = dV
        self.numHeads = numHeads
        self.dropout = nn.Dropout(p)
        
        # 线性投影层
        self.wQ = nn.Linear(dModel, dK * numHeads)
        self.wK = nn.Linear(dModel, dK * numHeads)
        self.wV = nn.Linear(dModel, dV * numHeads)
        self.wOut = nn.Linear(dV * numHeads, dModel)

        # 初始化参数
        nn.init.normal_(self.wQ.weight, mean=0, std=np.sqrt(2.0 / (dModel + dK)))
        nn.init.normal_(self.wK.weight, mean=0, std=np.sqrt(2.0 / (dModel + dK)))
        nn.init.normal_(self.wV.weight, mean=0, std=np.sqrt(2.0 / (dModel + dV)))
        nn.init.normal_(self.wOut.weight, mean=0, std=np.sqrt(2.0 / (dModel + dV)))

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, attnMask: Optional[Tensor] = None) -> Tensor:
        N = Q.size(0)
        
        # 多头分割
        Q = self.wQ(Q).view(N, -1, self.numHeads, self.dK).transpose(1, 2)
        K = self.wK(K).view(N, -1, self.numHeads, self.dK).transpose(1, 2)
        V = self.wV(V).view(N, -1, self.numHeads, self.dV).transpose(1, 2)
        
        # 掩码预处理
        if attnMask is not None:
            attnMask = attnMask.unsqueeze(1).repeat(1, self.numHeads, 1, 1)
            attnMask = attnMask.bool()

        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dK)
        if attnMask is not None:
            scores.masked_fill_(attnMask, -1e4)
        attns = torch.softmax(scores, dim=-1)
        attns = self.dropout(attns)

        # 计算输出
        output = torch.matmul(attns, V)
        output = output.transpose(1, 2).contiguous().view(N, -1, self.dV * self.numHeads)
        output = self.wOut(output)

        return output 
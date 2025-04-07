import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dModel, numHeads):
        super(MultiHeadAttention, self).__init__()
        assert dModel % numHeads == 0, "dModel必须能被numHeads整除"
        
        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads
        
        self.wQ = nn.Linear(dModel, dModel)
        self.wK = nn.Linear(dModel, dModel)
        self.wV = nn.Linear(dModel, dModel)
        self.wO = nn.Linear(dModel, dModel)
        
    def forward(self, q, k, v, mask=None):
        batchSize = q.size(0)
        
        # 线性变换并分头
        q = self.wQ(q).view(batchSize, -1, self.numHeads, self.dK).transpose(1, 2)
        k = self.wK(k).view(batchSize, -1, self.numHeads, self.dK).transpose(1, 2)
        v = self.wV(v).view(batchSize, -1, self.numHeads, self.dK).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dK)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attention = torch.softmax(scores, dim=-1)
        
        # 计算输出
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batchSize, -1, self.dModel)
        
        return self.wO(output) 
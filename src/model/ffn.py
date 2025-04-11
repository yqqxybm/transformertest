import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import Tensor  # 导入Tensor类型

class PoswiseFFN(nn.Module):
    """位置前馈网络"""
    def __init__(self, dModel: int, dFF: int, p: float = 0.):
        """
        初始化位置前馈网络
        参数:
        - dModel: 模型维度
        - dFF: 前馈网络隐藏层维度
        - p: dropout比率，默认为0
        """
        super(PoswiseFFN, self).__init__()
        self.dModel = dModel  # 保存模型维度
        self.dFF = dFF  # 保存前馈网络隐藏层维度
        self.conv1 = nn.Conv1d(dModel, dFF, 1, 1, 0)  # 第一个1x1卷积层，相当于全连接层，将维度从dModel扩展到dFF
        self.conv2 = nn.Conv1d(dFF, dModel, 1, 1, 0)  # 第二个1x1卷积层，相当于全连接层，将维度从dFF恢复到dModel
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，inplace=True表示直接修改输入而不创建新的输出
        self.dropout = nn.Dropout(p=p)  # Dropout层，用于防止过拟合

    def forward(self, X: Tensor) -> Tensor:
        """
        前向传播
        参数:
        - X: 输入张量 [batch_size, seq_len, dModel]
        返回:
        - 前馈网络输出张量 [batch_size, seq_len, dModel]
        """
        out = self.conv1(X.transpose(1, 2))  # 将输入张量转置为[batch_size, dModel, seq_len]并通过第一个卷积层
        out = self.relu(out)  # 应用ReLU激活函数
        out = self.conv2(out).transpose(1, 2)  # 通过第二个卷积层并转置回[batch_size, seq_len, dModel]
        out = self.dropout(out)  # 应用dropout
        return out  # 返回前馈网络的输出 
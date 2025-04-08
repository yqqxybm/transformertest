import torch.nn as nn
from torch import Tensor

class PoswiseFFN(nn.Module):
    """位置前馈网络"""
    def __init__(self, dModel: int, dFF: int, p: float = 0.):
        super(PoswiseFFN, self).__init__()
        self.dModel = dModel
        self.dFF = dFF
        self.conv1 = nn.Conv1d(dModel, dFF, 1, 1, 0)
        self.conv2 = nn.Conv1d(dFF, dModel, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X: Tensor) -> Tensor:
        out = self.conv1(X.transpose(1, 2))
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)
        out = self.dropout(out)
        return out 
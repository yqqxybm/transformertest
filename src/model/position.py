import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库，用于数学计算
from torch import Tensor  # 导入Tensor类型

def posSinusoidEmbedding(seqLen: int, dModel: int) -> Tensor:
    """
    生成正弦位置编码
    参数:
    - seqLen: 序列长度
    - dModel: 模型维度
    返回:
    - 位置编码张量 [seqLen, dModel]
    """
    embeddings = torch.zeros((seqLen, dModel))  # 创建零张量，大小为[seqLen, dModel]
    for i in range(dModel):
        f = torch.sin if i % 2 == 0 else torch.cos  # 偶数索引使用sin函数，奇数索引使用cos函数
        # 使用公式 sin/cos(pos/10000^(2i/dModel)) 计算位置编码
        embeddings[:, i] = f(torch.arange(0, seqLen) / np.power(1e4, 2 * (i // 2) / dModel))
    return embeddings.float()  # 返回浮点型的位置编码张量 
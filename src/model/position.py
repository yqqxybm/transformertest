import torch
import numpy as np
from torch import Tensor

def posSinusoidEmbedding(seqLen: int, dModel: int) -> Tensor:
    """生成正弦位置编码"""
    embeddings = torch.zeros((seqLen, dModel))
    for i in range(dModel):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seqLen) / np.power(1e4, 2 * (i // 2) / dModel))
    return embeddings.float() 
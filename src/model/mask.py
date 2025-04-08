import torch
from torch import Tensor

def getLenMask(b: int, maxLen: int, featLens: Tensor, device: torch.device) -> Tensor:
    """生成长度掩码"""
    attnMask = torch.ones((b, maxLen, maxLen), device=device)
    for i in range(b):
        attnMask[i, :, :featLens[i]] = 0
    return attnMask.to(torch.bool)

def getSubsequentMask(b: int, maxLen: int, device: torch.device) -> Tensor:
    """生成后续掩码"""
    return torch.triu(torch.ones((b, maxLen, maxLen), device=device), diagonal=1).to(torch.bool)

def getEncDecMask(b: int, maxFeatLen: int, featLens: Tensor, maxLabelLen: int, device: torch.device) -> Tensor:
    """生成编码器-解码器掩码"""
    attnMask = torch.zeros((b, maxLabelLen, maxFeatLen), device=device)
    for i in range(b):
        attnMask[i, :, featLens[i]:] = 1
    return attnMask.to(torch.bool) 
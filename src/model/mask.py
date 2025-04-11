import torch  # 导入PyTorch库
from torch import Tensor  # 导入Tensor类型

def getLenMask(b: int, maxLen: int, featLens: Tensor, device: torch.device) -> Tensor:
    """
    生成长度掩码
    参数:
    - b: 批次大小
    - maxLen: 最大序列长度
    - featLens: 每个样本的实际特征长度
    - device: 计算设备
    返回:
    - 长度掩码张量 [b, maxLen, maxLen]，True表示被掩盖的位置
    """
    attnMask = torch.ones((b, maxLen, maxLen), device=device)  # 创建全1掩码张量
    for i in range(b):
        attnMask[i, :, :featLens[i]] = 0  # 将每个样本实际长度内的位置设为0（即不掩盖）
    return attnMask.to(torch.bool)  # 转换为布尔型掩码并返回

def getSubsequentMask(b: int, maxLen: int, device: torch.device) -> Tensor:
    """
    生成后续掩码（用于解码器自注意力，防止看到未来信息）
    参数:
    - b: 批次大小
    - maxLen: 最大序列长度
    - device: 计算设备
    返回:
    - 后续掩码张量 [b, maxLen, maxLen]，True表示被掩盖的位置
    """
    return torch.triu(torch.ones((b, maxLen, maxLen), device=device), diagonal=1).to(torch.bool)  # 创建上三角矩阵（对角线偏移1）作为掩码

def getEncDecMask(b: int, maxFeatLen: int, featLens: Tensor, maxLabelLen: int, device: torch.device) -> Tensor:
    """
    生成编码器-解码器掩码
    参数:
    - b: 批次大小
    - maxFeatLen: 最大特征长度（源序列）
    - featLens: 每个样本的实际特征长度
    - maxLabelLen: 最大标签长度（目标序列）
    - device: 计算设备
    返回:
    - 编码器-解码器掩码张量 [b, maxLabelLen, maxFeatLen]，True表示被掩盖的位置
    """
    attnMask = torch.zeros((b, maxLabelLen, maxFeatLen), device=device)  # 创建全0掩码张量
    for i in range(b):
        attnMask[i, :, featLens[i]:] = 1  # 将每个样本实际长度之外的位置设为1（即掩盖）
    return attnMask.to(torch.bool)  # 转换为布尔型掩码并返回 
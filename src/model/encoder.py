import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import Tensor  # 导入Tensor类型
from typing import Optional  # 导入Optional类型，用于类型提示
from .attention import MultiHeadAttention  # 导入多头注意力机制
from .ffn import PoswiseFFN  # 导入位置前馈网络
from .position import posSinusoidEmbedding  # 导入正弦位置编码函数

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, dim: int, n: int, dff: int, dropoutPosffn: float, dropoutAttn: float):
        """
        初始化编码器层
        参数:
        - dim: 模型维度
        - n: 注意力头数量
        - dff: 前馈网络隐藏层维度
        - dropoutPosffn: 前馈网络的dropout比率
        - dropoutAttn: 注意力的dropout比率
        """
        super(EncoderLayer, self).__init__()
        assert dim % n == 0  # 确保模型维度可以被头数量整除
        hdim = dim // n  # 计算每个头的维度
        
        self.norm1 = nn.LayerNorm(dim)  # 第一个层归一化，用于注意力子层后的归一化
        self.norm2 = nn.LayerNorm(dim)  # 第二个层归一化，用于前馈网络子层后的归一化
        self.multiHeadAttn = MultiHeadAttention(hdim, hdim, dim, n, dropoutAttn)  # 多头自注意力机制
        self.poswiseFFN = PoswiseFFN(dim, dff, p=dropoutPosffn)  # 位置前馈网络

    def forward(self, encIn: Tensor, attnMask: Optional[Tensor]) -> Tensor:
        """
        编码器层的前向传播
        参数:
        - encIn: 输入张量 [batch_size, seq_len, dim]
        - attnMask: 注意力掩码 (可选)
        返回:
        - 编码器层输出张量
        """
        residual = encIn  # 保存残差连接的输入
        context = self.multiHeadAttn(encIn, encIn, encIn, attnMask)  # 多头自注意力计算
        out = self.norm1(residual + context)  # 第一个子层的残差连接和层归一化
        
        residual = out  # 保存第二个残差连接的输入
        out = self.poswiseFFN(out)  # 通过位置前馈网络
        out = self.norm2(residual + out)  # 第二个子层的残差连接和层归一化
        return out  # 返回编码器层的输出

class Encoder(nn.Module):
    """编码器"""
    def __init__(self, dropoutEmb: float, dropoutPosffn: float, dropoutAttn: float,
                 numLayers: int, encDim: int, numHeads: int, dff: int, tgtLen: int):
        """
        初始化编码器
        参数:
        - dropoutEmb: 嵌入层的dropout比率
        - dropoutPosffn: 前馈网络的dropout比率
        - dropoutAttn: 注意力机制的dropout比率
        - numLayers: 编码器层的数量
        - encDim: 编码器的维度
        - numHeads: 注意力头的数量
        - dff: 前馈网络隐藏层的维度
        - tgtLen: 目标序列最大长度
        """
        super(Encoder, self).__init__()
        self.tgtLen = tgtLen  # 保存目标序列最大长度
        self.posEmb = nn.Embedding.from_pretrained(posSinusoidEmbedding(tgtLen, encDim), freeze=True)  # 位置编码嵌入，使用预训练权重并冻结
        self.embDropout = nn.Dropout(dropoutEmb)  # 嵌入层的dropout
        self.layers = nn.ModuleList([
            EncoderLayer(encDim, numHeads, dff, dropoutPosffn, dropoutAttn) 
            for _ in range(numLayers)
        ])  # 创建多个编码器层
    
    def forward(self, X: Tensor, X_lens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        编码器的前向传播
        参数:
        - X: 输入序列张量 [batch_size, seq_len, dim]
        - X_lens: 输入序列的实际长度
        - mask: 掩码 (可选)
        返回:
        - 编码器的输出张量
        """
        batchSize, seqLen, dModel = X.shape  # 获取批次大小、序列长度和模型维度
        out = X + self.posEmb(torch.arange(seqLen, device=X.device))  # 添加位置编码
        out = self.embDropout(out)  # 应用dropout到嵌入和位置编码的和
        
        for layer in self.layers:
            out = layer(out, mask)  # 通过每个编码器层
        return out  # 返回编码器的最终输出 
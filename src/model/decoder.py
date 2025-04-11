import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import Tensor  # 导入Tensor类型
from typing import Optional  # 导入Optional类型，用于类型提示
from .attention import MultiHeadAttention  # 导入多头注意力机制
from .ffn import PoswiseFFN  # 导入位置前馈网络
from .position import posSinusoidEmbedding  # 导入正弦位置编码函数

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, dim: int, n: int, dff: int, dropoutPosffn: float, dropoutAttn: float):
        """
        初始化解码器层
        参数:
        - dim: 模型维度
        - n: 注意力头数量
        - dff: 前馈网络隐藏层维度
        - dropoutPosffn: 前馈网络的dropout比率
        - dropoutAttn: 注意力的dropout比率
        """
        super(DecoderLayer, self).__init__()
        assert dim % n == 0  # 确保模型维度可以被头数量整除
        hdim = dim // n  # 计算每个头的维度
        
        self.norm1 = nn.LayerNorm(dim)  # 第一个层归一化，用于自注意力子层后的归一化
        self.norm2 = nn.LayerNorm(dim)  # 第二个层归一化，用于编码器-解码器注意力子层后的归一化
        self.norm3 = nn.LayerNorm(dim)  # 第三个层归一化，用于前馈网络子层后的归一化
        self.maskedMultiHeadAttn = MultiHeadAttention(hdim, hdim, dim, n, dropoutAttn)  # 带掩码的多头自注意力
        self.multiHeadAttn = MultiHeadAttention(hdim, hdim, dim, n, dropoutAttn)  # 编码器-解码器多头注意力
        self.poswiseFFN = PoswiseFFN(dim, dff, p=dropoutPosffn)  # 位置前馈网络

    def forward(self, decIn: Tensor, encOut: Tensor, 
                selfAttnMask: Optional[Tensor], encDecAttnMask: Optional[Tensor]) -> Tensor:
        """
        解码器层的前向传播
        参数:
        - decIn: 解码器输入张量 [batch_size, seq_len, dim]
        - encOut: 编码器输出张量 [batch_size, src_seq_len, dim]
        - selfAttnMask: 自注意力掩码 (可选)
        - encDecAttnMask: 编码器-解码器注意力掩码 (可选)
        返回:
        - 解码器层输出张量
        """
        residual = decIn  # 保存第一个残差连接的输入
        context = self.maskedMultiHeadAttn(decIn, decIn, decIn, selfAttnMask)  # 带掩码的多头自注意力计算
        out = self.norm1(residual + context)  # 第一个子层的残差连接和层归一化
        
        residual = out  # 保存第二个残差连接的输入
        context = self.multiHeadAttn(out, encOut, encOut, encDecAttnMask)  # 编码器-解码器注意力计算
        out = self.norm2(residual + context)  # 第二个子层的残差连接和层归一化
        
        residual = out  # 保存第三个残差连接的输入
        out = self.poswiseFFN(out)  # 通过位置前馈网络
        out = self.norm3(residual + out)  # 第三个子层的残差连接和层归一化
        return out  # 返回解码器层的输出

class Decoder(nn.Module):
    """解码器"""
    def __init__(self, dropoutEmb: float, dropoutPosffn: float, dropoutAttn: float,
                 numLayers: int, decDim: int, numHeads: int, dff: int, tgtLen: int):
        """
        初始化解码器
        参数:
        - dropoutEmb: 嵌入层的dropout比率
        - dropoutPosffn: 前馈网络的dropout比率
        - dropoutAttn: 注意力机制的dropout比率
        - numLayers: 解码器层的数量
        - decDim: 解码器的维度
        - numHeads: 注意力头的数量
        - dff: 前馈网络隐藏层的维度
        - tgtLen: 目标序列最大长度
        """
        super(Decoder, self).__init__()
        self.tgtLen = tgtLen  # 保存目标序列最大长度
        self.posEmb = nn.Embedding.from_pretrained(posSinusoidEmbedding(tgtLen, decDim), freeze=True)  # 位置编码嵌入，使用预训练权重并冻结
        self.embDropout = nn.Dropout(dropoutEmb)  # 嵌入层的dropout
        self.layers = nn.ModuleList([
            DecoderLayer(decDim, numHeads, dff, dropoutPosffn, dropoutAttn) 
            for _ in range(numLayers)
        ])  # 创建多个解码器层
    
    def forward(self, Y: Tensor, encOut: Tensor, 
                selfAttnMask: Optional[Tensor] = None, 
                encDecAttnMask: Optional[Tensor] = None) -> Tensor:
        """
        解码器的前向传播
        参数:
        - Y: 目标序列张量 [batch_size, seq_len, dim]
        - encOut: 编码器输出张量 [batch_size, src_seq_len, dim]
        - selfAttnMask: 自注意力掩码 (可选)
        - encDecAttnMask: 编码器-解码器注意力掩码 (可选)
        返回:
        - 解码器的输出张量
        """
        batchSize, seqLen, dModel = Y.shape  # 获取批次大小、序列长度和模型维度
        out = Y + self.posEmb(torch.arange(seqLen, device=Y.device))  # 添加位置编码
        out = self.embDropout(out)  # 应用dropout到嵌入和位置编码的和
        
        for layer in self.layers:
            out = layer(out, encOut, selfAttnMask, encDecAttnMask)  # 通过每个解码器层
        return out  # 返回解码器的最终输出 
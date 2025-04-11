import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import math  # 导入数学函数库
from .attention import MultiHeadAttention  # 从attention模块导入多头注意力机制

# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxLen=5000):
        """
        初始化位置编码层
        参数:
        - dModel: 模型维度
        - maxLen: 最大序列长度，默认5000
        """
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵，大小为[maxLen, dModel]
        pe = torch.zeros(maxLen, dModel)
        # 生成位置索引向量 [0, 1, 2, ..., maxLen-1]，并扩展为列向量
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        # 计算分母项 10000^(2i/dModel)
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        
        # 使用正弦函数计算偶数索引位置的编码
        pe[:, 0::2] = torch.sin(position * divTerm)
        # 使用余弦函数计算奇数索引位置的编码
        pe[:, 1::2] = torch.cos(position * divTerm)
        # 添加批次维度 [1, maxLen, dModel]
        pe = pe.unsqueeze(0)
        # 将位置编码注册为缓冲区（非模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播: 将位置编码添加到输入张量
        """
        return x + self.pe[:, :x.size(1)]  # 将位置编码切片到输入序列长度并相加

class TransformerBlock(nn.Module):
    def __init__(self, dModel, numHeads, dFf, dropout=0.1):
        """
        初始化Transformer块
        参数:
        - dModel: 模型维度
        - numHeads: 注意力头数
        - dFf: 前馈网络隐藏层维度
        - dropout: Dropout比率，默认0.1
        """
        super(TransformerBlock, self).__init__()
        
        # 计算每个头的键和值的维度
        dK = dModel // numHeads  # 每个头的键维度
        dV = dModel // numHeads  # 每个头的值维度
        
        # 多头注意力层
        self.attention = MultiHeadAttention(dK, dV, dModel, numHeads, dropout)
        # 第一个归一化层，用于注意力子层的残差连接
        self.norm1 = nn.LayerNorm(dModel)
        # 第二个归一化层，用于前馈子层的残差连接
        self.norm2 = nn.LayerNorm(dModel)
        # 位置前馈网络，包含两个线性变换和一个ReLU激活函数
        self.feedForward = nn.Sequential(
            nn.Linear(dModel, dFf),  # 第一个线性层，升维
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(dFf, dModel)  # 第二个线性层，降维回原始维度
        )
        
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        参数:
        - x: 输入张量
        - mask: 注意力掩码，默认为None
        """
        # 多头注意力计算
        attentionOutput = self.attention(x, x, x, mask)
        # 第一个子层的残差连接和层归一化
        x = self.norm1(x + self.dropout(attentionOutput))
        
        # 前馈网络计算
        ffOutput = self.feedForward(x)
        # 第二个子层的残差连接和层归一化
        x = self.norm2(x + self.dropout(ffOutput))
        
        return x  # 返回处理后的张量

class Transformer(nn.Module):
    def __init__(self, srcVocabSize, tgtVocabSize, dModel=512, numHeads=8, 
                 numLayers=6, dFf=2048, dropout=0.1):
        """
        初始化Transformer模型
        参数:
        - srcVocabSize: 源语言词汇表大小
        - tgtVocabSize: 目标语言词汇表大小
        - dModel: 模型维度，默认512
        - numHeads: 注意力头数，默认8
        - numLayers: 编码器和解码器层数，默认6
        - dFf: 前馈网络隐藏层维度，默认2048
        - dropout: Dropout比率，默认0.1
        """
        super(Transformer, self).__init__()
        # 源语言词嵌入层
        self.srcEmbedding = nn.Embedding(srcVocabSize, dModel)
        # 目标语言词嵌入层
        self.tgtEmbedding = nn.Embedding(tgtVocabSize, dModel)
        # 位置编码层
        self.positionalEncoding = PositionalEncoding(dModel)
        
        # 创建编码器层列表
        self.encoderLayers = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFf, dropout)
            for _ in range(numLayers)
        ])
        
        # 创建解码器层列表
        self.decoderLayers = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFf, dropout)
            for _ in range(numLayers)
        ])
        
        # 输出线性层，将解码器输出映射到目标语言词汇表
        self.finalLayer = nn.Linear(dModel, tgtVocabSize)
        
    def forward(self, src, tgt, srcMask=None, tgtMask=None):
        """
        前向传播
        参数:
        - src: 源语言输入
        - tgt: 目标语言输入
        - srcMask: 源语言掩码，默认为None
        - tgtMask: 目标语言掩码，默认为None
        """
        # 编码器处理
        # 1. 源语言词嵌入并添加位置编码
        src = self.positionalEncoding(self.srcEmbedding(src))
        # 2. 通过所有编码器层
        for layer in self.encoderLayers:
            src = layer(src, srcMask)
            
        # 解码器处理
        # 1. 目标语言词嵌入并添加位置编码
        tgt = self.positionalEncoding(self.tgtEmbedding(tgt))
        # 2. 通过所有解码器层
        for layer in self.decoderLayers:
            tgt = layer(tgt, tgtMask)
            
        # 最终线性输出层转换为词汇表概率
        output = self.finalLayer(tgt)
        return output  # 返回输出预测 
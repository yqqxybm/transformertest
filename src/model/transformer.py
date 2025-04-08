import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxLen=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, dModel, numHeads, dFf, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # 计算dK和dV
        dK = dModel // numHeads
        dV = dModel // numHeads
        
        self.attention = MultiHeadAttention(dK, dV, dModel, numHeads, dropout)
        # 归一化，为了避免padding token的影响选择layerNorm
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        # 位置前馈网络
        self.feedForward = nn.Sequential(
            nn.Linear(dModel, dFf),
            nn.ReLU(),
            nn.Linear(dFf, dModel)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头注意力
        attentionOutput = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attentionOutput))
        
        # 前馈网络
        ffOutput = self.feedForward(x)
        #跳连 add&norm
        x = self.norm2(x + self.dropout(ffOutput))
        
        return x

class Transformer(nn.Module):
    def __init__(self, srcVocabSize, tgtVocabSize, dModel=512, numHeads=8, 
                 numLayers=6, dFf=2048, dropout=0.1):
        super(Transformer, self).__init__()
        # 创建源语言和目标语言的嵌入层，实现对输入序列的词嵌入
        self.srcEmbedding = nn.Embedding(srcVocabSize, dModel)
        self.tgtEmbedding = nn.Embedding(tgtVocabSize, dModel)
        self.positionalEncoding = PositionalEncoding(dModel)
        
        self.encoderLayers = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFf, dropout)
            for _ in range(numLayers)
        ])
        
        self.decoderLayers = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFf, dropout)
            for _ in range(numLayers)
        ])
        
        self.finalLayer = nn.Linear(dModel, tgtVocabSize)
        
    def forward(self, src, tgt, srcMask=None, tgtMask=None):
        # 编码器
        src = self.positionalEncoding(self.srcEmbedding(src))
        for layer in self.encoderLayers:
            src = layer(src, srcMask)
            
        # 解码器
        tgt = self.positionalEncoding(self.tgtEmbedding(tgt))
        for layer in self.decoderLayers:
            tgt = layer(tgt, tgtMask)
            
        # 输出层
        output = self.finalLayer(tgt)
        return output 
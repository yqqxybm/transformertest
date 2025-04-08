import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .ffn import PoswiseFFN
from .position import posSinusoidEmbedding

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, dim: int, n: int, dff: int, dropoutPosffn: float, dropoutAttn: float):
        super(EncoderLayer, self).__init__()
        assert dim % n == 0
        hdim = dim // n
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multiHeadAttn = MultiHeadAttention(hdim, hdim, dim, n, dropoutAttn)
        self.poswiseFFN = PoswiseFFN(dim, dff, p=dropoutPosffn)

    def forward(self, encIn: Tensor, attnMask: Optional[Tensor]) -> Tensor:
        residual = encIn
        context = self.multiHeadAttn(encIn, encIn, encIn, attnMask)
        out = self.norm1(residual + context)
        
        residual = out
        out = self.poswiseFFN(out)
        out = self.norm2(residual + out)
        return out

class Encoder(nn.Module):
    """编码器"""
    def __init__(self, dropoutEmb: float, dropoutPosffn: float, dropoutAttn: float,
                 numLayers: int, encDim: int, numHeads: int, dff: int, tgtLen: int):
        super(Encoder, self).__init__()
        self.tgtLen = tgtLen
        self.posEmb = nn.Embedding.from_pretrained(posSinusoidEmbedding(tgtLen, encDim), freeze=True)
        self.embDropout = nn.Dropout(dropoutEmb)
        self.layers = nn.ModuleList([
            EncoderLayer(encDim, numHeads, dff, dropoutPosffn, dropoutAttn) 
            for _ in range(numLayers)
        ])
    
    def forward(self, X: Tensor, X_lens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batchSize, seqLen, dModel = X.shape
        out = X + self.posEmb(torch.arange(seqLen, device=X.device))
        out = self.embDropout(out)
        
        for layer in self.layers:
            out = layer(out, mask)
        return out 
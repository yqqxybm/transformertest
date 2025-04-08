import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .ffn import PoswiseFFN
from .position import posSinusoidEmbedding

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, dim: int, n: int, dff: int, dropoutPosffn: float, dropoutAttn: float):
        super(DecoderLayer, self).__init__()
        assert dim % n == 0
        hdim = dim // n
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.maskedMultiHeadAttn = MultiHeadAttention(hdim, hdim, dim, n, dropoutAttn)
        self.multiHeadAttn = MultiHeadAttention(hdim, hdim, dim, n, dropoutAttn)
        self.poswiseFFN = PoswiseFFN(dim, dff, p=dropoutPosffn)

    def forward(self, decIn: Tensor, encOut: Tensor, 
                selfAttnMask: Optional[Tensor], encDecAttnMask: Optional[Tensor]) -> Tensor:
        residual = decIn
        context = self.maskedMultiHeadAttn(decIn, decIn, decIn, selfAttnMask)
        out = self.norm1(residual + context)
        
        residual = out
        context = self.multiHeadAttn(out, encOut, encOut, encDecAttnMask)
        out = self.norm2(residual + context)
        
        residual = out
        out = self.poswiseFFN(out)
        out = self.norm3(residual + out)
        return out

class Decoder(nn.Module):
    """解码器"""
    def __init__(self, dropoutEmb: float, dropoutPosffn: float, dropoutAttn: float,
                 numLayers: int, decDim: int, numHeads: int, dff: int, tgtLen: int):
        super(Decoder, self).__init__()
        self.tgtLen = tgtLen
        self.posEmb = nn.Embedding.from_pretrained(posSinusoidEmbedding(tgtLen, decDim), freeze=True)
        self.embDropout = nn.Dropout(dropoutEmb)
        self.layers = nn.ModuleList([
            DecoderLayer(decDim, numHeads, dff, dropoutPosffn, dropoutAttn) 
            for _ in range(numLayers)
        ])
    
    def forward(self, Y: Tensor, encOut: Tensor, 
                selfAttnMask: Optional[Tensor] = None, 
                encDecAttnMask: Optional[Tensor] = None) -> Tensor:
        batchSize, seqLen, dModel = Y.shape
        out = Y + self.posEmb(torch.arange(seqLen, device=Y.device))
        out = self.embDropout(out)
        
        for layer in self.layers:
            out = layer(out, encOut, selfAttnMask, encDecAttnMask)
        return out 
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Transformer模型配置类"""
    # 模型基础参数
    batchSize: int = 16
    maxFeatLen: int = 100
    fbankDim: int = 80
    hiddenDim: int = 512
    vocabSize: int = 26
    maxLabelLen: int = 100
    
    # Transformer参数
    numEncoderLayers: int = 6
    numDecoderLayers: int = 6
    numHeads: int = 8
    dFF: int = 2048
    
    # Dropout参数
    dropoutEmb: float = 0.1
    dropoutPosffn: float = 0.1
    dropoutAttn: float = 0.0 
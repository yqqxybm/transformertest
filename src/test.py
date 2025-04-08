import torch
from model.transformer import Transformer
from config.model_config import ModelConfig

def test_transformer():
    # 创建配置实例
    config = ModelConfig()
    
    # 创建模型实例
    model = Transformer(
        srcVocabSize=config.vocabSize,
        tgtVocabSize=config.vocabSize,
        dModel=config.hiddenDim,
        numHeads=config.numHeads,
        numLayers=config.numEncoderLayers,
        dFf=config.dFF,
        dropout=config.dropoutEmb
    )
    
    # 创建测试数据
    batchSize = 2
    src = torch.randint(0, config.vocabSize, (batchSize, config.maxFeatLen))
    tgt = torch.randint(0, config.vocabSize, (batchSize, config.maxLabelLen))
    
    # 前向传播
    output = model(src, tgt)
    
    print("模型测试成功！")
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状: {output.shape}")

if __name__ == "__main__":
    test_transformer() 
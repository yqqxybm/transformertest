import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import time

from model.transformer import Transformer
from utils.data_loader import createDataLoader

def train(model, trainLoader, criterion, optimizer, device, epoch, writer):
    model.train()
    totalLoss = 0
    
    with tqdm(trainLoader, desc=f'Epoch {epoch}') as pbar:
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # 创建掩码
            srcMask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgtMask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            
            # 前向传播
            output = model(src, tgt, srcMask, tgtMask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            totalLoss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    avgLoss = totalLoss / len(trainLoader)
    writer.add_scalar('Loss/train', avgLoss, epoch)
    return avgLoss

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数
    srcVocabSize = 10000  # 源语言词汇表大小
    tgtVocabSize = 10000  # 目标语言词汇表大小
    dModel = 512
    numHeads = 8
    numLayers = 6
    dFf = 2048
    dropout = 0.1
    
    # 创建模型
    model = Transformer(srcVocabSize, tgtVocabSize, dModel, numHeads, 
                       numLayers, dFf, dropout).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter('runs/transformer_experiment')
    
    # 训练参数
    numEpochs = 100
    batchSize = 32
    
    # 这里需要替换为实际的数据
    # srcSequences = ...
    # tgtSequences = ...
    # srcVocab = ...
    # tgtVocab = ...
    
    # trainLoader = createDataLoader(srcSequences, tgtSequences, srcVocab, tgtVocab, batchSize)
    
    # 训练循环
    for epoch in range(numEpochs):
        startTime = time.time()
        trainLoss = train(model, trainLoader, criterion, optimizer, device, epoch, writer)
        endTime = time.time()
        
        print(f'Epoch {epoch}: Loss = {trainLoss:.4f}, Time = {endTime - startTime:.2f}s')
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')
    
    writer.close()

if __name__ == '__main__':
    main() 
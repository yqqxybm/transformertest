import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, srcSequences, tgtSequences, srcVocab, tgtVocab):
        self.srcSequences = srcSequences
        self.tgtSequences = tgtSequences
        self.srcVocab = srcVocab
        self.tgtVocab = tgtVocab
        
    def __len__(self):
        return len(self.srcSequences)
    
    def __getitem__(self, idx):
        srcSequence = self.srcSequences[idx]
        tgtSequence = self.tgtSequences[idx]
        
        # 将序列转换为索引
        srcIndices = [self.srcVocab[token] for token in srcSequence]
        tgtIndices = [self.tgtVocab[token] for token in tgtSequence]
        
        return torch.tensor(srcIndices), torch.tensor(tgtIndices)

def createDataLoader(srcSequences, tgtSequences, srcVocab, tgtVocab, 
                    batchSize=32, shuffle=True):
    dataset = TranslationDataset(srcSequences, tgtSequences, srcVocab, tgtVocab)
    return DataLoader(dataset, batchSize=batchSize, shuffle=shuffle) 
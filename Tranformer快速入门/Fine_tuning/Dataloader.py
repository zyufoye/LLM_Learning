# 通过 DataLoader 库按批 (batch) 加载数据，并且将样本转换成模型可以接受的输入格式
# 对于 NLP 任务，这个环节就是将每个 batch 中的文本按照预训练模型的格式进行编码（包括 Padding、截断等操作）

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 首先加载分词器，然后对每个 batch 中的所有句子对进行编码，同时把标签转换为张量格式

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

from torch.utils.data import IterableDataset
import json

class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt',encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

train_data = IterableAFQMC('data/train.json')

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
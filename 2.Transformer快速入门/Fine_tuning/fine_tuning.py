# 同义语句判断任务，构建Transformer

# 择蚂蚁金融语义相似度数据集 AFQMC 作为语料

# 数据集处理

from torch.utils.data import Dataset
import json

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt',encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('data/train.json')
valid_data = AFQMC('data/dev.json')

# print(train_data[0])
"""
{'sentence1': '蚂蚁借呗等额还款可以换成先息后本吗', 'sentence2': '借呗有先息到期还本吗', 'label': '0'}
"""

# 当数据集非常大时，无法一次性全部加载进内存，可以使用IterableDataset类迭代构建数据集

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

print(next(iter(train_data)))
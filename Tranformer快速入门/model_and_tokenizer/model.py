
# 直接使用模型对应的 Model 类，例如 BERT 对应的就是 BertModel

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")

# 方式2： 使用 AutoModel 根据 checkpoint 自动加载模型

from transformers import AutoModel, AutoTokenizer

ckpt = "bert-base-uncased"          # 也可以是本地路径或私有仓库名
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt)  # 按 checkpoint 自动挑对的模型类
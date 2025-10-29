# 通过 Pytorch 来手工实现 Scaled Dot-product Attention

# 1.首先将文本分词为 token 序列
# 2.将每个词语转换为对应的词向量 embedding

from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer

# bert-base-uncased 是 Hugging Face 上的 BERT 基础英文模型
# uncased 表示该模型在训练时已将所有文本转换为小写，因此 不会区分大小写
model_ckpt = "bert-base-uncased"

# 自动下载并加载与 bert-base-uncased 匹配的分词器
# 分词器的作用是把人类可读的文本（字符串）转换成模型可以理解的数字 token ID
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"

# 对文本进行分词，将文本转换为token id
# return_tensors="pt"：返回 PyTorch 张量（tensor），而不是 Python 列表
# add_special_tokens=False：不添加 [CLS] 和 [SEP] 等特殊标记
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

# 每个数字是一个单词或子词的 token id ：tensor([[ 2051, 10029,  2066,  2019,  8612]])
print(inputs.input_ids)

# 加载模型配置，从checkpoint中读取配置
# 例如config.vocab_size：词表大小（=30522）
# config.hidden_size：每个 token 的向量维度（=768）
config = AutoConfig.from_pretrained(model_ckpt)

# 定义一个 Embedding 层
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())
"""
tensor([[ 2051, 10029,  2066,  2019,  8612]])
Embedding(30522, 768)
torch.Size([1, 5, 768])
"""
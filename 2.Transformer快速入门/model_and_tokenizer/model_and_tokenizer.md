## 模型加载

所有存储在 HuggingFace Model Hub 上的模型都可以通过 Model.from_pretrained() 来加载权重，参数可以像上面一样是 checkpoint 的名称，也可以是本地路径（预先下载的模型目录）

Model.from_pretrained() 会自动缓存下载的模型权重。

## 分词器

由于神经网络模型不能直接处理文本，因此我们需要先将文本转换为数字，这个过程被称为编码 (Encoding)，其包含两个步骤：

1. 使用分词器 (tokenizer) 将文本按词、子词、字符切分为 tokens；
2. 将所有的 token 映射到对应的 token ID。


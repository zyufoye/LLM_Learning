# Transformer快速入门

因为大语言模型大多基于Transformer，但是我并没有很好理解Transformer是什么，所以在此处进行一些相关学习和实践，加深相关理解。

## 一、自然语言处理

### 1.1 Word2Vec模型

把神经网络语言模型发扬光大的是2013年Google公司提出的Word2Vec模型。Word2Vec的训练方法分为CBOW (Continuous Bag-of-Words) 和 Skip-gram 两种。CBOW使用周围的词语预测当前词，Skip-gram使用当前词来预测周围的词语。  
Word2Vec 模型在结构上更加自由，训练目标也更多地是为获得词向量服务。特别是同时通过上文和下文来预测当前词语的 CBOW 训练方法打破了语言模型“只通过上文来预测当前词”的固定思维，为后续一系列神经网络语言模型的发展奠定了基础。  
Word2Vec存在一个弊端就是无法解决一词多义问题。

### 1.2 ELMo模型

ELMo 模型（Embeddings from Language Models）。与 Word2Vec 模型只能提供静态词向量不同，ELMo 模型会根据上下文动态地调整词语的词向量。

## 二、Transformer模型

自BERT和GPT模型取得重大成功之后， Transformer 结构已经替代了循环神经网络 (RNN) 和卷积神经网络 (CNN)，成为了当前 NLP 模型的标配。 
虽然新的 Transformer 模型层出不穷，它们采用不同的预训练目标在不同的数据集上进行训练，但是依然可以按模型结构将它们大致分为三类：  
- 纯Encoder模型（例如 BERT），又称自编码 (auto-encoding) Transformer 模型；
- 纯Decoder模型（例如 GPT），又称自回归 (auto-regressive) Transformer 模型；
- Encoder-Decoder模型（例如 BART、T5），又称 Seq2Seq (sequence-to-sequence) Transformer 模型。

### 2.1 语言模型

Transformer 模型本质上都是预训练语言模型，大都采用自监督学习 (Self-supervised learning) 的方式在大量生语料上进行训练，也就是说，训练这些 Transformer 模型完全不需要人工标注数据。  

> 自监督学习是一种训练目标可以根据模型的输入自动计算的训练方法。  

我们可以在预训练好的模型权重上构建模型，就可以大幅地降低计算成本和碳排放。  

> 现在也有一些工作致力于在尽可能保持模型性能的情况下大幅减少参数量，达到用“小模型”获得媲美“大模型”的效果（例如模型蒸馏）。

因此，大部分情况下，我们都不会从头训练模型，而是将别人预训练好的模型权重通过迁移学习应用到自己的模型中，即使用自己的任务语料对模型进行“二次训练”，通过微调参数使模型适用于新任务。  

### 2.2 Transformer结构

标准的Transformer 模型主要由两个模块构成：  
1. Encoder：负责理解输入文本，为每个输入构造对应的语义表示（语义特征）；
2. Decoder：负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列。

纯Encoder模型适用于只需要理解输入语义的任务，例如句子分类、命名实体识别。  
纯 Decoder 模型：适用于生成式任务，例如文本生成；   
Encoder-Decoder 模型或 Seq2Seq 模型：适用于需要基于输入的生成式任务，例如翻译、摘要。

### 2.3 注意力机制

NLP 神经网络模型的本质就是对输入文本进行编码，常规的做法是首先对句子进行分词，然后将每个词语 (token) 都转化为对应的词向量 (token embeddings)，这样文本就转换为一个由词语向量组成的矩阵。  
RNN和CNN都存在一些弊端，取而代之的是Transformer，其直接使用 Attention 机制编码整个文本。相比 RNN 要逐步递归才能获得全局信息（因此一般使用双向 RNN），而 CNN 实际只能获取局部信息，需要通过层叠来增大感受视野，Attention 机制一步到位获取了全局信息。  

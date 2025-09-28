# Transformer快速入门

因为大语言模型大多基于Transformer，但是我并没有很好理解Transformer是什么，所以在此处进行一些相关学习和实践，加深相关理解。

## 一、自然语言处理

### 1.1 Word2Vec模型

把神经网络语言模型发扬光大的是2013年Google公司提出的Word2Vec模型。Word2Vec的训练方法分为CBOW (Continuous Bag-of-Words) 和 Skip-gram 两种。CBOW使用周围的词语预测当前词，Skip-gram使用当前词来预测周围的词语。  
Word2Vec 模型在结构上更加自由，训练目标也更多地是为获得词向量服务。特别是同时通过上文和下文来预测当前词语的 CBOW 训练方法打破了语言模型“只通过上文来预测当前词”的固定思维，为后续一系列神经网络语言模型的发展奠定了基础。  
Word2Vec存在一个弊端就是无法解决一词多义问题。

### 1.2 ELMo模型


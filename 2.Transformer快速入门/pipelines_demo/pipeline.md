# Pipeline使用方法记录

简要记录 Transformers 库的组件以及使用方法。

Transformers 库将目前的 NLP 任务归纳为几下几类：  

文本分类：例如情感分析、句子对关系判断等；  
对文本中的词语进行分类：例如词性标注 (POS)、命名实体识别 (NER) 等；  
文本生成：例如填充预设的模板 (prompt)、预测文本中被遮掩掉 (masked) 的词语；  
从文本中抽取答案：例如根据给定的问题从一段文本中抽取出对应的答案；  
根据输入文本生成新的句子：例如文本翻译、自动摘要等。  


Transformers 库最基础的对象就是 pipeline() 函数，它封装了预训练模型和对应的前处理和后处理环节。我们只需输入文本，就能得到预期的答案。目前常用的 pipelines 有：  


feature-extraction （获得文本的向量化表示）  
fill-mask （填充被遮盖的词、片段）  
ner（命名实体识别）  
question-answering （自动问答）  
sentiment-analysis （情感分析）  
summarization （自动摘要）  
text-generation （文本生成）  
translation （机器翻译）  
zero-shot-classification （零训练样本分类）  
下面我们以常见的几个 NLP 任务为例，展示如何调用这些 pipeline 模型。

pipeline 模型会自动完成以下三个步骤：

1. 将文本预处理为模型可以理解的格式；
2. 将预处理好的文本送入模型；
3. 对模型的预测值进行后处理，输出人类可以理解的格式。
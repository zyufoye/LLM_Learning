from transformers import pipeline

# 情感分析
# 只需要输入文本，就可以得到其情感标签（积极/消极）以及对应的概率
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
results = classifier(
  ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(results)

"""
[{'label': 'POSITIVE', 'score': 0.9598049521446228}]
[{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
"""
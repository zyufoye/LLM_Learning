import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    # 计算注意力得分 (Attention Scores)
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    
    # 构造掩码
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))

    # 计算注意力权重
    weights = F.softmax(scores, dim=-1)

    # 加权输出
    return torch.bmm(weights, value)
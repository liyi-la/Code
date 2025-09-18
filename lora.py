import torch
import torch.nn as nn
import numpy as np

# ------------ 超参同你给的 --------------
d_model = 512
d_ff = 2048
n_heads = 8
d_v = d_k = d_model // n_heads
r = 4               # LoRA 秩
alpha = 4           # 缩放系数

class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super(scaled_dot_product_attention, self).__init__()                # Q: [batch_size, n_heads, len_q, d_k]
                                                      # K: [batch_size, n_heads, len_k, d_k]
    def forward(self,Q,K,V,attn_mask):                                      # V: [batch_size, n_heads, len_v(=len_k), d_v]
        scores = torch.matmul(Q,K.transpose(-1,-2)/np.sqrt(d_k))       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)#注意力掩码，标记为true的位置为极小值         # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V)                                       # context: [batch_size, n_heads, len_q, d_v]
        return context, attn
# ========== 1.  LoRA 旁路 ==========
class LoRALinear(nn.Module):
    """
    替换 nn.Linear 的 LoRA 版：冻结原权重，只训 A、B 两个低秩矩阵
    """
    def __init__(self, original_linear: nn.Linear, rank=r, alpha=alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        in_f, out_f = original_linear.in_features, original_linear.out_features

        # 原权重冻结
        self.weight = original_linear.weight          # shape: [out_f, in_f]
        self.weight.requires_grad = False
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # 低秩旁路
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)   # [r, in]
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))         # [out, r]

    def forward(self, x):
        # 原前向 + 旁路
        orig_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
        return orig_out + lora_out


# ========== 2.  把 MHA 里三个线性层换成 LoRA ==========
class MHA(nn.Module):
    def __init__(self):
        super(MHA, self).__init__()
        # 先建原始层
        self._q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self._k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self._v = nn.Linear(d_model, d_v * n_heads, bias=False)

        # 原地包一层 LoRA
        self.W_Q = LoRALinear(self._q)
        self.W_K = LoRALinear(self._k)
        self.W_V = LoRALinear(self._v)

        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = scaled_dot_product_attention()(Q, K, V, attn_mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)

        return nn.LayerNorm(d_model).cuda()(output + residual), attn
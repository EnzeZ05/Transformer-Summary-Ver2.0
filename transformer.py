import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q = self.wq(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask[:, :, :L, :L]

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)

        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention = Attention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, mask=None):
        h = x + self.attention(self.ln_1(x), mask)
        out = h + self.feed_forward(self.ln_2(h))
        return out


class ModernDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=1600,
            layers=48,
            heads=25,
            d_ff=None,
            max_len=1024,
            dropout=0.1,
            token_id=50256,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.drop = nn.Dropout(dropout)

        if d_ff is None:
            d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )

        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)

        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, L = input_ids.shape
        device = input_ids.device

        pos = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        causal_mask = (1.0 - self.bias[:, :, :L, :L]) * -1e9

        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask).view(B, 1, 1, L) * -1e9
            combined_mask = causal_mask + pad_mask
        else:
            combined_mask = causal_mask

        for layer in self.layers:
            x = layer(x, mask = combined_mask)

        x = self.ln_f(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

"""
Transformer do transformers.ipynb, com duas correcoes:
- mascara de PAD na atencao e no pooling (no notebook o PAD entrava na media);
- dropout no embedding, para nao decorar o dataset (o notebook treinava 100 epocas sem validacao).
So e usado no treino. A API roda o mesmo calculo em numpy (inferencia.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoAtencao(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        # Uma unica cabeca, para conseguir explicar quais palavras pesaram (XAI)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x, mascara):
        Q, K, V = self.queries(x), self.keys(x), self.values(x)
        score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.embed_size)
        # Nenhuma palavra presta atencao no PAD
        score = score.masked_fill(~mascara[:, None, :], float("-inf"))
        pesos = F.softmax(score, dim=-1)
        return torch.bmm(pesos, V), pesos


class ClassificadorOfensivo(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = AutoAtencao(embed_size)
        self.fc = nn.Linear(embed_size, 2)  # 0 = normal, 1 = ofensivo

    def forward(self, x):
        N, L = x.shape
        mascara = x != 0
        pos = torch.arange(L, device=x.device).expand(N, L)
        out = self.dropout(self.token_embedding(x) + self.position_embedding(pos))
        out, pesos = self.attention(out, mascara)
        # Media so das palavras reais
        m = mascara.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(pooled), pesos

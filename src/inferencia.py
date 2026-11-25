"""
Inferencia em numpy puro: o mesmo calculo do modelo.py, sem precisar do torch na imagem
(imagem menor e start mais rapido). O treino confere que os dois batem.
"""
import json
import os

import numpy as np

import config
import texto


class Classificador:
    def __init__(self, pasta=config.PASTA_ARTEFATOS):
        pesos = np.load(os.path.join(pasta, "modelo.npz"))
        with open(os.path.join(pasta, "metadados.json"), encoding="utf-8") as f:
            meta = json.load(f)

        self.E = pesos["token_embedding"]
        self.P = pesos["position_embedding"]
        self.Wq, self.Wk, self.Wv = pesos["queries"], pesos["keys"], pesos["values"]
        self.Wfc, self.bfc = pesos["fc_w"], pesos["fc_b"]
        self.vocab = meta["vocab"]
        self.idx2word = {i: w for w, i in self.vocab.items()}
        self.max_len = meta["max_seq_length"]
        self.versao = meta["versao"]
        self.limiar = float(os.getenv("LIMIAR", meta.get("limiar", config.LIMIAR_PADRAO)))

    def _janela(self, ids):
        """Probabilidade de ofensivo e atencao recebida por cada palavra de uma janela."""
        x = self.E[ids] + self.P[: len(ids)]
        Q, K, V = x @ self.Wq.T, x @ self.Wk.T, x @ self.Wv.T
        score = Q @ K.T / np.sqrt(x.shape[1])
        score -= score.max(axis=1, keepdims=True)
        A = np.exp(score)
        A /= A.sum(axis=1, keepdims=True)
        pooled = (A @ V).mean(axis=0)
        logits = pooled @ self.Wfc.T + self.bfc
        logits -= logits.max()
        prob = np.exp(logits) / np.exp(logits).sum()
        return float(prob[1]), A.mean(axis=0)

    def classificar(self, descricao, top_palavras=3):
        tokens = texto.tokenizar(descricao)
        if not tokens:
            return {"ofensivo": False, "probabilidade": 0.0, "palavras_relevantes": [],
                    "limiar": self.limiar, "modelo_versao": self.versao}

        # Texto longo: avalia cada janela e fica com a pior
        melhor = (-1.0, None, None)
        for jan in texto.janelas(tokens, self.max_len):
            ids = [self.vocab.get(t, texto.UNK) for t in jan]
            prob, atencao = self._janela(ids)
            if prob > melhor[0]:
                melhor = (prob, jan, atencao)

        prob, jan, atencao = melhor
        ordem = np.argsort(-atencao)
        palavras = []
        for i in ordem:
            if jan[i] not in palavras:
                palavras.append(jan[i])
            if len(palavras) == top_palavras:
                break

        return {
            "ofensivo": prob >= self.limiar,
            "probabilidade": round(prob, 4),
            "palavras_relevantes": palavras,
            "limiar": self.limiar,
            "modelo_versao": self.versao,
        }

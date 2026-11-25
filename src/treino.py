"""
Treina o classificador de ofensivo com o HateBR e salva os artefatos usados pela API.
Uso: python treino.py
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
import texto
from inferencia import Classificador
from modelo import ClassificadorOfensivo

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preparar(df):
    df["tokens"] = df["texto"].apply(texto.tokenizar)
    return df[df["tokens"].str.len() > 0].reset_index(drop=True)


def carregar_dados():
    df = pd.read_csv(config.CAMINHO_DATASET)
    # label_final ja e o voto da maioria dos 3 anotadores: 1 = ofensivo, 0 = nao ofensivo
    df = df[["comentario", "label_final"]].dropna()
    df.columns = ["texto", "label"]
    return _preparar(df)


def carregar_csv(caminho):
    # Separador ";" porque as frases tem virgula
    return _preparar(pd.read_csv(caminho, sep=";")[["texto", "label"]].dropna())


def dividir(df):
    """70% treino, 15% validacao (early stopping e limiar), 15% teste (numero final)."""
    treino, resto = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=config.SEED)
    val, teste = train_test_split(resto, test_size=0.50, stratify=resto["label"], random_state=config.SEED)
    return treino, val, teste


def tensores(df, vocab):
    X = torch.tensor([texto.codificar(t, vocab, config.MAX_SEQ_LENGTH) for t in df["tokens"]], dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return X, y


@torch.no_grad()
def probabilidades(model, X):
    model.eval()
    logits, _ = model(X.to(device))
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def treinar(X_tr, y_tr, X_val, y_val, vocab_size):
    model = ClassificadorOfensivo(vocab_size, config.EMBED_SIZE, config.MAX_SEQ_LENGTH, config.DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)

    melhor_f1, melhor_estado, sem_melhora = -1.0, None, 0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        f1_val = f1_score(y_val.numpy(), probabilidades(model, X_val) >= 0.5)
        print(f"Epoca {epoch + 1:2d} | loss {total_loss / len(loader):.4f} | F1 validacao {f1_val:.4f}")

        if f1_val > melhor_f1:
            melhor_f1, sem_melhora = f1_val, 0
            melhor_estado = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= config.PACIENCIA:
                print(f"Early stopping: {config.PACIENCIA} epocas sem melhorar")
                break

    model.load_state_dict(melhor_estado)
    return model


def metricas(y, prob, limiar):
    pred = prob >= limiar
    return {"limiar": limiar,
            "acuracia": round(accuracy_score(y, pred), 4),
            "precisao": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4)}


def salvar(model, vocab, versao, avaliacao):
    os.makedirs(config.PASTA_ARTEFATOS, exist_ok=True)
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(
        os.path.join(config.PASTA_ARTEFATOS, "modelo.npz"),
        token_embedding=sd["token_embedding.weight"],
        position_embedding=sd["position_embedding.weight"],
        queries=sd["attention.queries.weight"],
        keys=sd["attention.keys.weight"],
        values=sd["attention.values.weight"],
        fc_w=sd["fc.weight"],
        fc_b=sd["fc.bias"],
    )
    meta = {
        "versao": versao,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "embed_size": config.EMBED_SIZE,
        "limiar": config.LIMIAR_PADRAO,
        "dados_dominio": config.USAR_DADOS_DOMINIO,
        "avaliacao_teste": avaliacao,
        "vocab": vocab,
    }
    with open(os.path.join(config.PASTA_ARTEFATOS, "metadados.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def conferir_numpy(model, teste):
    """A API usa numpy; garante que da o mesmo resultado que o torch."""
    clf = Classificador()
    clf.limiar = 0.5
    amostra = teste[teste["tokens"].str.len() <= config.MAX_SEQ_LENGTH].head(300)
    X, _ = tensores(amostra, clf.vocab)
    p_torch = probabilidades(model, X)
    p_numpy = np.array([clf.classificar(t)["probabilidade"] for t in amostra["texto"]])
    diff = np.abs(p_torch - p_numpy).max()
    print(f"Diferenca maxima torch x numpy: {diff:.6f}")
    assert diff < 1e-3, "Inferencia em numpy nao bate com o torch"


def main():
    versao = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    print(f"Treinando no dispositivo: {device} | versao {versao}")

    df = carregar_dados()
    treino, val, teste = dividir(df)
    if config.USAR_DADOS_DOMINIO:
        dominio = carregar_csv(config.CAMINHO_DOMINIO)
        treino = pd.concat([treino] + [dominio] * config.REPETICOES_DOMINIO, ignore_index=True)
        print(f"Dados de dominio: {len(dominio)} frases x{config.REPETICOES_DOMINIO} no treino")
    print(f"Treino {len(treino)} | validacao {len(val)} | teste {len(teste)}")

    # Vocabulario so com o treino, para o teste ser honesto
    vocab = texto.construir_vocab(treino["tokens"], config.VOCAB_SIZE, config.MIN_FREQ)
    print(f"Vocabulario: {len(vocab)} palavras")

    X_tr, y_tr = tensores(treino, vocab)
    X_val, y_val = tensores(val, vocab)
    X_te, y_te = tensores(teste, vocab)

    model = treinar(X_tr, y_tr, X_val, y_val, len(vocab))

    p_te = probabilidades(model, X_te)
    print("\nRelatorio no TESTE (limiar 0.5):")
    print(classification_report(y_te.numpy(), p_te >= 0.5, target_names=["normal", "ofensivo"], digits=4))

    print("Trade-off por limiar (teste) - limiar maior = menos falso positivo, mais ofensivo passa:")
    avaliacao = [metricas(y_te.numpy(), p_te, l) for l in (0.5, 0.6, 0.7, 0.8, 0.9)]
    print(pd.DataFrame(avaliacao).to_string(index=False))

    aval_dom = carregar_csv(config.CAMINHO_AVALIACAO_DOMINIO)
    X_dom, y_dom = tensores(aval_dom, vocab)
    p_dom = probabilidades(model, X_dom)
    print("\nFrases de review NUNCA vistas no treino (dados/avaliacao_dominio.csv):")
    print(pd.DataFrame([metricas(y_dom.numpy(), p_dom, l) for l in (0.5, 0.7, 0.9)]).to_string(index=False))
    erros = aval_dom[(p_dom >= 0.5) != aval_dom["label"].astype(bool)]
    for t, l, p in zip(erros["texto"], erros["label"], p_dom[erros.index]):
        print(f"  erro: label={l} prob={p:.2f} | {t}")

    salvar(model, vocab, versao, avaliacao)
    conferir_numpy(model, teste)
    print(f"\nArtefatos salvos em {os.path.abspath(config.PASTA_ARTEFATOS)}")


if __name__ == "__main__":
    main()

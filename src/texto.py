"""Tokenizacao e vocabulario. Usado igual no treino e na inferencia."""
import re
import unicodedata
from collections import Counter

PAD, UNK = 0, 1


def normalizar(texto):
    """Minusculo, sem acento, sem letras repetidas demais ("lixooooo" -> "lixoo")."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"(.)\1{2,}", r"\1\1", texto)


def tokenizar(texto):
    # Diferente do notebook (split por espaco), separa a pontuacao: "lixo!" vira "lixo"
    return re.findall(r"[a-z0-9]+", normalizar(texto))


def construir_vocab(lista_tokens, tamanho, min_freq):
    contagem = Counter(t for tokens in lista_tokens for t in tokens)
    comuns = [w for w, c in contagem.most_common(tamanho - 2) if c >= min_freq]
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    vocab.update({w: i + 2 for i, w in enumerate(comuns)})
    return vocab


def codificar(tokens, vocab, max_len):
    """Ids com PAD ate max_len (para o treino em lote)."""
    ids = [vocab.get(t, UNK) for t in tokens[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


def janelas(tokens, max_len):
    """
    Divide textos longos em janelas sobrepostas. O notebook cortava o texto,
    entao um palavrao no fim de uma review longa passaria batido.
    """
    if len(tokens) <= max_len:
        return [tokens]
    passo = max_len // 2
    return [tokens[i:i + max_len] for i in range(0, len(tokens) - passo, passo)]

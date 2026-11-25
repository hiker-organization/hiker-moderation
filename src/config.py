import os
from dotenv import load_dotenv

load_dotenv()

PASTA_ARTEFATOS = os.getenv("PASTA_ARTEFATOS", os.path.join(os.path.dirname(__file__), "..", "artefatos"))
_PASTA_DADOS = os.path.join(os.path.dirname(__file__), "..", "dados")
CAMINHO_DATASET = os.getenv("CAMINHO_DATASET", os.path.join(_PASTA_DADOS, "HateBR.csv"))

# Frases no estilo de review (criticas legitimas = 0, ofensas = 1) somadas ao treino.
# Sem elas, o modelo treinado so no HateBR marca qualquer review negativa como ofensiva.
USAR_DADOS_DOMINIO = os.getenv("USAR_DADOS_DOMINIO", "1") == "1"
CAMINHO_DOMINIO = os.path.join(_PASTA_DADOS, "reviews_dominio.csv")
REPETICOES_DOMINIO = 5  # o arquivo e pequeno perto do HateBR, entao e repetido no treino
# Frases diferentes das de treino, so para avaliar o comportamento em reviews
CAMINHO_AVALIACAO_DOMINIO = os.path.join(_PASTA_DADOS, "avaliacao_dominio.csv")

# Parametros do modelo (mesma arquitetura do transformers.ipynb)
MAX_SEQ_LENGTH = 64   # 99% dos comentarios do HateBR tem ate ~73 palavras
VOCAB_SIZE = 6000
EMBED_SIZE = 64
MIN_FREQ = 2          # palavra precisa aparecer 2x no treino para entrar no vocabulario

# Treino
EPOCHS = 40
PACIENCIA = 5         # early stopping: epocas sem melhorar o F1 de validacao
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.2
SEED = 42

# Probabilidade minima para considerar ofensivo. Pode ser sobrescrita na API pela env LIMIAR.
# Alto de proposito: falso positivo esconde uma review legitima sem ninguem ver, enquanto
# a ofensa que passar ainda pode ser denunciada pelos usuarios. Veja o trade-off impresso no treino.
LIMIAR_PADRAO = 0.9

# API
API_KEY = os.getenv("API_KEY", "")

# Backfill
DATABASE_URL = os.getenv("DATABASE_URL", "")

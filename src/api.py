"""
API de moderacao. A hiker-api chama POST /classificar antes de salvar a review.
Rodar local: uvicorn api:app --reload --port 8080
"""
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import config
from inferencia import Classificador

app = FastAPI(title="Hiker Moderacao")
classificador = Classificador()  # carregado uma vez, na subida do processo


class Entrada(BaseModel):
    texto: str = Field(max_length=10000)


class Resultado(BaseModel):
    ofensivo: bool
    probabilidade: float
    limiar: float
    modelo_versao: str
    palavras_relevantes: list[str]


def verificar_chave(x_api_key: str = Header(default="")):
    # Sem API_KEY configurada (desenvolvimento local) a rota fica aberta
    if config.API_KEY and not secrets.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="API key invalida")


@app.get("/health")
def health():
    return {"status": "ok", "modelo_versao": classificador.versao}


@app.post("/classificar", response_model=Resultado, dependencies=[Depends(verificar_chave)])
def classificar(entrada: Entrada):
    return classificador.classificar(entrada.texto)

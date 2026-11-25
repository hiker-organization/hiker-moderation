# hiker-moderation

Classifica a descrição de uma review como ofensiva ou não. Roda como API no App Engine standard (GCP); a hiker-api chama a API antes de salvar cada review.

## Como funciona

```
App -> POST /review (hiker-api)
         |-- POST /classificar (este serviço, App Engine) -> { ofensivo, probabilidade, ... }
         |-- salva Review (status OFENSIVA se ofensivo) + linha em Classificacao_review
```

- Review ofensiva é salva com `status = 'OFENSIVA'`. Todas as consultas públicas da hiker-api já filtram `status IN ('AUTORIZADA', 'ANALISE')`, então ela não aparece pra ninguém.
- `Classificacao_review` guarda a probabilidade e a versão do modelo de toda review classificada, para auditoria.
- Se este serviço estiver fora ou demorar mais que `MODERACAO_TIMEOUT_MS`, a review é publicada normalmente, sem linha em `Classificacao_review`. O `backfill.py` pega essas depois.

### Revisão pelo admin (hiker-api)

| Rota | O que faz |
|---|---|
| `GET /admin/reviews/auto-blocked?pendentes=true` | Reviews bloqueadas pelo modelo que nenhum admin revisou (`pendentes=false` lista todas), com probabilidade e versão do modelo |
| `PATCH /admin/review/:id/auto-block` `{ "status": "AUTORIZADA" }` | Reverte o bloqueio: a review volta a aparecer |
| `PATCH /admin/review/:id/auto-block` `{ "status": "OFENSIVA" }` | Confirma o bloqueio |

A decisão fica em `Classificacao_review.revisado_em` / `decisao_admin`.

**Automático x manual:** bloqueio automático é `status = OFENSIVA` com `Classificacao_review.ofensivo = true` e sem reversão do admin (confirmar mantém como automático). Todo o resto com `OFENSIVA` é manual (admin moderando denúncia). A regra está em `src/modules/admin/utils/constants/bloqueio.constants.ts` da hiker-api.

## Modelo

É a arquitetura do `transformers.ipynb` (embedding + posição + uma cabeça de self-attention + média + camada linear), treinada no `HateBR.csv`. Mudanças em relação ao notebook:

| Notebook | Aqui | Por quê |
|---|---|---|
| Acurácia só no treino, 100 épocas | Treino/validação/teste (70/15/15), early stopping pelo F1 de validação | Medir de verdade e não decorar o dataset |
| `split()` por espaço | Minúsculo, sem acento, pontuação separada | "Lixo!" e "lixo" viravam palavras diferentes |
| PAD entra na atenção e na média | Máscara de PAD | O tamanho do texto mudava o resultado |
| Corta em 20 palavras | 64 palavras + janelas sobrepostas para textos maiores | Ofensa no fim de uma review longa passava |
| Inferência em torch | Inferência em numpy (o treino confere que bate com o torch) | Imagem de ~200 MB a menos e start rápido |

### Dados de domínio

Treinado só no HateBR (comentários políticos do Instagram), o modelo marcava **crítica** como ofensa: "a praia estava um lixo", "comida veio fria" e "lugar horrível" saíam todas como ofensivas. Por isso o treino soma `dados/reviews_dominio.csv`: ~110 frases no estilo de review (críticas legítimas = 0, ofensas = 1). `dados/avaliacao_dominio.csv` tem outras 30 frases, que nunca entram no treino, só para medir.

| | F1 no teste do HateBR | Acerto nas 30 reviews de avaliação (limiar 0.5) |
|---|---|---|
| Só HateBR | 0.83 | 70% (9 críticas legítimas bloqueadas) |
| HateBR + domínio | 0.82 | 93% |

As frases de domínio foram escritas à mão, então são poucas e a avaliação com 30 frases é indicativa, não conclusiva. O ideal é trocar/ampliar esses CSVs com reviews reais do app (por exemplo as de `Classificacao_review` que um admin revisou).

### Limiar

O padrão é **0.9**: só bloqueia quando o modelo tem muita certeza. No teste do HateBR isso dá precisão de 0.90 e recall de 0.63, ou seja, poucas reviews legítimas bloqueadas, mas parte das ofensivas passa (e continua podendo ser denunciada pelos usuários). O treino imprime a tabela completa por limiar. Para mudar sem retreinar, defina a env `LIMIAR` no serviço.

## Rodando local

```bash
pip install -r requirements-treino.txt
cd src
python treino.py                      # gera artefatos/modelo.npz e artefatos/metadados.json
uvicorn api:app --reload --port 8080  # http://localhost:8080/docs
```

```bash
curl -X POST localhost:8080/classificar -H "Content-Type: application/json" -d '{"texto": "Trilha linda, recomendo"}'
```

O `HateBR.csv` fica em `dados/`; o caminho pode ser trocado com a env `CAMINHO_DATASET`.

## Deploy no App Engine standard

Escolhido por custo: desliga quando ninguém usa (`min_instances: 0`) e a instância F1 entra na cota gratuita do App Engine standard. **Não use o App Engine flexible**: ele mantém uma instância ligada o tempo todo.

O modelo sobe junto com o código, então **treine antes** e faça commit da pasta `artefatos/`.

Primeira vez (uma vez por projeto):

```bash
gcloud config set project SEU_PROJETO
gcloud services enable appengine.googleapis.com cloudbuild.googleapis.com
gcloud app create --region=REGIAO   # a regiao NAO pode ser trocada depois
```

Deploy (nesta pasta):

```bash
cp env.example.yaml env.yaml        # e coloque a API_KEY (openssl rand -hex 32)
gcloud app deploy
gcloud app browse                   # mostra a URL: https://SEU_PROJETO.<cod>.r.appspot.com
```

Teste: `curl https://<url>/health`.

Notas:
- `env.yaml` está no `.gitignore`: a chave não vai pro repositório. Guarde o valor, a hiker-api usa o mesmo.
- O que sobe é controlado pelo `.gcloudignore`: só a API (`src/api.py`, `config.py`, `inferencia.py`, `texto.py`), `artefatos/`, `requirements.txt` e os yaml. Treino, backfill e `dados/` ficam só no GitHub. Confira com `gcloud meta list-files-for-upload`.
- A URL é pública porque a hiker-api roda no Azure e não tem identidade do Google; a proteção é o header `X-API-Key`. O serviço não acessa o banco, então o pior caso de vazar a chave é alguém usar o classificador.
- A primeira chamada depois de um tempo parado demora alguns segundos (a instância sobe). O timeout de 10 s da hiker-api cobre; se estourar, a review é publicada e o `backfill.py` classifica depois.
- `max_instances: 2` no `app.yaml` é um teto de gasto.

### Na hiker-api (Azure App Service)

1. Aplicar a migration: `npx prisma migrate deploy` (cria `Classificacao_review`).
2. Configurar as variáveis:
   - `MODERACAO_URL` = URL do App Engine (sem `/` no final)
   - `MODERACAO_API_KEY` = a mesma `API_KEY` do `env.yaml`
   - `MODERACAO_TIMEOUT_MS` = `10000`

Sem `MODERACAO_URL` a hiker-api simplesmente não classifica (útil em desenvolvimento).

## Backfill

Classifica reviews que não têm linha em `Classificacao_review` (antigas, ou em que a chamada falhou). Usa o modelo local e escreve direto no banco.

```bash
cd src
DATABASE_URL=postgresql://... python backfill.py            # só mostra
DATABASE_URL=postgresql://... python backfill.py --aplicar  # grava
```

Ele não bloqueia review que um admin já manteve (denúncia `FECHADA` e status `AUTORIZADA`). Dá pra rodar manualmente de vez em quando ou agendar, igual ao hiker-recommendation.

## Retreinar

1. Editar/ampliar `dados/reviews_dominio.csv` (separador `;`).
2. `python src/treino.py` e conferir as métricas impressas.
3. Commit de `artefatos/` e `gcloud app deploy`. A nova versão aparece em `modelo_versao` nas próximas classificações.

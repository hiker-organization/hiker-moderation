"""
Classifica as reviews que ainda nao tem linha em Classificacao_review:
- reviews antigas, criadas antes da moderacao existir;
- reviews em que a chamada da hiker-api para a API falhou (timeout, servico fora).
Roda direto no banco, com o modelo local (nao chama a API).

Uso: python backfill.py            -> so mostra o que faria
     python backfill.py --aplicar  -> grava no banco
"""
import sys

import psycopg

import config
from inferencia import Classificador

PENDENTES = """
    SELECT r.id, r.descricao
    FROM "Review" r
    LEFT JOIN "Classificacao_review" c ON c.id_review = r.id
    WHERE c.id IS NULL AND r."deletedAt" IS NULL
    ORDER BY r.id
"""

INSERIR = """
    INSERT INTO "Classificacao_review" (id_review, ofensivo, probabilidade, modelo_versao)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (id_review) DO NOTHING
"""

# So bloqueia review que nenhum admin ja analisou: se existe denuncia FECHADA e a review
# continua AUTORIZADA, um admin decidiu manter, e o modelo nao passa por cima disso.
BLOQUEAR = """
    UPDATE "Review" r SET status = 'OFENSIVA'
    WHERE r.id = %s AND r.status = 'AUTORIZADA'
      AND NOT EXISTS (SELECT 1 FROM "Denuncia" d WHERE d.id_review = r.id AND d.status = 'FECHADA')
"""


def main(aplicar):
    clf = Classificador()
    print(f"Modelo {clf.versao} | limiar {clf.limiar}")

    with psycopg.connect(config.DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(PENDENTES)
        pendentes = cur.fetchall()
        print(f"Reviews sem classificacao: {len(pendentes)}")

        ofensivas = 0
        for id_review, descricao in pendentes:
            r = clf.classificar(descricao or "")
            if r["ofensivo"]:
                ofensivas += 1
                print(f"  ofensiva #{id_review} ({r['probabilidade']:.2f}) {r['palavras_relevantes']}")
            if aplicar:
                cur.execute(INSERIR, (id_review, r["ofensivo"], r["probabilidade"], r["modelo_versao"]))
                if r["ofensivo"]:
                    cur.execute(BLOQUEAR, (id_review,))

        print(f"Ofensivas: {ofensivas} de {len(pendentes)}")
        if not aplicar:
            print("Nada foi gravado. Rode com --aplicar para gravar.")


if __name__ == "__main__":
    main("--aplicar" in sys.argv)

"""Confere o carregador do TAWOS contra a Tabela 2 de Tawosi, Moussa & Sarro (TSE 2022).

Para cada projeto do conjunto de replicacao, compara n, minimo, maximo, media,
mediana e desvio-padrao do story point com os valores publicados. E o mesmo tipo
de prova que o 2,59 contra 2,61 deu para o conjunto Deep-SE: se o carregador
errasse a leitura do banco, as estatisticas nao bateriam.

Tolerancias: media e desvio no arredondamento de duas casas da tabela (0,005);
o resto exato. O desvio e o amostral (ddof=1), como no pandas.

O que a conferencia prova e o que nao prova: as CHAVES das issues vem dos CSVs dos
autores, entao a contagem bate por construcao (exceto a issue do TIMOB que o banco
1.1 nao tem). O que se testa de fato e que os valores de story point lidos do
banco reproduzem minimo, maximo, media, mediana e desvio publicados.

Uso:  python src/conferir_tawos.py
"""
import sys
import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PUBLICADO = RAIZ / "referencia" / "tawosi_tabela2.csv"
DADOS = RAIZ / "tawos" / "dados"


def main():
    pub = pd.read_csv(PUBLICADO).set_index("projeto")
    linhas, falhas = [], 0
    for projeto, ref in pub.iterrows():
        sp = pd.read_csv(DADOS / f"{projeto}.csv").storypoint
        obtido = {"issues": len(sp), "sp_min": sp.min(), "sp_max": sp.max(),
                  "sp_medio": sp.mean(), "sp_mediana": sp.median(), "sp_desvio": sp.std()}
        tol = {"sp_medio": 0.005, "sp_desvio": 0.005}
        # o TIMOB perde 1 issue que o banco 1.1 nao tem: tolera-se so a contagem
        tol_n = 1 if projeto == "TIMOB" else 0
        dif = {k: abs(obtido[k] - ref[k]) for k in ref.index}
        ok = all(dif[k] <= tol.get(k, 0) + (tol_n if k == "issues" else 0) + 1e-9
                 for k in ref.index)
        falhas += not ok
        linhas.append({"projeto": projeto, "ok": "sim" if ok else "NAO",
                       **{f"{k}_obt": round(float(v), 2) for k, v in obtido.items()},
                       **{f"{k}_pub": ref[k] for k in ("sp_medio", "sp_mediana")}})
    r = pd.DataFrame(linhas).set_index("projeto")
    print(r[["ok", "issues_obt", "sp_medio_obt", "sp_medio_pub",
             "sp_mediana_obt", "sp_mediana_pub"]].to_string())
    print(f"\n{len(r) - falhas}/{len(r)} projetos batem com a Tabela 2")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()

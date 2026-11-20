"""Apontar as historias grandes: o valor calculado pelo modelo poe as grandes acima?

Criterio fixado no PLANO.md antes de rodar (2026-11-10). Nasceu de usar o
software como PO: no RoomBook, uma historia de 20 pontos recebeu 5 do modelo e 5
da mediana. O cartao do modelo quase nunca passa do da mediana, mas o valor que
ele calcula antes de virar cartao muda de historia para historia. A pergunta e se
esse valor coloca as historias grandes acima das outras -- coisa que a mediana,
que da o mesmo valor a todas, nao consegue fazer.

    historia grande   valor real pelo menos dois cartoes acima do cartao da
                      mediana da equipe (mediana dos 80% mais antigos; empate de
                      cartao fica com o menor). Alternativa: 13 ou mais.
    chance de ordenar sorteando uma grande e uma nao grande do mesmo projeto, a
    certo (AUC)       chance de a grande ter o valor calculado maior (empate
                      conta meio). 0,5 = sortear; a mediana tira 0,5.
    teste             Wilcoxon de (chance - 0,5) entre os projetos com pelo
                      menos 5 grandes e 5 nao grandes no teste.

Usa so previsoes ja gravadas (previsoes.py, base_sintetica.py); nada e treinado.

Uso:  python src/apontar_grandes.py
      (grava resultados/apontar_grandes.csv e apontar_grandes_resumo.csv)
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from caracterizar import carregar
from conjunto import RAIZ
from sonda_comum import CARTOES, dividir_historico_teste

RES = RAIZ / "resultados"
MIN_GRUPO = 5
SEMENTE = 42
REAMOSTRAGENS = 10_000

CONJUNTOS = [
    ("Deep-SE", RAIZ / "dados", RES / "previsoes_teste.csv"),
    ("TAWOS", RAIZ / "tawos" / "dados", RES / "tawos" / "previsoes_teste.csv"),
    ("RoomBook", RAIZ / "base_sintetica" / "dados", RES / "sintetica" / "base_sintetica_previsoes.csv"),
]
# o modelo do software e o principal; os outros sao reportados sem decidir nada
PRINCIPAL = ("base (TF-IDF + SVR)", "TF-IDF + SVR")
OUTROS = ("TF-IDF original",)


def indice_cartao(v):
    """Posicao do cartao de Planning Poker mais proximo (empate: o menor)."""
    v = np.asarray(v, dtype=float)
    return np.abs(v[:, None] - CARTOES).argmin(axis=1)


def chance_ordenar(grandes, outras):
    """Chance de uma grande sorteada ter valor maior que uma nao grande sorteada."""
    g, o = np.asarray(grandes)[:, None], np.asarray(outras)[None, :]
    return float(((g > o).sum() + 0.5 * (g == o).sum()) / (g.size * o.size))


def so_no_tawos():
    """Projetos do TAWOS sem nenhuma issue em comum com o Deep-SE."""
    deepse = set().union(*(set(pd.read_csv(a, usecols=["issuekey"]).issuekey)
                           for a in (RAIZ / "dados").glob("*.csv")))
    return {a.stem for a in (RAIZ / "tawos" / "dados").glob("*.csv")
            if not set(pd.read_csv(a, usecols=["issuekey"]).issuekey) & deepse}


def por_projeto(conjunto, dados, prev, condicao):
    linhas = []
    for arq in sorted(dados.glob("*.csv")):
        hist, teste = dividir_historico_teste(carregar(arq))
        p = prev[(prev.projeto == arq.stem) & (prev.condicao == condicao)]
        if p.empty:
            continue
        p = teste[["issuekey"]].merge(p, on="issuekey", validate="one_to_one")
        assert len(p) == len(teste), f"{arq.stem}: previsoes nao cobrem o teste"
        y, valor = p.storypoint.values.astype(float), p.previsao.values.astype(float)
        mediana = float(hist.storypoint.median())
        c_med = indice_cartao([mediana])[0]
        c_real, c_mod = indice_cartao(y), indice_cartao(valor)

        linha = {"conjunto": conjunto, "projeto": arq.stem, "condicao": condicao,
                 "n_teste": len(y), "mediana": mediana, "cartao_mediana": CARTOES[c_med]}
        for nome, grande in [("dois_cartoes", c_real >= c_med + 2),
                             ("13_ou_mais", y >= 13),
                             ("um_cartao", c_real >= c_med + 1)]:
            ng = int(grande.sum())
            ok = ng >= MIN_GRUPO and (len(y) - ng) >= MIN_GRUPO
            acima = c_mod > c_med
            mesmo = c_mod == c_med
            linha.update({
                f"{nome}_grandes": ng,
                f"{nome}_chance": chance_ordenar(valor[grande], valor[~grande]) if ng and ng < len(y) else np.nan,
                f"{nome}_valido": ok,
                # o que a pagina faz hoje: o cartao do modelo passa do da mediana?
                f"{nome}_grandes_com_cartao_acima": int((grande & acima).sum()),
                f"{nome}_cartao_acima": int(acima.sum()),
                f"{nome}_cartao_acima_e_grande": int((acima & grande).sum()),
                # a pergunta do Lucas: dentro do mesmo cartao da mediana
                f"{nome}_valor_medio_grandes_mesmo_cartao": valor[mesmo & grande].mean() if (mesmo & grande).any() else np.nan,
                f"{nome}_valor_medio_tipicas_mesmo_cartao": valor[mesmo & (c_real == c_med)].mean() if (mesmo & (c_real == c_med)).any() else np.nan,
                f"{nome}_n_grandes_mesmo_cartao": int((mesmo & grande).sum()),
            })
        linhas.append(linha)
    return linhas


def resumir(t, rotulo, definicao):
    v = t[t[f"{definicao}_valido"]]
    linha = {"grupo": rotulo, "definicao": definicao, "projetos": len(v),
             "grandes": int(t[f"{definicao}_grandes"].sum()), "historias": int(t.n_teste.sum())}
    if len(v):
        c = v[f"{definicao}_chance"].values
        rng = np.random.default_rng(SEMENTE)
        medias = [rng.choice(c, len(c)).mean() for _ in range(REAMOSTRAGENS)]
        linha.update({"chance_media": c.mean(),
                      "ic95_inf": np.percentile(medias, 2.5), "ic95_sup": np.percentile(medias, 97.5),
                      "projetos_acima_de_0_5": int((c > 0.5).sum()),
                      "p_wilcoxon": wilcoxon(c - 0.5).pvalue if len(c) > 1 and (c != 0.5).any() else np.nan})
    g = t[f"{definicao}_grandes"].sum()
    a = t[f"{definicao}_cartao_acima"].sum()
    linha.update({
        "grandes_com_cartao_acima_pct": 100 * t[f"{definicao}_grandes_com_cartao_acima"].sum() / g if g else np.nan,
        "cartao_acima_que_eram_grandes_pct": 100 * t[f"{definicao}_cartao_acima_e_grande"].sum() / a if a else np.nan,
        "valor_medio_grandes_mesmo_cartao": t[f"{definicao}_valor_medio_grandes_mesmo_cartao"].mean(),
        "valor_medio_tipicas_mesmo_cartao": t[f"{definicao}_valor_medio_tipicas_mesmo_cartao"].mean(),
    })
    return linha


def main():
    novos = so_no_tawos()
    projetos, resumo = [], []
    for conjunto, dados, arq in CONJUNTOS:
        prev = pd.read_csv(arq)
        condicoes = [c for c in prev.condicao.unique()
                     if c in PRINCIPAL or c.startswith(OUTROS)]
        for cond in condicoes:
            t = pd.DataFrame(por_projeto(conjunto, dados, prev, cond))
            projetos.append(t)
            principal = "principal" if cond in PRINCIPAL else "sem decidir"
            grupos = [(conjunto, t)]
            if conjunto == "TAWOS":
                grupos.append(("TAWOS, so os 13 projetos novos", t[t.projeto.isin(novos)]))
            for rotulo, g in grupos:
                for definicao in ["dois_cartoes", "13_ou_mais", "um_cartao"]:
                    resumo.append({"condicao": cond, "papel": principal,
                                   **resumir(g, rotulo, definicao)})

    r = pd.DataFrame(resumo)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 40):
        print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pd.concat(projetos, ignore_index=True).to_csv(RES / "apontar_grandes.csv", index=False,
                                                  float_format="%.4f")
    r.to_csv(RES / "apontar_grandes_resumo.csv", index=False, float_format="%.4f")
    print(f"\n-> {RES / 'apontar_grandes.csv'}\n-> {RES / 'apontar_grandes_resumo.csv'}")


if __name__ == "__main__":
    main()

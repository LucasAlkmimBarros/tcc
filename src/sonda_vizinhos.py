"""Sonda 7: as historias mais parecidas do passado dizem o story point da nova?

Por que esta sonda existe. Um RAG busca as historias parecidas do passado e as
entrega ao LLM junto com a historia nova. Antes de gastar chamadas de LLM, da para
medir so a parte da busca: se a mediana dos pontos das k vizinhas mais parecidas
nao erra menos que a mediana do historico inteiro, a busca nao entrega ao LLM nada
que a mediana ja nao entregue.

Quatro previsoes por historia de teste (desenho fixado no PLANO.md antes de rodar):

    mediana    mediana do historico inteiro
    vizinhas   mediana dos pontos das k historias mais parecidas (cosseno, MiniLM)
    sorteadas  mediana dos pontos de k historias sorteadas (media de 20 sorteios)
    recentes   mediana dos pontos das k ultimas historias do historico

"recentes" separa "parecer-se" de "ser recente": em projeto que muda de escala no
tempo (Moodle), olhar so para o passado recente ja ajuda sem busca nenhuma.

Uso:  python src/sonda_vizinhos.py            (Deep-SE)
      CONJUNTO=tawos python src/sonda_vizinhos.py
      CONJUNTO=sintetica python src/sonda_vizinhos.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sonda_comum import (dividir_historico_teste, embeddings, imprimir,
                         projetos, salvar, SEMENTE)

KS = (5, 10, 20)
K_DECIDE = 10
SORTEIOS = 20


def erros_do_projeto(nome, d):
    """Erro absoluto de cada previsao em cada historia de teste, para cada k."""
    hist, teste = dividir_historico_teste(d)
    E = embeddings(nome, d["texto"]).astype(np.float64)
    corte = len(hist)
    y_hist = hist["storypoint"].to_numpy(dtype=float)
    y = teste["storypoint"].to_numpy(dtype=float)

    # mais parecidas primeiro: cosseno entre cada historia de teste e o historico
    ordem = np.argsort(-(E[corte:] @ E[:corte].T), axis=1)
    rng = np.random.default_rng(SEMENTE)

    linhas = {"projeto": nome, "issuekey": teste["issuekey"].to_numpy(),
              "real": y, "erro_mediana": np.abs(y - np.median(y_hist))}
    for k in KS:
        linhas[f"erro_vizinhas_{k}"] = np.abs(y - np.median(y_hist[ordem[:, :k]], axis=1))
        linhas[f"erro_recentes_{k}"] = np.abs(y - np.median(y_hist[-k:]))
        sorteios = [np.abs(y - np.median(rng.choice(y_hist, size=(len(y), k)), axis=1))
                    for _ in range(SORTEIOS)]
        linhas[f"erro_sorteadas_{k}"] = np.mean(sorteios, axis=0)
    return pd.DataFrame(linhas)


def p_wilcoxon(a, b):
    """Wilcoxon pareado, bilateral; 1,0 quando nao ha diferenca nenhuma."""
    a, b = np.asarray(a), np.asarray(b)
    return 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)


def main():
    por_historia = pd.concat([erros_do_projeto(n, d) for n, d in projetos()],
                             ignore_index=True)
    colunas = [c for c in por_historia.columns if c.startswith("erro_")]
    por_projeto = por_historia.groupby("projeto")[colunas].mean().reset_index()
    por_projeto.columns = [c.replace("erro_", "MAE_") for c in por_projeto.columns]

    k = K_DECIDE
    imprimir(por_projeto[["projeto", "MAE_mediana", f"MAE_vizinhas_{k}",
                          f"MAE_sorteadas_{k}", f"MAE_recentes_{k}"]],
             f"Erro medio por projeto, k = {k}")

    # um projeto so (RoomBook): o teste pareia as historias; senao, os projetos
    um_so = len(por_projeto) == 1
    base = por_historia if um_so else por_projeto
    pref = "erro_" if um_so else "MAE_"
    resumo = []
    for k in KS:
        viz = base[f"{pref}vizinhas_{k}"]
        linha = {"k": k, "decide": k == K_DECIDE,
                 "MAE_mediana": por_projeto["MAE_mediana"].mean(),
                 "MAE_vizinhas": por_projeto[f"MAE_vizinhas_{k}"].mean(),
                 "MAE_sorteadas": por_projeto[f"MAE_sorteadas_{k}"].mean(),
                 "MAE_recentes": por_projeto[f"MAE_recentes_{k}"].mean()}
        for outra in ("mediana", "sorteadas", "recentes"):
            col = f"{pref}{outra}" if outra == "mediana" else f"{pref}{outra}_{k}"
            linha[f"p_vs_{outra}"] = p_wilcoxon(viz, base[col])
            if not um_so:
                linha[f"vence_{outra}"] = int((viz < base[col]).sum())
        linha["n_projetos"] = len(por_projeto)
        resumo.append(linha)
    resumo = pd.DataFrame(resumo)
    imprimir(resumo, "Resumo (media dos projetos; p do Wilcoxon pareado)", casas=3)

    d = resumo[resumo["decide"]].iloc[0]
    passa = (d["MAE_vizinhas"] < d["MAE_mediana"] and d["p_vs_mediana"] < 0.05
             and d["MAE_vizinhas"] < d["MAE_recentes"] and d["p_vs_recentes"] < 0.05)
    print(f"\nRegra (k = {K_DECIDE}): a busca {'PASSA' if passa else 'NAO passa'} nesta base.")

    salvar(por_projeto, "sonda_vizinhos.csv")
    salvar(resumo, "sonda_vizinhos_resumo.csv")
    salvar(por_historia, "sonda_vizinhos_por_historia.csv")


if __name__ == "__main__":
    main()

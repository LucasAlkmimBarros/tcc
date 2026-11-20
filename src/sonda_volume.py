"""Sonda 1: quanta historia uma empresa precisa ter para o software funcionar?

Por que esta sonda existe. A comparacao ENTRE projetos nao responde essa
pergunta: os 16 projetos diferem em volume e em dificuldade ao mesmo tempo, e a
variacao de dificuldade engole o efeito do volume (correlacao entre tamanho da
base e ganho sobre a mediana: r = -0,12, ou seja, nada). Aquele resultado nao
mostra que volume e irrelevante -- mostra que aquele desenho experimental nao
consegue medir volume.

O desenho correto fixa o projeto e varia so o volume. Para cada projeto: teste
fixo (os ultimos 20% cronologicos, os mesmos do experimento principal) e treino
com as N historias MAIS RECENTES antes do teste -- que e exatamente o que uma
empresa tem ao adotar a ferramenta. Projeto e teste constantes, 16 replicacoes.

A mediana historica e recalculada sobre o mesmo N, senao a comparacao daria ao
baseline uma vantagem de informacao que o modelo nao teve.
"""
import numpy as np
import pandas as pd

from sonda_comum import (dividir_historico_teste, imprimir, limitador,
                         metricas, modelo_tfidf, projetos, salvar)

TAMANHOS = [25, 50, 100, 200, 400, 800, 1600, None]  # None = todo o historico


def main():
    linhas = []
    print("Curva de aprendizado por projeto (teste fixo, treino crescente):\n")
    for nome, d in projetos():
        hist, teste = dividir_historico_teste(d)
        y_te = teste.storypoint

        for n in TAMANHOS:
            if n is not None and n > len(hist):
                continue
            tr = hist if n is None else hist.iloc[-n:]   # as N mais recentes
            limitar = limitador(tr.storypoint)

            m = modelo_tfidf()
            m.fit(tr.texto, tr.storypoint)
            previsoes = {
                "PLN": limitar(m.predict(teste.texto)),
                "Mediana": np.full(len(y_te), tr.storypoint.median()),
            }
            for rotulo, p in previsoes.items():
                linhas.append({"projeto": nome, "n_treino": len(tr),
                               "tamanho": "tudo" if n is None else str(n),
                               "modelo": rotulo, **metricas(y_te, p)})

        print(f"  {nome:<20} historico={len(hist):>4}  teste={len(teste):>4}", flush=True)

    r = pd.DataFrame(linhas)
    ordem = ["25", "50", "100", "200", "400", "800", "1600", "tudo"]
    r["tamanho"] = pd.Categorical(r["tamanho"], ordem, ordered=True)

    for met in ["MAE", "dentro_1", "cartao_exato"]:
        piv = r.pivot_table(index="tamanho", columns="modelo", values=met,
                            aggfunc="mean", observed=True)
        piv["projetos"] = r[r.modelo == "PLN"].groupby("tamanho", observed=True).size()
        imprimir(piv, f"{met} medio por tamanho de treino")

    # so os projetos que possuem todos os tamanhos ate 200: comparacao sem vies de amostra
    sub = r[r.tamanho.isin(["25", "50", "100", "200"])]
    completos = sub.groupby("projeto").tamanho.nunique().eq(4)
    sub = sub[sub.projeto.isin(completos[completos].index)]
    imprimir(sub.pivot_table(index="tamanho", columns="modelo",
                             values=["MAE", "dentro_1", "cartao_exato"],
                             aggfunc="mean", observed=True),
             f"mesmos {int(completos.sum())} projetos em todos os tamanhos (<= 200)")

    pln = r[r.modelo == "PLN"]
    imprimir(pln.pivot_table(index="projeto", columns="tamanho", values="MAE",
                             observed=True),
             "MAE do modelo por projeto conforme o treino cresce")

    # o que a curva conclui, em uma linha
    p25 = pln[pln.tamanho == "25"].MAE.mean()
    ptudo = pln[pln.tamanho == "tudo"].MAE.mean()
    n_medio = pln[pln.tamanho == "tudo"].n_treino.mean()
    print(f"\nDe 25 para {n_medio:.0f} historias de treino (media), o MAE vai de "
          f"{p25:.3f} para {ptudo:.3f} -- ganho de {(1 - ptudo / p25) * 100:.1f}%.")

    salvar(r, "sonda_volume.csv")


if __name__ == "__main__":
    main()

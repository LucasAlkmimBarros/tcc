"""A partir de quantas historias o modelo de texto pode aparecer no software?

A curva de aprendizado (sonda_volume.py) comeca em 25 historias. Abaixo disso
ninguem mediu, e o software usava 25 como piso. Este script estende a curva para
baixo com o mesmo desenho: teste fixo (os ultimos 20% de cada projeto), treino
com as N historias MAIS RECENTES antes do teste, e a mediana calculada sobre as
mesmas N historias.

Regra fixada em 2026-11-07, antes de rodar. O modelo passa a aparecer a partir
do menor N (entre 5, 10, 15, 20 e 25) em que, na media dos projetos, ele nao erra
mais que a mediana das mesmas N historias, nos dois conjuntos (Deep-SE e TAWOS),
e em todos os N maiores da lista. Se nenhum N abaixo de 25 passar, fica 25.

Acrescentado antes de ver o Deep-SE (a primeira execucao quebrou nele com N = 5):
com poucas historias o modelo pode nem treinar, porque nenhuma palavra aparece em
pelo menos 2 historias (min_df=2). Se isso acontece em algum projeto, aquele N nao
passa: o software nao pode prometer o modelo ali. Nesses projetos, a linha do
modelo repete a mediana so para a media poder ser calculada.

Uso:  python src/limiar_modelo.py ; CONJUNTO=tawos python src/limiar_modelo.py
"""
import numpy as np
import pandas as pd

from sonda_comum import (dividir_historico_teste, limitador, metricas,
                         modelo_tfidf, projetos, salvar)
from conjunto import NOME

TAMANHOS = [5, 10, 15, 20, 25]


def main():
    linhas = []
    for nome, d in projetos():
        hist, teste = dividir_historico_teste(d)
        y = teste.storypoint
        for n in TAMANHOS:
            tr = hist.iloc[-n:]
            mediana = np.full(len(y), tr.storypoint.median())
            try:
                m = modelo_tfidf().fit(tr.texto, tr.storypoint)
                p_modelo, treinou = limitador(tr.storypoint)(m.predict(teste.texto)), True
            except ValueError:  # nenhuma palavra em 2 historias
                p_modelo, treinou = mediana, False
            for rotulo, p in {"modelo": p_modelo, "mediana": mediana}.items():
                linhas.append({"projeto": nome, "n": n, "quem": rotulo, "treinou": treinou,
                               **metricas(y, p)})

    r = pd.DataFrame(linhas)
    piv = r.pivot_table(index="n", columns="quem", values="MAE", aggfunc="mean")
    piv["nao_treinou"] = (r[r.quem == "modelo"].groupby("n").treinou
                          .apply(lambda t: int((~t).sum())))
    piv["modelo_nao_pior"] = (piv.modelo <= piv.mediana) & (piv.nao_treinou == 0)
    piv["projetos_modelo_melhor"] = (r.pivot_table(index=["n", "projeto"], columns="quem",
                                                   values="MAE")
                                     .assign(m=lambda t: t.modelo < t.mediana)
                                     .groupby("n").m.sum().astype(int))
    print(f"=== {NOME}: erro medio (MAE) com as N historias mais recentes ===")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    salvar(r, "limiar_modelo.csv")


if __name__ == "__main__":
    main()

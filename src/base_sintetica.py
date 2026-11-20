"""Base construida (BASE_SINTETICA.md): o modelo do software contra a mediana.

Um projeto so, entao a comparacao e historia a historia: Wilcoxon pareado no erro
absoluto das 50 historias de teste (as 20% mais recentes), no mesmo corte das
sondas. O criterio foi fixado no BASE_SINTETICA.md antes de rodar.

Le o que ja existe: modelo_tfidf (o TF-IDF + SVR do software), a divisao e as
metricas das sondas. Nenhum modelo novo.

Uso:  CONJUNTO=sintetica python src/base_sintetica.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sonda_comum import (dividir_historico_teste, limitador, metricas,
                         modelo_tfidf, projetos, salvar)


def main():
    linhas, previsoes = [], []
    for nome, d in projetos():
        hist, teste = dividir_historico_teste(d)
        y = teste.storypoint.values.astype(float)
        m = modelo_tfidf().fit(hist.texto, hist.storypoint)
        p = {"mediana historica": np.full(len(y), hist.storypoint.median()),
             "TF-IDF + SVR": limitador(hist.storypoint)(m.predict(teste.texto))}

        e_med, e_mod = np.abs(y - p["mediana historica"]), np.abs(y - p["TF-IDF + SVR"])
        teste_p = wilcoxon(e_mod, e_med).pvalue
        for rot, prev in p.items():
            e = np.abs(y - prev)
            linhas.append({"projeto": nome, "condicao": rot, "n_historico": len(hist),
                           "n_teste": len(y), **metricas(y, prev),
                           "RMSE": float(np.sqrt((e ** 2).mean())),
                           "modelo_erra_menos_em": int((e_mod < e_med).sum()),
                           "empates": int((e_mod == e_med).sum()),
                           "mediana_erra_menos_em": int((e_mod > e_med).sum()),
                           "p_wilcoxon": teste_p})
            previsoes.append(pd.DataFrame({"projeto": nome, "issuekey": teste.issuekey,
                                           "storypoint": y, "condicao": rot, "previsao": prev}))

    r = pd.DataFrame(linhas)
    with pd.option_context("display.width", 200):
        print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    salvar(r, "base_sintetica.csv")
    salvar(pd.concat(previsoes, ignore_index=True), "base_sintetica_previsoes.csv")


if __name__ == "__main__":
    main()

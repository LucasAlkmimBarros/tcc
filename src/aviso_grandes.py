"""Aviso de historia grande: um corte tirado do historico de cada equipe.

Criterio fixado no PLANO.md antes de rodar (2026-11-10). Em vez de escolher a mao
um corte ("acima de 5,6 vira 8"), a equipe diz quantos alarmes falsos aceita para
cada historia grande achada (k). O aviso precisa entao acertar pelo menos 1 em cada
k + 1 vezes, e o corte e o menor valor de pontuacao em que isso aconteceu no
historico da propria equipe, em historias que o modelo previu sem ter visto.

    historia grande   pelo menos dois cartoes acima do cartao da mediana (80%)
    calibracao        previsoes "as cegas" de 40-60% (treino 0-40%) e de 60-80%
                      (treino 0-60%); o corte sai dai
    teste             treino nos 80%, corte aplicado nas 20% mais recentes
    pontuacoes        1. valor calculado pelo modelo atual (TF-IDF + SVR)
                      2. C: regressao de quantil 80% (errar para baixo custa 4x)
                      3. D: classificador "e grande?" (regressao logistica)
    melhor k          a fracao da perda recuperada: deixar passar uma grande custa
                      k, um alarme falso custa 1; economia = k*achadas - falsos,
                      dividida pela perda sem aviso, k*grandes
    acaso             avisar ao acaso o mesmo numero de vezes em cada projeto

Uso:  python src/aviso_grandes.py
      (grava resultados/aviso_grandes.csv e aviso_grandes_resumo.csv)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, QuantileRegressor

from apontar_grandes import indice_cartao, so_no_tawos
from caracterizar import carregar
from conjunto import RAIZ
from experimento import SEMENTE
from sonda_comum import limitador, modelo_tfidf

RES = RAIZ / "resultados"
KS = [1, 2, 3, 4, 6]
MIN_AVISOS = 5
QUANTIL = 0.8
ALPHAS = [0.001, 0.0003, 0.0001]
SORTEIOS = 10_000
REAMOSTRAGENS = 10_000
PONTUACOES = ["valor calculado (modelo atual)", "C: quantil 80%", "D: classificador"]

CONJUNTOS = [
    ("Deep-SE", RAIZ / "dados", RES / "previsoes_teste.csv", "base (TF-IDF + SVR)"),
    ("TAWOS", RAIZ / "tawos" / "dados", RES / "tawos" / "previsoes_teste.csv", "base (TF-IDF + SVR)"),
    ("RoomBook", RAIZ / "base_sintetica" / "dados", RES / "sintetica" / "base_sintetica_previsoes.csv", "TF-IDF + SVR"),
]


def pinball(y, p, q=QUANTIL):
    """Erro de quantil: errar para baixo pesa q, para cima pesa 1 - q."""
    e = y - p
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def pontuar(texto_tr, y_tr, g_tr, texto_alvo, alpha):
    """As tres pontuacoes para o alvo, treinadas so no passado."""
    svr = modelo_tfidf().fit(texto_tr, y_tr)
    valor = limitador(y_tr)(svr.predict(texto_alvo))
    vet = clone(modelo_tfidf().named_steps["tfidfvectorizer"])
    X_tr, X_al = vet.fit_transform(texto_tr), vet.transform(texto_alvo)
    quantil = QuantileRegressor(quantile=QUANTIL, alpha=alpha, solver="highs").fit(X_tr, y_tr)
    if g_tr.any() and not g_tr.all():
        clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X_tr, g_tr)
        prob = clf.predict_proba(X_al)[:, 1]
    else:
        prob = np.zeros(len(texto_alvo))
    return {PONTUACOES[0]: valor, PONTUACOES[1]: quantil.predict(X_al),
            PONTUACOES[2]: prob, "_quantil_cartao": limitador(y_tr)(quantil.predict(X_al))}


def escolher_alpha(texto_tr, y_tr, texto_al, y_al):
    vet = clone(modelo_tfidf().named_steps["tfidfvectorizer"])
    X_tr, X_al = vet.fit_transform(texto_tr), vet.transform(texto_al)
    erros = {a: pinball(y_al, QuantileRegressor(quantile=QUANTIL, alpha=a, solver="highs")
                        .fit(X_tr, y_tr).predict(X_al)) for a in ALPHAS}
    return min(erros, key=erros.get)


def corte(pont, grande, k):
    """Menor valor em que, das historias com pontuacao >= ele, ao menos 1/(k+1) sao
    grandes, com pelo menos MIN_AVISOS avisos. Sem nenhum: o aviso nao dispara."""
    alvo = 1 / (k + 1)
    ordem = np.argsort(-pont, kind="stable")
    p, g = pont[ordem], grande[ordem].astype(float)
    avisos = np.arange(1, len(p) + 1)
    acertos = np.cumsum(g)
    fim_de_empate = np.r_[p[1:] != p[:-1], True]       # so corta entre valores diferentes
    ok = fim_de_empate & (avisos >= MIN_AVISOS) & (acertos / avisos >= alvo)
    return float(p[np.flatnonzero(ok)[-1]]) if ok.any() else np.inf


def por_projeto(conjunto, dados, prev, cond_ref):
    linhas, cartoes, conferencia = [], [], []
    for arq in sorted(dados.glob("*.csv")):
        d = carregar(arq)
        n = len(d)
        a, b, c = int(n * 0.4), int(n * 0.6), int(n * 0.8)
        y = d.storypoint.values.astype(float)
        texto = d.texto.reset_index(drop=True)
        c_med = indice_cartao([np.median(y[:c])])[0]
        grande = indice_cartao(y) >= c_med + 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = escolher_alpha(texto[:a], y[:a], texto[a:b], y[a:b])
            cal1 = pontuar(texto[:a], y[:a], grande[:a], texto[a:b], alpha)
            cal2 = pontuar(texto[:b], y[:b], grande[:b], texto[b:c], alpha)
            tes = pontuar(texto[:c], y[:c], grande[:c], texto[c:], alpha)

        # conferencia: o valor calculado no teste e o do previsoes.py
        ref = prev[(prev.projeto == arq.stem) & (prev.condicao == cond_ref)]
        ref = d.iloc[c:][["issuekey"]].merge(ref, on="issuekey")
        conferencia.append(float(np.max(np.abs(ref.previsao.values - tes[PONTUACOES[0]]))))

        g_cal, g_te = grande[a:c], grande[c:]
        for nome in PONTUACOES:
            p_cal = np.r_[cal1[nome], cal2[nome]]
            for k in KS:
                t = corte(p_cal, g_cal, k)
                aviso = tes[nome] >= t
                linhas.append({"conjunto": conjunto, "projeto": arq.stem, "pontuacao": nome,
                               "k": k, "alpha_C": alpha, "corte": t,
                               "historias_teste": len(g_te), "grandes": int(g_te.sum()),
                               "avisos": int(aviso.sum()), "acertos": int((aviso & g_te).sum()),
                               "alarmes_falsos": int((aviso & ~g_te).sum())})
        y_te, med = y[c:], np.median(y[:c])
        cartoes.append({"conjunto": conjunto, "projeto": arq.stem,
                        "MAE_mediana": np.abs(y_te - med).mean(),
                        "MAE_modelo_atual": np.abs(y_te - tes[PONTUACOES[0]]).mean(),
                        "MAE_C_quantil_80": np.abs(y_te - tes["_quantil_cartao"]).mean()})
        print(f"  {conjunto:<9} {arq.stem:<20} alpha C = {alpha:g}", flush=True)
    return linhas, cartoes, conferencia


def resumir(t, grupo, rng):
    linhas = []
    for (nome, k), g in t.groupby(["pontuacao", "k"], sort=False):
        G, ach, fal, av = g.grandes.sum(), g.acertos.sum(), g.alarmes_falsos.sum(), g.avisos.sum()
        fracao = (k * ach - fal) / (k * G) if G else np.nan
        # acaso: o mesmo numero de avisos em cada projeto, em historias sorteadas
        acaso = sum(rng.hypergeometric(r.grandes, r.historias_teste - r.grandes, r.avisos, SORTEIOS)
                    if r.avisos else np.zeros(SORTEIOS, int) for r in g.itertuples())
        frac_acaso = (k * acaso - (av - acaso)) / (k * G) if G else np.full(SORTEIOS, np.nan)
        # intervalo: reamostrando projetos
        idx = rng.integers(0, len(g), (REAMOSTRAGENS, len(g)))
        Gb, Ab, Fb = g.grandes.values[idx].sum(1), g.acertos.values[idx].sum(1), g.alarmes_falsos.values[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            fb = (k * Ab - Fb) / (k * Gb)
        linhas.append({"grupo": grupo, "pontuacao": nome, "k": k,
                       "promessa_acerto_pct": 100 / (k + 1), "projetos": len(g),
                       "historias": int(g.historias_teste.sum()), "grandes": int(G),
                       "avisos": int(av), "acertos": int(ach), "alarmes_falsos": int(fal),
                       "historias_com_aviso_pct": 100 * av / g.historias_teste.sum(),
                       "grandes_achadas_pct": 100 * ach / G if G else np.nan,
                       "avisos_certos_pct": 100 * ach / av if av else np.nan,
                       "projetos_sem_aviso": int((g.avisos == 0).sum()),
                       "fracao_recuperada": fracao,
                       "ic95_inf": np.nanpercentile(fb, 2.5), "ic95_sup": np.nanpercentile(fb, 97.5),
                       "fracao_acaso_media": float(np.mean(frac_acaso)),
                       "p_acaso": float(np.mean(acaso >= ach)) if av else np.nan})
    return linhas


def main():
    rng = np.random.default_rng(SEMENTE)
    novos = so_no_tawos()
    todos, cartoes, conf = [], [], []
    for conjunto, dados, arq_prev, cond_ref in CONJUNTOS:
        li, ca, co = por_projeto(conjunto, dados, pd.read_csv(arq_prev), cond_ref)
        todos += li
        cartoes += ca
        conf += co
    print(f"\nconferencia com as previsoes ja gravadas: maior diferenca {max(conf):.2e}")
    assert max(conf) < 1e-3, "o valor calculado nao bate com o previsoes.py"

    t = pd.DataFrame(todos)
    resumo = []
    for grupo, g in [("Deep-SE", t[t.conjunto == "Deep-SE"]), ("TAWOS", t[t.conjunto == "TAWOS"]),
                     ("TAWOS, so os 13 projetos novos", t[(t.conjunto == "TAWOS") & t.projeto.isin(novos)]),
                     ("RoomBook", t[t.conjunto == "RoomBook"])]:
        resumo += resumir(g, grupo, rng)
    r = pd.DataFrame(resumo)

    # decisao fixada no PLANO
    def funciona(grupo):
        x = r[r.grupo == grupo].set_index(["pontuacao", "k"])
        return (x.fracao_recuperada > 0) & (x.p_acaso < 0.05)
    r["funciona_no_grupo"] = [bool(funciona(gr).get((p, k), False)) if gr in ("Deep-SE", "TAWOS") else np.nan
                              for gr, p, k in zip(r.grupo, r.pontuacao, r.k)]

    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 32):
        cols = ["grupo", "pontuacao", "k", "promessa_acerto_pct", "historias_com_aviso_pct",
                "grandes_achadas_pct", "avisos_certos_pct", "projetos_sem_aviso",
                "fracao_recuperada", "ic95_inf", "ic95_sup", "fracao_acaso_media", "p_acaso",
                "funciona_no_grupo"]
        print(r[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ds, tw = funciona("Deep-SE"), funciona("TAWOS")
    ok = ds & tw
    x = r[r.grupo == "Deep-SE"].set_index(["pontuacao", "k"]).fracao_recuperada
    print("\n=== decisao (regra do PLANO) ===")
    if not ok.any():
        print("nenhum k funciona nos dois conjuntos: a pagina nao ganha o aviso")
    else:
        for nome in PONTUACOES:
            ks = [k for k in KS if ok.get((nome, k), False)]
            print(f"  {nome:<32} k que funcionam nos dois: {ks}")
        melhor_k = max(KS, key=lambda k: max((x[(p, k)] for p in PONTUACOES if ok.get((p, k), False)),
                                             default=-np.inf))
        melhor_p = max((p for p in PONTUACOES if ok.get((p, melhor_k), False)), key=lambda p: x[(p, melhor_k)])
        print(f"  k recomendado: {melhor_k} | pontuacao recomendada: {melhor_p} "
              f"(fracao recuperada no Deep-SE {x[(melhor_p, melhor_k)]:.3f})")

    c = pd.DataFrame(cartoes).groupby("conjunto")[["MAE_mediana", "MAE_modelo_atual", "MAE_C_quantil_80"]].mean()
    print("\n=== C usado como cartao (sem decidir): erro medio ===")
    print(c.to_string(float_format=lambda v: f"{v:.3f}"))

    t.to_csv(RES / "aviso_grandes.csv", index=False, float_format="%.4f")
    r.to_csv(RES / "aviso_grandes_resumo.csv", index=False, float_format="%.4f")
    pd.DataFrame(cartoes).to_csv(RES / "aviso_grandes_c_cartao.csv", index=False, float_format="%.4f")
    print(f"\n-> {RES / 'aviso_grandes.csv'}\n-> {RES / 'aviso_grandes_resumo.csv'}"
          f"\n-> {RES / 'aviso_grandes_c_cartao.csv'}")


if __name__ == "__main__":
    main()

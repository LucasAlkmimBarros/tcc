"""Sensibilidade: as conclusoes resistem a outras formas de fazer a mesma conta?

Analises feitas DEPOIS de ver os resultados (2026-11-10), a partir da pre-banca.
Nao ha criterio fixado antes: nada aqui escolhe modelo nem muda o software. Cada
bloco responde a uma pergunta que a banca pode fazer:

  1. mediana com o mesmo historico   A "mediana historica" dos experimentos usa os
                                     60% mais antigos (como a replicacao de Tawosi
                                     et al.), e o modelo treina com 80%. E se a
                                     mediana tambem usar os 80%?
  2. tamanho da diferenca            Em vez de so "p = 0,14": o ganho medio do
                                     modelo sobre a mediana e um intervalo de 95%
                                     (reamostrando os projetos).
  3. projetos repetidos              10 dos 23 projetos do TAWOS tambem estao no
                                     Deep-SE. O que acontece nos 13 que nao estao?
  4. um projeto so                   O Moodle e o projeto em que a escala mais
                                     mudou. O que muda sem ele?
  5. por historia                    A comparacao principal da o mesmo peso a cada
                                     projeto. E juntando todas as historias?
  6. dia zero                        O LLM com referencias contra a mediana da
                                     equipe e contra o chute fixo, com e sem o
                                     Moodle, e a mediana das n historias mais
                                     recentes.

  7. base construida                 O resultado do RoomBook (modelo passa da
                                     mediana) depende de onde se corta treino e
                                     teste? E qual o intervalo do ganho?

  8. volume de historico             Mais historias melhoram o modelo? E a diferenca
                                     dele para a mediana? Compara as bases publicas
                                     (Deep-SE) com a base construida, e conta quanto
                                     do texto de teste o modelo reconhece.

  9. as condicoes de cada base     As tres bases (Deep-SE, TAWOS e a construida)
                                     lado a lado no que a literatura diz que falta as
                                     bases publicas (2026-11-15): tamanho do texto,
                                     descricao vazia, criterios de aceitacao escritos,
                                     escala que muda e textos identicos com pontos
                                     diferentes. So conta; nao treina nada.

Os blocos 1 a 6 usam so previsoes ja gravadas (previsoes.py, cold_start_*.py). Os
blocos 7 e 8 treinam o modelo do software (segundos).

Uso:  python src/sensibilidade.py
      (grava resultados/sensibilidade.csv, sensibilidade_dia_zero.csv,
       sensibilidade_base_construida.csv, sensibilidade_volume.csv,
       sensibilidade_base_construida_medidas.csv e sensibilidade_condicoes.csv)
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from apontar_grandes import so_no_tawos
from caracterizar import carregar
from conjunto import RAIZ
from sonda_comum import cartao, dividir_historico_teste, limitador, modelo_tfidf

RES = RAIZ / "resultados"
SEMENTE = 42
REAMOSTRAGENS = 10_000
SPRINT = 10
MOODLE = {"moodle", "MDL"}

CONJUNTOS = [("Deep-SE", RAIZ / "dados", RES / "previsoes_teste.csv"),
             ("TAWOS", RAIZ / "tawos" / "dados", RES / "tawos" / "previsoes_teste.csv")]
MODELOS = {"TF-IDF original (experimento principal)": "TF-IDF original",
           "base (TF-IDF + SVR)": "modelo do software (TF-IDF + SVR)"}
MENOR_E_MELHOR = {"MAE": True, "RMSE": True, "erro_total_sprint": True,
                  "dentro_1": False, "cartao_exato": False}


def metricas(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    e = np.abs(y - p)
    n = len(y) // SPRINT * SPRINT
    blocos = np.abs(p[:n].reshape(-1, SPRINT).sum(1) - y[:n].reshape(-1, SPRINT).sum(1))
    return {"MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "dentro_1": (e <= 1).mean() * 100,
            "cartao_exato": (cartao(p) == cartao(y)).mean() * 100,
            "erro_total_sprint": blocos.mean() if len(blocos) else np.nan}


def comparar(tab, modelo, regua, metrica, rng):
    """Ganho do modelo sobre a regua, entre projetos, com intervalo e Wilcoxon."""
    x, r = tab[(modelo, metrica)], tab[(regua, metrica)]
    ok = x.notna() & r.notna()
    x, r = x[ok], r[ok]
    ganho = (r - x) if MENOR_E_MELHOR[metrica] else (x - r)     # positivo = modelo melhor
    medias = [rng.choice(ganho.values, len(ganho)).mean() for _ in range(REAMOSTRAGENS)]
    return {"modelo_media": x.mean(), "regua_media": r.mean(), "ganho_medio": ganho.mean(),
            "ic95_inf": np.percentile(medias, 2.5), "ic95_sup": np.percentile(medias, 97.5),
            "modelo_melhor_em": int((ganho > 0).sum()), "projetos": len(ganho),
            "p_wilcoxon": wilcoxon(x, r).pvalue if (x != r).any() else np.nan}


def tabela_do_conjunto(dados, prev):
    """Metricas por projeto de cada modelo e das duas medianas; e as historias."""
    linhas, historias = {}, []
    for arq in sorted(dados.glob("*.csv")):
        hist, teste = dividir_historico_teste(carregar(arq))
        p = prev[prev.projeto == arq.stem]
        y = p[p.condicao == "mediana historica"].storypoint.values.astype(float)
        assert len(y) == len(teste), f"{arq.stem}: previsoes nao cobrem o teste"
        m60 = p[p.condicao == "mediana historica"].previsao.values.astype(float)
        m80 = np.full(len(y), float(hist.storypoint.median()))
        cond = {"mediana com 60% (a da literatura)": m60,
                "mediana com 80% (mesmo historico do modelo)": m80}
        for c in p.condicao.unique():
            rot = next((v for k, v in MODELOS.items() if c.startswith(k)), None)
            if rot:
                cond[rot] = p[p.condicao == c].previsao.values.astype(float)
        for rot, v in cond.items():
            for k, val in metricas(y, v).items():
                linhas[(arq.stem, rot, k)] = val
            historias.append(pd.DataFrame({"projeto": arq.stem, "condicao": rot,
                                           "erro": np.abs(y - v)}))
    tab = pd.Series(linhas).unstack([1, 2])
    return tab, pd.concat(historias, ignore_index=True)


def bases_publicas(rng):
    novos = so_no_tawos()
    reguas = ["mediana com 60% (a da literatura)", "mediana com 80% (mesmo historico do modelo)"]
    saida = []
    for conj, dados, arq in CONJUNTOS:
        tab, hist = tabela_do_conjunto(dados, pd.read_csv(arq))
        grupos = [("todos os projetos", tab.index)]
        grupos.append(("sem o Moodle", [p for p in tab.index if p not in MOODLE]))
        if conj == "TAWOS":
            grupos.append(("so os 13 projetos que nao estao no Deep-SE",
                           [p for p in tab.index if p in novos]))
        for grupo, projs in grupos:
            for modelo in MODELOS.values():
                for regua in reguas:
                    for met in MENOR_E_MELHOR:
                        if grupo != "todos os projetos" and met not in ("MAE", "RMSE"):
                            continue
                        saida.append({"analise": "entre projetos", "conjunto": conj, "grupo": grupo,
                                      "modelo": modelo, "regua": regua, "metrica": met,
                                      **comparar(tab.loc[projs], modelo, regua, met, rng)})
        # por historia: todas as historias de teste juntas
        for modelo in MODELOS.values():
            a = hist[hist.condicao == modelo].erro.values
            for regua in reguas:
                b = hist[hist.condicao == regua].erro.values
                saida.append({"analise": "por historia (todas juntas)", "conjunto": conj,
                              "grupo": "todos os projetos", "modelo": modelo, "regua": regua,
                              "metrica": "MAE", "modelo_media": a.mean(), "regua_media": b.mean(),
                              "ganho_medio": b.mean() - a.mean(),
                              "modelo_melhor_em": int((a < b).sum()), "projetos": len(a),
                              "p_wilcoxon": wilcoxon(a, b).pvalue})
        # o ganho depende do tamanho do projeto? metade maior contra metade menor
        n_teste = hist[hist.condicao == reguas[0]].groupby("projeto").size()
        maiores = n_teste.sort_values().index[len(n_teste) // 2:]
        for modelo in MODELOS.values():
            g = tab[(reguas[1], "MAE")] - tab[(modelo, "MAE")]
            for rot, idx in [("metade maior dos projetos", maiores),
                             ("metade menor dos projetos", [p for p in tab.index if p not in maiores])]:
                saida.append({"analise": "tamanho do projeto", "conjunto": conj, "grupo": rot,
                              "modelo": modelo, "regua": reguas[1], "metrica": "MAE",
                              "modelo_media": tab.loc[idx, (modelo, "MAE")].mean(),
                              "regua_media": tab.loc[idx, (reguas[1], "MAE")].mean(),
                              "ganho_medio": g[idx].mean(),
                              "modelo_melhor_em": int((g[idx] > 0).sum()), "projetos": len(idx)})
    return pd.DataFrame(saida)


def dia_zero(rng):
    metodo = "LLM com referencias (prompt unico)"
    reguas = ["mediana da equipe", "mediana dos outros projetos", "LLM sem referencia"]
    prev_reguas = pd.read_csv(RES / "tawos" / "cold_start_previsoes_reguas.csv")
    saida = []
    for nome, sufixo in [("Gemini 3.5 Flash-Lite", "cold_start_lote_reduzida"),
                         ("Qwen 2.5 14B (local)", "cold_start_ollama_reduzida")]:
        m = pd.read_csv(RES / "tawos" / f"{sufixo}_metricas.csv")
        for met in ["MAE", "cartao_exato", "dentro_1"]:
            piv = m.pivot(index="projeto", columns="rotulo", values=met)
            for grupo, projs in [("12 projetos", piv.index),
                                 ("sem o Moodle (11)", [p for p in piv.index if p not in MOODLE])]:
                for regua in reguas:
                    x, r = piv.loc[projs, metodo], piv.loc[projs, regua]
                    ganho = (r - x) if met == "MAE" else (x - r)
                    medias = [rng.choice(ganho.values, len(ganho)).mean() for _ in range(REAMOSTRAGENS)]
                    saida.append({"analise": "LLM com referencias contra a regua", "llm": nome,
                                  "grupo": grupo, "regua": regua, "metrica": met,
                                  "llm_media": x.mean(), "regua_media": r.mean(),
                                  "ganho_medio": ganho.mean(),
                                  "ic95_inf": np.percentile(medias, 2.5),
                                  "ic95_sup": np.percentile(medias, 97.5),
                                  "llm_melhor_em": int((ganho > 0).sum()), "projetos": len(ganho),
                                  "p_wilcoxon": wilcoxon(x, r).pvalue if (x != r).any() else np.nan})
        # a mediana das n historias mais recentes, nas mesmas historias do LLM
        piv = m.pivot(index="projeto", columns="rotulo", values="MAE")
        llm = pd.read_csv(RES / "tawos" / f"{sufixo.replace('cold_start_', 'cold_start_previsoes_llm_')}_tipica.csv")
        cobertas = llm[["projeto", "issuekey"]].drop_duplicates()
        cur = prev_reguas[prev_reguas.condicao == "mediana das ultimas n"].merge(
            cobertas, on=["projeto", "issuekey"])
        cur["erro"] = (cur.storypoint - cur.previsao).abs()
        por_n = cur.groupby(["n_historico", "projeto"]).erro.mean().unstack()
        for n in [1, 2, 3, 5, 10, 15, 20, 30]:
            for grupo, projs in [("12 projetos", piv.index),
                                 ("sem o Moodle (11)", [p for p in piv.index if p not in MOODLE])]:
                x, r = piv.loc[projs, metodo], por_n.loc[n, projs]
                saida.append({"analise": "mediana das n historias mais recentes", "llm": nome,
                              "grupo": grupo, "regua": f"mediana das {n} mais recentes",
                              "metrica": "MAE", "llm_media": x.mean(), "regua_media": r.mean(),
                              "ganho_medio": (r - x).mean(),
                              "llm_melhor_em": int((x < r).sum()), "projetos": len(x)})
    return pd.DataFrame(saida)


def base_construida(rng):
    """O modelo do software contra a mediana no RoomBook, em tres cortes: treino nas
    primeiras N historias, teste nas 50 seguintes. O corte oficial e o de 200."""
    d = carregar(RAIZ / "base_sintetica" / "dados" / "roombook.csv")
    saida = []
    for n_treino in [150, 175, 200]:
        hist, teste = d.iloc[:n_treino], d.iloc[n_treino:n_treino + 50]
        y = teste.storypoint.values.astype(float)
        m = modelo_tfidf().fit(hist.texto, hist.storypoint)
        e_mod = np.abs(y - limitador(hist.storypoint)(m.predict(teste.texto)))
        e_med = np.abs(y - float(hist.storypoint.median()))
        dif = e_med - e_mod
        medias = [rng.choice(dif, len(dif)).mean() for _ in range(REAMOSTRAGENS)]
        saida.append({"treino": f"historias 1 a {n_treino}",
                      "teste": f"historias {n_treino + 1} a {n_treino + 50}",
                      "corte_oficial": n_treino == 200,
                      "MAE_modelo": e_mod.mean(), "MAE_mediana": e_med.mean(),
                      "ganho_medio": dif.mean(), "ic95_inf": np.percentile(medias, 2.5),
                      "ic95_sup": np.percentile(medias, 97.5),
                      "modelo_erra_menos_em": int((e_mod < e_med).sum()),
                      "mediana_erra_menos_em": int((e_mod > e_med).sum()),
                      "p_wilcoxon_historias": wilcoxon(e_mod, e_med).pvalue})
    return pd.DataFrame(saida)


def reconhecidos(vec, textos):
    """% dos termos (palavras e pares de palavras) dos textos que o modelo conhece."""
    analisar, voc = vec.build_analyzer(), vec.vocabulary_
    termos = [t for x in textos for t in analisar(x)]
    return float(np.mean([t in voc for t in termos]) * 100) if termos else np.nan


def volume():
    """Erro do modelo e da mediana quando o treino usa so as N historias mais recentes
    antes do teste, e quanto do texto do teste o modelo reconhece."""
    linhas = []
    # base construida: teste fixo = historias 201 a 250; e a media de tres janelas
    d = carregar(RAIZ / "base_sintetica" / "dados" / "roombook.csv")
    for N in [25, 50, 100, 150, 200]:
        tr, te = d.iloc[200 - N:200], d.iloc[200:]
        m = modelo_tfidf().fit(tr.texto, tr.storypoint.astype(float))
        v, y = m.named_steps["tfidfvectorizer"], te.storypoint.values.astype(float)
        linhas.append({"base": "base construida (teste: historias 201 a 250)", "N": N, "projetos": 1,
                       "termos_no_vocabulario": len(v.vocabulary_),
                       "termos_do_teste_reconhecidos_pct": reconhecidos(v, te.texto),
                       "MAE_modelo": np.abs(y - limitador(tr.storypoint)(m.predict(te.texto))).mean(),
                       "MAE_mediana": np.abs(y - float(tr.storypoint.median())).mean()})
    for N in [25, 50, 100, 150]:
        a, b, c, r = [], [], [], []
        for ini in (150, 175, 200):
            tr, te = d.iloc[ini - N:ini], d.iloc[ini:ini + 50]
            m = modelo_tfidf().fit(tr.texto, tr.storypoint.astype(float))
            y = te.storypoint.values.astype(float)
            a.append(np.abs(y - limitador(tr.storypoint)(m.predict(te.texto))).mean())
            b.append(np.abs(y - float(tr.storypoint.median())).mean())
            c.append(len(m.named_steps["tfidfvectorizer"].vocabulary_))
            r.append(reconhecidos(m.named_steps["tfidfvectorizer"], te.texto))
        linhas.append({"base": "base construida (media de 3 janelas de teste de 50)", "N": N, "projetos": 1,
                       "termos_no_vocabulario": np.mean(c), "termos_do_teste_reconhecidos_pct": np.mean(r),
                       "MAE_modelo": np.mean(a), "MAE_mediana": np.mean(b)})
    # Deep-SE: teste = 20% finais de cada projeto; projetos com historico >= N
    res = {}
    for arq in sorted((RAIZ / "dados").glob("*.csv")):
        hist, te = dividir_historico_teste(carregar(arq))
        y = te.storypoint.values.astype(float)
        for N in [25, 50, 100, 200, 400, 800]:
            if N > len(hist):
                continue
            tr = hist.iloc[-N:]
            m = modelo_tfidf().fit(tr.texto, tr.storypoint.astype(float))
            v = m.named_steps["tfidfvectorizer"]
            res.setdefault(N, []).append((len(v.vocabulary_), reconhecidos(v, te.texto),
                                          np.abs(y - limitador(tr.storypoint)(m.predict(te.texto))).mean(),
                                          np.abs(y - float(tr.storypoint.median())).mean()))
    for N, r in res.items():
        r = np.array(r)
        linhas.append({"base": "Deep-SE (teste: 20% finais; so projetos com N historias)", "N": N,
                       "projetos": len(r), "termos_no_vocabulario": r[:, 0].mean(),
                       "termos_do_teste_reconhecidos_pct": r[:, 1].mean(),
                       "MAE_modelo": r[:, 2].mean(), "MAE_mediana": r[:, 3].mean()})
    return pd.DataFrame(linhas)


def condicoes_das_bases():
    """Bloco 9. Cada projeto pesa igual nas medias; as contagens somam os projetos.
    A escala e a mesma conta da sonda de consistencia: a mediana do primeiro terco
    contra a do ultimo terco do projeto, e 'andou' quando o cartao das duas difere."""
    bases = {"Deep-SE": sorted((RAIZ / "dados").glob("*.csv")),
             "TAWOS": sorted((RAIZ / "tawos" / "dados").glob("*.csv")),
             "base construida": [RAIZ / "base_sintetica" / "dados" / "roombook.csv"]}
    linhas = []
    for base, arquivos in bases.items():
        por = []
        for arq in arquivos:
            d = carregar(arq)
            n, terco = len(d), len(d) // 3
            ini = float(d.storypoint.iloc[:terco].median())
            fim = float(d.storypoint.iloc[-terco:].median())
            g = (d.assign(chave=d.texto.str.lower().str.strip())
                 .groupby("chave").storypoint.agg(["count", "nunique"]))
            desc = d.description.fillna("").str.strip()
            por.append({"palavras": d.texto.str.split().str.len().median(),
                        "descricao_vazia_pct": (desc == "").mean() * 100,
                        "criterios_pct": d.texto.str.contains(r"acceptance criteri", case=False).mean() * 100,
                        "escala_andou": cartao([ini])[0] != cartao([fim])[0],
                        "textos_divergentes": int(((g["count"] > 1) & (g["nunique"] > 1)).sum())})
        t = pd.DataFrame(por)
        linhas.append({"base": base, "projetos": len(t),
                       "palavras_por_historia_mediana": t.palavras.mean(),
                       "descricao_vazia_pct": t.descricao_vazia_pct.mean(),
                       "com_criterios_de_aceitacao_pct": t.criterios_pct.mean(),
                       "projetos_em_que_a_escala_andou": int(t.escala_andou.sum()),
                       "grupos_de_texto_identico_com_pontos_diferentes": int(t.textos_divergentes.sum()),
                       "projetos_com_esses_grupos": int((t.textos_divergentes > 0).sum())})
    return pd.DataFrame(linhas)


def base_construida_medidas(rng):
    """As quatro medidas do modelo contra a mediana na base construida (corte oficial:
    treino nas 200 primeiras, teste nas 50 ultimas). So ha um projeto, entao o intervalo
    de 95% reamostra as 50 historias de teste (nas bases publicas ele reamostra projetos)."""
    d = carregar(RAIZ / "base_sintetica" / "dados" / "roombook.csv")
    hist, te = d.iloc[:200], d.iloc[200:]
    y = te.storypoint.values.astype(float)
    m = modelo_tfidf().fit(hist.texto, hist.storypoint.astype(float))
    p_mod = limitador(hist.storypoint)(m.predict(te.texto))
    p_med = np.full(len(y), float(hist.storypoint.median()))
    e_mod, e_med = np.abs(y - p_mod), np.abs(y - p_med)
    acerto_mod, acerto_med = (cartao(p_mod) == cartao(y)).astype(float), (cartao(p_med) == cartao(y)).astype(float)
    idx = rng.integers(0, len(y), (REAMOSTRAGENS, len(y)))
    # cada medida: (valor do modelo, valor da mediana, ganho do modelo em cada reamostragem)
    medidas = {
        "MAE": (e_mod.mean(), e_med.mean(), e_med[idx].mean(1) - e_mod[idx].mean(1)),
        "RMSE": (np.sqrt((e_mod ** 2).mean()), np.sqrt((e_med ** 2).mean()),
                 np.sqrt((e_med[idx] ** 2).mean(1)) - np.sqrt((e_mod[idx] ** 2).mean(1))),
        "dentro_1": ((e_mod <= 1).mean() * 100, (e_med <= 1).mean() * 100,
                     ((e_mod[idx] <= 1).mean(1) - (e_med[idx] <= 1).mean(1)) * 100),
        "cartao_exato": (acerto_mod.mean() * 100, acerto_med.mean() * 100,
                         (acerto_mod[idx].mean(1) - acerto_med[idx].mean(1)) * 100)}
    linhas = []
    for k, (a, b, boot) in medidas.items():
        ganho = (b - a) if k in ("MAE", "RMSE") else (a - b)
        linhas.append({"medida": k, "modelo": a, "mediana": b, "ganho_medio": ganho,
                       "ic95_inf": np.percentile(boot, 2.5), "ic95_sup": np.percentile(boot, 97.5),
                       "historias_teste": len(y)})
    return pd.DataFrame(linhas)


def main():
    rng = np.random.default_rng(SEMENTE)
    vol = volume()
    b = bases_publicas(rng)
    z = dia_zero(rng)
    c = base_construida(rng)
    cm = base_construida_medidas(rng)
    cond = condicoes_das_bases()
    # o erro medio tem um unico intervalo na base construida: o do corte oficial do bloco 7
    oficial = c[c.corte_oficial].iloc[0]
    cm.loc[cm.medida == "MAE", ["ic95_inf", "ic95_sup"]] = [oficial.ic95_inf, oficial.ic95_sup]
    fmt = lambda v: f"{v:.3f}"
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 46):
        soft = "modelo do software (TF-IDF + SVR)"
        print("=== bases publicas: modelo do software, entre projetos ===")
        x = b[(b.analise == "entre projetos") & (b.modelo == soft)]
        print(x.drop(columns=["analise", "modelo"]).to_string(index=False, float_format=fmt))
        print("\n=== por historia e por tamanho do projeto ===")
        x = b[b.analise != "entre projetos"]
        print(x.drop(columns=["ic95_inf", "ic95_sup"]).to_string(index=False, float_format=fmt))
        print("\n=== dia zero: LLM com referencias (prompt unico), erro medio ===")
        x = z[(z.analise == "LLM com referencias contra a regua") & (z.metrica == "MAE")]
        print(x.drop(columns=["analise", "metrica"]).to_string(index=False, float_format=fmt))
        print("\n=== dia zero: mediana das n mais recentes ===")
        x = z[z.analise == "mediana das n historias mais recentes"]
        print(x.drop(columns=["analise", "metrica", "ic95_inf", "ic95_sup", "p_wilcoxon"])
              .to_string(index=False, float_format=fmt))
        print("\n=== base construida (RoomBook): outros cortes ===")
        print(c.to_string(index=False, float_format=fmt))
        print("\n=== base construida: medidas com intervalo por historia ===")
        print(cm.to_string(index=False, float_format=fmt))
        print("\n=== as condicoes de cada base ===")
        print(cond.T.to_string(float_format=fmt))
        print("\n=== volume de historico ===")
        print(vol.assign(dif=vol.MAE_modelo - vol.MAE_mediana).to_string(index=False, float_format=fmt))
    b.to_csv(RES / "sensibilidade.csv", index=False, float_format="%.4f")
    z.to_csv(RES / "sensibilidade_dia_zero.csv", index=False, float_format="%.4f")
    c.to_csv(RES / "sensibilidade_base_construida.csv", index=False, float_format="%.4f")
    vol.to_csv(RES / "sensibilidade_volume.csv", index=False, float_format="%.4f")
    cm.to_csv(RES / "sensibilidade_base_construida_medidas.csv", index=False, float_format="%.4f")
    cond.to_csv(RES / "sensibilidade_condicoes.csv", index=False, float_format="%.4f")
    print(f"\n-> {RES / 'sensibilidade.csv'}\n-> {RES / 'sensibilidade_dia_zero.csv'}"
          f"\n-> {RES / 'sensibilidade_base_construida.csv'}\n-> {RES / 'sensibilidade_volume.csv'}")


if __name__ == "__main__":
    main()

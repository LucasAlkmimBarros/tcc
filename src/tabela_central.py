"""Tabela central: todas as solucoes, as mesmas colunas, nos dois conjuntos.

A pergunta do trabalho e "quando e por que um modelo de texto ajuda". Para
responder, cada solucao -- as reguas simples, os modelos de texto daqui, os
publicados por outros autores e os LLMs -- entra numa tabela so, separada por
conjunto (Deep-SE, 16 projetos; TAWOS, 23 projetos).

Colunas, todas como media dos projetos (cada projeto pesa igual):
    MAE, MdAE           erro medio e erro mediano, em story points
    cartao_exato        % de previsoes no mesmo cartao de Planning Poker da verdade
    dentro_1            % de previsoes que erram por no maximo 1 ponto
    correlacao          se a previsao sobe quando a verdade sobe (acerta a ordem);
                        vazia para previsao constante
    vence_mediana       em quantos projetos o MAE fica abaixo do da mediana
                        historica da equipe, calculada nas MESMAS historias
    p_wilcoxon          teste pareado entre projetos contra essa mediana

Regras de origem:
- Toda linha nossa vem de previsoes historia a historia gravadas por um script
  (previsoes.py, cold_start_*.py).
- Linhas de terceiros vem so de referencia/: tabelas transcritas e conferidas
  (Deep-SE publicado e replicado, Tawosi Tab. 5) ou previsoes publicadas e
  recalculadas aqui (Shetty et al.). Delas so existe o que a fonte da: as
  tabelas trazem so MAE; o resto fica vazio.
- As linhas do cold start usam a amostra de teste do cold start (ate 100
  historias por projeto; ou a amostra reduzida, se a cota do LLM nao deu para
  tudo). A coluna `historias` diz isso, e a comparacao com a mediana e feita nas
  mesmas historias.

Uso:  python src/tabela_central.py   (le os dois conjuntos; grava resultados/tabela_central.csv)
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

from caracterizar import carregar
from experimento import dividir
from sonda_comum import cartao

from conjunto import RAIZ
RES, REF = RAIZ / "resultados", RAIZ / "referencia"
PASTAS = {"Deep-SE": (RAIZ / "dados", RES), "TAWOS": (RAIZ / "tawos" / "dados", RES / "tawos")}


def metricas(g):
    y, p = g.storypoint.values.astype(float), g.previsao.values.astype(float)
    e = np.abs(y - p)
    r = pearsonr(p, y)[0] if p.std() > 1e-9 and y.std() > 1e-9 else np.nan
    return pd.Series({"MAE": e.mean(), "MdAE": np.median(e),
                      "cartao_exato": (cartao(p) == cartao(y)).mean() * 100,
                      "dentro_1": (e <= 1).mean() * 100, "correlacao": r})


def contra_mediana(mae, mae_med):
    comuns = mae.index.intersection(mae_med.index)
    a, b = mae[comuns], mae_med[comuns]
    p = wilcoxon(a, b).pvalue if (a != b).any() else np.nan
    return int((a < b).sum()), len(comuns), p


def linha_previsoes(conjunto, nome, origem, prev, med, historias):
    """Linha a partir de previsoes historia a historia; `med` sao as previsoes da
    mediana historica, e so as historias presentes em `prev` entram na conta."""
    m = prev.groupby("projeto")[["storypoint", "previsao"]].apply(metricas)
    med = med.merge(prev[["projeto", "issuekey"]], on=["projeto", "issuekey"])
    mae_med = med.groupby("projeto")[["storypoint", "previsao"]].apply(metricas).MAE
    v, n, p = contra_mediana(m.MAE, mae_med)
    return {"conjunto": conjunto, "solucao": nome, "origem": origem,
            "historias": historias, "projetos": len(m), **m.mean().to_dict(),
            "vence_mediana": v, "de": n, "p_wilcoxon": p}


def linha_mae(conjunto, nome, origem, mae, mae_med, historias):
    """Linha so com MAE por projeto (tabelas publicadas)."""
    v, n, p = contra_mediana(mae, mae_med)
    return {"conjunto": conjunto, "solucao": nome, "origem": origem,
            "historias": historias, "projetos": len(mae), "MAE": mae.mean(),
            "vence_mediana": v, "de": n, "p_wilcoxon": p}


def reguas_constantes(dados):
    """Media historica, mediana dos outros projetos e mediana com 80% do historico
    (o mesmo que o modelo usa para treinar), no teste do experimento."""
    proj = {arq.stem: dividir(carregar(arq)) for arq in sorted(dados.glob("*.csv"))}
    linhas = []
    for nome, (tr, val, te) in proj.items():
        outros = pd.concat([t.storypoint for o, (t, _, _) in proj.items() if o != nome])
        for cond, v in [("media historica", tr.storypoint.mean()),
                        ("mediana dos outros projetos", outros.median()),
                        ("mediana com 80%", pd.concat([tr.storypoint, val.storypoint]).median())]:
            linhas += [{"projeto": nome, "issuekey": k, "storypoint": y,
                        "condicao": cond, "previsao": v}
                       for k, y in zip(te.issuekey, te.storypoint)]
    return pd.DataFrame(linhas)


def main():
    linhas = []
    for conj, (dados, saida) in PASTAS.items():
        TESTE = "teste completo (20% finais)"
        mel = pd.read_csv(saida / "previsoes_teste.csv")
        med = mel[mel.condicao == "mediana historica"]
        cond = lambda c: mel[mel.condicao == c]
        const = reguas_constantes(dados)

        exp = pd.read_csv(saida / "resultados_completos.csv")
        ale = exp[exp.modelo == "Palpite aleatorio"].set_index("projeto").MAE
        mae_med = exp[exp.modelo == "Mediana historica"].set_index("projeto").MAE
        linhas.append(linha_mae(conj, "palpite aleatorio", "experimento.py", ale, mae_med, TESTE))
        linhas.append(linha_previsoes(conj, "media historica", "experimento.py",
                                      const[const.condicao == "media historica"], med, TESTE))
        linhas.append(linha_previsoes(conj, "mediana historica", "experimento.py", med, med, TESTE))
        # a mediana historica usa os 60% mais antigos (como a replicacao); os modelos
        # treinam com 80%. Esta linha da a mediana com o mesmo historico dos modelos
        linhas.append(linha_previsoes(conj, "mediana com o mesmo historico do modelo (80%)",
                                      "tabela_central.py",
                                      const[const.condicao == "mediana com 80%"], med, TESTE))
        linhas.append(linha_previsoes(conj, "TF-IDF + SVR (modelo do software)", "previsoes.py",
                                      cond("base (TF-IDF + SVR)"), med, TESTE))
        linhas.append(linha_previsoes(conj, "TF-IDF + regressor (original)", "previsoes.py",
                                      cond("TF-IDF original (experimento principal)"), med, TESTE))
        if conj == "Deep-SE":
            pub = pd.read_csv(REF / "deepse_resultados_publicados.csv").pivot(
                index="projeto", columns="tecnica", values="MAE")
            linhas.append(linha_mae(conj, "Deep-SE como publicado (contestado)",
                                    "Choetkiertikul et al., Tab. II", pub["LD-RNN"],
                                    pub["Mediana"], "teste deles (corte no teste)"))
            rp = pd.read_csv(REF / "deepse_replicacao_tawosi.csv")
            for col, rot in [("MAE_NoCut", "Deep-SE replicado, sem teto"),
                             ("MAE_CutTrain", "Deep-SE replicado, teto so no treino")]:
                t = rp.pivot(index="projeto", columns="tecnica", values=col)
                linhas.append(linha_mae(conj, rot, "Tawosi et al. 2022, Tab. 3",
                                        t["Deep-SE"], t["Mediana"], "mesma divisao cronologica"))
            # o GPT2SP com a conta do erro corrigida (relatorio de replicacao, arXiv:2209.00437)
            g2 = pd.read_csv(REF / "gpt2sp_replicacao_tawosi.csv").set_index("projeto")
            linhas.append(linha_mae(conj, "GPT2SP corrigido (replicacao)",
                                    "Tawosi et al. 2022 (relatorio GPT2SP), Tab. 1",
                                    g2.GPT2SP, g2.Mediana, "mesma divisao cronologica"))
            sh = pd.read_csv(REF / "shetty_llm_teste.csv")
            for c in sh.condicao.unique():
                linhas.append(linha_previsoes(conj, f"LLM: {c}", "Shetty et al. 2026, recalculado",
                                              sh[sh.condicao == c], med, TESTE))
        else:
            t5 = pd.read_csv(REF / "tawosi_tabela5.csv").set_index("projeto")
            linhas.append(linha_mae(conj, "Deep-SE (replicacao)", "Tawosi et al. 2022, Tab. 5",
                                    t5.DeepSE, t5.Mediana, "mesma divisao cronologica"))
            linhas.append(linha_mae(conj, "TF/IDF-SVM (replicacao)", "Tawosi et al. 2022, Tab. 5",
                                    t5.TFIDF_SVM, t5.Mediana, "mesma divisao cronologica"))

        linhas.append(linha_previsoes(conj, "regua do dia zero: mediana dos outros projetos",
                                      "tabela_central.py",
                                      const[const.condicao == "mediana dos outros projetos"],
                                      med, TESTE))

        # cold start com LLM: so no TAWOS, na amostra que o LLM cobriu
        arq_llm = next((saida / f"cold_start_previsoes_llm{suf}_tipica.csv"
                        for suf in ["_lote", "_lote_reduzida", "", "_reduzida"]
                        if (saida / f"cold_start_previsoes_llm{suf}_tipica.csv").exists()), None)
        if arq_llm is not None:
            llm = pd.read_csv(arq_llm)
            reg = pd.read_csv(saida / "cold_start_previsoes_reguas.csv")
            cob = llm[["projeto", "issuekey"]].drop_duplicates()
            reg = reg.merge(cob, on=["projeto", "issuekey"])
            amostra = f"amostra do cold start ({len(cob)} historias)"
            med_cs = reg[reg.condicao == "mediana da equipe"]
            modelo = llm.modelo_respondeu.dropna().iloc[0]
            # o modelo local, uma historia por chamada, nas mesmas historias
            local = saida / "cold_start_previsoes_llm_ollama_reduzida_tipica.csv"
            fontes = [(llm, modelo, f"{arq_llm.name}")]
            if local.exists():
                loc = pd.read_csv(local).merge(cob, on=["projeto", "issuekey"])
                fontes.append((loc, f"{loc.modelo.iloc[0]}, local", local.name))
            for fonte, rot_modelo, origem in fontes:
                for c in ["LLM sem referencia", "LLM com referencias (prompt unico)",
                          "LLM com referencias (busca binaria)"]:
                    linhas.append(linha_previsoes(conj, f"{c} [{rot_modelo}]", origem,
                                                  fonte[fonte.condicao == c], med_cs, amostra))
            for c, rot in [("mediana da equipe", "mediana historica (mesma amostra)"),
                           ("mediana dos outros projetos", "regua do dia zero (mesma amostra)"),
                           ("referencia mais parecida (TF-IDF)",
                            "referencia mais parecida por TF-IDF (mesma amostra)")]:
                sel = reg[(reg.condicao == c) & (reg.referencias.isin(["-", "tipica"]))]
                linhas.append(linha_previsoes(conj, rot, "cold_start_reguas.py", sel, med_cs, amostra))

    t = pd.DataFrame(linhas)
    cols = ["conjunto", "solucao", "origem", "historias", "projetos", "MAE", "MdAE",
            "cartao_exato", "dentro_1", "correlacao", "vence_mediana", "de", "p_wilcoxon"]
    t = t[cols]
    t.to_csv(RES / "tabela_central.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        for conj, g in t.groupby("conjunto", sort=False):
            print(f"\n=== {conj} ===")
            print(g.drop(columns=["conjunto", "origem"]).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {RES / 'tabela_central.csv'}")


if __name__ == "__main__":
    main()

"""Cold start, analise: a escada de 4 degraus e a curva de cruzamento.

Le as previsoes das reguas (cold_start_reguas.py) e do LLM (cold_start_llm.py)
e responde, para cada forma de usar o LLM com referencias (prompt unico e busca
binaria), quantos degraus da escada fixada no PLANO.md ela sobe:

  degrau 0  vence a mediana dos outros projetos      serve para alguma coisa
  degrau 1  vence o mesmo LLM sem referencia         as referencias importam
  degrau 2  vence copiar a referencia mais parecida  o LLM importa
  degrau 3  vence a mediana da propria equipe        cold start resolvido

Um degrau e vencido com Wilcoxon pareado nos projetos (p < 0,05) e o metodo a
frente no MAE medio. Se o LLM nao cobriu todas as historias (cota), todas as
condicoes sao avaliadas so nas historias que ele cobriu, para a comparacao ser
pareada.

Curva de cruzamento: a mediana das n historias mais recentes antes do teste,
com n crescente. O ponto de cruzamento e o menor n em que o MAE medio dessa
mediana fica igual ou abaixo do MAE medio do metodo. E a duracao do cold start:
a partir dali, a equipe faz melhor so com a propria mediana.

Como a mediana de poucas historias oscila muito, a curva nao e monotona e a
definicao acima pode cruzar por sorte em n = 1. Por isso reporta-se tambem o
cruzamento estavel: o menor n a partir do qual a mediana fica sempre igual ou
abaixo do metodo. Essa segunda definicao foi acrescentada depois de ver o piloto
local (2 projetos), antes da execucao com o Gemini.

Uso:  CONJUNTO=tawos python src/cold_start_analise.py [--provedor gemini|ollama] [--piloto|--reduzida]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

from conjunto import SAIDA
from sonda_comum import cartao, salvar

METODOS = ["LLM com referencias (prompt unico)", "LLM com referencias (busca binaria)"]
DEGRAUS = [
    (0, "serve para alguma coisa", "mediana dos outros projetos", "-"),
    (1, "as referencias importam", "LLM sem referencia", "tipica"),
    (2, "o LLM importa", "referencia mais parecida (TF-IDF)", "tipica"),
    (3, "cold start resolvido", "mediana da equipe", "-"),
]


def metricas_projeto(g):
    y, p = g.storypoint.values.astype(float), g.previsao.values.astype(float)
    e = np.abs(y - p)
    r = pearsonr(p, y)[0] if p.std() > 1e-9 and y.std() > 1e-9 else np.nan
    return pd.Series({"MAE": e.mean(), "MdAE": np.median(e),
                      "cartao_exato": (cartao(p) == cartao(y)).mean() * 100,
                      "dentro_1": (e <= 1).mean() * 100, "correlacao": r, "n": len(g)})


def comparar(mae, a, b):
    """Wilcoxon pareado entre projetos; None se as duas colunas forem identicas."""
    d = mae[a] - mae[b]
    p = wilcoxon(mae[a], mae[b]).pvalue if (d != 0).any() else np.nan
    return {"vitorias": int((d < 0).sum()), "empates": int((d == 0).sum()),
            "MAE_metodo": mae[a].mean(), "MAE_regua": mae[b].mean(), "p": p}


def lote_vs_unitario(lote, prov, suf):
    """Efeito de mandar 20 historias por chamada: compara, historia a historia, com
    as mesmas historias feitas uma por chamada (cold_start_llm.py --so-cache)."""
    p = prov.replace("_lote", "")
    candidatos = [SAIDA / f"cold_start_previsoes_llm{p}{s}_tipica.csv"
                  for s in ["_reduzida", "_unitario_cache"]]
    existentes = [c for c in candidatos if c.exists()]
    if not existentes:
        print("\n(sem previsoes unitarias: comparacao lote x unitario nao feita)")
        return
    uni = pd.concat([pd.read_csv(c) for c in existentes]).drop_duplicates(
        subset=["projeto", "issuekey", "condicao"])
    m = lote.merge(uni, on=["projeto", "issuekey", "condicao"], suffixes=("_lote", "_unit"))
    m["erro_lote"] = (m.storypoint_lote - m.previsao_lote).abs()
    m["erro_unit"] = (m.storypoint_lote - m.previsao_unit).abs()
    linhas = []
    for cond, g in m.groupby("condicao"):
        d = g.erro_lote - g.erro_unit
        linhas.append({"condicao": cond, "historias": len(g), "projetos": g.projeto.nunique(),
                       "MAE_lote": g.erro_lote.mean(), "MAE_unitario": g.erro_unit.mean(),
                       "mesma_previsao_pct": (g.previsao_lote == g.previsao_unit).mean() * 100,
                       "p_wilcoxon_historias": wilcoxon(g.erro_lote, g.erro_unit).pvalue
                       if (d != 0).any() else np.nan})
    r = pd.DataFrame(linhas)
    print("\n=== efeito do lote: mesmas historias, uma por chamada x 20 por chamada ===")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    salvar(r, f"cold_start{prov}{suf}_lote_vs_unitario.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", default="gemini")
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--reduzida", action="store_true")
    ap.add_argument("--lote", action="store_true", help="previsoes de cold_start_llm_lote.py")
    a = ap.parse_args()
    prov = ("_lote" if a.lote else "") + ("" if a.provedor == "gemini" else f"_{a.provedor}")
    suf = "_piloto" if a.piloto else ("_reduzida" if a.reduzida else "")

    reguas = pd.read_csv(SAIDA / "cold_start_previsoes_reguas.csv")
    llm = pd.read_csv(SAIDA / f"cold_start_previsoes_llm{prov}{suf}_tipica.csv")
    modelo = llm.modelo.iloc[0]
    cobertas = llm[["projeto", "issuekey"]].drop_duplicates()
    # so as historias que o LLM cobriu, em todas as condicoes
    reguas = reguas.merge(cobertas, on=["projeto", "issuekey"])
    llm = llm.assign(referencias=np.where(llm.condicao == "LLM sem referencia", "tipica",
                                          llm.referencias))
    todas = pd.concat([reguas, llm[reguas.columns.intersection(llm.columns)]],
                      ignore_index=True)
    todas["rotulo"] = np.where(todas.referencias.isin(["-", "tipica"]), todas.condicao,
                               todas.condicao + " [" + todas.referencias + "]")

    base = todas[todas.condicao != "mediana das ultimas n"]
    met = (base.groupby(["rotulo", "projeto"])[["storypoint", "previsao"]]
           .apply(metricas_projeto).reset_index())
    mae = met.pivot(index="projeto", columns="rotulo", values="MAE")
    n_proj, n_hist = mae.shape[0], len(cobertas)

    resumo = met.groupby("rotulo")[["MAE", "MdAE", "cartao_exato", "dentro_1",
                                    "correlacao"]].mean().sort_values("MAE")
    print(f"=== {modelo}: {n_hist} historias em {n_proj} projetos ===")
    print(resumo.to_string(float_format=lambda v: f"{v:.3f}"))

    invalidas = llm.groupby("condicao").valida.apply(lambda s: int((~s).sum()))
    chamadas = llm.groupby("condicao").chamadas.mean()
    print("\nrespostas invalidas:", invalidas.to_dict())
    print("chamadas por historia:", chamadas.round(2).to_dict())

    escada = []
    for metodo in METODOS:
        for deg, sentido, regua, _ in DEGRAUS:
            c = comparar(mae, metodo, regua)
            passou = bool(c["MAE_metodo"] < c["MAE_regua"] and c["p"] < 0.05)
            escada.append({"metodo": metodo, "degrau": deg, "pergunta": sentido,
                           "regua": regua, **c, "projetos": n_proj, "passou": passou})
    esc = pd.DataFrame(escada)
    print("\n=== escada ===")
    print(esc.drop(columns=["pergunta"]).to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))

    # curva de cruzamento: mediana das n mais recentes contra cada metodo
    cur = todas[todas.condicao == "mediana das ultimas n"].copy()
    cur["erro"] = (cur.storypoint - cur.previsao).abs()
    por_n = cur.groupby(["n_historico", "projeto"]).erro.mean().unstack()
    curva = []
    for n, linha in por_n.iterrows():
        comuns = linha.dropna().index
        item = {"n": int(n), "projetos": len(comuns),
                "MAE_mediana_ultimas_n": linha[comuns].mean()}
        for m in METODOS + ["LLM sem referencia", "mediana dos outros projetos"]:
            item[f"MAE {m}"] = mae.loc[comuns, m].mean()
        curva.append(item)
    curva = pd.DataFrame(curva)
    # o maior n de cada projeto e o historico inteiro; para a media so entram os n
    # que todos os projetos tem, para nao misturar projetos diferentes
    completos = curva[curva.projetos == n_proj]
    print("\n=== curva (n com todos os projetos) ===")
    print(completos.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    def estavel(serie, alvo):
        """Menor n a partir do qual a serie (indexada por n) fica sempre <= alvo."""
        acima = serie[serie > alvo]
        if not len(acima):
            return serie.index.min()
        depois = serie[serie.index > acima.index.max()]
        return depois.index.min() if len(depois) else np.nan

    cruz = []
    for m in METODOS + ["LLM sem referencia", "mediana dos outros projetos"]:
        alvo = completos[f"MAE {m}"].iloc[0]
        serie = completos.set_index("n").MAE_mediana_ultimas_n
        ok = serie[serie <= alvo]
        primeiro, sempre = [], []
        for proj in mae.index:
            s_p = por_n[proj].dropna()
            abaixo = s_p[s_p <= mae.loc[proj, m]]
            primeiro.append(abaixo.index.min() if len(abaixo) else np.nan)
            sempre.append(estavel(s_p, mae.loc[proj, m]))
        primeiro, sempre = pd.Series(primeiro, index=mae.index), pd.Series(sempre, index=mae.index)
        cruz.append({"metodo": m, "MAE_metodo": alvo,
                     "cruzamento_n_medio": ok.index.min() if len(ok) else np.nan,
                     "cruzamento_estavel_n_medio": estavel(serie, alvo),
                     "projetos_que_cruzam": int(primeiro.notna().sum()),
                     "cruzamento_n_mediana_projetos": primeiro.median(),
                     "cruzamento_estavel_n_mediana_projetos": sempre.median(),
                     "projetos_ja_com_1_historia": int((primeiro == 1).sum()),
                     "projetos": n_proj})
    cruz = pd.DataFrame(cruz)
    print("\n=== ponto de cruzamento ===")
    print(cruz.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if a.lote:
        lote_vs_unitario(llm, prov, suf)

    nome = f"cold_start{prov}{suf}"
    salvar(met, f"{nome}_metricas.csv")
    salvar(esc, f"{nome}_escada.csv")
    salvar(curva, f"{nome}_curva.csv")
    salvar(cruz, f"{nome}_cruzamento.csv")


if __name__ == "__main__":
    main()

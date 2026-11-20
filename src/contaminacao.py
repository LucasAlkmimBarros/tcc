"""Teste de contaminacao: o LLM decorou o story point das issues publicas?

As issues do TAWOS estao em paginas publicas do Jira, que mostram o story point.
Um LLM treinado com a web pode ter visto essas paginas. Se ele decorou, qualquer
acerto do cold start pode ser memoria, nao julgamento.

O teste pergunta o story point so pelo identificador da issue (por exemplo,
MESOS-336), sem texto nenhum. Sem o texto, a unica forma de acertar acima do
acaso e lembrar daquela issue especifica.

Como se mede "acima do acaso". Um LLM que nao lembra de nada ainda pode acertar
bastante se chutar valores comuns no projeto (o identificador revela o projeto,
e o LLM pode saber que o Mesos usa 3 com frequencia). Por isso a comparacao nao
e com o sorteio uniforme: embaralham-se as respostas do proprio LLM entre as
issues do mesmo projeto, 10.000 vezes. Isso mantem tudo o que ele sabe do
projeto e destroi so a ligacao com a issue especifica. Se os acertos reais nao
passarem do percentil 95 dos embaralhados, nao ha evidencia de memoria.

Amostra: 5 issues por projeto (115), sorteadas com semente fixa da amostra de
teste do cold start.

Uso:  CONJUNTO=tawos python src/contaminacao.py [--provedor ollama --modelo ...]
"""
import argparse
from string import Template

import numpy as np
import pandas as pd

from cold_start_comum import projetos_cold_start
from cold_start_llm import MODELO_GEMINI
from experimento import SEMENTE
from llm import CotaEsgotada, carregar_prompt, extrair_json, perguntar
from sonda_comum import salvar

POR_PROJETO = 5
PERMUTACOES = 10_000
VERSAO = "contaminacao_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--modelo", default=None)
    a = ap.parse_args()
    modelo = a.modelo or (MODELO_GEMINI if a.provedor == "gemini" else "qwen2.5:14b")
    t = Template(carregar_prompt(VERSAO))

    rng = np.random.default_rng(SEMENTE)
    linhas = []
    try:
        for sigla, treino, _, teste in projetos_cold_start():
            escolhidas = teste.iloc[np.sort(rng.choice(len(teste), POR_PROJETO, replace=False))]
            for _, h in escolhidas.iterrows():
                r = perguntar(t.substitute(chave=h.issuekey), modelo, VERSAO, a.provedor)
                j = extrair_json(r["texto"])
                try:
                    v = float(j["story_points"])
                except (TypeError, KeyError, ValueError):
                    v = np.nan
                linhas.append({"projeto": sigla, "issuekey": h.issuekey,
                               "storypoint": h.storypoint, "resposta": v,
                               "mediana_equipe": treino.storypoint.median(),
                               "modelo_respondeu": r["modelo_respondeu"]})
            print(f"  {sigla:<12} feito", flush=True)
    except CotaEsgotada as e:
        print(f"!! cota esgotada; rode de novo depois\n{str(e)[:300]}")
        return

    r = pd.DataFrame(linhas)
    validas = r.dropna(subset=["resposta"]).copy()
    acertos = int((validas.resposta == validas.storypoint).sum())

    # embaralha as respostas dentro de cada projeto, mantendo o que o LLM sabe do
    # projeto e destruindo so a ligacao com a issue especifica
    rng = np.random.default_rng(SEMENTE)
    grupos = [g for _, g in validas.groupby("projeto")]
    nulos = np.empty(PERMUTACOES)
    for k in range(PERMUTACOES):
        nulos[k] = sum(int((rng.permutation(g.resposta.values) == g.storypoint.values).sum())
                       for g in grupos)
    p = float((nulos >= acertos).mean())

    mae_llm = float((validas.resposta - validas.storypoint).abs().mean())
    mae_med = float((validas.mediana_equipe - validas.storypoint).abs().mean())
    acerto_med = float((validas.mediana_equipe == validas.storypoint).mean() * 100)
    resumo = pd.DataFrame([{
        "modelo": modelo, "provedor": a.provedor, "issues": len(r),
        "respostas_validas": len(validas),
        "acertos_exatos": acertos, "acerto_exato_pct": acertos / len(validas) * 100,
        "acertos_embaralhado_media": nulos.mean(),
        "acertos_embaralhado_p95": np.percentile(nulos, 95),
        "p_valor_permutacao": p,
        "acerto_exato_mediana_equipe_pct": acerto_med,
        "MAE_llm_so_identificador": mae_llm, "MAE_mediana_equipe": mae_med,
        "respostas_distintas": validas.resposta.nunique(),
    }])
    print(resumo.T.to_string(header=False))
    print("\nLeitura: memoria so se os acertos reais passarem do p95 dos embaralhados "
          f"(p = {p:.3f}).")

    prov = "" if a.provedor == "gemini" else f"_{a.provedor}"
    salvar(r, f"contaminacao_respostas{prov}.csv")
    salvar(resumo, f"contaminacao{prov}.csv")


if __name__ == "__main__":
    main()

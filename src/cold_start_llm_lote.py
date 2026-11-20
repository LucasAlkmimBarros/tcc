"""Cold start com LLM em lote: 20 historias do mesmo projeto por chamada.

Por que existe. A faixa gratuita do Gemini da 500 chamadas por dia ao
gemini-3.5-flash-lite, e as respostas 503 (servidor sobrecarregado) parecem contar
na cota. Com uma historia por chamada (cold_start_llm.py), a amostra minima do
plano (12 projetos x 60 historias) precisaria de cerca de 2.600 chamadas: mais de
uma semana. Em lote, a mesma amostra cabe em cerca de 250 chamadas.

O que muda e o que nao muda. Mesmo modelo, mesmas referencias tipicas, mesmo texto
limpo e as mesmas tres formas (sem referencia, prompt unico, busca binaria). O que
muda: o prompt traz ate 20 historias novas (S1 ... S20), todas do mesmo projeto,
e pede uma resposta para cada, dizendo que elas sao independentes. Na busca
binaria, cada rodada agrupa as historias que precisam ser comparadas com a mesma
referencia. O LLM continua sem ver nenhum story point.

O risco do lote e uma historia influenciar a outra (o modelo pode querer
"espalhar" as respostas pela escala). Por isso o efeito e medido: as historias
que ja tinham sido feitas uma por chamada sao comparadas com a versao em lote
(cold_start_lote_vs_unitario.csv, gerado por cold_start_analise.py).

Uso:
    python src/cold_start_llm_lote.py --so-reduzida     # 12 projetos x 60 historias
    python src/cold_start_llm_lote.py                   # tudo (1.869 historias)
"""
import argparse
from string import Template

import numpy as np
import pandas as pd

from cold_start_comum import conjuntos_de_referencia, projetos_cold_start
from cold_start_llm import MODELO_GEMINI, ordem_de_execucao
from llm import CotaEsgotada, carregar_prompt, em_cache, extrair_json, perguntar
from sonda_comum import salvar

TAMANHO_LOTE = 20
P_SEM = "estimar_sem_referencia_lote_v1"
P_UNICO = "escolher_referencia_lote_v1"
P_COMP = "comparar_lote_v1"


def bloco_historias(textos, ids):
    return "\n\n".join(f"<{i}>\n{t}\n</{i}>" for i, t in zip(ids, textos))


def respostas_por_id(j, chave_lista, campo):
    """{"S1": valor, ...} a partir da resposta; ids ausentes ficam de fora."""
    out = {}
    for item in (j or {}).get(chave_lista, []) or []:
        if isinstance(item, dict) and "story" in item:
            out[str(item["story"]).strip().upper()] = item.get(campo)
    return out


class EstimadorLote:
    def __init__(self, provedor, modelo):
        self.provedor, self.modelo = provedor, modelo
        self.t = {v: Template(carregar_prompt(v)) for v in (P_SEM, P_UNICO, P_COMP)}
        self.chamadas_novas = 0
        self.chamadas = 0

    def _perguntar(self, prompt, versao):
        if not em_cache(prompt, self.modelo, versao, self.provedor):
            self.chamadas_novas += 1
        self.chamadas += 1
        r = perguntar(prompt, self.modelo, versao, self.provedor)
        return extrair_json(r["texto"]), r["modelo_respondeu"]

    def sem_referencia(self, textos):
        ids = [f"S{i + 1}" for i in range(len(textos))]
        j, mr = self._perguntar(self.t[P_SEM].substitute(historias=bloco_historias(textos, ids)), P_SEM)
        resp = respostas_por_id(j, "estimates", "story_points")
        out = []
        for i in ids:
            try:
                out.append((float(resp[i]), True))
            except (KeyError, TypeError, ValueError):
                out.append((None, False))
        return out, mr

    def prompt_unico(self, textos, textos_refs, valores, versao=P_UNICO):
        """versao: outro prompt com os mesmos campos (o de referencias agrupadas)."""
        if versao not in self.t:
            self.t[versao] = Template(carregar_prompt(versao))
        ids = [f"S{i + 1}" for i in range(len(textos))]
        refs = "\n\n".join(f"<R{i + 1}>\n{t}\n</R{i + 1}>" for i, t in enumerate(textos_refs))
        prompt = self.t[versao].substitute(referencias=refs, ultima=len(textos_refs),
                                           historias=bloco_historias(textos, ids))
        j, mr = self._perguntar(prompt, versao)
        resp = respostas_por_id(j, "answers", "reference")
        out = []
        for i in ids:
            try:
                k = int(str(resp[i]).strip().upper().lstrip("R")) - 1
                if not 0 <= k < len(valores):
                    raise ValueError
                out.append((valores[k], True))
            except (KeyError, TypeError, ValueError):
                out.append((None, False))
        return out, mr

    def busca_binaria(self, textos, textos_refs, valores):
        """Busca binaria para todas as historias do lote ao mesmo tempo.

        A cada rodada, as historias ainda em busca sao agrupadas pela referencia
        do meio do seu intervalo, e cada grupo vira uma chamada. Mesma regra da
        versao unitaria: "same" para; fora da escala fica no extremo; entre duas
        referencias, o ponto medio dos valores.
        """
        n, k = len(textos), len(valores)
        lo, hi = [0] * n, [k - 1] * n
        fim, valida, chamadas = [None] * n, [True] * n, [0] * n
        mr = None
        while True:
            ativos = [i for i in range(n) if fim[i] is None and lo[i] <= hi[i]]
            if not ativos:
                break
            grupos = {}
            for i in ativos:
                grupos.setdefault((lo[i] + hi[i]) // 2, []).append(i)
            for meio, membros in sorted(grupos.items()):
                ids = [f"S{m + 1}" for m in range(len(membros))]
                prompt = self.t[P_COMP].substitute(
                    referencia=textos_refs[meio],
                    historias=bloco_historias([textos[i] for i in membros], ids))
                j, mr = self._perguntar(prompt, P_COMP)
                resp = respostas_por_id(j, "answers", "answer")
                for sid, i in zip(ids, membros):
                    chamadas[i] += 1
                    a = str(resp.get(sid, "")).strip().lower()
                    if a not in ("less", "same", "more"):
                        valida[i], a = False, "same"
                    if a == "same":
                        fim[i] = valores[meio]
                    elif a == "less":
                        hi[i] = meio - 1
                    else:
                        lo[i] = meio + 1
        for i in range(n):
            if fim[i] is None:
                if hi[i] < 0:
                    fim[i] = valores[0]
                elif lo[i] > k - 1:
                    fim[i] = valores[-1]
                else:
                    fim[i] = (valores[hi[i]] + valores[lo[i]]) / 2
        return [(fim[i], valida[i]) for i in range(n)], chamadas, mr


def lotes(ordem):
    """Agrupa a ordem de execucao em lotes de ate 20 historias do mesmo projeto,
    sem misturar a amostra reduzida com o resto."""
    out, atual = [], []
    for s, i in ordem:
        if atual and (atual[0][0] != s or len(atual) == TAMANHO_LOTE):
            out.append(atual)
            atual = []
        atual.append((s, i))
    if atual:
        out.append(atual)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--so-reduzida", action="store_true")
    ap.add_argument("--max-lotes", type=int, default=None, help="para testes")
    a = ap.parse_args()
    modelo = a.modelo or (MODELO_GEMINI if a.provedor == "gemini" else "qwen2.5:14b")

    dados = list(projetos_cold_start())
    por_sigla = {s: (tr, te) for s, tr, _, te in dados}
    mediana_outros = {s: float(pd.concat([t.storypoint for o, t, _, _ in dados if o != s]).median())
                      for s, _, _, _ in dados}
    refs = {s: conjuntos_de_referencia(tr)["tipica"] for s, (tr, _) in por_sigla.items()}
    ordem, reduzida = ordem_de_execucao(dados, False)
    if a.so_reduzida:
        ordem = ordem[:len(reduzida)]

    est = EstimadorLote(a.provedor, modelo)
    linhas, parou = [], None
    todos = lotes(ordem)[:a.max_lotes]
    try:
        for n_lote, lote in enumerate(todos):
            sigla = lote[0][0]
            treino, teste = por_sigla[sigla]
            hs = [teste.iloc[i] for _, i in lote]
            textos = [h.texto_limpo for h in hs]
            valores = sorted(refs[sigla])
            textos_refs = [treino.loc[refs[sigla][v], "texto_limpo"] for v in valores]

            sem, mr1 = est.sem_referencia(textos)
            uni, mr2 = est.prompt_unico(textos, textos_refs, valores)
            bin_, ch, mr3 = est.busca_binaria(textos, textos_refs, valores)
            for k, h in enumerate(hs):
                base = {"projeto": sigla, "issuekey": h.issuekey, "storypoint": h.storypoint,
                        "referencias": "tipica", "provedor": a.provedor, "modelo": modelo,
                        "amostra_reduzida": (sigla, lote[k][1]) in reduzida, "lote": n_lote}
                for cond, (v, ok), c, mr in [
                        ("LLM sem referencia", sem[k], 1 / len(hs), mr1),
                        ("LLM com referencias (prompt unico)", uni[k], 1 / len(hs), mr2),
                        ("LLM com referencias (busca binaria)", bin_[k], ch[k] / len(hs), mr3)]:
                    linhas.append({**base, "condicao": cond, "chamadas": c, "valida": ok,
                                   "modelo_respondeu": mr,
                                   "previsao": v if v is not None else mediana_outros[sigla]})
            print(f"  lote {n_lote + 1}/{len(todos)} {sigla}  "
                  f"(chamadas novas: {est.chamadas_novas})", flush=True)
    except CotaEsgotada as e:
        parou = str(e)
        print(f"\n!! cota diaria esgotada no lote {n_lote + 1}/{len(todos)}; "
              f"rode de novo depois (o cache guarda o que ja foi feito)\n{parou[:300]}")

    r = pd.DataFrame(linhas)
    prov = "" if a.provedor == "gemini" else f"_{a.provedor}"
    suf = "_reduzida" if a.so_reduzida else ""
    salvar(r, f"cold_start_previsoes_llm_lote{prov}{suf}_tipica.csv")
    if len(r):
        r["erro"] = (r.storypoint - r.previsao).abs()
        print(r.groupby("condicao").agg(MAE=("erro", "mean"), validas=("valida", "mean"),
                                        n=("erro", "size"))
              .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"chamadas nesta execucao: {est.chamadas} (novas: {est.chamadas_novas})"
          + ("  (INCOMPLETA)" if parou else ""))


if __name__ == "__main__":
    main()

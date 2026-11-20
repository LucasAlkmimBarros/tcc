"""Cold start, parte com LLM: tres formas de estimar sem historico.

    sem referencia   o LLM recebe so a historia e escolhe um cartao de Planning
                     Poker. E o degrau 1 da escada: o mesmo LLM, sem a escala
                     da equipe.
    prompt unico     o LLM recebe as referencias da equipe em ordem de esforco
                     (R1 menor ... Rk maior) e diz de qual a historia nova esta
                     mais perto. O codigo troca a referencia pelo valor dela.
    busca binaria    o LLM compara a historia com uma referencia de cada vez
                     ("menos, parecido ou mais esforco?"), comecando pela do
                     meio, como numa busca binaria: cerca de 3 perguntas por
                     historia. Se terminar entre duas referencias, a previsao
                     e o ponto medio entre os valores delas (regra fixada antes
                     de rodar).

O LLM nunca ve um story point: nem o da historia avaliada nem o das referencias.
As referencias aparecem so em ordem, e e o codigo que sabe o valor de cada uma.
Por isso a resposta do LLM so pode ser um julgamento de tamanho relativo.

Ordem das chamadas. A cota gratuita pode acabar antes do fim. Para que um corte
deixe uma amostra utilizavel, as historias sao processadas numa ordem fixa: antes
a amostra reduzida (12 projetos sorteados x 60 historias, o minimo aceito no
plano), depois o resto. Cada historia passa pelas tres formas antes da proxima.

Resposta invalida (sem JSON legivel): a previsao cai para a regua do dia zero
(mediana dos outros projetos) e a linha fica marcada em `valida`.

Uso:
    python src/cold_start_llm.py                       # Gemini, execucao completa
    python src/cold_start_llm.py --piloto              # 2 projetos, 25 historias cada
    python src/cold_start_llm.py --provedor ollama --modelo qwen2.5:14b --piloto
    python src/cold_start_llm.py --referencias sorteio0  # outras referencias
    python src/cold_start_llm.py --so-reduzida           # so a amostra reduzida
    python src/cold_start_llm.py --so-reduzida --projetos MDL,XD   # encher o cache em paralelo
    python src/cold_start_llm.py --so-cache              # so o que ja esta no cache
"""
import argparse
from string import Template

import numpy as np
import pandas as pd

from cold_start_comum import conjuntos_de_referencia, projetos_cold_start
from llm import CotaEsgotada, carregar_prompt, extrair_json, perguntar
from sonda_comum import salvar

MODELO_GEMINI = "gemini-3.5-flash-lite"  # o 3.6-flash tem cota de 20 chamadas/dia (PLANO.md)
PILOTO = {"MESOS": 25, "TISTUD": 25}
REDUZIDA_PROJETOS, REDUZIDA_HISTORIAS, SEMENTE_REDUZIDA = 12, 60, 7

P_SEM_REF = "estimar_sem_referencia_v1"
P_UNICO = "escolher_referencia_v1"
P_COMPARAR = "comparar_v1"
CARTOES = [0.5, 1, 2, 3, 5, 8, 13, 20, 40, 100]


class ForaDoCache(Exception):
    """Modo --so-cache: a historia tem pergunta ainda nao respondida."""


class Estimador:
    def __init__(self, provedor, modelo):
        self.provedor, self.modelo = provedor, modelo
        self.t_sem = Template(carregar_prompt(P_SEM_REF))
        self.t_unico = Template(carregar_prompt(P_UNICO))
        self.t_comp = Template(carregar_prompt(P_COMPARAR))
        self.chamadas_novas = 0
        self.so_cache = False

    def _perguntar(self, prompt, versao):
        from llm import em_cache
        if not em_cache(prompt, self.modelo, versao, self.provedor):
            if self.so_cache:
                raise ForaDoCache
            self.chamadas_novas += 1
        r = perguntar(prompt, self.modelo, versao, self.provedor)
        return extrair_json(r["texto"]), r

    def sem_referencia(self, historia):
        j, r = self._perguntar(self.t_sem.substitute(historia=historia), P_SEM_REF)
        try:
            v = float(j["story_points"])
            return v, 1, True, r["modelo_respondeu"]
        except (TypeError, KeyError, ValueError):
            return None, 1, False, r["modelo_respondeu"]

    def prompt_unico(self, historia, textos_refs, valores):
        bloco = "\n\n".join(f"<R{i + 1}>\n{t}\n</R{i + 1}>" for i, t in enumerate(textos_refs))
        prompt = self.t_unico.substitute(referencias=bloco, historia=historia,
                                         ultima=len(textos_refs))
        j, r = self._perguntar(prompt, P_UNICO)
        try:
            i = int(str(j["reference"]).strip().upper().lstrip("R")) - 1
            if not 0 <= i < len(valores):
                raise ValueError
            return valores[i], 1, True, r["modelo_respondeu"]
        except (TypeError, KeyError, ValueError):
            return None, 1, False, r["modelo_respondeu"]

    def busca_binaria(self, historia, textos_refs, valores):
        lo, hi, chamadas, valida, modelo = 0, len(valores) - 1, 0, True, None
        while lo <= hi:
            meio = (lo + hi) // 2
            prompt = self.t_comp.substitute(referencia=textos_refs[meio], historia=historia)
            j, r = self._perguntar(prompt, P_COMPARAR)
            chamadas, modelo = chamadas + 1, r["modelo_respondeu"]
            resp = str((j or {}).get("answer", "")).strip().lower()
            if resp not in ("less", "same", "more"):
                valida, resp = False, "same"
            if resp == "same":
                return valores[meio], chamadas, valida, modelo
            if resp == "less":
                hi = meio - 1
            else:
                lo = meio + 1
        if hi < 0:
            return valores[0], chamadas, valida, modelo
        if lo > len(valores) - 1:
            return valores[-1], chamadas, valida, modelo
        return (valores[hi] + valores[lo]) / 2, chamadas, valida, modelo


def ordem_de_execucao(dados, piloto):
    """Lista (sigla, posicao na amostra) na ordem em que as historias serao feitas."""
    if piloto:
        return [(s, i) for s, n in PILOTO.items() for i in range(n)]
    tamanhos = {s: len(te) for s, _, _, te in dados}
    rng = np.random.default_rng(SEMENTE_REDUZIDA)
    elegiveis = sorted(s for s, n in tamanhos.items() if n >= REDUZIDA_HISTORIAS)
    reduzida = sorted(rng.choice(elegiveis, REDUZIDA_PROJETOS, replace=False))
    primeiro = []
    for s in reduzida:
        idx = sorted(rng.choice(tamanhos[s], REDUZIDA_HISTORIAS, replace=False))
        primeiro += [(s, int(i)) for i in idx]
    feitos = set(primeiro)
    resto = [(s, i) for s in sorted(tamanhos) for i in range(tamanhos[s])
             if (s, i) not in feitos]
    return primeiro + resto, set(primeiro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--referencias", default="tipica")
    ap.add_argument("--formas", default="sem,unico,binaria")
    ap.add_argument("--so-reduzida", action="store_true",
                    help="para ao fim da amostra reduzida (12 projetos x 60 historias)")
    ap.add_argument("--so-cache", action="store_true",
                    help="nao chama a API: so as historias ja respondidas por inteiro")
    ap.add_argument("--projetos", default=None,
                    help="so estes projetos (separados por virgula), so para encher o cache "
                         "em paralelo; nao grava CSV")
    a = ap.parse_args()
    modelo = a.modelo or (MODELO_GEMINI if a.provedor == "gemini" else "qwen2.5:14b")
    formas = a.formas.split(",")

    dados = list(projetos_cold_start())
    por_sigla = {s: (tr, te) for s, tr, _, te in dados}
    mediana_outros = {s: float(pd.concat([t.storypoint for o, t, _, _ in dados if o != s]).median())
                      for s, _, _, _ in dados}
    refs = {s: conjuntos_de_referencia(tr)[a.referencias] for s, (tr, _) in por_sigla.items()}

    if a.piloto:
        ordem, reduzida = ordem_de_execucao(dados, True), set()
    else:
        ordem, reduzida = ordem_de_execucao(dados, False)
        if a.so_reduzida:
            ordem = ordem[:len(reduzida)]
    if a.projetos:
        ordem = [(s, i) for s, i in ordem if s in a.projetos.split(",")]

    est = Estimador(a.provedor, modelo)
    est.so_cache = a.so_cache
    linhas, parou = [], None
    try:
        for k, (sigla, i) in enumerate(ordem):
            antes = len(linhas)
            treino, teste = por_sigla[sigla]
            h = teste.iloc[i]
            valores = sorted(refs[sigla])
            textos = [treino.loc[refs[sigla][v], "texto_limpo"] for v in valores]
            base = {"projeto": sigla, "issuekey": h.issuekey, "storypoint": h.storypoint,
                    "referencias": a.referencias, "provedor": a.provedor,
                    "modelo": modelo, "amostra_reduzida": (sigla, i) in reduzida}
            try:
                for forma in formas:
                    if forma == "sem":
                        v, c, ok, mr = est.sem_referencia(h.texto_limpo)
                        cond = "LLM sem referencia"
                    elif forma == "unico":
                        v, c, ok, mr = est.prompt_unico(h.texto_limpo, textos, valores)
                        cond = "LLM com referencias (prompt unico)"
                    else:
                        v, c, ok, mr = est.busca_binaria(h.texto_limpo, textos, valores)
                        cond = "LLM com referencias (busca binaria)"
                    linhas.append({**base, "condicao": cond, "chamadas": c, "valida": ok,
                                   "modelo_respondeu": mr,
                                   "previsao": v if v is not None else mediana_outros[sigla]})
            except ForaDoCache:
                # modo --so-cache: a historia entra inteira ou nao entra
                del linhas[antes:]
                continue
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(ordem)} historias  (chamadas novas: {est.chamadas_novas})",
                      flush=True)
    except CotaEsgotada as e:
        parou = str(e)
        print(f"\n!! cota diaria esgotada depois de {len(linhas)} previsoes; "
              f"rode de novo amanha (o cache guarda o que ja foi feito)\n{parou[:300]}")

    if a.projetos:
        print(f"cache preenchido para {a.projetos}: {len(linhas)} previsoes, "
              f"{est.chamadas_novas} chamadas novas" + ("  (INCOMPLETA)" if parou else ""))
        return
    r = pd.DataFrame(linhas)
    sufixo = "_piloto" if a.piloto else ("_reduzida" if a.so_reduzida else "")
    if a.so_cache:
        sufixo = "_unitario_cache"
    prov = "" if a.provedor == "gemini" else f"_{a.provedor}"
    nome = f"cold_start_previsoes_llm{prov}{sufixo}_{a.referencias}.csv"
    salvar(r, nome)
    if len(r):
        r["erro"] = (r.storypoint - r.previsao).abs()
        print(r.groupby("condicao").agg(MAE=("erro", "mean"), chamadas=("chamadas", "mean"),
                                        validas=("valida", "mean"), n=("erro", "size"))
              .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"chamadas novas nesta execucao: {est.chamadas_novas}"
          + ("  (INCOMPLETA)" if parou else ""))


if __name__ == "__main__":
    main()

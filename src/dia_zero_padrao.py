"""Dia zero com historias padrao (PLANO.md, "Dia zero com historias padrao: fixado
antes de rodar", 2026-11-15).

Na base construida (RoomBook), o mesmo LLM do dia zero medido no TAWOS compara as
50 historias de teste com tres tipos de referencia:

    proprias      a historia mais tipica de cada cartao nas 200 mais antigas
                  (o que a pagina monta em "Simular com o historico carregado")
    padrao (en)   as 10 historias padrao, com os pontos que um Gemini deu a elas
                  numa conversa nova do app (um voto por modelo: 3.5 Flash-Lite e
                  4 Hydrogen; o 3.1 Pro, que estimou a base, nao estava disponivel)
    padrao (pt)   as mesmas, em portugues, com os mesmos votos (so se reporta)

Duas historias no mesmo cartao: "primeira" usa a primeira da lista; "todas" junta
as duas na mesma referencia, com o prompt de referencias agrupadas.

e com tres reguas: o LLM sem referencia, o valor tipico de outras equipes (a
mediana dos pontos dos 39 projetos publicos) e a mediana da propria equipe.

Os votos ficam em base_sintetica/votos_padrao/voto_<k>.jsonl, uma linha JSON por
historia ({"id": "PAD-01", "story_point": 5}), coladas da resposta do Gemini.
Rodar sem votos faz so as proprias e as reguas.

Uso:  CONJUNTO=sintetica python src/dia_zero_padrao.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from caracterizar import carregar
from cold_start_comum import limpar, referencias
from cold_start_llm import MODELO_GEMINI
from cold_start_llm_lote import P_UNICO, TAMANHO_LOTE, EstimadorLote
from conjunto import RAIZ
from historias_padrao import ler_votos, referencias_dos_votos
from sonda_comum import dividir_historico_teste, metricas, projetos, salvar

VOTOS = RAIZ / "base_sintetica" / "votos_padrao"
PROPRIAS = "proprias"
P_GRUPO = "escolher_referencia_grupo_lote_v1"


def valor_tipico_de_outras_equipes():
    """A mediana dos pontos dos 39 projetos publicos (16 do Deep-SE e 23 do TAWOS)."""
    arqs = sorted((RAIZ / "dados").glob("*.csv")) + sorted((RAIZ / "tawos" / "dados").glob("*.csv"))
    assert len(arqs) == 39, len(arqs)
    return float(pd.concat([carregar(a).storypoint for a in arqs]).median())


def p_wilcoxon(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:               # todas as diferencas zero
        return 1.0


def lotes(n):
    return [list(range(i, min(i + TAMANHO_LOTE, n))) for i in range(0, n, TAMANHO_LOTE)]


def comparar(est, textos, refs, reserva, versao=P_UNICO):
    """Previsao do prompt unico, em lotes de 20. Resposta ilegivel vira a reserva."""
    valores = sorted(refs)
    textos_refs = [limpar(refs[v]) for v in valores]
    prev, ok = [], []
    for idx in lotes(len(textos)):
        out, _ = est.prompt_unico([textos[i] for i in idx], textos_refs, valores, versao)
        prev += [v if v is not None else reserva for v, _ in out]
        ok += [o for _, o in out]
    return prev, ok


def main():
    (nome, d), = list(projetos())
    hist, teste = dividir_historico_teste(d)
    y = teste.storypoint.values.astype(float)
    textos = [limpar(t) for t in teste.texto]
    tipico = valor_tipico_de_outras_equipes()
    print(f"{nome}: {len(hist)} no historico, {len(teste)} no teste; "
          f"valor tipico de outras equipes = {tipico:g}")

    h = hist.copy()
    h["texto_limpo"] = [limpar(t) for t in h.texto]
    conjuntos = {PROPRIAS: ({v: h.loc[i, "texto"] for v, i in referencias(h, "tipica").items()},
                            P_UNICO)}
    for arq in sorted(VOTOS.glob("voto_*.jsonl")):
        k = arq.stem.split("_", 1)[1]
        votos = ler_votos(arq)
        assert len(votos) == 10, (arq, votos)
        print(f"voto {k}: " + ", ".join(f"{i[-2:]}={v:g}" for i, v in votos.items()))
        for lingua in ("en", "pt"):
            for forma, todas, versao in (("primeira", False, P_UNICO), ("todas", True, P_GRUPO)):
                conjuntos[f"padrao ({lingua}, {forma}) voto {k}"] = (
                    referencias_dos_votos(votos, lingua, todas=todas), versao)

    est = EstimadorLote("gemini", MODELO_GEMINI)
    prev = {"valor tipico de outras equipes": (np.full(len(y), tipico), [True] * len(y)),
            "mediana da equipe": (np.full(len(y), float(hist.storypoint.median())), [True] * len(y))}
    sem, ok_sem = [], []
    for idx in lotes(len(textos)):
        out, _ = est.sem_referencia([textos[i] for i in idx])
        sem += [v if v is not None else tipico for v, _ in out]
        ok_sem += [o for _, o in out]
    prev["LLM sem referencia"] = (np.array(sem), ok_sem)
    cartoes = {}
    for cond, (refs, versao) in conjuntos.items():
        cartoes[cond] = sorted(refs)
        p, ok = comparar(est, textos, refs, tipico, versao)
        prev[cond] = (np.array(p, dtype=float), ok)
    print(f"chamadas: {est.chamadas} (novas: {est.chamadas_novas})")

    e_prop = np.abs(y - prev[PROPRIAS][0])
    linhas, por_historia = [], []
    for cond, (p, ok) in prev.items():
        e = np.abs(y - p)
        pv = p_wilcoxon(e, e_prop) if cond != PROPRIAS else np.nan
        par = cond.replace(", todas)", ", primeira)")
        e_par = np.abs(y - prev[par][0]) if par != cond else None
        linhas.append({"condicao": cond, "cartoes_de_referencia": " ".join(f"{c:g}" for c in cartoes.get(cond, [])),
                       **metricas(y, p), "validas": float(np.mean(ok)),
                       "erra_menos_que_proprias_em": int((e < e_prop).sum()),
                       "erra_mais_que_proprias_em": int((e > e_prop).sum()),
                       "p_wilcoxon_contra_proprias": pv,
                       "MAE_primeira": float(e_par.mean()) if e_par is not None else np.nan,
                       "p_wilcoxon_todas_contra_primeira":
                           p_wilcoxon(e, e_par) if e_par is not None else np.nan})
        por_historia.append(pd.DataFrame({"issuekey": teste.issuekey.values, "storypoint": y,
                                          "condicao": cond, "previsao": p, "valida": ok}))
    r = pd.DataFrame(linhas)
    with pd.option_context("display.width", 220, "display.max_colwidth", 30):
        print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    salvar(r, "dia_zero_padrao.csv")
    salvar(pd.concat(por_historia, ignore_index=True), "dia_zero_padrao_previsoes.csv")


if __name__ == "__main__":
    main()

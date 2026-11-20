"""LLMs sem treino no Deep-SE: as previsoes publicadas por Shetty et al. (2026),
recalculadas nas nossas historias de teste.

Por que existe. Shetty et al. ("Agile Story Point Estimation with Large Language
Models", arXiv:2603.06276) pediram a quatro LLMs que estimassem as historias do
benchmark Deep-SE sem nenhum treino, e publicaram as previsoes historia a
historia (github.com/hil-se/LLM_Story_Point). O artigo reporta so correlacao, e
calculada sobre o projeto inteiro. Para entrar na mesma tabela que o resto,
recalcula-se o erro so nas historias do nosso teste (os ultimos 20% de cada
projeto), casando pelo identificador da issue.

O que se baixa: as previsoes sem exemplo (RQ1, quatro modelos) e as do DeepSeek
com 5 exemplos espalhados pela escala (RQ2, "Scale"), que e o mais proximo de
"LLM com referencias". Fixa-se o commit do repositorio para o numero nao mudar
se os autores atualizarem os arquivos. Grava-se so o necessario (identificador,
modelo, previsao) em referencia/shetty_llm_teste.csv; os arquivos completos ficam
em cache/shetty/ (nao versionado).

Uso:  python src/llm_publicado.py
"""
import io
import urllib.request

import pandas as pd

from conjunto import RAIZ
from sonda_comum import dividir_historico_teste, projetos

COMMIT = "37e73fac6a112d568878606da201ce0d22aed5d6"
BASE = f"https://raw.githubusercontent.com/hil-se/LLM_Story_Point/{COMMIT}/results"
CONDICOES = {
    "DeepSeek-V3.2 sem exemplo": ("RQ1_ZeroShot_Direct/deepseek", "deepseek", "ZeroShot"),
    "Kimi K2 sem exemplo": ("RQ1_ZeroShot_Direct/kimi", "kimi", "ZeroShot"),
    "OpenAI sem exemplo": ("RQ1_ZeroShot_Direct/openai", "openai", "ZeroShot"),
    "Gemini sem exemplo": ("RQ1_ZeroShot_Direct/gemini", "gemini", "ZeroShot"),
    "DeepSeek-V3.2 com 5 exemplos (Scale)": ("RQ2_FewShot_Direct/deepseek/Scale", "deepseek", "Scale"),
}
CACHE = RAIZ / "cache" / "shetty"
SAIDA = RAIZ / "referencia" / "shetty_llm_teste.csv"


def baixar(pasta, prefixo, projeto, sufixo):
    """Le um arquivo de previsoes, do cache local ou do GitHub (commit fixo)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for nome in [f"{projeto}_{prefixo}_{sufixo}.csv",
                 f"{projeto}_filtered_{prefixo}_{sufixo}.csv"]:
        local = CACHE / pasta.replace("/", "_") / nome
        if local.exists():
            return pd.read_csv(local)
        try:
            with urllib.request.urlopen(f"{BASE}/{pasta}/{nome}", timeout=60) as r:
                bruto = r.read()
        except urllib.error.HTTPError:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(bruto)
        return pd.read_csv(io.BytesIO(bruto))
    raise FileNotFoundError(f"{pasta} {projeto}")


def main():
    linhas, cobertura = [], []
    for projeto, d in projetos():
        _, teste = dividir_historico_teste(d)
        for cond, (pasta, prefixo, sufixo) in CONDICOES.items():
            pub = baixar(pasta, prefixo, projeto, sufixo)
            # os arquivos variam de formato entre modelos
            col = [c for c in pub.columns
                   if c.endswith("_sp") or "predicted" in c and "raw" not in c][0]
            real = "storypoint" if "storypoint" in pub else "actual_storypoint"
            pub = pub[["issuekey", col, real]].rename(columns={col: "previsao", real: "sp_deles"})
            pub["previsao"] = pd.to_numeric(pub.previsao, errors="coerce")
            m = teste[["issuekey", "storypoint"]].merge(pub, on="issuekey", how="left")
            cobertura.append({"projeto": projeto, "condicao": cond, "teste": len(teste),
                              "com_previsao": int(m.previsao.notna().sum()),
                              "sp_diferente": int((m.sp_deles.notna()
                                                   & (m.sp_deles != m.storypoint)).sum())})
            m = m.dropna(subset=["previsao"])
            linhas += [{"projeto": projeto, "issuekey": k, "storypoint": y,
                        "condicao": cond, "previsao": float(p)}
                       for k, y, p in zip(m.issuekey, m.storypoint, m.previsao)]
        print(f"  {projeto}", flush=True)

    r = pd.DataFrame(linhas)
    r.to_csv(SAIDA, index=False)
    cob = pd.DataFrame(cobertura)
    print(f"\nstory point deles diferente do nosso: {cob.sp_diferente.sum()} historias")
    print("historias de teste sem previsao publicada:")
    falta = cob[cob.com_previsao < cob.teste]
    print(falta.to_string(index=False) if len(falta) else "  nenhuma")
    r["erro"] = (r.storypoint - r.previsao).abs()
    print("\nMAE medio por projeto:")
    print(r.groupby(["condicao", "projeto"]).erro.mean().groupby("condicao").mean()
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {SAIDA}")


if __name__ == "__main__":
    main()

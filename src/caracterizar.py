"""Carregador dos dados e caracterizacao do benchmark Deep-SE (Choetkiertikul et al.,
IEEE TSE 2018). `carregar` le um projeto (usado por todos os scripts); rodado
sozinho, grava as estatisticas dos 16 projetos em tabela1_caracterizacao.csv."""
import re
from pathlib import Path
import pandas as pd

from conjunto import DADOS, SAIDA

# repositorio de origem de cada projeto, conforme o artigo original
REPOSITORIO = {
    "appceleratorstudio": "Appcelerator", "aptanastudio": "Appcelerator",
    "titanium": "Appcelerator", "bamboo": "Atlassian", "clover": "Atlassian",
    "jirasoftware": "Atlassian", "duracloud": "DuraSpace", "mesos": "Apache",
    "usergrid": "Apache", "moodle": "Moodle", "mule": "MuleSoft",
    "mulestudio": "MuleSoft", "springxd": "Spring", "talenddataquality": "Talend",
    "talendesb": "Talend", "datamanagement": "Talend",
}


def carregar(caminho):
    """Le um CSV de projeto e o ordena cronologicamente.

    O benchmark Deep-SE nao traz data de criacao, mas o numero sequencial da issue
    (PROJ-123) e monotonico na ordem de abertura, servindo de proxy temporal.
    O TAWOS traz a data real (coluna `created`), e quando ela existe e ela que
    ordena -- a replicacao de Tawosi et al. tambem ordena por criacao.
    """
    d = pd.read_csv(caminho)
    d["ordem"] = d["issuekey"].str.extract(r"-(\d+)$").astype(int)
    if "created" in d.columns:
        d = d.sort_values(["created", "ordem"], kind="stable").reset_index(drop=True)
    else:
        d = d.sort_values("ordem").reset_index(drop=True)
    d["texto"] = (d["title"].fillna("") + " " + d["description"].fillna("")).str.strip()
    return d


def main():
    linhas = []
    for arq in sorted(DADOS.glob("*.csv")):
        nome = arq.stem
        d = carregar(arq)
        sp = d["storypoint"]
        linhas.append({
            "projeto": nome,
            "repositorio": REPOSITORIO.get(nome, "-"),
            "issues": len(d),
            "sp_min": sp.min(),
            "sp_max": sp.max(),
            "sp_medio": sp.mean(),
            "sp_mediana": sp.median(),
            "sp_desvio": sp.std(),
            "palavras_medias": d["texto"].str.split().str.len().mean(),
        })

    t = pd.DataFrame(linhas).sort_values("issues", ascending=False)
    t.to_csv(SAIDA / "tabela1_caracterizacao.csv", index=False)

    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nTOTAL: {t['issues'].sum()} issues em {len(t)} projetos")



if __name__ == "__main__":
    main()

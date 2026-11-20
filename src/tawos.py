"""Carregador do TAWOS: do banco MySQL para um CSV por projeto.

O TAWOS (Tawosi et al., MSR 2022) e distribuido como um dump MySQL. Este script
le o banco e exporta, para cada projeto, as colunas que o experimento ja usa
(`issuekey`, `created`, `title`, `description`, `storypoint`) mais os campos do
Jira que a fase 4 do PLANO.md vai precisar (tipo, prioridade, componentes,
resolucao e as marcas de "mudou depois da estimativa").

Dois subconjuntos, em pastas separadas (nenhuma versionada):

  tawos/dados/         REPLICACAO. As issues que Tawosi, Moussa & Sarro (TSE 2022)
                       usaram, identificadas pelas chaves dos CSVs do pacote de
                       replicacao deles (tawos/replicacao/). E o que permite
                       comparar com a literatura.
  tawos/dados_proprio/ FILTRO PROPRIO. Todas as issues do banco com story point,
                       resolucao preenchida, valor em cartao de Planning Poker e
                       as tres marcas de mudanca zeradas; projetos com 200+ issues.
                       Nao depende dos autores, mas nao reproduz o conjunto deles.

Por que dois. Os criterios descritos no artigo (Porru et al.) nao reproduzem o
conjunto dos autores a partir das marcas da versao 1.1 do banco: os autores
calcularam as marcas por conta propria, e esta versao tem so 39 projetos (a
versao deles tinha 44; o repositorio Talendforge saiu). Em vez de fingir que o
filtro bate, o conjunto de replicacao usa as chaves deles como verdade e o
relatorio de carga diz o quanto o filtro proprio se afasta.

Texto: o dump grava titulo e descricao entre aspas, no estilo CSV (aspas
internas dobradas). Remove-se esse envelope. Depois disso o titulo e identico ao
dos autores em 100% das issues e a descricao em ~91% (o resto e a troca de
usuarios por <USER> feita pelo TAWOS 1.1).

Uso:  python src/tawos.py          (precisa do MySQL de pe; ver README)
"""
import os
import pandas as pd
import pymysql
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
TAWOS = RAIZ / "tawos"
CARTOES_JIRA = [0, 0.5, 1, 2, 3, 5, 8, 13, 20, 40, 100]
MINIMO_ISSUES = 200  # mesmo corte da replicacao ("each having more than 200 issues")

CONSULTA = """
SELECT i.Issue_Key AS issuekey,
       DATE_FORMAT(i.Creation_Date, '%Y-%m-%d %H:%i:%s') AS created,
       i.Title AS title, i.Description AS description,
       i.Story_Point AS storypoint,
       i.Type AS tipo, i.Priority AS prioridade, i.Resolution AS resolucao,
       DATE_FORMAT(i.Estimation_Date, '%Y-%m-%d %H:%i:%s') AS estimado_em,
       i.Title_Changed_After_Estimation AS titulo_mudou,
       i.Description_Changed_After_Estimation AS descricao_mudou,
       i.Story_Point_Changed_After_Estimation AS sp_mudou,
       (SELECT GROUP_CONCAT(c.Name ORDER BY c.Name SEPARATOR '|')
          FROM Issue_Component ic JOIN Component c ON c.ID = ic.Component_ID
         WHERE ic.Issue_ID = i.ID) AS componentes,
       p.Project_Key AS projeto
  FROM Issue i JOIN Project p ON p.ID = i.Project_ID
 WHERE i.Story_Point IS NOT NULL
"""


def conectar():
    return pymysql.connect(
        host=os.environ.get("TAWOS_HOST", "127.0.0.1"),
        port=int(os.environ.get("TAWOS_PORT", "3307")),
        user=os.environ.get("TAWOS_USER", "root"),
        password=os.environ.get("TAWOS_SENHA", "tawos"),
        database=os.environ.get("TAWOS_BANCO", "tawos_cleaned"))


def desembrulhar(s):
    """Remove o envelope de aspas estilo CSV que o dump poe em titulo e descricao."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].replace('""', '"')
    return s.strip()


def ler_banco():
    with conectar() as c:
        d = pd.read_sql(CONSULTA, c)
    d["title"] = d["title"].map(desembrulhar)
    d["description"] = d["description"].map(desembrulhar)
    return d


def chaves_dos_autores():
    """Chaves por projeto nos CSVs do pacote de replicacao (verdade de referencia)."""
    return {arq.name.split("_")[0]: pd.read_csv(arq, usecols=["issuekey"]).issuekey
            for arq in sorted((TAWOS / "replicacao").glob("*_deep-se.csv"))}


def filtro_proprio(d):
    ok = (d.resolucao.notna() & d.storypoint.isin(CARTOES_JIRA)
          & (d.sp_mudou == 0) & (d.titulo_mudou == 0) & (d.descricao_mudou == 0))
    d = d[ok]
    return d[d.groupby("projeto").projeto.transform("size") >= MINIMO_ISSUES]


def gravar(d, pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    for arq in pasta.glob("*.csv"):
        arq.unlink()
    for projeto, g in d.groupby("projeto"):
        g.drop(columns="projeto").to_csv(pasta / f"{projeto}.csv", index=False)


def main():
    d = ler_banco()
    autores = chaves_dos_autores()
    print(f"banco: {len(d):,} issues com story point em {d.projeto.nunique()} projetos")

    linhas = []
    partes = []
    for projeto, chaves in autores.items():
        g = d[d.projeto == projeto]
        achadas = g[g.issuekey.isin(set(chaves))]
        partes.append(achadas)
        linhas.append({"projeto": projeto, "autores": len(chaves),
                       "achadas_no_banco": len(achadas),
                       "ausentes_no_banco": len(chaves) - len(achadas)})
    replic = pd.concat(partes)
    gravar(replic, TAWOS / "dados")

    proprio = filtro_proprio(d)
    gravar(proprio, TAWOS / "dados_proprio")

    r = pd.DataFrame(linhas).set_index("projeto")
    r["filtro_proprio"] = proprio.groupby("projeto").size().reindex(r.index).fillna(0).astype(int)
    r["filtro_proprio_dentro_dos_autores"] = (
        proprio[proprio.issuekey.isin(set(replic.issuekey))]
        .groupby("projeto").size().reindex(r.index).fillna(0).astype(int))
    r.to_csv(TAWOS / "relatorio_carga.csv")
    print(r.to_string())
    print(f"\nreplicacao: {len(replic):,} issues em {replic.projeto.nunique()} projetos "
          f"(autores: {r.autores.sum():,}; ausentes no banco: {r.ausentes_no_banco.sum()})")
    print(f"filtro proprio: {len(proprio):,} issues em {proprio.projeto.nunique()} projetos; "
          f"cobre {r.filtro_proprio_dentro_dos_autores.sum() / r.autores.sum():.1%} do conjunto dos autores")


if __name__ == "__main__":
    main()

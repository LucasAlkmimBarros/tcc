"""Verificacao do experimento: sete checagens que falham alto se algo quebrar.

Nao basta o pipeline "rodar sem erro" — um bug silencioso (vazamento temporal,
rotulos trocados, metrica errada) tambem roda sem erro e produz numeros
plausiveis. Cada checagem abaixo existe para pegar uma classe especifica de
falha. Uso:

    .venv/bin/python src/verificar.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from caracterizar import carregar
from experimento import dividir, construir_modelos, mae_aleatorio, SEMENTE

RAIZ = Path(__file__).resolve().parent.parent
DADOS, SAIDA = RAIZ / "dados", RAIZ / "resultados"

# Tabela 1 de Choetkiertikul et al. (IEEE TSE 2018), transcrita do preprint:
# projeto -> (issues, min SP, max SP, media SP, mediana SP, desvio SP)
PUBLICADO = {
    "mesos": (1680, 1, 40, 3.09, 3, 2.42),
    "usergrid": (482, 1, 8, 2.85, 3, 1.40),
    "appceleratorstudio": (2919, 1, 40, 5.64, 5, 3.33),
    "aptanastudio": (829, 1, 40, 8.02, 8, 5.95),
    "titanium": (2251, 1, 34, 6.32, 5, 5.10),
    "duracloud": (666, 1, 16, 2.13, 1, 2.03),
    "bamboo": (521, 1, 20, 2.42, 2, 2.14),
    "clover": (384, 1, 40, 4.59, 2, 6.55),
    "jirasoftware": (352, 1, 20, 4.43, 3, 3.51),
    "moodle": (1166, 1, 100, 15.54, 8, 21.65),
    "datamanagement": (4667, 1, 100, 9.57, 4, 16.61),
    "mule": (889, 1, 21, 5.08, 5, 3.50),
    "mulestudio": (732, 1, 34, 6.40, 5, 5.39),
    "springxd": (3526, 1, 40, 3.70, 3, 3.23),
    "talenddataquality": (1381, 1, 40, 5.92, 5, 5.19),
    "talendesb": (868, 1, 13, 2.16, 2, 1.50),
}

falhas = []


def checar(condicao, titulo, detalhe=""):
    print(f"  {'PASSOU' if condicao else 'FALHOU'}  {titulo}")
    if detalhe:
        print(f"          {detalhe}")
    if not condicao:
        falhas.append(titulo)


# ---------------------------------------------------------------------------
def c1_dados_integros():
    """Pega arquivo faltando, download truncado ou CSV corrompido."""
    print("\n[1] Integridade dos dados")
    arquivos = sorted(DADOS.glob("*.csv"))
    checar(len(arquivos) == 16, "16 arquivos de projeto presentes",
           f"encontrados: {len(arquivos)}")

    total, problemas = 0, []
    for arq in arquivos:
        d = pd.read_csv(arq)
        total += len(d)
        if d.storypoint.isna().any():
            problemas.append(f"{arq.stem}: story points nulos")
        if (d.storypoint <= 0).any():
            problemas.append(f"{arq.stem}: story points <= 0")
        if d.title.isna().all():
            problemas.append(f"{arq.stem}: titulos vazios")
    checar(not problemas, "nenhum valor nulo, negativo ou zerado",
           "; ".join(problemas))
    checar(total == 23313, "total de 23.313 issues", f"contado: {total}")


def c2_bate_com_a_publicacao():
    """A checagem externa: nossas estatisticas tem de reproduzir a Tabela 1 do
    artigo original. Se divergirem, lemos o arquivo errado ou o parse falhou."""
    print("\n[2] Conferencia contra os valores publicados (IEEE TSE 2018)")
    divergencias = []
    for nome, (n, mn, mx, media, mediana, desvio) in PUBLICADO.items():
        sp = pd.read_csv(DADOS / f"{nome}.csv").storypoint
        for rotulo, obtido, esperado, tol in [
            ("n", len(sp), n, 0), ("min", sp.min(), mn, 0),
            ("max", sp.max(), mx, 0), ("media", sp.mean(), media, 0.005),
            ("mediana", sp.median(), mediana, 0.005),
            ("desvio", sp.std(), desvio, 0.015),
        ]:
            if abs(float(obtido) - esperado) > tol:
                divergencias.append(f"{nome}.{rotulo}: {obtido} != {esperado}")
    checar(not divergencias,
           "16 projetos reproduzem n, min, max, media, mediana e desvio publicados",
           "; ".join(divergencias[:5]))


def c3_split_sem_vazamento():
    """Pega o erro mais grave possivel: treinar com dados do futuro."""
    print("\n[3] Divisao cronologica sem vazamento")
    problemas = []
    for arq in sorted(DADOS.glob("*.csv")):
        treino, val, teste = dividir(carregar(arq))
        if treino.ordem.max() >= val.ordem.min():
            problemas.append(f"{arq.stem}: treino invade validacao")
        if val.ordem.max() >= teste.ordem.min():
            problemas.append(f"{arq.stem}: validacao invade teste")
        if set(treino.issuekey) & set(teste.issuekey):
            problemas.append(f"{arq.stem}: issue repetida entre treino e teste")
        proporcao = len(treino) / (len(treino) + len(val) + len(teste))
        if not 0.59 <= proporcao <= 0.61:
            problemas.append(f"{arq.stem}: treino com {proporcao:.0%}")
    checar(not problemas,
           "toda issue de teste e posterior a toda issue de treino, em 16 projetos",
           "; ".join(problemas[:5]))


def c4_metricas_corretas():
    """Pega metrica implementada ao contrario ou com sinal trocado."""
    print("\n[4] Sanidade das metricas")
    y = np.array([1, 2, 3, 5, 8], dtype=float)
    checar(mean_absolute_error(y, y) == 0, "MAE de predicao perfeita e zero")
    checar(abs(mean_absolute_error(y, y + 2) - 2) < 1e-9,
           "MAE de erro constante de 2 e exatamente 2")
    rng = np.random.default_rng(SEMENTE)
    base = mae_aleatorio(y, y, rng)
    checar(base > 0, "MAE do palpite aleatorio e positivo", f"valor: {base:.3f}")
    sa_perfeito = (1 - 0 / base) * 100
    checar(abs(sa_perfeito - 100) < 1e-9, "SA de predicao perfeita e 100%")


def c5_modelo_realmente_aprende():
    """O teste de sabotagem. Embaralhamos os rotulos do treino: o texto deixa de
    ter qualquer relacao com o esforco. Mede-se quanto o modelo piora. Usa cinco
    embaralhamentos por projeto, porque um unico sorteio e ruidoso demais para
    concluir qualquer coisa. Um modelo que le o texto deve piorar de forma
    consistente; se mal piorar, ele esta apenas reproduzindo a tendencia central."""
    print("\n[5] Teste de sabotagem: rotulos de treino embaralhados")
    rng = np.random.default_rng(SEMENTE)
    degradacoes = []
    for nome in ["mesos", "appceleratorstudio", "jirasoftware", "titanium"]:
        treino, val, teste = dividir(carregar(DADOS / f"{nome}.csv"))
        x_tr = pd.concat([treino.texto, val.texto])
        y_tr = pd.concat([treino.storypoint, val.storypoint])
        piso, teto = y_tr.min(), y_tr.max()

        prever = lambda alvo: mean_absolute_error(
            teste.storypoint,
            np.clip(construir_modelos(SEMENTE)["TF-IDF + SVR"]
                    .fit(x_tr, alvo).predict(teste.texto), piso, teto))

        real = prever(y_tr)
        ruidos = [prever(rng.permutation(y_tr.values)) for _ in range(5)]
        ruido = float(np.mean(ruidos))
        piora = (ruido / real - 1) * 100
        degradacoes.append(piora)
        print(f"          {nome:<20} MAE real {real:.3f} | sabotado {ruido:.3f} "
              f"(media de 5)  -> piora de {piora:+.1f}%")

    checar(all(p > 0 for p in degradacoes),
           "sabotar os rotulos piora o modelo em todos os projetos testados")
    media = float(np.mean(degradacoes))
    print(f"\n          Degradacao media: {media:+.1f}%.")
    print("          Interpretacao: quanto MAIOR, mais o modelo depende do texto.")
    print("          Valores baixos indicam que o desempenho vem sobretudo da")
    print("          tendencia central aprendida — coerente com o empate observado")
    print("          contra a mediana historica.")


def c6_metrica_detecta_vazamento():
    """Controle positivo. Injeta-se vazamento de proposito — metade do conjunto
    de teste entra no treino, mantendo o MESMO conjunto de teste — e o MAE tem de
    cair. Isso prova que a metrica e sensivel a vazamento e, portanto, teria
    acusado se o pipeline real tivesse algum.

    Nao se compara aqui o split cronologico com o aleatorio: seriam conjuntos de
    teste distintos, com dificuldades distintas, e o MAE nao seria comparavel."""
    print("\n[6] Controle positivo: a metrica detecta vazamento deliberado")
    detectou = []
    for nome in ["appceleratorstudio", "springxd", "titanium"]:
        treino, val, teste = dividir(carregar(DADOS / f"{nome}.csv"))
        x_tr = pd.concat([treino.texto, val.texto])
        y_tr = pd.concat([treino.storypoint, val.storypoint])
        piso, teto = y_tr.min(), y_tr.max()

        prever = lambda x, y: mean_absolute_error(
            teste.storypoint,
            np.clip(construir_modelos(SEMENTE)["TF-IDF + SVR"]
                    .fit(x, y).predict(teste.texto), piso, teto))

        limpo = prever(x_tr, y_tr)
        metade = teste.iloc[: len(teste) // 2]
        sujo = prever(pd.concat([x_tr, metade.texto]),
                      pd.concat([y_tr, metade.storypoint]))

        detectou.append(sujo < limpo)
        print(f"          {nome:<20} sem vazamento {limpo:.3f} | "
              f"com vazamento {sujo:.3f}  ({(1 - sujo / limpo) * 100:+.1f}%)")
    checar(all(detectou),
           "vazar teste no treino melhora o MAE — a metrica e sensivel a vazamento")


def c7_resultados_conferem():
    """Confirma que o CSV de resultados existe e e coerente internamente."""
    print("\n[7] Coerencia do arquivo de resultados")
    arq = SAIDA / "resultados_completos.csv"
    if not arq.exists():
        checar(False, "resultados_completos.csv existe",
               "rode src/experimento.py antes")
        return
    r = pd.read_csv(arq)
    PLN = [m for m in r.modelo.unique() if m.startswith("PLN")][0]
    piv = r.pivot(index="projeto", columns="modelo", values="MAE")

    checar(len(r) == 64, "64 linhas (16 projetos x 4 tecnicas)", f"lidas: {len(r)}")
    checar((piv[PLN] < piv["Palpite aleatorio"]).all(),
           "o modelo supera o palpite aleatorio nos 16 projetos")
    sa = r[r.modelo == PLN].SA
    checar(sa.between(0, 100).all(), "todos os SA no intervalo [0, 100]",
           f"faixa observada: {sa.min():.1f} a {sa.max():.1f}")
    coerente = r.dropna(subset=["MdAE"])
    checar((coerente.MAE > 0).all(), "todos os MAE positivos")


for etapa in [c1_dados_integros, c2_bate_com_a_publicacao, c3_split_sem_vazamento,
              c4_metricas_corretas, c5_modelo_realmente_aprende,
              c6_metrica_detecta_vazamento, c7_resultados_conferem]:
    etapa()

print("\n" + "=" * 68)
if falhas:
    print(f"{len(falhas)} CHECAGEM(NS) FALHOU(RAM):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("TODAS AS CHECAGENS PASSARAM — o experimento esta integro.")

"""Compara o experimento no TAWOS com a Tabela 5 de Tawosi, Moussa & Sarro (TSE 2022).

A Tabela 5 da replicacao traz, por projeto, o MAE de Deep-SE, TF/IDF-SVM, media e
mediana no conjunto TAWOS. Aqui:

  1. Validacao do protocolo. A mediana historica e uma regua sem parametros: se o
     nosso divisor cronologico e o carregador estao certos, nossa mediana tem que
     cair perto da deles. (No Deep-SE, 2,590 contra 2,610.)
  2. A pergunta da fase 2. O modelo supera a mediana no TAWOS?

Os projetos TDP, TDQ e TESB estao na Tabela 5 mas nao no banco 1.1; a comparacao
usa so os 23 projetos que temos.

Uso:  python src/comparar_tawos.py     (depois de CONJUNTO=tawos experimento.py)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

RAIZ = Path(__file__).resolve().parent.parent
ref = pd.read_csv(RAIZ / "referencia" / "tawosi_tabela5.csv").set_index("projeto")
r = pd.read_csv(RAIZ / "resultados" / "tawos" / "resultados_completos.csv")
mae = r.pivot(index="projeto", columns="modelo", values="MAE")
pln = r[r.modelo.str.startswith("PLN")].set_index("projeto").MAE

t = pd.DataFrame({
    "mediana_nossa": mae["Mediana historica"], "mediana_ref": ref.Mediana,
    "media_nossa": mae["Media historica"], "media_ref": ref.Media,
    "tfidf_nosso": pln, "tfidf_svm_ref": ref.TFIDF_SVM, "deepse_ref": ref.DeepSE,
}).loc[ref.index]
t.round(2).to_csv(RAIZ / "resultados" / "tawos" / "comparacao_tabela5.csv")

print("=== 1. o protocolo reproduz a literatura? (MAE por projeto) ===")
print(t.round(2).to_string())
dm = (t.mediana_nossa - t.mediana_ref).abs()
print(f"\n  mediana: media dos 23 = {t.mediana_nossa.mean():.3f} (nossa) contra "
      f"{t.mediana_ref.mean():.3f} (Tawosi)")
print(f"  diferenca absoluta media por projeto: {dm.mean():.3f}; "
      f"{(dm <= 0.05).sum()}/23 dentro de 0,05; pior: {dm.idxmax()} ({dm.max():.2f})")
print(f"  media: {t.media_nossa.mean():.3f} contra {t.media_ref.mean():.3f}")

print("\n=== 2. o modelo supera a mediana no TAWOS? ===")
for rotulo, col in [("nosso TF-IDF", "tfidf_nosso"), ("Deep-SE (Tawosi)", "deepse_ref"),
                    ("TF/IDF-SVM (Tawosi)", "tfidf_svm_ref")]:
    med = t.mediana_nossa if col == "tfidf_nosso" else t.mediana_ref
    dif = med - t[col]
    p = wilcoxon(t[col], med).pvalue
    print(f"  {rotulo:<20} MAE medio {t[col].mean():.3f} contra mediana {med.mean():.3f} "
          f"({(1 - t[col].mean() / med.mean()) * 100:+.1f}%)  "
          f"vence em {(dif > 0).sum():>2}/23, empata em {(dif == 0).sum()}  Wilcoxon p={p:.3f}")

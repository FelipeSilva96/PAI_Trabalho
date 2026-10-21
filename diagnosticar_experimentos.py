
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score
)

PASTA = Path("resultados/metricas")

# Lemos apenas os resultados produzidos pelo
# novo gerenciador de experimentos.
arquivos = sorted(
    PASTA.glob("cnn7_*_previsoes.csv")
)

if not arquivos:
    raise FileNotFoundError(
        "Nenhum arquivo de previsões foi encontrado."
    )

for caminho in arquivos:
    dados = pd.read_csv(caminho)

    y = dados["pCR_real"].to_numpy(dtype=int)
    p = dados["probabilidade"].to_numpy(dtype=float)

    previstos = (p >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, previstos, labels=[0, 1]
    ).ravel()

    print("\n" + "=" * 65)
    print("EXPERIMENTO:", caminho.stem)
    print("=" * 65)

    print("Pacientes avaliados:", len(y))
    print("Previsões positivas:", int(previstos.sum()))
    print("Previsões negativas:", int((previstos == 0).sum()))

    print("\nMatriz de confusão:")
    print("VP:", vp, "VN:", vn, "FP:", fp, "FN:", fn)

    print("\nMétricas:")
    print(f"AUC-ROC: {roc_auc_score(y, p):.4f}")
    print(f"F1: {f1_score(y, previstos):.4f}")

    print(
        "Acurácia balanceada:",
        f"{balanced_accuracy_score(y, previstos):.4f}"
    )

    print("\nDistribuição das probabilidades:")
    print(f"Mínima: {p.min():.4f}")
    print(f"Máxima: {p.max():.4f}")
    print(f"Média: {p.mean():.4f}")
    print(f"Desvio padrão: {p.std():.4f}")

    print("\nProbabilidade média por classe real:")
    print(
        dados.groupby("pCR_real")["probabilidade"]
        .agg(["mean", "min", "max"])
    )

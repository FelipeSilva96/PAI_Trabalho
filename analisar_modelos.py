
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix
)

PASTA = Path("resultados/metricas")

ARQUIVOS = {
    "CNN7 - Uma fase": (
        "avaliacao_val_cnn7_early_central_nenhuma.csv"
    ),
    "CNN7 - Tres fases": (
        "avaliacao_val_cnn7_three_central_nenhuma.csv"
    )
}

previsoes = {}

for nome, arquivo in ARQUIVOS.items():
    caminho = PASTA / arquivo

    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    dados = pd.read_csv(caminho)

    if dados["pid"].duplicated().any():
        raise ValueError(
            f"Pacientes duplicados: {nome}"
        )

    if len(dados) != 33:
        raise ValueError(
            f"{nome}: esperados 33 pacientes, "
            f"encontrados {len(dados)}"
        )

    y = dados["pCR_real"].astype(int)
    p = dados["probabilidade"].astype(float)

    previsto = (p >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, previsto, labels=[0, 1]
    ).ravel()

    metricas = {
        "AUC-ROC": roc_auc_score(y, p),
        "Average Precision": average_precision_score(y, p),
        "Acuracia Balanceada": balanced_accuracy_score(
            y, previsto
        ),
        "Brier Score": brier_score_loss(y, p),
        "Probabilidade Media": p.mean(),
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn)
    }

    print("\n" + "=" * 55)
    print(nome)
    print("=" * 55)

    for metrica, valor in metricas.items():
        print(f"{metrica}: {valor:.4f}")

    print("\nProbabilidades por classe real:")

    print(
        dados.groupby("pCR_real")["probabilidade"]
        .agg(["mean", "min", "max"])
    )

    previsoes[nome] = dados.set_index("pid")[
        ["pCR_real", "probabilidade"]
    ]


# Comparação pareada dos mesmos pacientes.
uma = previsoes["CNN7 - Uma fase"]
tres = previsoes["CNN7 - Tres fases"]

comparacao = uma.join(
    tres,
    lsuffix="_uma",
    rsuffix="_tres",
    how="inner"
)

if len(comparacao) != 33:
    raise ValueError(
        "As avaliações não contêm os mesmos 33 pacientes."
    )

if not (
    comparacao["pCR_real_uma"]
    == comparacao["pCR_real_tres"]
).all():
    raise ValueError(
        "Rótulos inconsistentes entre as avaliações."
    )

comparacao["prev_uma"] = (
    comparacao["probabilidade_uma"] >= 0.5
).astype(int)

comparacao["prev_tres"] = (
    comparacao["probabilidade_tres"] >= 0.5
).astype(int)

diferentes = comparacao[
    comparacao["prev_uma"] != comparacao["prev_tres"]
]

print("\n" + "=" * 55)
print("PACIENTES COM PREVISOES DIFERENTES")
print("=" * 55)
print(diferentes.to_string())

destino = PASTA / "comparacao_cnn7_fases.csv"
comparacao.to_csv(destino)

print(f"\nComparação salva em: {destino}")

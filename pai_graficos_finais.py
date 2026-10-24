
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)


PASTA = Path("resultados/avaliacao_final")
SAIDA = PASTA / "graficos"

ARQUIVOS = {
    "CNN7 E08": "previsoes_teste_cnn7_E08.csv",
    "ResNet50 R01": "previsoes_teste_resnet50_R01.csv",
    "Fusão FUS02": "previsoes_teste_fusao_FUS02.csv"
}


def carregar_dados():
    if not (PASTA / "execucao_teste.json").is_file():
        raise FileNotFoundError(
            "A avaliação final ainda não foi registrada."
        )

    dados = {}
    referencia = None

    for nome, arquivo in ARQUIVOS.items():
        caminho = PASTA / arquivo

        df = pd.read_csv(
            caminho,
            dtype={"pid": str}
        )

        if len(df) != 35:
            raise RuntimeError(
                f"{nome}: esperados 35 pacientes."
            )

        if df["pid"].duplicated().any():
            raise RuntimeError(
                f"{nome}: pacientes duplicados."
            )

        df = df.sort_values("pid").reset_index(drop=True)

        y = df["pCR_real"].to_numpy(dtype=int)
        p = df["probabilidade"].to_numpy(dtype=float)

        if int(y.sum()) != 12:
            raise RuntimeError(
                f"{nome}: distribuição de classes incorreta."
            )

        if not np.isfinite(p).all():
            raise RuntimeError(
                f"{nome}: probabilidades inválidas."
            )

        assinatura = (
            tuple(df["pid"]),
            tuple(y)
        )

        if referencia is None:
            referencia = assinatura
        elif assinatura != referencia:
            raise RuntimeError(
                "Os modelos não avaliaram os mesmos pacientes."
            )

        dados[nome] = (y, p)

    return dados


def gerar_curvas_roc(dados):
    plt.figure(figsize=(8, 6))

    for nome, (y, p) in dados.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        plt.plot(
            fpr,
            tpr,
            label=f"{nome} | AUC = {auc:.4f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Referência aleatória"
    )

    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Sensibilidade")
    plt.title("Curvas ROC — Teste final (35 pacientes)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        SAIDA / "curvas_roc.png",
        dpi=200
    )
    plt.close()


def gerar_curvas_pr(dados):
    plt.figure(figsize=(8, 6))

    prevalencia = 12 / 35

    plt.axhline(
        prevalencia,
        linestyle="--",
        label=f"Prevalência = {prevalencia:.4f}"
    )

    for nome, (y, p) in dados.items():
        precisao, revocacao, _ = (
            precision_recall_curve(y, p)
        )

        ap = average_precision_score(y, p)

        plt.step(
            revocacao,
            precisao,
            where="post",
            label=f"{nome} | AP = {ap:.4f}"
        )

    plt.xlabel("Revocação / Sensibilidade")
    plt.ylabel("Precisão")
    plt.title(
        "Curvas Precisão–Recall — Teste final"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        SAIDA / "curvas_precisao_recall.png",
        dpi=200
    )
    plt.close()


def gerar_matrizes(dados):
    for nome, (y, p) in dados.items():
        previstos = (p >= 0.5).astype(int)

        matriz = confusion_matrix(
            y,
            previstos,
            labels=[0, 1]
        )

        plt.figure(figsize=(6, 5))
        plt.imshow(matriz, cmap="Blues")

        for linha in range(2):
            for coluna in range(2):
                valor = matriz[linha, coluna]

               
                plt.text(
                coluna,
                linha,
                str(valor),
                ha="center",
                va="center",
                fontsize=18,
                color=(
                    "white"
                    if valor > matriz.max() / 2
                    else "black"
                    )
                )


        plt.xticks(
            [0, 1],
            ["pCR 0", "pCR 1"]
        )

        plt.yticks(
            [0, 1],
            ["pCR 0", "pCR 1"]
        )

        plt.xlabel("Classe prevista")
        plt.ylabel("Classe real")
        plt.title(f"Matriz de confusão — {nome}")
        plt.tight_layout()

        identificador = (
            nome.lower()
            .replace(" ", "_")
            .replace("ã", "a")
            .replace("ú", "u")
        )

        plt.savefig(
            SAIDA / f"matriz_{identificador}.png",
            dpi=200
        )

        plt.close()


def gerar_comparacao(dados):
    nomes = list(dados)

    aucs = [
        roc_auc_score(*dados[nome])
        for nome in nomes
    ]

    aps = [
        average_precision_score(*dados[nome])
        for nome in nomes
    ]

    x = np.arange(len(nomes))
    largura = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - largura / 2,
        aucs,
        largura,
        label="AUC-ROC"
    )

    plt.bar(
        x + largura / 2,
        aps,
        largura,
        label="Average Precision"
    )

    
    plt.axhline(
        0.5,
        linestyle=":",
        label="Referência AUC-ROC = 0,5"
    )

    plt.axhline(
        12 / 35,
        linestyle="--",
        label="Referência AP = prevalência (0,3429)"
    )


    plt.xticks(x, nomes)
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.title("Comparação final das arquiteturas")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        SAIDA / "comparacao_final.png",
        dpi=200
    )

    plt.close()


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)

    dados = carregar_dados()

    gerar_curvas_roc(dados)
    gerar_curvas_pr(dados)
    gerar_matrizes(dados)
    gerar_comparacao(dados)

    print("\nGRÁFICOS GERADOS COM SUCESSO")
    print("Pasta:", SAIDA.resolve())
    print("Arquivos PNG:", len(list(SAIDA.glob("*.png"))))
    print("\nNenhum modelo foi treinado.")
    print("Nenhuma inferência foi executada.")
    print("Nenhum limiar foi modificado.")


if __name__ == "__main__":
    main()

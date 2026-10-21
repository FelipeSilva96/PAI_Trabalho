
# Experimentos da CNN7 - BreastDCEDL
#
# Este arquivo permite testar diferentes configurações
# de treinamento sem sobrescrever os modelos anteriores.

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

from pai_etapa2 import (
    CNN7,
    DatasetMRI,
    definir_semente,
    criar_pastas,
    MODELOS,
    METRICAS
)


def criar_modelo(canais, normalizacao):
    modelo = CNN7(canais_entrada=canais)

    if normalizacao == "group":
        # Substituímos apenas as normalizações.
        # As sete convoluções permanecem iguais.
        for indice, camada in enumerate(modelo.extrator):
            if isinstance(camada, nn.BatchNorm2d):
                canais_saida = camada.num_features

                modelo.extrator[indice] = nn.GroupNorm(
                    num_groups=8,
                    num_channels=canais_saida
                )

    return modelo


def identificar_experimento(args):
    taxa = f"{args.lr:g}".replace(".", "p")
    peso = f"{args.pos_weight:g}".replace(".", "p")

    partes = [
        "cnn7",
        args.modo,
        args.selecao,
        args.norm,
        f"lr{taxa}",
        f"pw{peso}",
        f"seed{args.seed}",
        f"size{args.tamanho}"
    ]

    if args.tag:
        if not all(
            caractere.isalnum() or caractere in "_-"
            for caractere in args.tag
        ):
            raise ValueError(
                "A tag deve conter apenas letras, "
                "números, sublinhados ou hífens."
            )

        partes.append(args.tag)

    return "_".join(partes)


def construir_loader(dataset, batch, embaralhar, seed):
    gerador = torch.Generator()
    gerador.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=embaralhar,
        num_workers=0,
        generator=gerador
    )


def calcular_metricas(rotulos, probabilidades):
    y = np.asarray(rotulos, dtype=int)
    p = np.asarray(probabilidades, dtype=float)

    classes = (p >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, classes, labels=[0, 1]
    ).ravel()

    return {
        "acuracia": accuracy_score(y, classes),
        "acuracia_balanceada": balanced_accuracy_score(
            y, classes
        ),
        "sensibilidade": recall_score(
            y, classes, zero_division=0
        ),
        "especificidade": (
            float(vn / (vn + fp))
            if vn + fp else 0.0
        ),
        "precisao": precision_score(
            y, classes, zero_division=0
        ),
        "f1": f1_score(
            y, classes, zero_division=0
        ),
        "auc_roc": roc_auc_score(y, p),
        "average_precision": average_precision_score(
            y, p
        ),
        "brier": brier_score_loss(y, p),
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn)
    }


@torch.no_grad()
def validar(modelo, loader, criterio, dispositivo):
    modelo.eval()

    perda_total = 0.0
    rotulos = []
    probabilidades = []
    pacientes = []

    for imagens, y, ids in loader:
        imagens = imagens.to(dispositivo)
        y = y.to(dispositivo)

        logits = modelo(imagens)
        perda = criterio(logits, y)

        perda_total += perda.item() * len(y)

        probs = torch.sigmoid(logits)

        rotulos.extend(y.cpu().numpy().tolist())
        probabilidades.extend(
            probs.cpu().numpy().tolist()
        )
        pacientes.extend(ids)

    perda_media = perda_total / len(loader.dataset)

    metricas = calcular_metricas(
        rotulos, probabilidades
    )

    previsoes = pd.DataFrame({
        "pid": pacientes,
        "pCR_real": rotulos,
        "probabilidade": probabilidades
    })

    return perda_media, metricas, previsoes


def executar(args):
    criar_pastas()
    definir_semente(args.seed)

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    treino = DatasetMRI(
        particao=0,
        selecao=args.selecao,
        modo=args.modo,
        tamanho=args.tamanho,
        aumentar=True
    )

    validacao = DatasetMRI(
        particao=2,
        selecao=args.selecao,
        modo=args.modo,
        tamanho=args.tamanho,
        aumentar=False
    )

    if len(treino.registros) != 104:
        raise RuntimeError(
            "O treinamento precisa conter 104 pacientes."
        )

    if len(validacao.registros) != 33:
        raise RuntimeError(
            "A validação precisa conter 33 pacientes."
        )

    nome = identificar_experimento(args)

    checkpoint = MODELOS / f"{nome}.pth"
    historico_csv = METRICAS / f"{nome}_historico.csv"
    previsoes_csv = METRICAS / f"{nome}_previsoes.csv"
    resumo_json = METRICAS / f"{nome}_resumo.json"

    # Impedimos a substituição de resultados anteriores.
    destinos = [
        checkpoint, historico_csv,
        previsoes_csv, resumo_json
    ]

    existentes = [
        caminho for caminho in destinos
        if caminho.exists()
    ]

    if existentes:
        raise FileExistsError(
            "O experimento já possui arquivos salvos. "
            "Escolha outra tag ou configuração.\n"
            + "\n".join(map(str, existentes))
        )

    loader_treino = construir_loader(
        treino, args.batch, True, args.seed
    )

    loader_val = construir_loader(
        validacao, args.batch, False, args.seed
    )

    canais = 3 if args.modo == "three" else 1

    dispositivo = torch.device("cpu")

    modelo = criar_modelo(
        canais, args.norm
    ).to(dispositivo)

    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [args.pos_weight],
            dtype=torch.float32,
            device=dispositivo
        )
    )

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    melhor_perda = float("inf")
    melhor_epoca = None
    sem_melhora = 0
    historico = []
    melhor_metricas = None
    melhores_previsoes = None

    print("\nExperimento:", nome)
    print("Pacientes de treinamento:", len(treino.registros))
    print("Pacientes de validação:", len(validacao.registros))
    print("Amostras aumentadas:", len(treino))
    print("Dispositivo:", dispositivo)

    inicio = time.perf_counter()

    for epoca in range(1, args.epocas + 1):
        modelo.train()

        perda_acumulada = 0.0

        for imagens, y, _ in loader_treino:
            imagens = imagens.to(dispositivo)
            y = y.to(dispositivo)

            otimizador.zero_grad(
                set_to_none=True
            )

            logits = modelo(imagens)
            perda = criterio(logits, y)

            perda.backward()
            otimizador.step()

            perda_acumulada += perda.item() * len(y)

        perda_treino = perda_acumulada / len(treino)

        perda_val, metricas, previsoes = validar(
            modelo, loader_val,
            criterio, dispositivo
        )

        historico.append({
            "epoca": epoca,
            "loss_treino": perda_treino,
            "loss_validacao": perda_val,
            **metricas
        })

        print(
            f"Epoca {epoca:03d} | "
            f"Treino {perda_treino:.4f} | "
            f"Val {perda_val:.4f} | "
            f"F1 {metricas['f1']:.4f} | "
            f"AUC {metricas['auc_roc']:.4f}"
        )

        if perda_val < melhor_perda - 1e-5:
            melhor_perda = perda_val
            melhor_epoca = epoca
            melhor_metricas = metricas.copy()
            melhores_previsoes = previsoes.copy()
            sem_melhora = 0

            torch.save({
                "pesos": modelo.state_dict(),
                "arquitetura": "CNN7",
                "normalizacao": args.norm,
                "canais": canais,
                "modo": args.modo,
                "selecao": args.selecao,
                "tamanho": args.tamanho,
                "seed": args.seed,
                "pos_weight": args.pos_weight,
                "lr": args.lr,
                "epoca": epoca,
                "loss_validacao": perda_val
            }, checkpoint)

        else:
            sem_melhora += 1

        if sem_melhora >= args.paciencia:
            print("Early stopping ativado.")
            break

    duracao = time.perf_counter() - inicio

    pd.DataFrame(historico).to_csv(
        historico_csv, index=False
    )

    melhores_previsoes.to_csv(
        previsoes_csv, index=False
    )

    resumo = {
        "experimento": nome,
        "melhor_epoca": melhor_epoca,
        "loss_validacao": melhor_perda,
        "tempo_segundos": duracao,
        "configuracao": vars(args),
        "metricas": melhor_metricas
    }

    with open(
        resumo_json, "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            resumo, arquivo,
            indent=2,
            ensure_ascii=False
        )

    print("\nTreinamento concluido.")
    print("Melhor epoca:", melhor_epoca)
    print("AUC-ROC:", f"{melhor_metricas['auc_roc']:.4f}")
    print("F1:", f"{melhor_metricas['f1']:.4f}")
    print("Acuracia balanceada:",
          f"{melhor_metricas['acuracia_balanceada']:.4f}")
    print("Tempo:", f"{duracao / 60:.2f} minutos")
    print("Checkpoint:", checkpoint)
    print("Resumo:", resumo_json)


def main():
    parser = argparse.ArgumentParser(
        description="Experimentos controlados da CNN7"
    )

    parser.add_argument(
        "--modo",
        choices=["pre", "early", "late", "three"],
        default="early"
    )

    parser.add_argument(
        "--selecao",
        choices=["central", "maior_tumor"],
        default="central"
    )

    parser.add_argument(
        "--norm",
        choices=["batch", "group"],
        default="batch"
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--pos-weight", type=float, default=3.0
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--epocas", type=int, default=30
    )

    parser.add_argument(
        "--paciencia", type=int, default=8
    )

    parser.add_argument(
        "--batch", type=int, default=8
    )

    parser.add_argument(
        "--tamanho", type=int, default=128
    )

    parser.add_argument(
        "--tag", type=str, default=""
    )

    args = parser.parse_args()

    if args.lr <= 0:
        parser.error("--lr precisa ser positivo")

    if args.pos_weight <= 0:
        parser.error("--pos-weight precisa ser positivo")

    if args.batch < 1 or args.epocas < 1:
        parser.error(
            "Batch e quantidade de epocas precisam ser positivos"
        )

    executar(args)


if __name__ == "__main__":
    main()

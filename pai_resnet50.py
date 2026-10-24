
"""
PAI - BreastDCEDL I-SPY1
ResNet50 pré-treinada no ImageNet.

Primeira estratégia:
- Extrator ResNet50 congelado.
- Cache de características.
- Treinamento da cabeça binária.
- Validação por paciente.
- Conjunto de teste não utilizado.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

from pai_etapa2 import (
    DatasetMRI,
    definir_semente,
    criar_pastas,
    DADOS,
    MODELOS,
    METRICAS
)


PESOS = ResNet50_Weights.IMAGENET1K_V2

CACHE_CARACTERISTICAS = (
    DADOS / "cache" / "resnet50_features"
)


def nome_experimento(args):
    taxa = f"{args.lr:g}".replace(".", "p")
    peso = f"{args.pos_weight:g}".replace(".", "p")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("Tag inválida.")

    return (
        f"resnet50_{args.modo}_{args.selecao}"
        f"_sz{args.tamanho}"
        f"_lr{taxa}_pw{peso}"
        f"_seed{args.seed}_{args.tag}"
    )


def preparar_entrada_imagenet(x):
    """
    Converte a entrada MRI normalizada pelo DatasetMRI
    para uma entrada compatível com a ResNet50 ImageNet.

    Em modo early: 1 canal repetido em 3 canais.
    Em modo three: 3 fases utilizadas como 3 canais.
    """

    if x.ndim != 4:
        raise ValueError(
            f"Tensor esperado [B,C,H,W], encontrado {x.shape}"
        )

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    elif x.shape[1] != 3:
        raise ValueError(
            f"Quantidade de canais inválida: {x.shape[1]}"
        )

    # Resolução espacial utilizada no backbone ImageNet.
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

    # O DatasetMRI fornece intensidades normalizadas.
    x = torch.clamp(x, 0.0, 1.0)

    media = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=x.dtype,
        device=x.device
    ).view(1, 3, 1, 1)

    desvio = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=x.dtype,
        device=x.device
    ).view(1, 3, 1, 1)

    return (x - media) / desvio


def criar_extrator():
    # Na primeira execução, os pesos poderão ser
    # baixados automaticamente pelo TorchVision.
    modelo = resnet50(weights=PESOS)

    # Removemos a camada classificadora do ImageNet.
    # A saída passa a ser um vetor de 2048 elementos.
    modelo.fc = nn.Identity()

    # Congelamos os parâmetros convolucionais.
    modelo.requires_grad_(False)

    # Importante: mantém BatchNorm em modo de avaliação.
    modelo.eval()

    return modelo


def caminho_cache(args, particao):
    nome = (
        f"v1_resnet50_imagenetv2_{args.modo}_"
        f"{args.selecao}_sz{args.tamanho}_"
        f"{particao}.npz"
    )

    return CACHE_CARACTERISTICAS / nome


@torch.no_grad()
def extrair_caracteristicas(extrator, dataset, batch):
    extrator.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=0
    )

    caracteristicas = []
    rotulos = []
    pacientes = []

    total = len(loader)

    for indice, (imagens, y, ids) in enumerate(
        loader, start=1
    ):
        imagens = preparar_entrada_imagenet(imagens)

        vetores = extrator(imagens)

        caracteristicas.append(
            vetores.cpu().numpy().astype(np.float32)
        )

        rotulos.extend(
            y.cpu().numpy().astype(np.float32).tolist()
        )

        pacientes.extend(ids)

        if indice == 1 or indice % 10 == 0 or indice == total:
            print(
                f"Extração: lote {indice}/{total}",
                flush=True
            )

    return (
        np.concatenate(caracteristicas, axis=0),
        np.asarray(rotulos, dtype=np.float32),
        np.asarray(pacientes, dtype=str)
    )


def obter_caracteristicas(args, particao):
    CACHE_CARACTERISTICAS.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho = caminho_cache(args, particao)

    if caminho.exists():
        print(f"Reutilizando cache: {caminho}")

        with np.load(caminho, allow_pickle=False) as dados:
            x = dados["x"].copy()
            y = dados["y"].copy()
            ids = dados["pid"].copy()

    else:
        aumentar = particao == "treino"

        codigo_particao = 0 if aumentar else 2

        dataset = DatasetMRI(
            particao=codigo_particao,
            selecao=args.selecao,
            modo=args.modo,
            tamanho=args.tamanho,
            aumentar=aumentar
        )

        esperado = 104 if aumentar else 33

        if len(dataset.registros) != esperado:
            raise RuntimeError(
                f"Partição {particao} incompleta: "
                f"{len(dataset.registros)} pacientes."
            )

        print(
            f"\nExtraindo características: {particao}",
            flush=True
        )

        extrator = criar_extrator()

        x, y, ids = extrair_caracteristicas(
            extrator,
            dataset,
            args.batch_extracao
        )

        del extrator

        np.savez_compressed(
            caminho,
            x=x,
            y=y,
            pid=ids
        )

        print(f"Cache salvo: {caminho}", flush=True)

    quantidade = 1040 if particao == "treino" else 33

    if (
        len(x) != quantidade
        or len(y) != quantidade
        or len(ids) != quantidade
        or x.shape[1] != 2048
    ):
        raise RuntimeError(
            f"Cache inválido: {particao}, shape={x.shape}"
        )

    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
        ids
    )


def calcular_metricas(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)

    previstos = (prob >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, previstos, labels=[0, 1]
    ).ravel()

    return {
        "acuracia": float(
            accuracy_score(y, previstos)
        ),
        "acuracia_balanceada": float(
            balanced_accuracy_score(y, previstos)
        ),
        "sensibilidade": float(
            recall_score(y, previstos, zero_division=0)
        ),
        "especificidade": (
            float(vn / (vn + fp)) if vn + fp else 0.0
        ),
        "precisao": float(
            precision_score(y, previstos, zero_division=0)
        ),
        "f1": float(
            f1_score(y, previstos, zero_division=0)
        ),
        "auc_roc": float(
            roc_auc_score(y, prob)
        ),
        "average_precision": float(
            average_precision_score(y, prob)
        ),
        "brier": float(
            brier_score_loss(y, prob)
        ),
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn)
    }


def executar(args):
    criar_pastas()
    definir_semente(args.seed)

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    nome = nome_experimento(args)

    destino_modelo = MODELOS / f"{nome}.pth"
    destino_historico = METRICAS / f"{nome}_historico.csv"
    destino_previsoes = METRICAS / f"{nome}_previsoes.csv"
    destino_resumo = METRICAS / f"{nome}_resumo.json"

    # Não permitimos sobrescrever um experimento existente.
    for destino in (
        destino_modelo,
        destino_historico,
        destino_previsoes,
        destino_resumo
    ):
        if destino.exists():
            raise FileExistsError(
                f"Resultado já existe: {destino}"
            )

    inicio = time.perf_counter()

    x_treino, y_treino, _ = obter_caracteristicas(
        args, "treino"
    )

    x_val, y_val, ids_val = obter_caracteristicas(
        args, "validacao"
    )

    tempo_extracao = time.perf_counter() - inicio

    print("\nCaracterísticas de treino:", tuple(x_treino.shape))
    print("Características de validação:", tuple(x_val.shape))

    definir_semente(args.seed)

    # Apenas esta cabeça será treinada.
    cabeca = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(2048, 1)
    )

    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [args.pos_weight],
            dtype=torch.float32
        )
    )

    otimizador = torch.optim.AdamW(
        cabeca.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    gerador = torch.Generator()
    gerador.manual_seed(args.seed)

    loader = DataLoader(
        TensorDataset(x_treino, y_treino),
        batch_size=args.batch,
        shuffle=True,
        generator=gerador,
        num_workers=0
    )

    melhor_loss = float("inf")
    melhor_epoca = 0
    melhor_metricas = None
    melhor_probabilidades = None
    sem_melhora = 0
    historico = []

    inicio_treino = time.perf_counter()

    for epoca in range(1, args.epocas + 1):
        cabeca.train()
        perda_total = 0.0

        for x, y in loader:
            otimizador.zero_grad(set_to_none=True)

            logits = cabeca(x).squeeze(1)
            perda = criterio(logits, y)

            perda.backward()
            otimizador.step()

            perda_total += perda.item() * len(y)

        perda_treino = perda_total / len(x_treino)

        cabeca.eval()

        with torch.no_grad():
            logits_val = cabeca(x_val).squeeze(1)

            loss_val = float(
                criterio(logits_val, y_val).item()
            )

            probabilidades = (
                torch.sigmoid(logits_val).numpy()
            )

        metricas = calcular_metricas(
            y_val.numpy(),
            probabilidades
        )

        historico.append({
            "epoca": epoca,
            "loss_treino": perda_treino,
            "loss_validacao": loss_val,
            **metricas
        })

        print(
            f"Época {epoca:03d} | "
            f"Treino {perda_treino:.4f} | "
            f"Val {loss_val:.4f} | "
            f"F1 {metricas['f1']:.4f} | "
            f"AUC {metricas['auc_roc']:.4f}"
        )

        if loss_val < melhor_loss - 1e-5:
            melhor_loss = loss_val
            melhor_epoca = epoca
            melhor_metricas = metricas.copy()
            melhor_probabilidades = probabilidades.copy()
            sem_melhora = 0

            torch.save({
                "pesos_cabeca": cabeca.state_dict(),
                "arquitetura": "ResNet50",
                "pesos_backbone": "IMAGENET1K_V2",
                "backbone_congelado": True,
                "modo": args.modo,
                "selecao": args.selecao,
                "tamanho": args.tamanho,
                "pos_weight": args.pos_weight,
                "seed": args.seed,
                "epoca": epoca,
                "loss_validacao": loss_val
            }, destino_modelo)

        else:
            sem_melhora += 1

        if sem_melhora >= args.paciencia:
            print("Early stopping ativado.")
            break

    duracao_treino = (
        time.perf_counter() - inicio_treino
    )

    pd.DataFrame(historico).to_csv(
        destino_historico,
        index=False
    )

    pd.DataFrame({
        "pid": ids_val,
        "pCR_real": y_val.numpy().astype(int),
        "probabilidade": melhor_probabilidades
    }).to_csv(
        destino_previsoes,
        index=False
    )

    resumo = {
        "experimento": nome,
        "arquitetura": "ResNet50",
        "estrategia": "extrator_congelado",
        "melhor_epoca": melhor_epoca,
        "melhor_loss_validacao": melhor_loss,
        "tempo_extracao_segundos": tempo_extracao,
        "tempo_treinamento_segundos": duracao_treino,
        "parametros": vars(args),
        "metricas": melhor_metricas
    }

    with open(
        destino_resumo,
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            indent=2,
            ensure_ascii=False
        )

    print("\nTreinamento concluído.")
    print("Melhor época:", melhor_epoca)
    print("AUC-ROC:", f"{melhor_metricas['auc_roc']:.4f}")
    print("F1:", f"{melhor_metricas['f1']:.4f}")
    print(
        "Acurácia balanceada:",
        f"{melhor_metricas['acuracia_balanceada']:.4f}"
    )
    print(
        "Tempo de extração:",
        f"{tempo_extracao / 60:.2f} minutos"
    )
    print(
        "Tempo da cabeça:",
        f"{duracao_treino / 60:.2f} minutos"
    )
    print("Checkpoint:", destino_modelo)
    print("Previsões:", destino_previsoes)


def main():
    parser = argparse.ArgumentParser(
        description="ResNet50 ImageNet para pCR"
    )

    parser.add_argument(
        "--modo",
        choices=["early", "three"],
        default="early"
    )

    parser.add_argument(
        "--selecao",
        choices=["central", "maior_tumor"],
        default="central"
    )

    parser.add_argument(
        "--tamanho", type=int, default=128
    )

    parser.add_argument(
        "--batch-extracao", type=int, default=4
    )

    parser.add_argument(
        "--batch", type=int, default=64
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--pos-weight", type=float, default=1.0
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
        "--tag", type=str, default="R01"
    )

    args = parser.parse_args()

    if args.lr <= 0 or args.pos_weight <= 0:
        parser.error(
            "Learning rate e peso positivo devem ser maiores que zero."
        )

    if args.batch < 1 or args.batch_extracao < 1:
        parser.error("Batch precisa ser positivo.")

    if args.tamanho < 32 or args.epocas < 1:
        parser.error("Tamanho ou épocas inválidos.")

    executar(args)


if __name__ == "__main__":
    main()

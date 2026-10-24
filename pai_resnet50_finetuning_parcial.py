
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)

from pai_etapa2 import (
    DatasetMRI,
    MODELOS,
    METRICAS,
    criar_pastas,
    definir_semente
)

from pai_resnet50 import preparar_entrada_imagenet


def manter_batchnorm_congelado(modelo):
    for camada in modelo.modules():
        if isinstance(camada, nn.BatchNorm2d):
            camada.eval()


@torch.no_grad()
def avaliar(modelo, loader, criterio):
    modelo.eval()

    perdas = 0.0
    rotulos = []
    probabilidades = []
    pacientes = []

    for imagens, y, ids in loader:
        x = preparar_entrada_imagenet(imagens)

        logits = modelo(x).squeeze(1)
        perda = criterio(logits, y)

        perdas += perda.item() * len(y)

        probs = torch.sigmoid(logits)

        rotulos.extend(y.numpy().astype(int))
        probabilidades.extend(probs.numpy())
        pacientes.extend(ids)

    y = np.asarray(rotulos, dtype=int)
    p = np.asarray(probabilidades, dtype=float)
    previstos = (p >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, previstos, labels=[0, 1]
    ).ravel()

    metricas = {
        "auc_roc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, previstos, zero_division=0)),
        "acuracia_balanceada": float(
            balanced_accuracy_score(y, previstos)
        ),
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn)
    }

    previsoes = pd.DataFrame({
        "pid": pacientes,
        "pCR_real": y,
        "probabilidade": p
    })

    return perdas / len(loader.dataset), metricas, previsoes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="FT01")

    args = parser.parse_args()

    if args.epocas < 1 or args.paciencia < 1 or args.batch < 1:
        parser.error("Épocas, paciência e batch devem ser positivos.")

    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Tag inválida.")

    criar_pastas()
    definir_semente(args.seed)

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    nome = f"resnet50_finetune_early_central_{args.tag}"

    saida_modelo = MODELOS / f"{nome}.pth"
    saida_historico = METRICAS / f"{nome}_historico.csv"
    saida_previsoes = METRICAS / f"{nome}_previsoes.csv"
    saida_resumo = METRICAS / f"{nome}_resumo.json"

    for caminho in (
        saida_modelo,
        saida_historico,
        saida_previsoes,
        saida_resumo
    ):
        if caminho.exists():
            raise FileExistsError(
                f"Resultado já existe: {caminho}"
            )

    print("Carregando datasets...", flush=True)

    treino = DatasetMRI(
        particao=0,
        selecao="central",
        modo="early",
        tamanho=128,
        aumentar=True
    )

    validacao = DatasetMRI(
        particao=2,
        selecao="central",
        modo="early",
        tamanho=128,
        aumentar=False
    )

    if len(treino.registros) != 104:
        raise RuntimeError("Treinamento incompleto.")

    if len(validacao.registros) != 33:
        raise RuntimeError("Validação incompleta.")

    gerador = torch.Generator()
    gerador.manual_seed(args.seed)

    loader_treino = DataLoader(
        treino,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        generator=gerador
    )

    loader_val = DataLoader(
        validacao,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    print("Carregando ResNet50...", flush=True)

    modelo = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )

    modelo.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(2048, 1)
    )

    checkpoint_inicial = (
        MODELOS /
        "resnet50_early_central_sz128_lr0p001_pw1_seed42_R01_corrigido.pth"
    )

    checkpoint = torch.load(
        checkpoint_inicial,
        map_location="cpu",
        weights_only=True
    )

    modelo.fc.load_state_dict(
        checkpoint["pesos_cabeca"]
    )

    # Congelamos tudo inicialmente.
    modelo.requires_grad_(False)

    # Liberamos somente o último bloco e a cabeça.
    modelo.layer4.requires_grad_(True)
    modelo.fc.requires_grad_(True)

    criterio = nn.BCEWithLogitsLoss()

    otimizador = torch.optim.AdamW(
        [
            {
                "params": modelo.layer4.parameters(),
                "lr": 0.00001
            },
            {
                "params": modelo.fc.parameters(),
                "lr": 0.0001
            }
        ],
        weight_decay=1e-4
    )

    melhor_loss = float("inf")
    melhor_epoca = 0
    melhor_metricas = None
    melhores_previsoes = None
    sem_melhora = 0
    historico = []

    inicio_total = time.perf_counter()

    print("\nINICIANDO FINE-TUNING COMPLETO.")
    print("Pacientes de treino:", len(treino.registros))
    print("Amostras aumentadas:", len(treino))
    print("Pacientes de validação:", len(validacao.registros))
    print("Lotes por época:", len(loader_treino))
    print("Máximo de épocas:", args.epocas)
    print()

    for epoca in range(1, args.epocas + 1):
        inicio_epoca = time.perf_counter()

        modelo.train()
        manter_batchnorm_congelado(modelo)

        perda_acumulada = 0.0

        for indice, (imagens, y, _) in enumerate(
            loader_treino, start=1
        ):
            x = preparar_entrada_imagenet(imagens)

            otimizador.zero_grad(
                set_to_none=True
            )

            logits = modelo(x).squeeze(1)

            perda = criterio(logits, y)

            perda.backward()
            otimizador.step()

            perda_acumulada += perda.item() * len(y)

            if indice % 100 == 0:
                print(
                    f"Época {epoca:03d} | "
                    f"Lote {indice}/{len(loader_treino)}",
                    flush=True
                )

        loss_treino = perda_acumulada / len(treino)

        loss_val, metricas, previsoes = avaliar(
            modelo, loader_val, criterio
        )

        duracao = time.perf_counter() - inicio_epoca

        historico.append({
            "epoca": epoca,
            "loss_treino": loss_treino,
            "loss_validacao": loss_val,
            "duracao_segundos": duracao,
            **metricas
        })

        print(
            f"\nÉpoca {epoca:03d} | "
            f"Treino {loss_treino:.4f} | "
            f"Val {loss_val:.4f} | "
            f"F1 {metricas['f1']:.4f} | "
            f"AUC {metricas['auc_roc']:.4f} | "
            f"Tempo {duracao / 60:.2f} min\n",
            flush=True
        )

        if loss_val < melhor_loss - 1e-5:
            melhor_loss = loss_val
            melhor_epoca = epoca
            melhor_metricas = metricas.copy()
            melhores_previsoes = previsoes.copy()
            sem_melhora = 0

            torch.save({
                "pesos_modelo": modelo.state_dict(),
                "arquitetura": "ResNet50",
                "estrategia": "fine_tuning_layer4",
                "modo": "early",
                "selecao": "central",
                "tamanho": 128,
                "seed": args.seed,
                "epoca": epoca,
                "loss_validacao": loss_val
            }, saida_modelo)

            print(
                "Novo melhor checkpoint salvo.",
                flush=True
            )
        else:
            sem_melhora += 1

        if sem_melhora >= args.paciencia:
            print("Early stopping ativado.")
            break

    tempo_total = time.perf_counter() - inicio_total

    pd.DataFrame(historico).to_csv(
        saida_historico,
        index=False
    )

    melhores_previsoes.to_csv(
        saida_previsoes,
        index=False
    )

    resumo = {
        "experimento": nome,
        "melhor_epoca": melhor_epoca,
        "melhor_loss_validacao": melhor_loss,
        "metricas": melhor_metricas,
        "tempo_total_segundos": tempo_total,
        "parametros": vars(args)
    }

    with open(
        saida_resumo,
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            resumo, arquivo,
            indent=2,
            ensure_ascii=False
        )

    print("\nTREINAMENTO CONCLUÍDO.")
    print("Melhor época:", melhor_epoca)
    print("AUC-ROC:", melhor_metricas["auc_roc"])
    print("F1:", melhor_metricas["f1"])
    print("Tempo total:", f"{tempo_total / 60:.2f} min")
    print("Modelo:", saida_modelo)
    print("Previsões:", saida_previsoes)


if __name__ == "__main__":
    main()

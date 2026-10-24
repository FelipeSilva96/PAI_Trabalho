
import argparse
import json
import re
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


VERSAO = "FT_PARCIAL_V2"


def congelar_estatisticas_batchnorm(modelo):
    """
    Mantém as estatísticas acumuladas do BatchNorm fixas.
    """
    for camada in modelo.modules():
        if isinstance(camada, nn.BatchNorm2d):
            camada.eval()


@torch.no_grad()
def avaliar(modelo, loader, criterio):
    modelo.eval()

    perda_total = 0.0
    rotulos = []
    probabilidades = []
    pacientes = []

    for imagens, y, ids in loader:
        x = preparar_entrada_imagenet(imagens)

        logits = modelo(x).squeeze(1)
        perda = criterio(logits, y)

        perda_total += perda.item() * len(y)

        probabilidades.extend(
            torch.sigmoid(logits).cpu().numpy().tolist()
        )
        rotulos.extend(y.cpu().numpy().astype(int).tolist())
        pacientes.extend(ids)

    y = np.asarray(rotulos, dtype=int)
    p = np.asarray(probabilidades, dtype=float)

    previstos = (p >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, previstos, labels=[0, 1]
    ).ravel()

    metricas = {
        "auc_roc": float(roc_auc_score(y, p)),
        "f1": float(
            f1_score(y, previstos, zero_division=0)
        ),
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

    loss_media = perda_total / len(loader.dataset)

    return loss_media, metricas, previsoes


def salvar_checkpoint(
    modelo, destino, epoca, loss, args
):
    torch.save({
        "pesos_modelo": modelo.state_dict(),
        "arquitetura": "ResNet50",
        "estrategia": "fine_tuning_ultimo_bloco",
        "versao": VERSAO,
        "modo": "early",
        "selecao": "central",
        "tamanho": 128,
        "seed": args.seed,
        "epoca": epoca,
        "loss_validacao": loss
    }, destino)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning parcial da ResNet50"
    )

    parser.add_argument(
        "--epocas", type=int, default=20
    )
    parser.add_argument(
        "--paciencia", type=int, default=4
    )
    parser.add_argument(
        "--batch", type=int, default=2
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--tag", default="FT02_CORRIGIDO"
    )
    parser.add_argument(
        "--diagnostico", action="store_true"
    )

    args = parser.parse_args()

    if args.epocas < 1 or args.paciencia < 1:
        parser.error("Épocas e paciência devem ser positivas.")

    if args.batch < 1:
        parser.error("Batch deve ser positivo.")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("Tag inválida.")

    criar_pastas()
    definir_semente(args.seed)

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    print("=" * 65)
    print("VERSAO DO SCRIPT:", VERSAO)
    print("ARQUIVO EXECUTADO:", Path(__file__).resolve())
    print("ESTRATEGIA: SOMENTE LAYER4[-1] + FC")
    print("=" * 65, flush=True)

    nome = (
        f"resnet50_finetune_early_central_{args.tag}"
    )

    saida_modelo = MODELOS / f"{nome}.pth"
    saida_historico = METRICAS / f"{nome}_historico.csv"
    saida_previsoes = METRICAS / f"{nome}_previsoes.csv"
    saida_resumo = METRICAS / f"{nome}_resumo.json"

    # No modo diagnóstico, nenhum resultado será salvo.
    if not args.diagnostico:
        for caminho in (
            saida_modelo,
            saida_historico,
            saida_previsoes,
            saida_resumo
        ):
            if caminho.exists():
                raise FileExistsError(
                    f"Arquivo já existe: {caminho}"
                )

    print("\nCarregando datasets...", flush=True)

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
        raise RuntimeError(
            "Treinamento precisa conter 104 pacientes."
        )

    if len(validacao.registros) != 33:
        raise RuntimeError(
            "Validação precisa conter 33 pacientes."
        )

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

    print("Carregando ResNet50 ImageNet V2...", flush=True)

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

    if not checkpoint_inicial.is_file():
        raise FileNotFoundError(checkpoint_inicial)

    checkpoint = torch.load(
        checkpoint_inicial,
        map_location="cpu",
        weights_only=True
    )

    modelo.fc.load_state_dict(
        checkpoint["pesos_cabeca"]
    )

    # -----------------------------------------------
    # CONGELAMENTO CORRETO
    # -----------------------------------------------

    # Primeiro: congelar absolutamente toda a rede.
    modelo.requires_grad_(False)

    # Segundo: liberar apenas o último bloco residual.
    modelo.layer4[-1].requires_grad_(True)

    # Terceiro: liberar a cabeça classificadora.
    modelo.fc.requires_grad_(True)

    parametros_totais = sum(
        p.numel() for p in modelo.parameters()
    )

    parametros_treinaveis = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    nomes_treinaveis = [
        nome_parametro
        for nome_parametro, parametro in modelo.named_parameters()
        if parametro.requires_grad
    ]

    # Confirma que não liberamos outras camadas por engano.
    for nome_parametro in nomes_treinaveis:
        if not (
            nome_parametro.startswith("layer4.2.")
            or nome_parametro.startswith("fc.")
        ):
            raise RuntimeError(
                "Parâmetro indevidamente liberado: "
                + nome_parametro
            )

    print("\nDIAGNOSTICO DE PARAMETROS")
    print("Parâmetros totais:", f"{parametros_totais:,}")
    print(
        "Parâmetros treináveis:",
        f"{parametros_treinaveis:,}"
    )
    print(
        "Percentual treinável:",
        f"{100 * parametros_treinaveis / parametros_totais:.2f}%"
    )
    print(
        "Exemplos de parâmetros treináveis:",
        nomes_treinaveis[:5]
    )

    if parametros_treinaveis >= 10_000_000:
        raise RuntimeError(
            "Muitos parâmetros liberados. "
            "Verifique o congelamento da rede."
        )

    criterio = nn.BCEWithLogitsLoss()

    otimizador = torch.optim.AdamW(
        [
            {
                "params": modelo.layer4[-1].parameters(),
                "lr": 0.000001
            },
            {
                "params": modelo.fc.parameters(),
                "lr": 0.00001
            }
        ],
        weight_decay=1e-3
    )

    # -----------------------------------------------
    # ÉPOCA ZERO: ANTES DO TREINAMENTO
    # -----------------------------------------------

    loss_inicial, metricas_iniciais, previsoes_iniciais = (
        avaliar(modelo, loader_val, criterio)
    )

    print("\n" + "=" * 65)
    print("AVALIACAO INICIAL - EPOCA ZERO")
    print("=" * 65)
    print(f"Loss inicial: {loss_inicial:.4f}")
    print(
        f"AUC inicial: "
        f"{metricas_iniciais['auc_roc']:.4f}"
    )
    print(
        f"F1 inicial: "
        f"{metricas_iniciais['f1']:.4f}"
    )
    print(
        "Acurácia balanceada inicial:",
        f"{metricas_iniciais['acuracia_balanceada']:.4f}"
    )
    print("=" * 65, flush=True)

    if args.diagnostico:
        print("\nDIAGNOSTICO CONCLUIDO.")
        print("Nenhum lote foi treinado.")
        print("Nenhum checkpoint foi salvo.")
        return

    melhor_loss = loss_inicial
    melhor_epoca = 0
    melhor_metricas = metricas_iniciais.copy()
    melhores_previsoes = previsoes_iniciais.copy()

    sem_melhora = 0

    historico = [{
        "epoca": 0,
        "loss_treino": float("nan"),
        "loss_validacao": loss_inicial,
        "duracao_segundos": 0.0,
        **metricas_iniciais
    }]

    # A época zero é nosso primeiro checkpoint.
    salvar_checkpoint(
        modelo,
        saida_modelo,
        0,
        loss_inicial,
        args
    )

    inicio_total = time.perf_counter()

    print("\nINICIANDO FINE-TUNING PARCIAL.")
    print("Pacientes de treino:", len(treino.registros))
    print("Amostras aumentadas:", len(treino))
    print("Pacientes de validação:", len(validacao.registros))
    print("Lotes por época:", len(loader_treino))
    print("Máximo de épocas:", args.epocas)
    print()

    for epoca in range(1, args.epocas + 1):
        inicio_epoca = time.perf_counter()

        modelo.train()

        # BatchNorm não atualiza estatísticas acumuladas.
        congelar_estatisticas_batchnorm(modelo)

        perda_acumulada = 0.0

        for indice, (imagens, y, _) in enumerate(
            loader_treino, start=1
        ):
            x = preparar_entrada_imagenet(imagens)

            otimizador.zero_grad(set_to_none=True)

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
            modelo,
            loader_val,
            criterio
        )

        duracao_epoca = (
            time.perf_counter() - inicio_epoca
        )

        historico.append({
            "epoca": epoca,
            "loss_treino": loss_treino,
            "loss_validacao": loss_val,
            "duracao_segundos": duracao_epoca,
            **metricas
        })

        print(
            f"\nÉpoca {epoca:03d} | "
            f"Treino {loss_treino:.4f} | "
            f"Val {loss_val:.4f} | "
            f"F1 {metricas['f1']:.4f} | "
            f"AUC {metricas['auc_roc']:.4f} | "
            f"Tempo {duracao_epoca / 60:.2f} min\n",
            flush=True
        )

        if loss_val < melhor_loss - 1e-5:
            melhor_loss = loss_val
            melhor_epoca = epoca
            melhor_metricas = metricas.copy()
            melhores_previsoes = previsoes.copy()
            sem_melhora = 0

            salvar_checkpoint(
                modelo,
                saida_modelo,
                epoca,
                loss_val,
                args
            )

            print(
                "Novo melhor checkpoint salvo.",
                flush=True
            )

        else:
            sem_melhora += 1

        if sem_melhora >= args.paciencia:
            print("Early stopping ativado.")
            break

    duracao_total = (
        time.perf_counter() - inicio_total
    )

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
        "versao": VERSAO,
        "estrategia": "fine_tuning_ultimo_bloco",
        "parametros_treinaveis": parametros_treinaveis,
        "melhor_epoca": melhor_epoca,
        "loss_validacao": melhor_loss,
        "metricas": melhor_metricas,
        "tempo_total_segundos": duracao_total,
        "parametros": vars(args)
    }

    with open(
        saida_resumo,
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("FINE-TUNING PARCIAL CONCLUIDO")
    print("=" * 65)

    print("Melhor época:", melhor_epoca)
    print("Melhor loss:", f"{melhor_loss:.4f}")
    print(
        "AUC-ROC:",
        f"{melhor_metricas['auc_roc']:.4f}"
    )
    print("F1:", f"{melhor_metricas['f1']:.4f}")
    print(
        "Acurácia balanceada:",
        f"{melhor_metricas['acuracia_balanceada']:.4f}"
    )
    print(
        "Tempo total:",
        f"{duracao_total / 60:.2f} minutos"
    )
    print("Checkpoint:", saida_modelo)
    print("Previsões:", saida_previsoes)


if __name__ == "__main__":
    main()


import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from pai_etapa2 import (
    DatasetMRI,
    definir_semente,
    MODELOS
)

from pai_resnet50 import preparar_entrada_imagenet


def main():
    definir_semente(42)

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    print("Preparando dataset...", flush=True)

    dataset = DatasetMRI(
        particao=0,
        selecao="central",
        modo="early",
        tamanho=128,
        aumentar=True
    )

    if len(dataset.registros) != 104:
        raise RuntimeError(
            "Quantidade incorreta de pacientes."
        )

    batch = 2

    gerador = torch.Generator()
    gerador.manual_seed(42)

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        generator=gerador
    )

    print("Carregando ResNet50 ImageNet V2...", flush=True)

    modelo = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )

    # Congelamos inicialmente toda a ResNet50.
    modelo.requires_grad_(False)

    # Reconstruímos a cabeça usada no experimento R01.
    definir_semente(42)

    modelo.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(2048, 1)
    )

    checkpoint = (
        MODELOS /
        "resnet50_early_central_sz128_lr0p001_pw1_seed42_R01_corrigido.pth"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    dados = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    modelo.fc.load_state_dict(
        dados["pesos_cabeca"]
    )

    # Liberamos apenas o último bloco residual
    # e a cabeça de classificação.
    modelo.layer4.requires_grad_(True)
    modelo.fc.requires_grad_(True)

    total = sum(
        p.numel() for p in modelo.parameters()
    )

    treinaveis = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print("\nParâmetros totais:", f"{total:,}")
    print("Parâmetros treináveis:", f"{treinaveis:,}")

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

    criterio = nn.BCEWithLogitsLoss()

    modelo.train()

    # Manter BatchNorm em modo eval evita atualizar
    # suas médias e variâncias acumuladas.
    for modulo in modelo.modules():
        if isinstance(modulo, nn.BatchNorm2d):
            modulo.eval()

    print("\nINICIANDO TESTE COM APENAS 3 LOTES.")
    print("Nenhum checkpoint será modificado.\n")

    tempos = []

    for indice, (imagens, y, ids) in enumerate(loader):
        if indice >= 3:
            break

        inicio = time.perf_counter()

        imagens = preparar_entrada_imagenet(
            imagens
        )

        otimizador.zero_grad(
            set_to_none=True
        )

        logits = modelo(imagens).squeeze(1)

        perda = criterio(
            logits,
            y.float()
        )

        perda.backward()
        otimizador.step()

        duracao = time.perf_counter() - inicio
        tempos.append(duracao)

        print(
            f"Lote {indice + 1}/3 | "
            f"Tempo: {duracao:.2f} s | "
            f"Loss: {perda.item():.4f}",
            flush=True
        )

    print("\nTESTE CONCLUÍDO.")

    if len(tempos) >= 2:
        # Descartamos o primeiro lote por poder conter
        # custos adicionais de inicialização.
        media = sum(tempos[1:]) / len(tempos[1:])
    else:
        media = tempos[0]

    lotes_por_epoca = math.ceil(
        len(dataset) / batch
    )

    estimativa = media * lotes_por_epoca

    print(
        "Tempo médio dos lotes considerados:",
        f"{media:.2f} segundos"
    )

    print(
        "Lotes por época completa:",
        lotes_por_epoca
    )

    print(
        "Estimativa aproximada de treinamento por época:",
        f"{estimativa / 60:.2f} minutos"
    )

    print(
        "\nAVISO: Esta estimativa não inclui "
        "uma validação completa e pode variar."
    )


if __name__ == "__main__":
    main()

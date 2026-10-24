
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from pai_etapa2 import (
    CNN7,
    DatasetMRI,
    CACHE,
    MODELOS,
    METRICAS,
    GRAFICOS
)

from pai_resnet50 import preparar_entrada_imagenet
from pai_fusao_temporal import RedeFusaoTemporal


CHECKPOINTS = {
    "cnn7": (
        MODELOS / "cnn7_early_central_nenhuma.pth"
    ),
    "resnet50": (
        MODELOS /
        "resnet50_early_central_sz128_lr0p001_pw1_seed42_R01_corrigido.pth"
    ),
    "fusao": (
        MODELOS /
        "fusao_temporal_central_sz128_lr0p0003_pw3_seed42_FUS02.pth"
    )
}

PASTA_SAIDA = GRAFICOS / "gradcam"


class CapturaGradCAM:
    """
    Registra as ativações de uma camada convolucional
    e seus gradientes durante a retropropagação.
    """

    def __init__(self, camada):
        self.ativacoes = None
        self.gradientes = None

        self.handle = camada.register_forward_hook(
            self._capturar
        )

    def _receber_gradientes(self, gradiente):
        self.gradientes = gradiente.detach()

    def _capturar(self, modulo, entradas, saida):
        if not isinstance(saida, torch.Tensor):
            raise TypeError(
                "A camada selecionada não retornou um tensor."
            )

        self.ativacoes = saida.detach()
        self.gradientes = None

        if not saida.requires_grad:
            raise RuntimeError(
                "A saída da camada não participa do "
                "grafo de gradientes."
            )

        saida.register_hook(
            self._receber_gradientes
        )

    def fechar(self):
        self.handle.remove()


def carregar_modelo(tipo, caminho):
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {caminho}"
        )

    checkpoint = torch.load(
        caminho,
        map_location="cpu",
        weights_only=True
    )

    if tipo == "cnn7":
        modo = checkpoint.get("modo", "early")
        selecao = checkpoint.get("selecao", "central")
        tamanho = int(checkpoint.get("tamanho", 128))

        if checkpoint.get("normalizacao", "batch") != "batch":
            raise ValueError(
                "Este script suporta checkpoints CNN7 "
                "com BatchNorm, não GroupNorm."
            )

        if checkpoint.get("mascara", "nenhuma") != "nenhuma":
            raise ValueError(
                "Checkpoint CNN7 usa uma entrada com "
                "máscara que este exemplo não suporta."
            )

        canais = 3 if modo == "three" else 1

        modelo = CNN7(
            canais_entrada=canais
        )

        modelo.load_state_dict(
            checkpoint["pesos"]
        )

        convolucoes = [
            modulo
            for modulo in modelo.extrator.modules()
            if isinstance(modulo, nn.Conv2d)
        ]

        camada_alvo = convolucoes[-1]

    elif tipo == "resnet50":
        modo = checkpoint.get("modo", "early")
        selecao = checkpoint.get("selecao", "central")
        tamanho = int(checkpoint.get("tamanho", 128))

        modelo = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        modelo.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(2048, 1)
        )

        if "pesos_cabeca" in checkpoint:
            modelo.fc.load_state_dict(
                checkpoint["pesos_cabeca"]
            )
        elif "pesos_modelo" in checkpoint:
            modelo.load_state_dict(
                checkpoint["pesos_modelo"]
            )
        else:
            raise KeyError(
                "Pesos da ResNet50 não encontrados."
            )

        camada_alvo = modelo.layer4[-1]

    elif tipo == "fusao":
        modo = "three"
        selecao = checkpoint.get("selecao", "central")
        tamanho = int(checkpoint.get("tamanho", 128))

        modelo = RedeFusaoTemporal()

        modelo.load_state_dict(
            checkpoint["pesos_modelo"]
        )

        convolucoes = [
            modulo
            for modulo in modelo.extrator.modules()
            if isinstance(modulo, nn.Conv2d)
        ]

        camada_alvo = convolucoes[-1]

    else:
        raise ValueError(
            f"Arquitetura desconhecida: {tipo}"
        )

    if modo not in ("early", "three"):
        raise ValueError(
            f"Modo temporal não suportado: {modo}"
        )

    if selecao not in ("central", "maior_tumor"):
        raise ValueError(
            f"Seleção de corte inválida: {selecao}"
        )

    # Não queremos atualizar parâmetros.
    # As entradas terão requires_grad=True para
    # permitir calcular o Grad-CAM.
    modelo.requires_grad_(False)
    modelo.eval()

    return modelo, camada_alvo, modo, selecao, tamanho


def carregar_mascara(pid, selecao, tamanho):
    caminho = CACHE / selecao / f"{pid}.npz"

    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    with np.load(caminho, allow_pickle=False) as dados:
        mascara = dados["mascara"].astype(np.float32)

    tensor = torch.from_numpy(
        np.ascontiguousarray(mascara)
    )[None, None]

    redimensionada = F.interpolate(
        tensor,
        size=(tamanho, tamanho),
        mode="nearest"
    )

    return (
        redimensionada[0, 0].numpy() > 0.5
    )


def calcular_gradcam(
    modelo,
    camada_alvo,
    imagem,
    tipo,
    alvo
):
    """
    Calcula Grad-CAM para uma única amostra.

    Na fusão temporal, o extrator recebe as três
    fases como três elementos do batch interno.
    Por isso, retornamos três mapas.
    """

    x = imagem.unsqueeze(0).clone().detach()
    x.requires_grad_(True)

    captura = CapturaGradCAM(camada_alvo)

    try:
        modelo.zero_grad(set_to_none=True)

        if tipo == "resnet50":
            entrada_modelo = preparar_entrada_imagenet(x)
        else:
            entrada_modelo = x

        logits = modelo(entrada_modelo).reshape(-1)

        if logits.numel() != 1:
            raise RuntimeError(
                "Esperada uma única previsão por paciente."
            )

        logit = logits[0]
        probabilidade = float(
            torch.sigmoid(logit).detach().item()
        )

        previsao = int(probabilidade >= 0.5)

        # Explicação da classe pCR positiva ou negativa.
        if alvo == "pcr1":
            escore_alvo = logit
        elif alvo == "pcr0":
            escore_alvo = -logit
        elif alvo == "previsto":
            escore_alvo = (
                logit if previsao == 1 else -logit
            )
        else:
            raise ValueError("Alvo Grad-CAM inválido.")

        escore_alvo.backward()

        ativacoes = captura.ativacoes
        gradientes = captura.gradientes

        if ativacoes is None or gradientes is None:
            raise RuntimeError(
                "Não foi possível capturar ativações "
                "e gradientes."
            )

        # Peso médio de cada canal convolucional.
        pesos = gradientes.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Soma ponderada das ativações.
        mapas = torch.relu(
            (pesos * ativacoes).sum(
                dim=1,
                keepdim=True
            )
        )

        mapas = F.interpolate(
            mapas,
            size=imagem.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        mapas = (
            mapas[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        esperado = 3 if tipo == "fusao" else 1

        if mapas.shape[0] != esperado:
            raise RuntimeError(
                "Número inesperado de mapas Grad-CAM: "
                f"{mapas.shape[0]}"
            )

        # Utilizamos a mesma escala para as três
        # fases de um mesmo paciente.
        maximo = float(np.max(mapas))

        if maximo > 1e-12:
            mapas = mapas / maximo
        else:
            mapas = np.zeros_like(mapas)

        return probabilidade, previsao, mapas

    finally:
        captura.fechar()


def verificar_previsao_salva(
    caminho_checkpoint,
    pid,
    probabilidade
):
    """
    Confere a probabilidade com o CSV salvo durante
    o treinamento, quando esse CSV existe.
    """

    caminho = (
        METRICAS /
        f"{caminho_checkpoint.stem}_previsoes.csv"
    )

    if not caminho.is_file():
        return

    dados = pd.read_csv(caminho)

    linha = dados[
        dados["pid"].astype(str) == str(pid)
    ]

    if len(linha) != 1:
        raise RuntimeError(
            f"Paciente {pid} não encontrado uma "
            "única vez no CSV de previsões."
        )

    esperado = float(
        linha.iloc[0]["probabilidade"]
    )

    diferenca = abs(
        esperado - probabilidade
    )

    print(
        f"  Verificação do checkpoint: "
        f"diferença = {diferenca:.8f}"
    )

    if diferenca > 0.001:
        raise RuntimeError(
            "A previsão atual não reproduz o CSV "
            "original. Verifique os pesos e o "
            "pré-processamento."
        )


def salvar_figura(
    imagens,
    mapas,
    mascara,
    pid,
    classe_real,
    probabilidade,
    previsao,
    tipo,
    alvo,
    destino
):
    quantidade = imagens.shape[0]

    if quantidade == 3:
        nomes = [
            "Pré-contraste",
            "Pós-precoce",
            "Pós-tardio"
        ]
    else:
        nomes = ["Pós-precoce"]

    fig, eixos = plt.subplots(
        2,
        quantidade,
        figsize=(4.3 * quantidade, 8),
        squeeze=False
    )

    for indice in range(quantidade):
        imagem = imagens[indice]

        # Na CNN7/ResNet50, usamos uma única fase.
        mapa = (
            mapas[indice]
            if len(mapas) == quantidade
            else mapas[0]
        )

        superior = eixos[0, indice]
        inferior = eixos[1, indice]

        superior.imshow(
            imagem,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        superior.set_title(
            f"{nomes[indice]} — Original"
        )

        inferior.imshow(
            imagem,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        inferior.imshow(
            mapa,
            cmap="jet",
            vmin=0,
            vmax=1,
            alpha=np.clip(
                mapa * 0.65, 0.0, 0.65
            )
        )

        inferior.set_title(
            f"{nomes[indice]} — Grad-CAM"
        )

        for eixo in (superior, inferior):
            # Contorno da região tumoral original.
            if mascara.any() and (~mascara).any():
                eixo.contour(
                    mascara.astype(float),
                    levels=[0.5],
                    colors=["lime"],
                    linewidths=1.0
                )

            eixo.axis("off")

    fig.suptitle(
        f"{tipo.upper()} | {pid}\n"
        f"pCR real={classe_real} | "
        f"P(pCR=1)={probabilidade:.4f} | "
        f"Previsto={previsao}\n"
        f"Alvo Grad-CAM={alvo}",
        fontsize=12
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.90)
    )

    fig.savefig(
        destino,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM das três arquiteturas de pCR"
    )

    parser.add_argument(
        "--modelo",
        choices=["cnn7", "resnet50", "fusao"],
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default=None
    )

    parser.add_argument(
        "--pid",
        default=None,
        help="Paciente específico da validação"
    )

    parser.add_argument(
        "--listar",
        action="store_true"
    )

    parser.add_argument(
        "--alvo",
        choices=["pcr1", "pcr0", "previsto"],
        default="pcr1"
    )

    args = parser.parse_args()

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    caminho_checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else CHECKPOINTS[args.modelo]
    )

    modelo, camada_alvo, modo, selecao, tamanho = (
        carregar_modelo(
            args.modelo,
            caminho_checkpoint
        )
    )

    dataset = DatasetMRI(
        particao=2,
        selecao=selecao,
        modo=modo,
        tamanho=tamanho,
        aumentar=False
    )

    if len(dataset.registros) != 33:
        raise RuntimeError(
            "A validação deveria conter 33 pacientes."
        )

    print("\nARQUITETURA:", args.modelo)
    print("CHECKPOINT:", caminho_checkpoint)
    print("MODO:", modo)
    print("SELEÇÃO:", selecao)
    print("PACIENTES DE VALIDAÇÃO:", len(dataset.registros))

    if args.listar:
        for registro in dataset.registros:
            print(
                registro["pid"],
                "pCR=",
                registro["pCR"]
            )
        return

    if args.pid:
        indices = [
            indice
            for indice, registro in enumerate(
                dataset.registros
            )
            if registro["pid"] == args.pid
        ]

        if len(indices) != 1:
            raise ValueError(
                "Paciente não encontrado na validação: "
                + args.pid
            )

    else:
        # Escolhe um negativo e um positivo.
        negativo = next(
            (
                i for i, r in enumerate(dataset.registros)
                if r["pCR"] == 0
            ),
            None
        )

        positivo = next(
            (
                i for i, r in enumerate(dataset.registros)
                if r["pCR"] == 1
            ),
            None
        )

        if negativo is None or positivo is None:
            raise RuntimeError(
                "Não há representantes das duas classes."
            )

        indices = [negativo, positivo]

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    for indice in indices:
        imagem, rotulo, pid = dataset[indice]

        probabilidade, previsao, mapas = (
            calcular_gradcam(
                modelo,
                camada_alvo,
                imagem,
                args.modelo,
                args.alvo
            )
        )

        verificar_previsao_salva(
            caminho_checkpoint,
            pid,
            probabilidade
        )

        mascara = carregar_mascara(
            pid,
            selecao,
            tamanho
        )

        imagens = imagem.detach().cpu().numpy()

        destino = (
            PASTA_SAIDA /
            f"gradcam_{args.modelo}_{pid}_{args.alvo}.png"
        )

        salvar_figura(
            imagens,
            mapas,
            mascara,
            pid,
            int(rotulo.item()),
            probabilidade,
            previsao,
            args.modelo,
            args.alvo,
            destino
        )

        print(
            f"\nPaciente: {pid}",
            f"| pCR real: {int(rotulo.item())}",
            f"| Probabilidade: {probabilidade:.4f}",
            f"| Previsto: {previsao}"
        )

        print("Figura salva:", destino)

        if float(mapas.max()) == 0:
            print(
                "AVISO: Grad-CAM sem ativação positiva "
                "para o alvo escolhido."
            )

    print("\nGRAD-CAM CONCLUÍDO.")
    print("Pasta de saída:", PASTA_SAIDA)
    print("Nenhum modelo foi treinado.")
    print("Nenhum paciente de teste foi acessado.")


if __name__ == "__main__":
    main()
    
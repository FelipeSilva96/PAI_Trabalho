
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score
)

from pai_etapa2 import (
    BASE,
    CSV,
    CACHE,
    MODELOS,
    METRICAS,
    DatasetMRI,
    ler_metadados
)

from pai_gradcam import carregar_modelo
from pai_resnet50 import preparar_entrada_imagenet


# CHECKPOINTS FIXADOS ANTES DA AVALIACAO FINAL.
MODELOS_FINAIS = {
    "cnn7_E08": {
        "tipo": "cnn7",
        "arquivo": MODELOS /
            "cnn7_early_central_batch_lr0p001_pw1_seed42_size128_E08.pth",
        "auc_referencia": 0.7025,
        "modo": "early"
    },
    "resnet50_R01": {
        "tipo": "resnet50",
        "arquivo": MODELOS /
            "resnet50_early_central_sz128_lr0p001_pw1_seed42_R01_corrigido.pth",
        "auc_referencia": 0.4917,
        "modo": "early"
    },
    "fusao_FUS02": {
        "tipo": "fusao",
        "arquivo": MODELOS /
            "fusao_temporal_central_sz128_lr0p0003_pw3_seed42_FUS02.pth",
        "auc_referencia": 0.5785,
        "modo": "three"
    }
}

DESTINO = BASE / "resultados" / "avaliacao_final"
PROTOCOLO = DESTINO / "protocolo_congelado.json"


def sha256(arquivo):
    digest = hashlib.sha256()

    with Path(arquivo).open("rb") as f:
        while True:
            bloco = f.read(1024 * 1024)
            if not bloco:
                break
            digest.update(bloco)

    return digest.hexdigest()


def conferir_dataset():
    tabela = ler_metadados()

    esperados = {
        0: 104,
        2: 33,
        1: 35
    }

    if len(tabela) != 172:
        raise RuntimeError(
            "Quantidade total de pacientes incorreta."
        )

    if tabela["pid"].duplicated().any():
        raise RuntimeError(
            "Existem identificadores duplicados."
        )

    print("\nCONFERENCIA DAS PARTICOES")

    for particao, quantidade in esperados.items():
        registros = tabela[
            tabela["test"] == particao
        ]

        encontrado = len(registros)

        print(
            f"Particao {particao}: {encontrado} pacientes"
        )

        if encontrado != quantidade:
            raise RuntimeError(
                f"Particao {particao} incompleta."
            )

    # Conferimos a existencia dos arquivos de cache,
    # mas NAO lemos as imagens ou rotulos do teste.
    tabela_teste = tabela[tabela["test"] == 1]

    faltantes = [
        str(linha["pid"])
        for _, linha in tabela_teste.iterrows()
        if not (
            CACHE / "central" / f"{linha['pid']}.npz"
        ).is_file()
    ]

    if faltantes:
        raise RuntimeError(
            "Cache do teste incompleto: "
            + ", ".join(faltantes)
        )

    print("Cache do teste: 35 arquivos presentes.")
    print("Nenhuma imagem de teste foi processada.")


@torch.no_grad()
def reproduzir_validacao(nome, configuracao):
    caminho = configuracao["arquivo"]

    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    tipo = configuracao["tipo"]

    modelo, _, modo, selecao, tamanho = (
        carregar_modelo(tipo, caminho)
    )

    if modo != configuracao["modo"]:
        raise RuntimeError(
            f"Modo incorreto no modelo {nome}."
        )

    if selecao != "central" or tamanho != 128:
        raise RuntimeError(
            f"Configuracao inesperada: {nome}."
        )

    dataset = DatasetMRI(
        particao=2,
        selecao=selecao,
        modo=modo,
        tamanho=tamanho,
        aumentar=False
    )

    if len(dataset) != 33:
        raise RuntimeError(
            f"Validacao incompleta no modelo {nome}."
        )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    ids = []
    rotulos = []
    probabilidades = []

    modelo.eval()

    for imagens, y, pacientes in loader:
        if tipo == "resnet50":
            imagens = preparar_entrada_imagenet(
                imagens
            )

        logits = modelo(imagens).reshape(-1)

        probs = torch.sigmoid(logits)

        ids.extend(pacientes)
        rotulos.extend(
            y.numpy().astype(int).tolist()
        )
        probabilidades.extend(
            probs.numpy().tolist()
        )

    y = np.asarray(rotulos, dtype=int)
    p = np.asarray(probabilidades, dtype=float)

    auc = float(roc_auc_score(y, p))
    previsto = (p >= 0.5).astype(int)

    f1 = float(
        f1_score(y, previsto, zero_division=0)
    )

    balacc = float(
        balanced_accuracy_score(y, previsto)
    )

    # Comparamos com o CSV salvo no treinamento.
    csv_previsoes = (
        METRICAS /
        f"{caminho.stem}_previsoes.csv"
    )

    if not csv_previsoes.is_file():
        raise FileNotFoundError(
            "CSV de previsoes ausente: "
            + str(csv_previsoes)
        )

    anterior = pd.read_csv(csv_previsoes)

    anterior["pid"] = anterior["pid"].astype(str)

    if (
        len(anterior) != 33
        or anterior["pid"].duplicated().any()
        or set(anterior["pid"]) != set(ids)
    ):
        raise RuntimeError(
            f"Pacientes divergentes em {nome}."
        )

    anterior = anterior.set_index("pid").loc[ids]

    y_anterior = (
        anterior["pCR_real"]
        .to_numpy(dtype=int)
    )

    if not np.array_equal(y, y_anterior):
        raise RuntimeError(
            f"Rotulos divergentes em {nome}."
        )

    probs_anteriores = (
        anterior["probabilidade"]
        .to_numpy(dtype=float)
    )

    diferenca = float(
        np.max(np.abs(p - probs_anteriores))
    )

    print(f"\nMODELO: {nome}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Acuracia balanceada: {balacc:.4f}")
    print(f"  Diferenca maxima de probabilidades: {diferenca:.8f}")

    if diferenca > 0.001:
        raise RuntimeError(
            f"Checkpoint nao reproduz o CSV: {nome}"
        )

    if abs(auc - configuracao["auc_referencia"]) > 0.001:
        raise RuntimeError(
            f"AUC inesperada no modelo {nome}."
        )

    return {
        "auc_roc": auc,
        "f1": f1,
        "acuracia_balanceada": balacc,
        "diferenca_maxima": diferenca
    }


def executar_verificacao():
    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    conferir_dataset()

    resultados = {}

    for nome, config in MODELOS_FINAIS.items():
        resultados[nome] = reproduzir_validacao(
            nome, config
        )

    print("\n" + "=" * 65)
    print("VERIFICACAO CONCLUIDA COM SUCESSO")
    print("=" * 65)

    print("Todos os checkpoints reproduziram a validacao.")
    print("Nenhum paciente de teste foi avaliado.")

    return resultados


def congelar_protocolo(resultados):
    if PROTOCOLO.exists():
        raise FileExistsError(
            "O protocolo ja foi congelado: "
            + str(PROTOCOLO)
        )

    arquivos_codigo = [
        BASE / "pai_etapa2.py",
        BASE / "pai_gradcam.py",
        BASE / "pai_resnet50.py",
        BASE / "pai_fusao_temporal.py",
        Path(__file__).resolve()
    ]

    hashes_codigo = {}

    for arquivo in arquivos_codigo:
        if not arquivo.is_file():
            raise FileNotFoundError(arquivo)

        hashes_codigo[arquivo.name] = sha256(arquivo)

    modelos_congelados = {}

    for nome, config in MODELOS_FINAIS.items():
        modelos_congelados[nome] = {
            "tipo": config["tipo"],
            "checkpoint": str(config["arquivo"].resolve()),
            "sha256": sha256(config["arquivo"]),
            "modo": config["modo"],
            "selecao": "central",
            "tamanho": 128,
            "metricas_validacao": resultados[nome]
        }

    protocolo = {
        "versao": "PAI_FINAL_V1",
        "criterio_selecao": "AUC-ROC de validacao",
        "limiar_classificacao": 0.5,
        "pacientes_treino": 104,
        "pacientes_validacao": 33,
        "pacientes_teste": 35,
        "teste_ja_avaliado": False,
        "observacao": (
            "O conjunto de teste nao participou "
            "da escolha dos checkpoints nem "
            "dos hiperparametros."
        ),
        "csv_metadados_sha256": sha256(CSV),
        "hashes_codigo": hashes_codigo,
        "modelos": modelos_congelados
    }

    DESTINO.mkdir(
        parents=True,
        exist_ok=True
    )

    with PROTOCOLO.open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            protocolo,
            arquivo,
            indent=2,
            ensure_ascii=False
        )

    print("\nPROTOCOLO CONGELADO!")
    print("Arquivo:", PROTOCOLO)
    print(
        "O teste ainda nao foi executado."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--acao",
        choices=["verificar", "congelar"],
        default="verificar"
    )

    args = parser.parse_args()

    if PROTOCOLO.exists():
        print(
            "AVISO: ja existe um protocolo congelado."
        )

        if args.acao == "congelar":
            raise FileExistsError(PROTOCOLO)

    resultados = executar_verificacao()

    if args.acao == "congelar":
        congelar_protocolo(resultados)


if __name__ == "__main__":
    main()

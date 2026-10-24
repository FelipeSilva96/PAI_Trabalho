
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from pai_etapa2 import (
    BASE,
    CSV,
    DatasetMRI,
    ler_metadados
)

from pai_gradcam import carregar_modelo
from pai_resnet50 import preparar_entrada_imagenet


PASTA = BASE / "resultados" / "avaliacao_final"

PROTOCOLO = PASTA / "protocolo_congelado.json"

MARCADOR = PASTA / "execucao_teste.json"

MODELOS_ESPERADOS = {
    "cnn7_E08": "cnn7",
    "resnet50_R01": "resnet50",
    "fusao_FUS02": "fusao"
}


def calcular_hash(caminho):
    digest = hashlib.sha256()

    with open(caminho, "rb") as arquivo:
        while True:
            bloco = arquivo.read(1024 * 1024)

            if not bloco:
                break

            digest.update(bloco)

    return digest.hexdigest()


def verificar_protocolo():
    if not PROTOCOLO.is_file():
        raise FileNotFoundError(
            "Protocolo congelado não encontrado."
        )

    with open(PROTOCOLO, encoding="utf-8") as arquivo:
        protocolo = json.load(arquivo)

    if protocolo.get("versao") != "PAI_FINAL_V1":
        raise RuntimeError("Versão de protocolo incorreta.")

    if protocolo.get("limiar_classificacao") != 0.5:
        raise RuntimeError("O limiar foi alterado.")

    if protocolo.get("pacientes_teste") != 35:
        raise RuntimeError("Quantidade de teste incorreta.")

    if protocolo.get("teste_ja_avaliado") is not False:
        raise RuntimeError(
            "O protocolo não confirma teste ainda reservado."
        )

    if calcular_hash(CSV) != protocolo["csv_metadados_sha256"]:
        raise RuntimeError(
            "O CSV foi modificado após o congelamento."
        )

    for nome, hash_esperado in protocolo["hashes_codigo"].items():
        caminho = BASE / nome

        if not caminho.is_file():
            raise FileNotFoundError(caminho)

        if calcular_hash(caminho) != hash_esperado:
            raise RuntimeError(
                f"Código alterado após congelamento: {nome}"
            )

    if set(protocolo["modelos"]) != set(MODELOS_ESPERADOS):
        raise RuntimeError(
            "Os modelos não correspondem ao protocolo."
        )

    for nome, tipo in MODELOS_ESPERADOS.items():
        config = protocolo["modelos"][nome]

        caminho = Path(config["checkpoint"])

        if not caminho.is_file():
            raise FileNotFoundError(caminho)

        if calcular_hash(caminho) != config["sha256"]:
            raise RuntimeError(
                f"Checkpoint alterado: {nome}"
            )

        if config["tipo"] != tipo:
            raise RuntimeError(
                f"Arquitetura incorreta: {nome}"
            )

        if config["selecao"] != "central":
            raise RuntimeError(
                f"Seleção de corte incorreta: {nome}"
            )

        if int(config["tamanho"]) != 128:
            raise RuntimeError(
                f"Resolução incorreta: {nome}"
            )

        modo_esperado = (
            "three" if tipo == "fusao" else "early"
        )

        if config["modo"] != modo_esperado:
            raise RuntimeError(
                f"Modo de entrada incorreto: {nome}"
            )

    tabela = ler_metadados()

    if len(tabela) != 172:
        raise RuntimeError(
            "Esperados 172 pacientes."
        )

    quantidades = {
        0: 104,
        2: 33,
        1: 35
    }

    for particao, esperado in quantidades.items():
        encontrados = int(
            (tabela["test"] == particao).sum()
        )

        if encontrados != esperado:
            raise RuntimeError(
                f"Partição {particao}: "
                f"{encontrados}/{esperado}"
            )

    ids_teste = set(
        tabela.loc[
            tabela["test"] == 1,
            "pid"
        ].astype(str)
    )

    if len(ids_teste) != 35:
        raise RuntimeError(
            "Identificadores duplicados no teste."
        )

    print("\nPROTOCOLO VERIFICADO COM SUCESSO.")
    print("Checkpoints: íntegros")
    print("Código: íntegro")
    print("Metadados: íntegros")
    print("Treino: 104 | Validação: 33 | Teste: 35")
    print("Limiar: 0.5")

    return protocolo, ids_teste


@torch.no_grad()
def avaliar_modelo(nome, config, ids_teste):
    print(f"\nAvaliando {nome}...", flush=True)

    caminho = Path(config["checkpoint"])

    modelo, _, modo, selecao, tamanho = carregar_modelo(
        config["tipo"],
        caminho
    )

    if (
        modo != config["modo"]
        or selecao != config["selecao"]
        or tamanho != config["tamanho"]
    ):
        raise RuntimeError(
            f"Configuração divergente: {nome}"
        )

    # Somente agora acessamos a partição 1.
    dataset = DatasetMRI(
        particao=1,
        selecao=selecao,
        modo=modo,
        tamanho=tamanho,
        aumentar=False,
        usar_mascara="nenhuma"
    )

    if len(dataset) != 35 or len(dataset.registros) != 35:
        raise RuntimeError(
            f"Teste incompleto em {nome}."
        )

    ids_dataset = [
        str(r["pid"])
        for r in dataset.registros
    ]

    if (
        len(set(ids_dataset)) != 35
        or set(ids_dataset) != ids_teste
    ):
        raise RuntimeError(
            f"Pacientes divergentes em {nome}."
        )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    modelo.eval()

    pacientes = []
    rotulos = []
    probabilidades = []

    for imagens, y, ids in loader:
        if config["tipo"] == "resnet50":
            imagens = preparar_entrada_imagenet(
                imagens
            )

        logits = modelo(imagens).reshape(-1)

        probs = torch.sigmoid(logits)

        pacientes.extend(ids)

        rotulos.extend(
            y.numpy().astype(int).tolist()
        )

        probabilidades.extend(
            probs.numpy().tolist()
        )

    previsoes = pd.DataFrame({
        "pid": pacientes,
        "pCR_real": rotulos,
        "probabilidade": probabilidades
    })

    if (
        len(previsoes) != 35
        or previsoes["pid"].duplicated().any()
        or set(previsoes["pid"]) != ids_teste
    ):
        raise RuntimeError(
            f"Previsões incompletas: {nome}"
        )

    if not previsoes["pCR_real"].isin([0, 1]).all():
        raise RuntimeError(
            f"Rótulos inválidos: {nome}"
        )

    probabilidades_array = (
        previsoes["probabilidade"]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(probabilidades_array).all():
        raise RuntimeError(
            f"Probabilidades inválidas: {nome}"
        )

    previsoes["previsto"] = (
        previsoes["probabilidade"] >= 0.5
    ).astype(int)

    return previsoes


def calcular_metricas(nome, previsoes):
    y = previsoes["pCR_real"].to_numpy(dtype=int)

    p = previsoes["probabilidade"].to_numpy(
        dtype=float
    )

    previstos = previsoes["previsto"].to_numpy(
        dtype=int
    )

    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError(
            "As duas classes precisam estar presentes."
        )

    vn, fp, fn, vp = confusion_matrix(
        y,
        previstos,
        labels=[0, 1]
    ).ravel()

    return {
        "modelo": nome,
        "pacientes": len(y),
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn),
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
            float(vn / (vn + fp))
            if vn + fp else 0.0
        ),
        "precisao": float(
            precision_score(
                y, previstos, zero_division=0
            )
        ),
        "f1": float(
            f1_score(y, previstos, zero_division=0)
        ),
        "auc_roc": float(
            roc_auc_score(y, p)
        ),
        "average_precision": float(
            average_precision_score(y, p)
        ),
        "brier": float(
            brier_score_loss(y, p)
        )
    }


def executar_avaliacao(protocolo, ids_teste):
    if MARCADOR.exists():
        raise FileExistsError(
            "O teste final já foi executado. "
            "Não realize uma nova avaliação."
        )

    nomes_saidas = [
        "metricas_teste.csv",
        "metricas_teste.json"
    ]

    for nome in MODELOS_ESPERADOS:
        nomes_saidas.append(
            f"previsoes_teste_{nome}.csv"
        )

    for arquivo in nomes_saidas:
        destino = PASTA / arquivo

        if destino.exists():
            raise FileExistsError(
                f"Resultado já existente: {destino}"
            )

    resultados = {}
    metricas = []
    rotulos_referencia = None

    print("\nINICIANDO AVALIAÇÃO FINAL")
    print("Partição: TESTE")
    print("Pacientes: 35")
    print("Modelos: 3")
    print("Nenhum parâmetro será atualizado.\n")

    for nome in MODELOS_ESPERADOS:
        config = protocolo["modelos"][nome]

        previsoes = avaliar_modelo(
            nome,
            config,
            ids_teste
        )

        rotulos_atual = (
            previsoes.set_index("pid")["pCR_real"]
            .sort_index()
        )

        if rotulos_referencia is None:
            rotulos_referencia = rotulos_atual

        elif not rotulos_atual.equals(rotulos_referencia):
            raise RuntimeError(
                "Rótulos divergentes entre os modelos."
            )

        resultados[nome] = previsoes

        metricas.append(
            calcular_metricas(nome, previsoes)
        )

    # Salva somente após avaliar os três modelos
    # e verificar a consistência dos pacientes.
    PASTA.mkdir(parents=True, exist_ok=True)

    for nome, previsoes in resultados.items():
        caminho = (
            PASTA / f"previsoes_teste_{nome}.csv"
        )

        previsoes.to_csv(
            caminho,
            index=False
        )

    tabela = pd.DataFrame(metricas)

    tabela.to_csv(
        PASTA / "metricas_teste.csv",
        index=False
    )

    with open(
        PASTA / "metricas_teste.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            {
                "versao": "PAI_TESTE_FINAL_V1",
                "limiar": 0.5,
                "metricas": metricas
            },
            arquivo,
            indent=2,
            ensure_ascii=False
        )

    # Registro separado: não alteramos
    # o protocolo original congelado.
    registro = {
        "avaliacao_realizada": True,
        "particao": 1,
        "pacientes": 35,
        "protocolo_sha256": calcular_hash(PROTOCOLO),
        "limiar": 0.5,
        "modelos": list(MODELOS_ESPERADOS)
    }

    with open(
        MARCADOR,
        "x",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            registro,
            arquivo,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("AVALIAÇÃO FINAL CONCLUÍDA")
    print("=" * 65)

    colunas = [
        "modelo",
        "auc_roc",
        "average_precision",
        "f1",
        "acuracia_balanceada",
        "sensibilidade",
        "especificidade",
        "VP",
        "VN",
        "FP",
        "FN"
    ]

    print(
        tabela[colunas].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nResultados:", PASTA)
    print("Registro da execução:", MARCADOR)


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação final das três arquiteturas"
    )

    parser.add_argument(
        "--acao",
        choices=["preflight", "avaliar"],
        default="preflight"
    )

    parser.add_argument(
        "--confirmar-teste",
        action="store_true"
    )

    args = parser.parse_args()

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    protocolo, ids_teste = verificar_protocolo()

    if args.acao == "preflight":
        print("\nPREFLIGHT CONCLUÍDO.")
        print("Nenhuma imagem de teste foi processada.")
        return

    if not args.confirmar_teste:
        parser.error(
            "Para avaliar o teste, acrescente "
            "--confirmar-teste."
        )

    executar_avaliacao(
        protocolo,
        ids_teste
    )


if __name__ == "__main__":
    main()

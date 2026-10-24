
import os

import numpy as np
import pandas as pd
import torch

from pai_etapa2 import DatasetMRI, METRICAS, MODELOS

from pai_gradcam import (
    carregar_modelo,
    calcular_gradcam,
    carregar_mascara
)


CHECKPOINT = (
    MODELOS /
    "cnn7_early_central_batch_lr0p001_pw1_seed42_size128_E08.pth"
)

CSV_PREVISOES = (
    METRICAS /
    f"{CHECKPOINT.stem}_previsoes.csv"
)

CSV_SAIDA = (
    METRICAS /
    "auditoria_gradcam_validacao_e08.csv"
)


def main():
    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    print("\n" + "=" * 65)
    print("AUDITORIA GRAD-CAM — CNN7 E08")
    print("=" * 65)

    # 1. Conferir e carregar o checkpoint E08.
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    print(
        "CHECKPOINT EFETIVAMENTE UTILIZADO:",
        CHECKPOINT.resolve(),
        flush=True
    )

    modelo, camada, modo, selecao, tamanho = (
        carregar_modelo("cnn7", CHECKPOINT)
    )

    if modo != "early":
        raise RuntimeError(
            f"Modo inesperado: {modo}"
        )

    if selecao != "central" or tamanho != 128:
        raise RuntimeError(
            "Configuração diferente da CNN7 E08."
        )

    # 2. Carregar somente pacientes da validação.
    dataset = DatasetMRI(
        particao=2,
        selecao=selecao,
        modo=modo,
        tamanho=tamanho,
        aumentar=False
    )

    if len(dataset) != 33:
        raise RuntimeError(
            f"Esperados 33 pacientes; encontrados {len(dataset)}."
        )

    print("Pacientes de validação:", len(dataset))
    print("Iniciando Grad-CAM...\n", flush=True)

    resultados = []

    # 3. Processar TODOS os 33 pacientes.
    for indice in range(len(dataset)):
        imagem, rotulo, pid = dataset[indice]

        probabilidade, previsto, mapas = (
            calcular_gradcam(
                modelo,
                camada,
                imagem,
                "cnn7",
                "pcr1"
            )
        )

        # A CNN7 de uma fase deve produzir um mapa.
        if mapas.shape[0] != 1:
            raise RuntimeError(
                f"Quantidade inesperada de mapas: {pid}"
            )

        mapa = np.asarray(
            mapas[0],
            dtype=np.float64
        )

        mascara = carregar_mascara(
            pid,
            selecao,
            tamanho
        ).astype(bool)

        if mapa.shape != mascara.shape:
            raise RuntimeError(
                f"Mapa e máscara desalinhados: {pid}"
            )

        if not np.isfinite(mapa).all():
            raise RuntimeError(
                f"Grad-CAM com valores inválidos: {pid}"
            )

        # 4. Calcular concentração espacial.
        fracao_area = float(mascara.mean())

        if fracao_area <= 0:
            raise RuntimeError(
                f"Máscara tumoral vazia: {pid}"
            )

        energia_total = float(mapa.sum())
        maximo = float(mapa.max())

        if energia_total <= 1e-10:
            fracao_energia = np.nan
            concentracao = np.nan
            status = "sem_ativacao_positiva"

        else:
            fracao_energia = float(
                mapa[mascara].sum() /
                energia_total
            )

            concentracao = float(
                fracao_energia / fracao_area
            )

            status = "valido"

        resultados.append({
            "modelo": "cnn7_E08",
            "checkpoint": CHECKPOINT.name,
            "pid": str(pid),
            "fase": "early",
            "pCR_real": int(rotulo.item()),
            "probabilidade": float(probabilidade),
            "previsto": int(previsto),
            "fracao_area_tumor": fracao_area,
            "fracao_energia_tumor": fracao_energia,
            "concentracao_relativa": concentracao,
            "maximo_gradcam": maximo,
            "status": status
        })

        if (indice + 1) % 10 == 0:
            print(
                f"  {indice + 1}/33 pacientes",
                flush=True
            )

    # 5. Verificar se a auditoria está completa.
    tabela = pd.DataFrame(resultados)

    if len(tabela) != 33:
        raise RuntimeError(
            "A auditoria não processou os 33 pacientes."
        )

    if tabela["pid"].duplicated().any():
        raise RuntimeError(
            "Foram encontrados pacientes duplicados."
        )

    # 6. Verificar as probabilidades usando o
    # CSV produzido no treinamento original E08.
    if not CSV_PREVISOES.is_file():
        raise FileNotFoundError(
            "Previsões originais E08 não encontradas: "
            + str(CSV_PREVISOES)
        )

    referencia = pd.read_csv(
        CSV_PREVISOES,
        dtype={"pid": str}
    )

    if (
        len(referencia) != 33
        or referencia["pid"].duplicated().any()
        or set(referencia["pid"]) != set(tabela["pid"])
    ):
        raise RuntimeError(
            "Pacientes da auditoria divergem do CSV E08."
        )

    referencia = (
        referencia.set_index("pid")
        .loc[tabela["pid"]]
    )

    y_original = (
        referencia["pCR_real"]
        .to_numpy(dtype=int)
    )

    y_auditoria = (
        tabela["pCR_real"]
        .to_numpy(dtype=int)
    )

    if not np.array_equal(
        y_original,
        y_auditoria
    ):
        raise RuntimeError(
            "Os rótulos da auditoria não coincidem "
            "com os da validação original."
        )

    probs_originais = (
        referencia["probabilidade"]
        .to_numpy(dtype=float)
    )

    probs_auditoria = (
        tabela["probabilidade"]
        .to_numpy(dtype=float)
    )

    diferenca_maxima = float(
        np.max(
            np.abs(
                probs_originais - probs_auditoria
            )
        )
    )

    print(
        "\nDiferença máxima de probabilidades:",
        f"{diferenca_maxima:.8f}"
    )

    if diferenca_maxima > 0.001:
        raise RuntimeError(
            "A auditoria não reproduziu as "
            "probabilidades do checkpoint E08."
        )

    # 7. Preparar o resumo.
    resumo = tabela.groupby(
        ["modelo", "fase"]
    ).agg(
        pacientes=("pid", "count"),
        mapas_validos=(
            "fracao_energia_tumor", "count"
        ),
        energia_mediana=(
            "fracao_energia_tumor", "median"
        ),
        concentracao_mediana=(
            "concentracao_relativa", "median"
        )
    )

    # 8. Salvar somente após todas as verificações.
    # Usamos um arquivo temporário para evitar que
    # uma falha deixe outro CSV vazio.
    CSV_SAIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporario = CSV_SAIDA.with_name(
        CSV_SAIDA.name + ".tmp"
    )

    try:
        tabela.to_csv(
            temporario,
            index=False
        )

        os.replace(
            temporario,
            CSV_SAIDA
        )

    finally:
        if temporario.exists():
            temporario.unlink()

    # 9. Mostrar resultados.
    print("\n" + "=" * 65)
    print("RESUMO DA AUDITORIA GRAD-CAM — E08")
    print("=" * 65)

    print(resumo.to_string())

    print("\nPROBABILIDADES DE CONFERÊNCIA")

    exemplos = tabela[
        tabela["pid"].isin([
            "ISPY1_1009",
            "ISPY1_1010"
        ])
    ]

    print(
        exemplos[
            [
                "pid",
                "probabilidade",
                "fracao_energia_tumor",
                "concentracao_relativa"
            ]
        ].to_string(index=False)
    )

    print("\nAUDITORIA CONCLUÍDA COM SUCESSO.")
    print("Arquivo salvo:", CSV_SAIDA.resolve())
    print("Nenhum modelo foi treinado.")
    print("Nenhum paciente de teste foi acessado.")


if __name__ == "__main__":
    main()

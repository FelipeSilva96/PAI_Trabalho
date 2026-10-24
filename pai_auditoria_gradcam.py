
import numpy as np
import pandas as pd
import torch

from pai_etapa2 import DatasetMRI, METRICAS

from pai_gradcam import (
    CHECKPOINTS,
    carregar_modelo,
    calcular_gradcam,
    carregar_mascara
)


def main():
    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    resultados = []

    for tipo in ("cnn7", "resnet50", "fusao"):
        print(f"\nANALISANDO: {tipo.upper()}", flush=True)

        checkpoint = CHECKPOINTS[tipo]

        modelo, camada, modo, selecao, tamanho = (
            carregar_modelo(tipo, checkpoint)
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
                "Esperados 33 pacientes de validação."
            )

        nomes_fases = (
            ["pre", "early", "late"]
            if tipo == "fusao"
            else [modo]
        )

        for indice in range(len(dataset)):
            imagem, rotulo, pid = dataset[indice]

            probabilidade, previsto, mapas = (
                calcular_gradcam(
                    modelo,
                    camada,
                    imagem,
                    tipo,
                    "pcr1"
                )
            )

            mascara = carregar_mascara(
                pid,
                selecao,
                tamanho
            ).astype(bool)

            fracao_area = float(mascara.mean())

            for nome_fase, mapa in zip(
                nomes_fases, mapas
            ):
                mapa = np.asarray(
                    mapa, dtype=np.float64
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

                    concentracao = (
                        fracao_energia / fracao_area
                        if fracao_area > 0
                        else np.nan
                    )

                    status = "valido"

                resultados.append({
                    "modelo": tipo,
                    "pid": pid,
                    "fase": nome_fase,
                    "pCR_real": int(rotulo.item()),
                    "probabilidade": probabilidade,
                    "previsto": previsto,
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

    tabela = pd.DataFrame(resultados)

    destino = (
        METRICAS /
        "auditoria_gradcam_validacao.csv"
    )

    tabela.to_csv(destino, index=False)

    resumo = tabela.groupby(
        ["modelo", "fase"]
    ).agg(
        pacientes=("pid", "count"),
        mapas_validos=("fracao_energia_tumor", "count"),
        energia_mediana=("fracao_energia_tumor", "median"),
        concentracao_mediana=("concentracao_relativa", "median")
    )

    print("\n" + "=" * 65)
    print("RESUMO DA AUDITORIA GRAD-CAM")
    print("=" * 65)
    print(resumo.to_string())

    print("\nArquivo salvo:", destino)


if __name__ == "__main__":
    main()

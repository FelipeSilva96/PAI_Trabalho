
# Processamento e Análise de Imagens — PUC Minas
# Projeto: Classificação da resposta ao tratamento do câncer de mama
#
# Felipe Pereira da Silva
# Rikerson Antônio Freitas Silva
# Mateus Resente Ottoni
#
# Etapa 2: auditoria, preparação do dataset e CNN de 7 camadas.
#
# Este arquivo será integrado ao visualizador gráfico depois
# que os testes de preparação e treinamento estiverem concluídos.

from pathlib import Path
import argparse
import json
import random
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import rotate

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# -----------------------------------------------------------
# Configurações gerais
# -----------------------------------------------------------

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
DCE = DADOS / "dce"
MASK = DADOS / "mask"
CSV = DADOS / "BreastDCEDL_SPY1_min_crop_metadata.csv"

CACHE = DADOS / "cache"
RESULTADOS = BASE / "resultados"
MODELOS = RESULTADOS / "modelos"
METRICAS = RESULTADOS / "metricas"
GRAFICOS = RESULTADOS / "graficos"
AUDITORIA = RESULTADOS / "auditoria"

FASES = ("pre", "post_early", "post_late")
ANGULOS = (-20, -10, 0, 10, 20)

SPLITS = {
    0: "treino",
    1: "teste",
    2: "validacao"
}


def criar_pastas():
    for pasta in (
        CACHE, MODELOS, METRICAS, GRAFICOS, AUDITORIA
    ):
        pasta.mkdir(parents=True, exist_ok=True)


def definir_semente(semente=42):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def ler_metadados():
    if not CSV.is_file():
        raise FileNotFoundError(
            f"CSV não encontrado: {CSV}"
        )

    tabela = pd.read_csv(CSV)

    obrigatorias = [
        "pid", "pCR", "test", "n_z",
        "pre", "post_early", "post_late"
    ]

    faltando = [
        coluna for coluna in obrigatorias
        if coluna not in tabela.columns
    ]

    if faltando:
        raise ValueError(
            f"Colunas ausentes no CSV: {faltando}"
        )

    if tabela["pid"].duplicated().any():
        raise ValueError(
            "Existem identificadores de pacientes repetidos."
        )

    return tabela


# -----------------------------------------------------------
# Localização das imagens
# -----------------------------------------------------------

def encontrar_arquivo(pasta, nome):
    # Aceitamos tanto NIfTI normal quanto comprimido.
    for extensao in (".nii.gz", ".nii"):
        caminho = pasta / f"{nome}{extensao}"

        if caminho.is_file():
            return caminho

    return None


def caminhos_paciente(linha):
    pid = str(linha["pid"])
    prefixo = f"{pid}_spy1_vis1"

    aquisicoes = {}

    for fase in FASES:
        indice = int(linha[fase])

        nome = f"{prefixo}_dce_aqc_{indice}"
        caminho = encontrar_arquivo(DCE, nome)

        aquisicoes[fase] = caminho

    nome_mask = f"{prefixo}_mask"
    mascara = encontrar_arquivo(MASK, nome_mask)

    return aquisicoes, mascara


def arquivos_necessarios(linha):
    aquisicoes, mascara = caminhos_paciente(linha)

    arquivos = list(aquisicoes.values()) + [mascara]
    faltantes = [x for x in arquivos if x is None]

    return aquisicoes, mascara, faltantes


# -----------------------------------------------------------
# Auditoria de arquivos e dimensões
# -----------------------------------------------------------

def encontrar_eixo_z(shape, n_z=None, n_xy=256):
    # Primeiro verificamos se o volume tem três dimensões.
    if len(shape) != 3:
        raise ValueError(
            f"O volume precisa ser 3D. Shape encontrado: {shape}"
        )

    # Procuramos os dois eixos do plano XY.
    eixos_xy = [
        eixo for eixo, tamanho in enumerate(shape)
        if tamanho == n_xy
    ]

    if len(eixos_xy) == 2:
        # O eixo restante corresponde à profundidade.
        eixo_z = next(
            eixo for eixo in range(3)
            if eixo not in eixos_xy
        )

        return eixo_z

    # Caso não seja possível identificar pelos eixos XY,
    # tentamos usar a quantidade de cortes do CSV.
    if n_z is not None:
        candidatos = [
            eixo for eixo, tamanho in enumerate(shape)
            if tamanho == n_z
        ]

        if len(candidatos) == 1:
            return candidatos[0]

    # Não escolhemos um eixo arbitrariamente.
    raise ValueError(
        f"Não foi possível identificar o eixo Z. "
        f"Shape: {shape}, n_z: {n_z}, n_xy: {n_xy}"
    )



def auditar():
    criar_pastas()
    tabela = ler_metadados()

    relatorio = []

    print(f"Pacientes no CSV: {len(tabela)}")
    print("Verificando arquivos e cabeçalhos NIfTI...")

    for numero, (_, linha) in enumerate(
        tabela.iterrows(), start=1
    ):
        pid = str(linha["pid"])
        problemas = []
        informacoes = []

        n_z_real = None
        eixo_z = None

        try:
            # Primeiro conferimos os campos necessários.
            if pd.isna(linha["pCR"]):
                problemas.append("Rótulo pCR ausente")

            elif int(linha["pCR"]) not in (0, 1):
                problemas.append("Rótulo pCR inválido")

            if pd.isna(linha["test"]):
                problemas.append("Partição ausente")

            elif int(linha["test"]) not in (0, 1, 2):
                problemas.append("Partição inválida")

            aquisicoes, mascara, faltantes = (
                arquivos_necessarios(linha)
            )

            if faltantes:
                problemas.append(
                    f"{len(faltantes)} arquivo(s) ausente(s)"
                )

            else:
                imagens = [
                    nib.load(str(aquisicoes[fase]))
                    for fase in FASES
                ]

                imagem_mask = nib.load(str(mascara))
                todos = imagens + [imagem_mask]

                formas = [img.shape for img in todos]

                if any(len(forma) != 3 for forma in formas):
                    problemas.append(
                        "Um ou mais volumes não são 3D"
                    )

                elif len(set(formas)) != 1:
                    problemas.append(
                        "Dimensões diferentes entre "
                        "ressonâncias e máscara"
                    )

                else:
                    # O eixo Z é identificado pelas dimensões
                    # reais dos arquivos da versão MinCrop.
                    n_z_csv = int(linha["n_z"])
                    n_xy_csv = int(linha["n_xy"])

                    eixo_z = encontrar_eixo_z(
                        formas[0],
                        n_z=n_z_csv,
                        n_xy=n_xy_csv
                    )

                    n_z_real = formas[0][eixo_z]

                    # O CSV descreve informações da imagem
                    # original. O MinCrop pode ter menos cortes.
                    if n_z_real != n_z_csv:
                        informacoes.append(
                            f"MinCrop: {n_z_real} cortes; "
                            f"CSV: {n_z_csv}"
                        )

                # Conferimos se os volumes foram armazenados
                # em referenciais espaciais compatíveis.
                referencia = imagens[0].affine

                for img in todos[1:]:
                    if not np.allclose(
                        img.affine,
                        referencia,
                        rtol=1e-4,
                        atol=1e-3
                    ):
                        problemas.append(
                            "Transformações espaciais diferentes"
                        )
                        break

        except Exception as erro:
            problemas.append(str(erro))

        estado = "OK" if not problemas else "PROBLEMA"

        detalhes = problemas if problemas else informacoes

        relatorio.append({
            "pid": pid,
            "pCR": linha["pCR"],
            "particao": linha["test"],
            "n_z_csv": linha["n_z"],
            "n_z_real": n_z_real,
            "eixo_z": eixo_z,
            "status": estado,
            "detalhes": " | ".join(detalhes)
        })

        print(
            f"[{numero:03d}/{len(tabela)}] "
            f"{pid}: {estado}"
        )

    resultado = pd.DataFrame(relatorio)

    destino = AUDITORIA / "auditoria_dataset.csv"
    resultado.to_csv(destino, index=False)

    print("\nDistribuição original:")
    print(pd.crosstab(tabela["test"], tabela["pCR"]))

    print("\nResultado da auditoria:")
    print(resultado["status"].value_counts())

    print(f"\nRelatório salvo em: {destino}")


# -----------------------------------------------------------
# Carregamento dos volumes
# -----------------------------------------------------------

def carregar_volume(caminho):
    imagem = nib.load(str(caminho))

    # float32 economiza memória em comparação com float64.
    dados = imagem.get_fdata(
        dtype=np.float32,
        caching="unchanged"
    )

    if dados.ndim != 3:
        raise ValueError("O arquivo não contém um volume 3D")

    if not np.isfinite(dados).all():
        raise ValueError(
            "O volume contém NaN ou valores infinitos"
        )

    return imagem, dados



def preparar_paciente(linha, selecao):
    aquisicoes, caminho_mask, faltantes = (
        arquivos_necessarios(linha)
    )

    if faltantes:
        raise FileNotFoundError(
            "Uma ou mais aquisições/máscara estão ausentes"
        )

    # Começamos pela máscara, pois ela será utilizada
    # para identificar o eixo dos cortes.
    imagem_mask, mascara = carregar_volume(caminho_mask)

    n_z_csv = int(linha["n_z"])
    n_xy_csv = int(linha["n_xy"])

    eixo = encontrar_eixo_z(
        mascara.shape,
        n_z=n_z_csv,
        n_xy=n_xy_csv
    )

    # A quantidade real de cortes vem do NIfTI recortado,
    # não do volume original descrito pelo CSV.
    n_z = mascara.shape[eixo]

    # Colocamos os cortes na primeira dimensão do array.
    mascara = np.moveaxis(mascara, eixo, 0)

    # A máscara passa a ter apenas valores 0 ou 1.
    mascara = (mascara > 0).astype(np.uint8)

    if not mascara.any():
        raise ValueError("Máscara tumoral vazia")

    # Selecionamos o corte conforme a estratégia escolhida.
    if selecao == "central":
        z = n_z // 2

    elif selecao == "maior_tumor":
        areas = np.count_nonzero(
            mascara,
            axis=(1, 2)
        )

        z = int(np.argmax(areas))

    else:
        raise ValueError(
            f"Estratégia de corte desconhecida: {selecao}"
        )

    cortes = []
    referencia = imagem_mask.affine

    # Carregamos as três fases temporais selecionadas.
    for fase in FASES:
        imagem, volume = carregar_volume(
            aquisicoes[fase]
        )

        if volume.shape != imagem_mask.shape:
            raise ValueError(
                f"Dimensões incompatíveis na fase {fase}: "
                f"{volume.shape} != {imagem_mask.shape}"
            )

        if not np.allclose(
            imagem.affine,
            referencia,
            rtol=1e-4,
            atol=1e-3
        ):
            raise ValueError(
                f"Alinhamento espacial incompatível: {fase}"
            )

        # Mantemos a mesma organização para todas as fases.
        volume = np.moveaxis(volume, eixo, 0)

        # Selecionamos exatamente o mesmo corte nas fases.
        cortes.append(volume[z])

    # Empilhamos os três cortes em canais diferentes.
    fases = np.stack(cortes, axis=0)

    return (
        fases.astype(np.float32),
        mascara[z].astype(np.uint8),
        z
    )



def preparar_cache(selecao):
    criar_pastas()
    tabela = ler_metadados()

    destino = CACHE / selecao
    destino.mkdir(parents=True, exist_ok=True)

    relatorio = []

    for numero, (_, linha) in enumerate(
        tabela.iterrows(), start=1
    ):
        pid = str(linha["pid"])

        try:
            fases, mascara, z = preparar_paciente(
                linha, selecao
            )

            np.savez_compressed(
                destino / f"{pid}.npz",
                fases=fases,
                mascara=mascara,
                z=np.int32(z)
            )

            estado = "OK"
            detalhes = ""

        except Exception as erro:
            estado = "PROBLEMA"
            detalhes = str(erro)

        relatorio.append({
            "pid": pid,
            "status": estado,
            "detalhes": detalhes
        })

        print(
            f"[{numero}/{len(tabela)}] "
            f"{pid}: {estado}"
        )

    resultado = pd.DataFrame(relatorio)

    caminho_relatorio = (
        AUDITORIA / f"preparacao_{selecao}.csv"
    )

    resultado.to_csv(
        caminho_relatorio, index=False
    )

    print("\nPreparação concluída.")
    print(resultado["status"].value_counts())
    print(f"Cache em: {destino}")


# -----------------------------------------------------------
# Normalização e Dataset do PyTorch
# -----------------------------------------------------------

def normalizar(canais):
    x = canais.astype(np.float32, copy=True)

    valores = x[np.isfinite(x) & (x != 0)]

    if valores.size == 0:
        return np.zeros_like(x)

    minimo, maximo = np.percentile(
        valores, [1, 99]
    )

    if maximo <= minimo:
        return np.zeros_like(x)

    fundo = x == 0

    x = np.clip(
        (x - minimo) / (maximo - minimo),
        0, 1
    )

    x[fundo] = 0
    return x.astype(np.float32)


def redimensionar(canais, tamanho, tipo="bilinear"):
    tensor = torch.from_numpy(
        np.ascontiguousarray(canais)
    ).unsqueeze(0)

    if tipo == "nearest":
        saida = F.interpolate(
            tensor, size=(tamanho, tamanho),
            mode="nearest"
        )
    else:
        saida = F.interpolate(
            tensor, size=(tamanho, tamanho),
            mode="bilinear",
            align_corners=False
        )

    return saida.squeeze(0).numpy()


class DatasetMRI(Dataset):
    def __init__(
        self, particao, selecao, modo,
        tamanho=128, aumentar=False,
        usar_mascara="nenhuma"
    ):
        self.tabela = ler_metadados()

        self.tabela = self.tabela[
            self.tabela["test"] == particao
        ].reset_index(drop=True)

        self.cache = CACHE / selecao
        self.modo = modo
        self.tamanho = tamanho
        self.aumentar = aumentar
        self.usar_mascara = usar_mascara

        self.registros = []

        for _, linha in self.tabela.iterrows():
            arquivo = self.cache / (
                f"{linha['pid']}.npz"
            )

            if arquivo.is_file():
                self.registros.append({
                    "pid": str(linha["pid"]),
                    "pCR": int(linha["pCR"]),
                    "arquivo": arquivo
                })

    def __len__(self):
        multiplicador = 10 if self.aumentar else 1
        return len(self.registros) * multiplicador

    def __getitem__(self, indice):
        if self.aumentar:
            paciente_idx, variante = divmod(
                indice, 10
            )
        else:
            paciente_idx, variante = indice, None

        registro = self.registros[paciente_idx]

        with np.load(registro["arquivo"]) as dados:
            fases = dados["fases"].copy()
            mascara = dados["mascara"].copy()

        if self.modo == "pre":
            x = fases[0:1]
        elif self.modo == "early":
            x = fases[1:2]
        elif self.modo == "late":
            x = fases[2:3]
        elif self.modo == "three":
            x = fases
        else:
            raise ValueError("Modo temporal inválido")

        # Usamos uma mesma escala para todos os canais
        # presentes nesta amostra.
        x = normalizar(x)

        x = redimensionar(
            x, self.tamanho
        )

        mascara = redimensionar(
            mascara[None].astype(np.float32),
            self.tamanho,
            tipo="nearest"
        )[0]

        if variante is not None:
            angulo = ANGULOS[variante // 2]
            espelhar = bool(variante % 2)

            x = rotate(
                x, angulo,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode="constant",
                cval=0
            )

            mascara = rotate(
                mascara, angulo,
                reshape=False,
                order=0,
                mode="constant",
                cval=0
            )

            if espelhar:
                x = np.flip(x, axis=2)
                mascara = np.flip(mascara, axis=1)

        mascara = (mascara > 0.5).astype(np.float32)

        if self.usar_mascara == "zerar_fundo":
            x = x * mascara[None]

        elif self.usar_mascara == "canal":
            x = np.concatenate(
                [x, mascara[None]], axis=0
            )

        x = torch.from_numpy(
            np.ascontiguousarray(x, dtype=np.float32)
        )

        y = torch.tensor(
            registro["pCR"], dtype=torch.float32
        )

        return x, y, registro["pid"]


# -----------------------------------------------------------
# CNN própria de sete camadas convolucionais
# -----------------------------------------------------------

class CNN7(nn.Module):
    def __init__(self, canais_entrada=1):
        super().__init__()

        canais = [
            canais_entrada, 16, 16, 32,
            32, 64, 64, 128
        ]

        camadas = []

        for i in range(7):
            camadas.extend([
                nn.Conv2d(
                    canais[i], canais[i + 1],
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(canais[i + 1]),
                nn.ReLU(inplace=True)
            ])

            if i in (1, 3, 5):
                camadas.append(nn.MaxPool2d(2))

        self.extrator = nn.Sequential(*camadas)

        self.classificador = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        caracteristicas = self.extrator(x)
        return self.classificador(
            caracteristicas
        ).squeeze(1)


# -----------------------------------------------------------
# Métricas de classificação
# -----------------------------------------------------------

def calcular_metricas(y_true, probabilidades):
    y_true = np.asarray(y_true, dtype=int)
    probabilidades = np.asarray(
        probabilidades, dtype=float
    )

    previstos = (
        probabilidades >= 0.5
    ).astype(int)

    vp = int(np.sum(
        (previstos == 1) & (y_true == 1)
    ))
    vn = int(np.sum(
        (previstos == 0) & (y_true == 0)
    ))
    fp = int(np.sum(
        (previstos == 1) & (y_true == 0)
    ))
    fn = int(np.sum(
        (previstos == 0) & (y_true == 1)
    ))

    def divisao(a, b):
        return a / b if b else 0.0

    precisao = divisao(vp, vp + fp)
    sensibilidade = divisao(vp, vp + fn)

    f1 = divisao(
        2 * precisao * sensibilidade,
        precisao + sensibilidade
    )

    return {
        "VP": vp,
        "VN": vn,
        "FP": fp,
        "FN": fn,
        "acuracia": divisao(
            vp + vn, len(y_true)
        ),
        "sensibilidade": sensibilidade,
        "especificidade": divisao(
            vn, vn + fp
        ),
        "precisao": precisao,
        "f1": f1
    }


@torch.no_grad()
def validar(modelo, loader, criterio, dispositivo):
    modelo.eval()

    perda_total = 0.0
    rotulos = []
    probabilidades = []
    ids = []

    for imagens, y, pacientes in loader:
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
        ids.extend(pacientes)

    perda_media = perda_total / len(loader.dataset)

    metricas = calcular_metricas(
        rotulos, probabilidades
    )

    previsoes = pd.DataFrame({
        "pid": ids,
        "pCR_real": rotulos,
        "probabilidade": probabilidades
    })

    return perda_media, metricas, previsoes


# -----------------------------------------------------------
# Treinamento e checkpoint
# -----------------------------------------------------------

def treinar(args):
    criar_pastas()
    definir_semente(42)

    torch.set_num_threads(min(6, torch.get_num_threads()))

    treino = DatasetMRI(
        particao=0,
        selecao=args.selecao,
        modo=args.modo,
        tamanho=args.tamanho,
        aumentar=True,
        usar_mascara=args.mascara
    )

    validacao = DatasetMRI(
        particao=2,
        selecao=args.selecao,
        modo=args.modo,
        tamanho=args.tamanho,
        aumentar=False,
        usar_mascara=args.mascara
    )

    if not treino.registros or not validacao.registros:
        raise RuntimeError(
            "Prepare o cache antes de treinar."
        )

    # Evitamos comparar resultados de conjuntos reduzidos
    # com os resultados completos sem informar a diferença.
    if not args.parcial:
        if len(treino.registros) != 104:
            raise RuntimeError(
                "Treinamento incompleto. "
                "Use --parcial apenas para testes."
            )

        if len(validacao.registros) != 33:
            raise RuntimeError(
                "Validação incompleta. "
                "Use --parcial apenas para testes."
            )

    print(
        "Pacientes de treino:",
        len(treino.registros)
    )
    print(
        "Pacientes de validação:",
        len(validacao.registros)
    )
    print(
        "Amostras após augmentation:",
        len(treino)
    )

    gerador = torch.Generator()
    gerador.manual_seed(42)

    loader_treino = DataLoader(
        treino,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        generator=gerador
    )

    loader_val = DataLoader(
        validacao,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    canais = 3 if args.modo == "three" else 1

    if args.mascara == "canal":
        canais += 1

    dispositivo = torch.device("cpu")

    modelo = CNN7(
        canais_entrada=canais
    ).to(dispositivo)

    labels = [
        registro["pCR"]
        for registro in treino.registros
    ]

    positivos = sum(labels)
    negativos = len(labels) - positivos

    if positivos == 0 or negativos == 0:
        raise RuntimeError(
            "As duas classes são necessárias."
        )

    peso_positivo = negativos / positivos

    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [peso_positivo],
            dtype=torch.float32
        )
    )

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    nome = (
        f"cnn7_{args.modo}_{args.selecao}_"
        f"{args.mascara}"
    )

    caminho_modelo = MODELOS / f"{nome}.pth"

    melhor_perda = float("inf")
    sem_melhora = 0
    historico = []

    inicio = time.perf_counter()

    for epoca in range(1, args.epocas + 1):
        modelo.train()

        perda_treino = 0.0

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

            perda_treino += (
                perda.item() * len(y)
            )

        perda_treino /= len(treino)

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
            f"Época {epoca:03d} | "
            f"Treino: {perda_treino:.4f} | "
            f"Validação: {perda_val:.4f} | "
            f"F1: {metricas['f1']:.4f}"
        )

        if perda_val < melhor_perda - 1e-5:
            melhor_perda = perda_val
            sem_melhora = 0

            torch.save({
                "pesos": modelo.state_dict(),
                "canais": canais,
                "modo": args.modo,
                "selecao": args.selecao,
                "mascara": args.mascara,
                "tamanho": args.tamanho,
                "epoca": epoca,
                "loss_validacao": perda_val
            }, caminho_modelo)

            previsoes.to_csv(
                METRICAS / f"previsoes_val_{nome}.csv",
                index=False
            )

        else:
            sem_melhora += 1

        if sem_melhora >= args.paciencia:
            print("Early stopping ativado.")
            break

    duracao = time.perf_counter() - inicio

    tabela = pd.DataFrame(historico)
    tabela.to_csv(
        METRICAS / f"historico_{nome}.csv",
        index=False
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(
        tabela["epoca"],
        tabela["loss_treino"],
        label="Treinamento"
    )
    plt.plot(
        tabela["epoca"],
        tabela["loss_validacao"],
        label="Validação"
    )
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        GRAFICOS / f"loss_{nome}.png",
        dpi=150
    )
    plt.close()

    resumo = {
        "modelo": nome,
        "tempo_segundos": duracao,
        "pacientes_treino": len(treino.registros),
        "pacientes_validacao": len(validacao.registros),
        "peso_classe_positiva": peso_positivo,
        "melhor_loss_validacao": melhor_perda,
        "hiperparametros": {
            "epocas_maximas": args.epocas,
            "batch": args.batch,
            "lr": args.lr,
            "tamanho": args.tamanho,
            "paciencia": args.paciencia
        }
    }

    with open(
        METRICAS / f"resumo_{nome}.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            resumo, arquivo,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nTempo total: {duracao / 60:.2f} min")
    print(f"Melhor modelo salvo em: {caminho_modelo}")


# -----------------------------------------------------------
# Avaliação de um checkpoint
# -----------------------------------------------------------

def avaliar_checkpoint(args):
    caminho = Path(args.checkpoint)

    checkpoint = torch.load(
        caminho,
        map_location="cpu",
        weights_only=True
    )

    modelo = CNN7(
        canais_entrada=checkpoint["canais"]
    )

    modelo.load_state_dict(
        checkpoint["pesos"]
    )
    modelo.eval()

    particao = 2 if args.particao == "val" else 1

    if particao == 1 and not args.confirmar_teste:
        raise RuntimeError(
            "O conjunto de teste deve ser usado apenas "
            "na avaliação final. Acrescente "
            "--confirmar-teste para prosseguir."
        )

    dataset = DatasetMRI(
        particao=particao,
        selecao=checkpoint["selecao"],
        modo=checkpoint["modo"],
        tamanho=checkpoint["tamanho"],
        aumentar=False,
        usar_mascara=checkpoint["mascara"]
    )

    esperado = 33 if particao == 2 else 35

    if len(dataset.registros) != esperado:
        raise RuntimeError(
            f"Partição incompleta: "
            f"{len(dataset.registros)}/{esperado}"
        )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    criterio = nn.BCEWithLogitsLoss()

    perda, metricas, previsoes = validar(
        modelo, loader, criterio,
        torch.device("cpu")
    )

    print("\nResultados:")
    print(f"Loss: {perda:.4f}")

    for chave, valor in metricas.items():
        print(f"{chave}: {valor:.4f}")

    destino = (
        METRICAS /
        f"avaliacao_{args.particao}_{caminho.stem}.csv"
    )

    previsoes.to_csv(destino, index=False)
    print(f"Previsões salvas em: {destino}")


# -----------------------------------------------------------
# Execução pelo terminal
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline BreastDCEDL I-SPY1"
    )

    parser.add_argument(
        "comando",
        choices=["audit", "prepare", "train", "evaluate"]
    )

    parser.add_argument(
        "--selecao",
        choices=["central", "maior_tumor"],
        default="central"
    )

    parser.add_argument(
        "--modo",
        choices=["pre", "early", "late", "three"],
        default="early"
    )

    parser.add_argument(
        "--mascara",
        choices=["nenhuma", "zerar_fundo", "canal"],
        default="nenhuma"
    )

    parser.add_argument(
        "--tamanho", type=int, default=128
    )

    parser.add_argument(
        "--epocas", type=int, default=30
    )

    parser.add_argument(
        "--batch", type=int, default=8
    )

    parser.add_argument(
        "--lr", type=float, default=0.001
    )

    parser.add_argument(
        "--paciencia", type=int, default=8
    )

    parser.add_argument(
        "--parcial", action="store_true"
    )

    parser.add_argument(
        "--checkpoint", default=""
    )

    parser.add_argument(
        "--particao",
        choices=["val", "test"],
        default="val"
    )

    parser.add_argument(
        "--confirmar-teste",
        action="store_true"
    )

    args = parser.parse_args()

    if args.comando == "audit":
        auditar()

    elif args.comando == "prepare":
        preparar_cache(args.selecao)

    elif args.comando == "train":
        treinar(args)

    elif args.comando == "evaluate":
        if not args.checkpoint:
            parser.error(
                "Informe --checkpoint."
            )
        avaliar_checkpoint(args)


if __name__ == "__main__":
    main()

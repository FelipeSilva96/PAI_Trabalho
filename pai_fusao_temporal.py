
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
    DatasetMRI,
    criar_pastas,
    definir_semente,
    MODELOS,
    METRICAS
)


VERSAO = "FUSAO_TEMPORAL_V1"


class ExtratorCompartilhado(nn.Module):
    def __init__(self):
        super().__init__()

        self.camadas = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.camadas(x)


class RedeFusaoTemporal(nn.Module):
    def __init__(self):
        super().__init__()

        self.extrator = ExtratorCompartilhado()

        self.classificador = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "A entrada deve ter formato [B, 3, H, W]."
            )

        # As três fases são processadas pelo mesmo
        # extrator, com os mesmos parâmetros.
        fases = torch.cat([
            x[:, 0:1],
            x[:, 1:2],
            x[:, 2:3]
        ], dim=0)

        caracteristicas = self.extrator(fases)

        f_pre, f_early, f_late = (
            caracteristicas.chunk(3, dim=0)
        )

        fusao = torch.cat([
            f_pre,
            f_early,
            f_late,
            f_early - f_pre,
            f_late - f_early
        ], dim=1)

        return self.classificador(fusao).squeeze(1)


def criar_datasets(args):
    treino = DatasetMRI(
        particao=0,
        selecao=args.selecao,
        modo="three",
        tamanho=args.tamanho,
        aumentar=True
    )

    validacao = DatasetMRI(
        particao=2,
        selecao=args.selecao,
        modo="three",
        tamanho=args.tamanho,
        aumentar=False
    )

    if len(treino.registros) != 104:
        raise RuntimeError(
            "Treino precisa conter 104 pacientes."
        )

    if len(validacao.registros) != 33:
        raise RuntimeError(
            "Validação precisa conter 33 pacientes."
        )

    ids_treino = {
        r["pid"] for r in treino.registros
    }

    ids_val = {
        r["pid"] for r in validacao.registros
    }

    if ids_treino & ids_val:
        raise RuntimeError(
            "Vazamento: paciente presente em treino e validação."
        )

    return treino, validacao


def criar_loaders(treino, validacao, args):
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
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    return loader_treino, loader_val


def metricas_classificacao(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    previsto = (p >= 0.5).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, previsto, labels=[0, 1]
    ).ravel()

    return {
        "acuracia": float(
            accuracy_score(y, previsto)
        ),
        "acuracia_balanceada": float(
            balanced_accuracy_score(y, previsto)
        ),
        "sensibilidade": float(
            recall_score(y, previsto, zero_division=0)
        ),
        "especificidade": float(
            vn / (vn + fp)
        ) if (vn + fp) else 0.0,
        "precisao": float(
            precision_score(y, previsto, zero_division=0)
        ),
        "f1": float(
            f1_score(y, previsto, zero_division=0)
        ),
        "auc_roc": float(
            roc_auc_score(y, p)
        ),
        "average_precision": float(
            average_precision_score(y, p)
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
        "probabilidade_media": float(p.mean()),
        "probabilidade_desvio": float(p.std()),
        "previsoes_positivas": int(previsto.sum()),
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn)
    }


@torch.no_grad()
def validar(modelo, loader, criterio):
    modelo.eval()

    perda_total = 0.0
    rotulos = []
    probabilidades = []
    pacientes = []

    for imagens, y, ids in loader:
        logits = modelo(imagens)
        perda = criterio(logits, y)

        perda_total += perda.item() * len(y)

        rotulos.extend(y.cpu().numpy().tolist())
        probabilidades.extend(
            torch.sigmoid(logits).cpu().numpy().tolist()
        )
        pacientes.extend(ids)

    resultados = pd.DataFrame({
        "pid": pacientes,
        "pCR_real": rotulos,
        "probabilidade": probabilidades
    })

    if resultados["pid"].duplicated().any():
        raise RuntimeError(
            "Pacientes duplicados na validação."
        )

    metricas = metricas_classificacao(
        rotulos, probabilidades
    )

    return (
        perda_total / len(loader.dataset),
        metricas,
        resultados
    )


def testar_velocidade(modelo, loader, criterio, args):
    print("\nTESTE DE VELOCIDADE — SEM SALVAR PESOS")
    print("Lotes de teste:", args.lotes_teste)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=1e-3
    )

    modelo.train()

    tempos = []

    for numero, (imagens, y, _) in enumerate(
        loader, start=1
    ):
        if numero > args.lotes_teste:
            break

        inicio = time.perf_counter()

        otimizador.zero_grad(set_to_none=True)

        logits = modelo(imagens)
        perda = criterio(logits, y)

        perda.backward()
        otimizador.step()

        duracao = time.perf_counter() - inicio
        tempos.append(duracao)

        print(
            f"Lote {numero}/{args.lotes_teste} | "
            f"Tempo {duracao:.3f}s | "
            f"Loss {perda.item():.4f}",
            flush=True
        )

    amostra = tempos[1:] if len(tempos) > 1 else tempos
    media = float(np.mean(amostra))

    estimativa_epoca = media * len(loader)

    print("\nTESTE CONCLUÍDO")
    print(
        "Tempo médio por lote:",
        f"{media:.3f} segundo(s)"
    )
    print("Lotes por época:", len(loader))
    print(
        "Tempo estimado apenas para lotes por época:",
        f"{estimativa_epoca / 60:.2f} minuto(s)"
    )

    print(
        "Essa estimativa não inclui validação, "
        "salvamento e eventuais variações de desempenho."
    )


def identificar_experimento(args):
    lr = f"{args.lr:g}".replace(".", "p")
    pw = f"{args.pos_weight:g}".replace(".", "p")

    return (
        f"fusao_temporal_{args.selecao}_"
        f"sz{args.tamanho}_lr{lr}_pw{pw}_"
        f"seed{args.seed}_{args.tag}"
    )


def treinar(modelo, loader_treino, loader_val, criterio, args):
    criar_pastas()

    nome = identificar_experimento(args)

    destino_modelo = MODELOS / f"{nome}.pth"
    destino_historico = METRICAS / f"{nome}_historico.csv"
    destino_previsoes = METRICAS / f"{nome}_previsoes.csv"
    destino_resumo = METRICAS / f"{nome}_resumo.json"

    for destino in [
        destino_modelo,
        destino_historico,
        destino_previsoes,
        destino_resumo
    ]:
        if destino.exists():
            raise FileExistsError(
                f"Já existe resultado com essa identificação: {destino}"
            )

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=1e-3
    )

    melhor_loss = float("inf")
    melhor_epoca = 0
    melhores_metricas = None
    melhores_previsoes = None

    sem_melhora = 0
    historico = []

    inicio_total = time.perf_counter()

    print("\nINICIANDO TREINAMENTO DA FUSÃO TEMPORAL")
    print("Versão:", VERSAO)
    print("Amostras de treino:", len(loader_treino.dataset))
    print("Pacientes de validação:", len(loader_val.dataset))
    print("Lotes por época:", len(loader_treino))
    print("Máximo de épocas:", args.epocas)
    print()

    for epoca in range(1, args.epocas + 1):
        inicio_epoca = time.perf_counter()

        modelo.train()
        perda_total = 0.0

        for imagens, y, _ in loader_treino:
            otimizador.zero_grad(set_to_none=True)

            logits = modelo(imagens)
            perda = criterio(logits, y)

            perda.backward()

            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(), max_norm=5.0
            )

            otimizador.step()

            perda_total += perda.item() * len(y)

        loss_treino = perda_total / len(loader_treino.dataset)

        loss_val, metricas, previsoes = validar(
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
            f"Época {epoca:03d} | "
            f"Treino {loss_treino:.4f} | "
            f"Val {loss_val:.4f} | "
            f"F1 {metricas['f1']:.4f} | "
            f"AUC {metricas['auc_roc']:.4f} | "
            f"BalAcc {metricas['acuracia_balanceada']:.4f} | "
            f"Tempo {duracao / 60:.2f} min",
            flush=True
        )

        if loss_val < melhor_loss - 1e-5:
            melhor_loss = loss_val
            melhor_epoca = epoca
            melhores_metricas = metricas.copy()
            melhores_previsoes = previsoes.copy()
            sem_melhora = 0

            torch.save({
                "pesos_modelo": modelo.state_dict(),
                "arquitetura": "FusaoTemporalCompartilhada",
                "versao": VERSAO,
                "epoca": epoca,
                "loss_validacao": loss_val,
                "selecao": args.selecao,
                "tamanho": args.tamanho,
                "pos_weight": args.pos_weight,
                "lr": args.lr,
                "seed": args.seed
            }, destino_modelo)

        else:
            sem_melhora += 1

        if sem_melhora >= args.paciencia:
            print("Early stopping ativado.")
            break

    duracao_total = time.perf_counter() - inicio_total

    pd.DataFrame(historico).to_csv(
        destino_historico, index=False
    )

    melhores_previsoes.to_csv(
        destino_previsoes, index=False
    )

    resumo = {
        "experimento": nome,
        "versao": VERSAO,
        "melhor_epoca": melhor_epoca,
        "melhor_loss": melhor_loss,
        "metricas": melhores_metricas,
        "tempo_segundos": duracao_total,
        "configuracao": vars(args)
    }

    with open(destino_resumo, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print("\nTREINAMENTO CONCLUÍDO")
    print("Melhor época:", melhor_epoca)
    print("Melhor loss:", f"{melhor_loss:.4f}")
    print("AUC-ROC:", f"{melhores_metricas['auc_roc']:.4f}")
    print("F1:", f"{melhores_metricas['f1']:.4f}")
    print(
        "Acurácia balanceada:",
        f"{melhores_metricas['acuracia_balanceada']:.4f}"
    )
    print(
        "Tempo total:",
        f"{duracao_total / 60:.2f} minutos"
    )
    print("Checkpoint:", destino_modelo)
    print("Previsões:", destino_previsoes)


def main():
    parser = argparse.ArgumentParser(
        description="Rede de Fusão Temporal de DCE-MRI"
    )

    parser.add_argument(
        "--acao", choices=["teste", "treino"],
        default="teste"
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
        "--batch", type=int, default=8
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003
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
        "--lotes-teste", type=int, default=5
    )
    parser.add_argument(
        "--tag", default="FUS01"
    )

    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("Tag inválida.")

    if (
        args.batch <= 0 or args.lr <= 0
        or args.pos_weight <= 0
        or args.epocas <= 0
        or args.paciencia <= 0
        or args.lotes_teste <= 0
        or args.tamanho < 32
    ):
        parser.error("Parâmetros numéricos inválidos.")

    definir_semente(args.seed)

    torch.set_num_threads(
        min(6, torch.get_num_threads())
    )

    treino, validacao = criar_datasets(args)

    loader_treino, loader_val = criar_loaders(
        treino, validacao, args
    )

    # Reinicializamos após preparar os loaders.
    definir_semente(args.seed)

    modelo = RedeFusaoTemporal()

    total = sum(p.numel() for p in modelo.parameters())

    print("Versão:", VERSAO)
    print("Dispositivo: CPU")
    print("Pacientes treino:", len(treino.registros))
    print("Pacientes validação:", len(validacao.registros))
    print("Parâmetros totais:", f"{total:,}")

    criterio = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [args.pos_weight],
            dtype=torch.float32
        )
    )

    if args.acao == "teste":
        testar_velocidade(
            modelo, loader_treino, criterio, args
        )
    else:
        treinar(
            modelo,
            loader_treino,
            loader_val,
            criterio,
            args
        )


if __name__ == "__main__":
    main()

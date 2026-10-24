
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import binary_erosion
from matplotlib import colormaps

from pai_etapa2 import DatasetMRI, MODELOS
from pai_gradcam import (
    CHECKPOINTS,
    carregar_modelo,
    calcular_gradcam,
    carregar_mascara,
    verificar_previsao_salva
)


OPCOES_MODELO = {
    "CNN7": "cnn7",
    "ResNet50": "resnet50",
    "Fusão Temporal": "fusao"
}


def criar_paineis(imagens, mapas, mascara):
    """
    Gera versões para visualização:
    imagem original e imagem com Grad-CAM.
    """
    if imagens.shape[0] == 3:
        nomes = [
            "Pré-contraste",
            "Pós-precoce",
            "Pós-tardio"
        ]
    else:
        nomes = ["Pós-precoce"]

    borda = mascara & ~binary_erosion(mascara)

    resultado = []
    mapa_cores = colormaps["jet"]

    for indice, nome in enumerate(nomes):
        cinza = np.clip(
            imagens[indice], 0, 1
        )

        base = np.repeat(
            (cinza * 255).astype(np.uint8)[:, :, None],
            3,
            axis=2
        )

        mapa = np.clip(
            mapas[indice], 0, 1
        )

        cores = (
            mapa_cores(mapa)[:, :, :3] * 255
        ).astype(np.float32)

        opacidade = (
            mapa * 0.65
        )[:, :, None]

        sobreposta = (
            base.astype(np.float32) * (1 - opacidade)
            + cores * opacidade
        ).astype(np.uint8)

        original = base.copy()

        for imagem in (original, sobreposta):
            imagem[borda] = (40, 255, 90)

        original_pil = Image.fromarray(
            original
        ).resize(
            (235, 235),
            Image.Resampling.BILINEAR
        )

        sobreposta_pil = Image.fromarray(
            sobreposta
        ).resize(
            (235, 235),
            Image.Resampling.BILINEAR
        )

        resultado.append((
            nome,
            original_pil,
            sobreposta_pil
        ))

    return resultado


class JanelaIA:
    def __init__(self, mestre):
        self.mestre = mestre
        self.topo = tk.Toplevel(mestre)

        self.topo.title(
            "Classificação pCR e Grad-CAM | PAI"
        )
        self.topo.geometry("1050x760")
        self.topo.minsize(780, 630)

        self.fila = queue.Queue()
        self.processando = False
        self.fotos = []

        # Somente pacientes da validação.
        dataset = DatasetMRI(
            particao=2,
            selecao="central",
            modo="early",
            tamanho=128,
            aumentar=False
        )

        if len(dataset.registros) != 33:
            raise RuntimeError(
                "Esperados 33 pacientes de validação."
            )

        self.pacientes = sorted(
            registro["pid"]
            for registro in dataset.registros
        )

        self.modelo = tk.StringVar(
            value="CNN7"
        )

        self.pid = tk.StringVar(
            value=self.pacientes[0]
        )

        self.alvo = tk.StringVar(
            value="pcr1"
        )

        self.status = tk.StringVar(
            value="Selecione o modelo e clique em Analisar."
        )

        self.resumo = tk.StringVar(
            value="Nenhuma análise realizada."
        )

        self._construir_interface()

        self.topo.after(
            100, self._verificar_fila
        )

    def _construir_interface(self):
        cabecalho = ttk.Frame(
            self.topo,
            padding=18
        )
        cabecalho.pack(fill="x")

        ttk.Label(
            cabecalho,
            text="Análise de pCR — Redes Neurais",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w")

        ttk.Label(
            cabecalho,
            text=(
                "Protótipo experimental e educacional. "
                "Não destinado a diagnóstico clínico."
            )
        ).pack(anchor="w", pady=(5, 0))

        controles = ttk.Frame(
            self.topo,
            padding=(18, 8)
        )
        controles.pack(fill="x")

        ttk.Label(
            controles,
            text="Arquitetura:"
        ).grid(row=0, column=0, sticky="w")

        ttk.Combobox(
            controles,
            textvariable=self.modelo,
            values=list(OPCOES_MODELO.keys()),
            state="readonly",
            width=18
        ).grid(
            row=1, column=0,
            padx=(0, 15),
            sticky="w"
        )

        ttk.Label(
            controles,
            text="Paciente da validação:"
        ).grid(row=0, column=1, sticky="w")

        ttk.Combobox(
            controles,
            textvariable=self.pid,
            values=self.pacientes,
            state="readonly",
            width=20
        ).grid(
            row=1, column=1,
            padx=(0, 15),
            sticky="w"
        )

        ttk.Label(
            controles,
            text="Alvo da explicação:"
        ).grid(row=0, column=2, sticky="w")

        ttk.Combobox(
            controles,
            textvariable=self.alvo,
            values=["pcr1", "pcr0", "previsto"],
            state="readonly",
            width=15
        ).grid(
            row=1, column=2,
            padx=(0, 15),
            sticky="w"
        )

        self.botao = ttk.Button(
            controles,
            text="Analisar paciente",
            command=self.analisar
        )

        self.botao.grid(
            row=1, column=3, sticky="w"
        )

        ttk.Label(
            self.topo,
            textvariable=self.status,
            padding=(18, 12)
        ).pack(anchor="w")

        ttk.Label(
            self.topo,
            textvariable=self.resumo,
            font=("Segoe UI", 12, "bold"),
            padding=(18, 8)
        ).pack(anchor="w")

        ttk.Separator(
            self.topo,
            orient="horizontal"
        ).pack(fill="x", padx=18, pady=8)

        self.painel_resultados = ttk.Frame(
            self.topo,
            padding=12
        )
        self.painel_resultados.pack(
            fill="both",
            expand=True
        )

        ttk.Label(
            self.topo,
            text=(
                "Verde: contorno da máscara tumoral. "
                "Cores quentes: ativação positiva do Grad-CAM. "
                "Análise limitada ao corte preparado no treinamento."
            ),
            padding=12,
            wraplength=950
        ).pack(anchor="w")

    def analisar(self):
        if self.processando:
            return

        self.processando = True
        self.botao.configure(state="disabled")

        tipo = OPCOES_MODELO[
            self.modelo.get()
        ]

        pid = self.pid.get()
        alvo = self.alvo.get()

        self.status.set(
            f"Processando {pid} com {tipo}..."
        )

        threading.Thread(
            target=self._trabalhar,
            args=(tipo, pid, alvo),
            daemon=True
        ).start()

    def _trabalhar(self, tipo, pid, alvo):
        try:

            # O painel utiliza os checkpoints da avaliação final.
            # Não modificamos os checkpoints nem o protocolo congelado.
            if tipo == "cnn7":
                checkpoint = (
                    MODELOS /
                    "cnn7_early_central_batch_lr0p001_pw1_seed42_size128_E08.pth"
                )
            else:
                checkpoint = CHECKPOINTS[tipo]

            modelo, camada, modo, selecao, tamanho = (
                carregar_modelo(
                    tipo,
                    checkpoint
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
                    "Validação incompleta."
                )

            indices = [
                i for i, registro
                in enumerate(dataset.registros)
                if registro["pid"] == pid
            ]

            if len(indices) != 1:
                raise RuntimeError(
                    "Paciente não encontrado na validação."
                )

            imagem, rotulo, paciente = (
                dataset[indices[0]]
            )

            prob, previsto, mapas = (
                calcular_gradcam(
                    modelo,
                    camada,
                    imagem,
                    tipo,
                    alvo
                )
            )

            verificar_previsao_salva(
                checkpoint,
                paciente,
                prob
            )

            mascara = carregar_mascara(
                paciente,
                selecao,
                tamanho
            )

            imagens = (
                imagem.detach().cpu().numpy()
            )

            paineis = criar_paineis(
                imagens,
                mapas,
                mascara
            )

            resultado = {
                "tipo": tipo,
                "pid": paciente,
                "real": int(rotulo.item()),
                "prob": float(prob),
                "previsto": int(previsto),
                "alvo": alvo,
                "paineis": paineis
            }

            self.fila.put((
                "sucesso",
                resultado
            ))

        except Exception as erro:
            self.fila.put((
                "erro",
                str(erro)
            ))

    def _verificar_fila(self):
        try:
            while True:
                tipo, resultado = (
                    self.fila.get_nowait()
                )

                self.processando = False
                self.botao.configure(
                    state="normal"
                )

                if tipo == "erro":
                    self.status.set(
                        "Falha na análise."
                    )
                    messagebox.showerror(
                        "Erro de inferência",
                        resultado,
                        parent=self.topo
                    )
                else:
                    self._mostrar_resultado(
                        resultado
                    )

        except queue.Empty:
            pass

        if self.topo.winfo_exists():
            self.topo.after(
                100,
                self._verificar_fila
            )

    def _mostrar_resultado(self, resultado):
        for widget in (
            self.painel_resultados.winfo_children()
        ):
            widget.destroy()

        self.fotos.clear()

        real = resultado["real"]
        previsto = resultado["previsto"]
        prob = resultado["prob"]

        acerto = (
            "CORRETO"
            if real == previsto
            else "INCORRETO"
        )

        
        self.resumo.set(
        f"MODELO ANALISADO: {resultado['tipo'].upper()}   |   "
        f"Paciente: {resultado['pid']}   |   "
        f"pCR real: {real}   |   "
        f"Score pCR=1: {prob:.4f}   |   "
        f"Previsto: {previsto}   |   "
        f"{acerto}"
        )

        for indice, (
            nome, original, gradcam
        ) in enumerate(resultado["paineis"]):
            ttk.Label(
                self.painel_resultados,
                text=nome,
                font=("Segoe UI", 11, "bold")
            ).grid(
                row=0,
                column=indice,
                padx=10,
                pady=8
            )

            foto_original = ImageTk.PhotoImage(
                original
            )
            foto_gradcam = ImageTk.PhotoImage(
                gradcam
            )

            self.fotos.extend([
                foto_original,
                foto_gradcam
            ])

            ttk.Label(
                self.painel_resultados,
                image=foto_original
            ).grid(
                row=1,
                column=indice,
                padx=10,
                pady=5
            )

            ttk.Label(
                self.painel_resultados,
                image=foto_gradcam
            ).grid(
                row=2,
                column=indice,
                padx=10,
                pady=5
            )

        self.status.set(
            f"Análise concluída: {resultado['tipo']} "
            f"| Alvo Grad-CAM: {resultado['alvo']} "
            "| Somente validação"
        )


if __name__ == "__main__":
    raiz = tk.Tk()
    raiz.withdraw()

    janela = JanelaIA(raiz)

    janela.topo.protocol(
        "WM_DELETE_WINDOW",
        raiz.destroy
    )

    raiz.mainloop()

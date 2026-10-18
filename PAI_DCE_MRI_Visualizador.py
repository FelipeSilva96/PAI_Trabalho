"""
PUC Minas - Pontifícia Universidade Católica de Minas Gerais
Campus Coração Eucarístico | Ciência da Computação
Disciplina: Processamento e Análise de Imagens

Integrantes:
- Felipe Pereira da Silva       | Matrícula: preencher
- Rikerson Antônio Freitas Silva | Matrícula: preencher
- Mateus Resente Ottoni         | Matrícula: preencher

PROTÓTIPO EDUCACIONAL DE REFERÊNCIA - ETAPA 1
Leitura e visualização de volumes NIfTI (.nii/.nii.gz) e máscaras 3D.

IMPORTANTE PARA O TRABALHO:
O enunciado exige autoria original do grupo e declaração do uso de IA.
Este protótipo é material de estudo, não um trabalho pronto para ser
entregue como código de autoria exclusiva dos estudantes.

Instalação:
    py -m pip install numpy pillow nibabel matplotlib scikit-image
    py PAI_DCE_MRI_Visualizador.py

As duas últimas bibliotecas são opcionais, usadas apenas na visualização
tridimensional da máscara. A interface principal usa tkinter, da biblioteca
padrão do Python, além de numpy, pillow e nibabel.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import re
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


# As cores ficam juntas aqui para facilitar os ajustes da interface.
FUNDO = "#0b1220"
PAINEL = "#131e30"
PAINEL_2 = "#172439"
BORDA = "#2b3950"
TEXTO = "#edf4fc"
SUAVE = "#9aabc0"
DESTAQUE = "#40d6c0"
AZUL = "#4da9ff"
AVISO = "#f1c16e"
FASES = ("Pré-contraste", "Pós-contraste precoce", "Pós-contraste tardio")
EXTENSOES = (("NIfTI", "*.nii *.nii.gz"), ("Todos os arquivos", "*.*"))
RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
RESAMPLE_NEAREST = Image.Resampling.NEAREST


@dataclass
class Volume:
    caminho: str
    dados: np.ndarray
    affine: np.ndarray
    espacamento: tuple[float, float, float]
    unidade: str
    fonte_formato: str

    @property
    def nome(self):
        return Path(self.caminho).name

    @property
    def shape(self):
        return self.dados.shape


def carregar_nifti(caminho: str, mascara: bool = False) -> Volume:
    """Abre um volume sem alterar a ordem dos eixos gravada no arquivo."""
    try:
        import nibabel as nib
    except ImportError as erro:
        raise RuntimeError(
            "A biblioteca nibabel não foi instalada. Execute: "
            "py -m pip install nibabel"
        ) from erro

    img = nib.load(caminho, mmap=True)
    if len(img.shape) != 3:
        raise ValueError(
            f"Esperado um volume 3D, mas o arquivo possui dimensões {img.shape}. "
            "Arquivos 4D precisam ser separados por fase antes da visualização."
        )

    estimativa = int(np.prod(img.shape)) * (1 if mascara else 4)
    if estimativa > 1_500_000_000:
        raise ValueError(
            "O volume precisa de mais de 1,5 GB somente para os pixels. "
            "Utilize uma versão recortada ou reduza o volume antes de abrir."
        )

    # O cabeçalho guarda o espaçamento e a transformação para coordenadas físicas.
    # Mantemos os índices originais; a interface permite escolher o eixo de cortes.
    arr = np.asanyarray(img.dataobj)
    if mascara:
        dados = np.asarray(arr != 0, dtype=np.uint8)
    else:
        dados = np.asarray(arr, dtype=np.float32)
        if not np.isfinite(dados).any():
            raise ValueError("A imagem não contém pixels com intensidades válidas.")
    header = img.header
    unidade = header.get_xyzt_units()[0] or "desconhecida"
    espacamento = tuple(float(v) for v in header.get_zooms()[:3])
    return Volume(str(caminho), dados, np.array(img.affine, dtype=float), espacamento, unidade, "NIfTI")


def detectar_eixo(dimensoes: tuple[int, int, int]) -> int:
    """Encontra uma dimensão curta quando as outras parecem ser planos XY."""
    menor = int(np.argmin(dimensoes))
    outros = [dimensoes[i] for i in range(3) if i != menor]
    if dimensoes[menor] < 0.75 * min(outros):
        return menor
    return 2


def extrair_plano(dados: np.ndarray, eixo: int, indice: int) -> np.ndarray:
    """A indexação acontece no eixo selecionado e não depende de sua posição."""
    plano = np.take(dados, int(indice), axis=int(eixo))
    # Quando o arquivo está no padrão (x, y, z), X deve aparecer horizontal.
    if eixo == 2:
        plano = plano.T
    return np.asarray(plano)


def ajustar_janela(plano: np.ndarray, centro: float, largura: float, gamma: float = 1.0) -> np.ndarray:
    """Converte intensidades médicas para tons de cinza de 8 bits."""
    largura = max(1e-6, float(largura))
    minimo = centro - largura / 2.0
    resultado = (np.asarray(plano, dtype=np.float32) - minimo) / largura
    np.nan_to_num(resultado, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(resultado, 0.0, 1.0, out=resultado)
    resultado **= 1.0 / max(0.1, float(gamma))
    return np.asarray(resultado * 255.0, dtype=np.uint8)


def janela_automatica(dados: np.ndarray) -> tuple[float, float]:
    """Obtém limites robustos, desconsiderando um pouco dos extremos."""
    amostra = dados.ravel()[::max(1, dados.size // 250_000)]
    valores = amostra[np.isfinite(amostra)]
    if valores.size == 0:
        return 0.0, 1.0
    baixo, alto = np.percentile(valores, (1, 99))
    if alto <= baixo:
        baixo, alto = np.min(valores), np.max(valores)
    return float((baixo + alto) / 2.0), float(max(alto - baixo, 1.0))


def aplicar_orientacao(plano: np.ndarray, rotacao: int, esp_h: bool, esp_v: bool) -> np.ndarray:
    """Só muda a apresentação na tela; nunca altera o volume carregado."""
    plano = np.rot90(plano, k=rotacao)
    if esp_h:
        plano = np.fliplr(plano)
    if esp_v:
        plano = np.flipud(plano)
    return np.ascontiguousarray(plano)


def criar_imagem(plano: np.ndarray, mascara: np.ndarray | None = None,
                 modo: str = "Original", opacidade: float = 0.48) -> Image.Image:
    """Produz as cores da imagem e, quando solicitado, desenha a segmentação por cima."""
    rgb = np.repeat(plano[:, :, None], 3, axis=2)
    if modo == "Máscara" and mascara is not None:
        rgb[:] = 12
        rgb[mascara > 0] = (47, 225, 190)
    elif mascara is not None and modo == "Sobreposição":
        regiao = mascara > 0
        cor = np.array([45.0, 240.0, 190.0])
        rgb[regiao] = np.asarray(
            (1.0 - opacidade) * rgb[regiao].astype(float) + opacidade * cor,
            dtype=np.uint8,
        )
    return Image.fromarray(rgb, mode="RGB")


class PainelImagem(tk.Frame):
    def __init__(self, mestre, titulo, cor=DESTAQUE, ao_clicar=None, ao_rolar=None):
        super().__init__(mestre, bg=PAINEL, highlightthickness=1, highlightbackground=BORDA)
        self.titulo = titulo
        self.ao_clicar = ao_clicar
        self.ao_rolar = ao_rolar
        self.zoom = 1.0
        self.imagem = None
        self.foto = None
        self.geometria = None
        self.coordenadas = None
        self.ponto = None
        topo = tk.Frame(self, bg=PAINEL)
        topo.pack(fill="x")
        tk.Label(topo, text="●", fg=cor, bg=PAINEL, font=("Segoe UI", 12)).pack(side="left", padx=(12, 5), pady=9)
        tk.Label(topo, text=titulo, fg=TEXTO, bg=PAINEL, font=("Segoe UI", 11, "bold")).pack(side="left")
        self.subtitulo = tk.Label(topo, text="AGUARDANDO", fg=SUAVE, bg=PAINEL, font=("Segoe UI", 9))
        self.subtitulo.pack(side="right", padx=12)
        self.canvas = tk.Canvas(self, bg="#050b13", bd=0, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        self.canvas.bind("<Configure>", lambda e: self.redesenhar())
        self.canvas.bind("<Button-1>", self._clicou)
        self.canvas.bind("<MouseWheel>", self._rolou)
        self.canvas.bind("<Button-4>", lambda e: self._rolou(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._rolou(e, -1))

    def _rolou(self, evento, sinal=None):
        if self.ao_rolar is not None:
            if sinal is None:
                sinal = 1 if evento.delta > 0 else -1
            self.ao_rolar(sinal, bool(evento.state & 0x0004))
        return "break"

    def _clicou(self, evento):
        if self.geometria is None or self.imagem is None:
            return
        esquerda, superior, larg, alt = self.geometria
        col = int((evento.x - esquerda) / larg * self.imagem.width)
        lin = int((evento.y - superior) / alt * self.imagem.height)
        if 0 <= col < self.imagem.width and 0 <= lin < self.imagem.height:
            self.ponto = (col, lin)
            if self.ao_clicar:
                self.ao_clicar(col, lin)
            self._marcar_ponto()

    def atualizar(self, imagem: Image.Image | None, texto="", ponto=None):
        self.imagem = imagem
        self.ponto = ponto
        self.subtitulo.config(text=texto)
        self.redesenhar()

    def _marcar_ponto(self):
        self.canvas.delete("marcacao")
        if self.ponto is None or self.geometria is None or self.imagem is None:
            return
        x0, y0, larg, alt = self.geometria
        x = x0 + (self.ponto[0] + .5) * larg / self.imagem.width
        y = y0 + (self.ponto[1] + .5) * alt / self.imagem.height
        self.canvas.create_line(x - 12, y, x + 12, y, fill="#f8d67c", width=1, tags="marcacao")
        self.canvas.create_line(x, y - 12, x, y + 12, fill="#f8d67c", width=1, tags="marcacao")

    def redesenhar(self):
        canvas = self.canvas
        canvas.delete("all")
        if self.imagem is None:
            w, h = canvas.winfo_width(), canvas.winfo_height()
            canvas.create_text(w / 2, h / 2 - 12, text="SEM VOLUME CARREGADO",
                               fill=SUAVE, font=("Segoe UI", 11, "bold"))
            canvas.create_text(w / 2, h / 2 + 14, text="Abra um arquivo NIfTI para começar",
                               fill="#64748b", font=("Segoe UI", 9))
            self.geometria = None
            return
        w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        razao = min((w - 18) / self.imagem.width, (h - 18) / self.imagem.height)
        escala = max(.05, min(razao * self.zoom, 15.0))
        largura = max(1, int(self.imagem.width * escala))
        altura = max(1, int(self.imagem.height * escala))
        # Evita estourar a memória ao ampliar excessivamente uma imagem grande.
        if largura * altura > 12_000_000:
            fator = math.sqrt(12_000_000 / (largura * altura))
            largura, altura = int(largura * fator), int(altura * fator)
        exibida = self.imagem.resize((largura, altura), RESAMPLE_BILINEAR)
        self.foto = ImageTk.PhotoImage(exibida)
        x0 = (w - largura) / 2
        y0 = (h - altura) / 2
        canvas.create_image(x0, y0, anchor="nw", image=self.foto)
        self.geometria = (x0, y0, largura, altura)
        self._marcar_ponto()


class Aplicacao(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Breast DCE-MRI Studio | PUC Minas · Processamento e Análise de Imagens")
        self.geometry("1450x850")
        self.minsize(1140, 690)
        self.configure(bg=FUNDO)
        self.fases: dict[str, Volume] = {}
        self.mascara: Volume | None = None
        self.metadados = {}
        self.imagem_atual: Image.Image | None = None
        self.mascara_atual: Image.Image | None = None
        self.rotacao = 0
        self.flip_h = False
        self.flip_v = False
        self.centro = 0.0
        self.largura = 1.0
        self.animando = False
        self.eixo = tk.IntVar(value=2)
        self.corte = tk.IntVar(value=0)
        self.fase_atual = tk.StringVar(value=FASES[0])
        self.modo = tk.StringVar(value="Original")
        self.opacidade = tk.DoubleVar(value=.50)
        self.gamma = tk.DoubleVar(value=1.0)
        self.estado = tk.StringVar(value="Pronto. Abra um volume NIfTI ou inicie a demonstração sintética.")
        self.identificacao = tk.StringVar(value="Nenhum paciente carregado")
        self.info_volume = tk.StringVar(value="Dimensões: —\nEspaçamento: —\nIntensidade: —")
        self.info_mascara = tk.StringVar(value="Máscara não carregada")
        self.info_coordenada = tk.StringVar(value="Clique na imagem para consultar a intensidade")
        self._fila: queue.Queue = queue.Queue()
        self._estilo()
        self._construir_menu()
        self._construir_interface()
        self.bind("<Left>", lambda e: self.mudar_corte(-1) if not isinstance(e.widget, (tk.Entry, ttk.Entry)) else None)
        self.bind("<Right>", lambda e: self.mudar_corte(1) if not isinstance(e.widget, (tk.Entry, ttk.Entry)) else None)
        self.bind("<Control-o>", lambda e: self.abrir_volume())
        self.bind("<Control-m>", lambda e: self.abrir_mascara())
        self.bind("<Control-s>", lambda e: self.exportar_imagem())
        self.bind("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.after(100, self._consultar_fila)

    def _estilo(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TCombobox", fieldbackground=PAINEL_2, background=PAINEL_2,
                     foreground=TEXTO, arrowcolor=DESTAQUE, bordercolor=BORDA, padding=7)
        st.map("TCombobox", fieldbackground=[("readonly", PAINEL_2)],
               foreground=[("readonly", TEXTO)])
        st.configure("TScale", background=FUNDO, troughcolor=PAINEL_2)
        st.configure("TSpinbox", fieldbackground=PAINEL_2, foreground=TEXTO)

    def _construir_menu(self):
        menu = tk.Menu(self)
        arquivo = tk.Menu(menu, tearoff=False)
        arquivo.add_command(label="Abrir volume NIfTI...     Ctrl+O", command=self.abrir_volume)
        arquivo.add_command(label="Abrir máscara tumoral... Ctrl+M", command=self.abrir_mascara)
        arquivo.add_command(label="Abrir pasta de um paciente...", command=self.abrir_pasta)
        arquivo.add_command(label="Carregar CSV de metadados...", command=self.abrir_csv)
        arquivo.add_separator()
        arquivo.add_command(label="Demonstração com dados sintéticos", command=self.abrir_demo)
        arquivo.add_command(label="Exportar corte como PNG... Ctrl+S", command=self.exportar_imagem)
        arquivo.add_command(label="Fechar volumes", command=self.fechar_volumes)
        arquivo.add_separator()
        arquivo.add_command(label="Sair", command=self.destroy)
        menu.add_cascade(label="Arquivo", menu=arquivo)
        vis = tk.Menu(menu, tearoff=False)
        vis.add_command(label="Contraste automático", command=self.contraste_auto)
        vis.add_command(label="Rotacionar 90°", command=lambda: self.ajustar_orientacao("rotar"))
        vis.add_command(label="Espelhar horizontalmente", command=lambda: self.ajustar_orientacao("h"))
        vis.add_command(label="Espelhar verticalmente", command=lambda: self.ajustar_orientacao("v"))
        vis.add_command(label="Restaurar apresentação", command=lambda: self.ajustar_orientacao("reset"))
        vis.add_separator()
        vis.add_command(label="Visualizar três planos ortogonais", command=self.abrir_ortogonais)
        vis.add_command(label="Exibir histograma de intensidades", command=self.abrir_histograma)
        vis.add_command(label="Visualização 3D da máscara (opcional)", command=self.abrir_3d)
        vis.add_command(label="Tela cheia     F11", command=lambda: self.attributes("-fullscreen", True))
        menu.add_cascade(label="Visualizar", menu=vis)
        ferramentas = tk.Menu(menu, tearoff=False)
        ferramentas.add_command(label="Estatísticas da máscara/ROI", command=self.abrir_estatisticas)
        ferramentas.add_command(label="Próximo corte", command=lambda: self.mudar_corte(1))
        ferramentas.add_command(label="Corte anterior", command=lambda: self.mudar_corte(-1))
        menu.add_cascade(label="Ferramentas", menu=ferramentas)
        ajuda = tk.Menu(menu, tearoff=False)
        ajuda.add_command(label="Como utilizar", command=self.mostrar_ajuda)
        ajuda.add_command(label="Equipe e informações", command=self.mostrar_sobre)
        menu.add_cascade(label="Ajuda", menu=ajuda)
        self.configure(menu=menu)

    def _texto(self, pai, texto, tamanho=10, cor=TEXTO, negrito=False, **kwargs):
        return tk.Label(pai, text=texto, bg=pai.cget("bg"), fg=cor,
                        font=("Segoe UI", tamanho, "bold" if negrito else "normal"), **kwargs)

    def _botao(self, pai, titulo, comando, primario=False):
        return tk.Button(pai, text=titulo, command=comando, cursor="hand2", relief="flat", bd=0,
                         bg=DESTAQUE if primario else PAINEL_2,
                         fg="#071821" if primario else TEXTO, activebackground=AZUL,
                         activeforeground="#061422", font=("Segoe UI", 10, "bold"),
                         anchor="center", pady=10, padx=9)

    def _construir_interface(self):
        principal = tk.Frame(self, bg=FUNDO)
        principal.pack(fill="both", expand=True)
        lateral = tk.Frame(principal, bg=PAINEL, width=264)
        lateral.pack(side="left", fill="y")
        lateral.pack_propagate(False)
        corpo = tk.Frame(principal, bg=FUNDO)
        corpo.pack(side="right", fill="both", expand=True)

        self._texto(lateral, "  PUC MINAS  /  PAI", 10, DESTAQUE, True).pack(anchor="w", padx=17, pady=(25, 7))
        self._texto(lateral, "Breast DCE-MRI", 17, TEXTO, True).pack(anchor="w", padx=20)
        self._texto(lateral, "STUDIO   •   VISUALIZADOR", 9, SUAVE, True).pack(anchor="w", padx=20, pady=(1, 22))
        self._botao(lateral, "+  Abrir volume NIfTI", self.abrir_volume, True).pack(fill="x", padx=15, pady=4)
        self._botao(lateral, "+  Abrir máscara tumoral", self.abrir_mascara).pack(fill="x", padx=15, pady=4)
        self._botao(lateral, "Abrir pasta do paciente", self.abrir_pasta).pack(fill="x", padx=15, pady=4)
        self._botao(lateral, "Demonstração sintética", self.abrir_demo).pack(fill="x", padx=15, pady=4)

        self._texto(lateral, "FASE DA AQUISIÇÃO", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(23, 6))
        self.seletor_fase = ttk.Combobox(lateral, values=FASES, textvariable=self.fase_atual, state="readonly")
        self.seletor_fase.pack(fill="x", padx=15)
        self.seletor_fase.bind("<<ComboboxSelected>>", lambda e: self.atualizar_fase())
        self._texto(lateral, "EIXO DE NAVEGAÇÃO", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(19, 5))
        self.seletor_eixo = ttk.Combobox(lateral, values=["0", "1", "2"], state="readonly")
        self.seletor_eixo.set("2")
        self.seletor_eixo.pack(fill="x", padx=15)
        self.seletor_eixo.bind("<<ComboboxSelected>>", lambda e: self.atualizar_eixo(int(self.seletor_eixo.get())))
        self._texto(lateral, "MODO DE EXIBIÇÃO", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(19, 5))
        self.seletor_modo = ttk.Combobox(lateral, textvariable=self.modo, state="readonly",
                                          values=["Original", "Sobreposição", "Subtração (fase - pré)"])
        self.seletor_modo.pack(fill="x", padx=15)
        self.seletor_modo.bind("<<ComboboxSelected>>", lambda e: self.atualizar_imagens())
        self._texto(lateral, "OPACIDADE DA MÁSCARA", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(20, 0))
        tk.Scale(lateral, from_=0.1, to=1.0, resolution=.05, orient="horizontal",
                 variable=self.opacidade, command=lambda s: self.atualizar_imagens(),
                 bg=PAINEL, fg=TEXTO, troughcolor=PAINEL_2, highlightthickness=0,
                 activebackground=DESTAQUE).pack(fill="x", padx=18)
        self._texto(lateral, "GAMMA", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(8, 0))
        tk.Scale(lateral, from_=0.4, to=2.5, resolution=.1, orient="horizontal", variable=self.gamma,
                 command=lambda s: self.atualizar_imagens(), bg=PAINEL, fg=TEXTO,
                 troughcolor=PAINEL_2, highlightthickness=0,
                 activebackground=DESTAQUE).pack(fill="x", padx=18)
        self._texto(lateral, "DIMENSÕES E ESPAÇAMENTO", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(15, 5))
        self._texto(lateral, "", 9, SUAVE, False, textvariable=self.info_volume, justify="left",
                    wraplength=230).pack(anchor="w", padx=19)
        self._texto(lateral, "MÁSCARA / ROI", 9, SUAVE, True).pack(anchor="w", padx=18, pady=(17, 4))
        self._texto(lateral, "", 9, DESTAQUE, False, textvariable=self.info_mascara,
                    justify="left", wraplength=232).pack(anchor="w", padx=19)
        self._texto(lateral, "PUC Minas  •  Coração Eucarístico", 8, SUAVE).pack(side="bottom", pady=12)

        topo = tk.Frame(corpo, bg=FUNDO)
        topo.pack(fill="x", padx=20, pady=(16, 13))
        self._texto(topo, "Visualização de volumes", 19, TEXTO, True).pack(anchor="w")
        self._texto(topo, "", 10, SUAVE, textvariable=self.identificacao).pack(anchor="w", pady=(3, 0))

        botoes = tk.Frame(corpo, bg=FUNDO)
        botoes.pack(fill="x", padx=20, pady=(0, 12))
        for rotulo, acao in [
            ("Auto contraste", self.contraste_auto),
            ("Girar 90°", lambda: self.ajustar_orientacao("rotar")),
            ("Espelhar", lambda: self.ajustar_orientacao("h")),
            ("Três planos", self.abrir_ortogonais),
            ("Histograma", self.abrir_histograma),
            ("3D da máscara", self.abrir_3d),
            ("Exportar PNG", self.exportar_imagem),
        ]:
            self._botao(botoes, rotulo, acao).pack(side="left", padx=(0, 7))

        imagens = tk.Frame(corpo, bg=FUNDO)
        imagens.pack(fill="both", expand=True, padx=19)
        imagens.columnconfigure(0, weight=1, uniform="p")
        imagens.columnconfigure(1, weight=1, uniform="p")
        imagens.rowconfigure(0, weight=1)
        self.painel_imagem = PainelImagem(imagens, "RESSONÂNCIA MAGNÉTICA", DESTAQUE,
                                          self.clicar_imagem, self._rolar)
        self.painel_imagem.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.painel_mascara = PainelImagem(imagens, "SEGMENTAÇÃO TUMORAL", AZUL,
                                           self.clicar_imagem, self._rolar)
        self.painel_mascara.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        controles = tk.Frame(corpo, bg=PAINEL)
        controles.pack(fill="x", padx=19, pady=(13, 12))
        faixa = tk.Frame(controles, bg=PAINEL)
        faixa.pack(fill="x", padx=15, pady=(11, 3))
        self._texto(faixa, "NAVEGAÇÃO PELOS CORTES", 10, TEXTO, True).pack(side="left")
        self.rotulo_corte = self._texto(faixa, "CORTE —", 10, DESTAQUE, True)
        self.rotulo_corte.pack(side="right")
        trilha = tk.Frame(controles, bg=PAINEL)
        trilha.pack(fill="x", padx=13, pady=(0, 6))
        self._botao(trilha, "◀", lambda: self.mudar_corte(-1)).pack(side="left", padx=(0, 6))
        self.escala_corte = ttk.Scale(trilha, from_=0, to=1, orient="horizontal",
                                     command=self._mover_escala)
        self.escala_corte.pack(side="left", fill="x", expand=True)
        self._botao(trilha, "▶", lambda: self.mudar_corte(1)).pack(side="left", padx=(6, 0))
        self.botao_animar = self._botao(trilha, "Reproduzir", self.alternar_animacao)
        self.botao_animar.pack(side="left", padx=(7, 0))
        janela = tk.Frame(controles, bg=PAINEL)
        janela.pack(fill="x", padx=16, pady=(2, 12))
        self._texto(janela, "Centro", 9, SUAVE).pack(side="left", padx=(0, 5))
        self.centro_entrada = tk.Entry(janela, width=11, relief="flat", bg=PAINEL_2, fg=TEXTO,
                                       insertbackground=TEXTO, font=("Consolas", 10))
        self.centro_entrada.pack(side="left", padx=(0, 15))
        self._texto(janela, "Largura", 9, SUAVE).pack(side="left", padx=(0, 5))
        self.largura_entrada = tk.Entry(janela, width=11, relief="flat", bg=PAINEL_2, fg=TEXTO,
                                        insertbackground=TEXTO, font=("Consolas", 10))
        self.largura_entrada.pack(side="left", padx=(0, 8))
        self._botao(janela, "Aplicar janela", self.aplicar_janela).pack(side="left")
        self.centro_entrada.bind("<Return>", lambda e: self.aplicar_janela())
        self.largura_entrada.bind("<Return>", lambda e: self.aplicar_janela())
        self._texto(janela, "", 9, SUAVE, textvariable=self.info_coordenada).pack(side="right", padx=(4, 0))

        status = tk.Frame(corpo, bg=FUNDO)
        status.pack(fill="x", padx=20, pady=(0, 9))
        self._texto(status, "", 9, SUAVE, textvariable=self.estado, anchor="w").pack(fill="x")

    @property
    def volume(self):
        return self.fases.get(self.fase_atual.get())

    def _executar_em_thread(self, arquivos: list[tuple[str, str]]):
        """A leitura demora em alguns exames, então fica fora do loop gráfico."""
        self.estado.set("Lendo arquivos NIfTI… aguarde.")
        def trabalho():
            try:
                volumes = [(tipo, carregar_nifti(caminho, tipo == "mascara")) for tipo, caminho in arquivos]
                self._fila.put(("ok", volumes))
            except Exception as erro:
                self._fila.put(("erro", str(erro)))
        threading.Thread(target=trabalho, daemon=True).start()

    def _consultar_fila(self):
        try:
            while True:
                tipo, resultado = self._fila.get_nowait()
                if tipo == "erro":
                    self.estado.set("Falha na leitura do arquivo.")
                    messagebox.showerror("Falha ao abrir o NIfTI", resultado)
                else:
                    self._receber_volumes(resultado)
        except queue.Empty:
            pass
        self.after(100, self._consultar_fila)

    def _receber_volumes(self, volumes):
        primeiro = not bool(self.fases)
        avisos = []
        for tipo, vol in volumes:
            if tipo == "mascara":
                self.mascara = vol
            else:
                self.fases[tipo] = vol
        if primeiro and self.fases:
            self.eixo.set(detectar_eixo(next(iter(self.fases.values())).shape))
            self.seletor_eixo.set(str(self.eixo.get()))
            self.fase_atual.set(next(iter(self.fases.keys())))
            self.corte.set(next(iter(self.fases.values())).shape[self.eixo.get()] // 2)
            self.contraste_auto(False)
        elif self.volume is None and self.fases:
            self.fase_atual.set(next(iter(self.fases.keys())))
        if self.volume is not None and self.mascara is not None:
            if self.volume.shape != self.mascara.shape:
                avisos.append("A máscara e a imagem possuem dimensões diferentes. "
                              "A máscara não será exibida até serem compatíveis.")
            elif not np.allclose(self.volume.affine, self.mascara.affine, rtol=1e-4, atol=1e-3):
                avisos.append("As transformações espaciais (affines) da imagem e da máscara diferem. "
                              "Confira o registro/alinhamento antes de interpretar a sobreposição.")
        self.atualizar_fase(recalcular_janela=False)
        self.estado.set(f"{len(volumes)} arquivo(s) carregado(s). " + " ".join(avisos))
        if avisos:
            messagebox.showwarning("Atenção ao alinhamento", "\n\n".join(avisos))

    def abrir_volume(self):
        caminho = filedialog.askopenfilename(title="Escolha uma fase DCE-MRI", filetypes=EXTENSOES)
        if caminho:
            self._executar_em_thread([(self.fase_atual.get(), caminho)])

    def abrir_mascara(self):
        caminho = filedialog.askopenfilename(title="Escolha a máscara NIfTI", filetypes=EXTENSOES)
        if caminho:
            self._executar_em_thread([("mascara", caminho)])

    def abrir_pasta(self):
        pasta = filedialog.askdirectory(title="Pasta com os arquivos de um paciente")
        if not pasta:
            return
        arquivos = sorted(p for p in Path(pasta).rglob("*")
                          if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz")))
        if not arquivos:
            messagebox.showinfo("Nenhum NIfTI", "A pasta não contém arquivos .nii ou .nii.gz.")
            return

        # Se houver exames de vários pacientes, não fazemos misturas automáticas.
        pacientes = {a.group(0).upper() for p in arquivos
                     if (a := re.search(r"ISPY1[_-]\d+", p.name, re.IGNORECASE))}
        if len(pacientes) > 1:
            messagebox.showwarning("Mais de um paciente",
                                   "Foram encontrados arquivos de pacientes diferentes. "
                                   "Selecione uma pasta com apenas um paciente para evitar "
                                   "combinar exames incompatíveis.")
            return
        pid = next(iter(pacientes), None)
        registro = self.metadados.get(pid) if pid is not None else None
        indices_csv = {}
        if registro:
            try:
                indices_csv = {
                    FASES[0]: int(float(registro["pre"])),
                    FASES[1]: int(float(registro["post_early"])),
                    FASES[2]: int(float(registro["post_late"])),
                }
            except (ValueError, KeyError, TypeError):
                indices_csv = {}

        escolhido = {}
        aquisicoes = {}
        for arq in arquivos:
            nome = arq.name.lower()
            if re.search(r"mask|segm|_seg\b|_roi\b", nome):
                escolhido.setdefault("mascara", str(arq))
                continue
            achado = re.search(r"acq[_-]?(\d+)|scan[_-]?(\d+)", nome)
            if achado:
                indice = int(next(g for g in achado.groups() if g is not None))
                aquisicoes.setdefault(indice, str(arq))
            elif re.search(r"pre[-_]|precontrast", nome):
                escolhido.setdefault(FASES[0], str(arq))
            elif re.search(r"early|post1", nome):
                escolhido.setdefault(FASES[1], str(arq))
            elif re.search(r"late|post2", nome):
                escolhido.setdefault(FASES[2], str(arq))

        if indices_csv:
            for fase, indice in indices_csv.items():
                if indice in aquisicoes:
                    escolhido[fase] = aquisicoes[indice]
        elif len(aquisicoes) == 3:
            # Esta é somente uma sugestão pela ordem, não uma identificação clínica.
            for fase, indice in zip(FASES, sorted(aquisicoes)):
                escolhido.setdefault(fase, aquisicoes[indice])
        elif len(aquisicoes) > 3:
            messagebox.showwarning("CSV necessário",
                                   "Existem mais de três aquisições, mas o CSV correspondente "
                                   "ao paciente ainda não foi carregado. Use Arquivo > Carregar CSV "
                                   "e reabra esta pasta para selecionar as fases corretas.")
            return

        if not escolhido:
            messagebox.showinfo("Seleção manual necessária",
                                f"Foram encontrados {len(arquivos)} arquivos NIfTI, mas seus nomes "
                                "não permitem reconhecer as fases. Selecione a fase no painel "
                                "lateral e abra cada volume manualmente.")
            return
        resumo = "\n".join(f"{rotulo}: {Path(path).name}" for rotulo, path in escolhido.items())
        observacao = ("Seleção orientada pelo CSV do paciente."
                      if indices_csv else
                      "IMPORTANTE: seleção sugerida apenas pelos nomes/ordem dos arquivos. "
                      "Compare com as colunas pre, post_early e post_late do CSV.")
        if messagebox.askyesno("Confirmar fases detectadas",
                               observacao + "\n\n" + resumo + "\n\nCarregar?"):
            self._executar_em_thread(list(escolhido.items()))

    def abrir_csv(self):
        caminho = filedialog.askopenfilename(title="Abrir metadados de I-SPY1", filetypes=[("CSV", "*.csv")])
        if not caminho:
            return
        try:
            with open(caminho, encoding="utf-8-sig", newline="") as arq:
                linhas = list(csv.DictReader(arq))
            self.metadados = {linha["pid"]: linha for linha in linhas if linha.get("pid")}
            self.estado.set(f"Metadados carregados: {len(self.metadados)} pacientes (CSV).")
            self.atualizar_imagens()
        except Exception as erro:
            messagebox.showerror("CSV inválido", str(erro))

    def abrir_demo(self):
        """O exemplo é inteiramente artificial: não é imagem de paciente real."""
        xx, yy, zz = np.ogrid[:256, :256, :56]
        corpo = ((xx - 128) / 95)**2 + ((yy - 128) / 88)**2 + ((zz - 27) / 27)**2
        tumor = ((xx - 156) / 23)**2 + ((yy - 110) / 19)**2 + ((zz - 30) / 12)**2
        base = 220 * np.exp(-corpo * 1.7) + 70 * np.exp(-((xx-91)**2+(yy-155)**2)/1350)
        ruido = np.random.default_rng(42).normal(0, 7, size=(256, 256, 56))
        base = np.maximum(0, base + ruido).astype(np.float32)
        mascara = np.broadcast_to(tumor < 1, base.shape).astype(np.uint8).copy()
        affine = np.diag([.8, .8, 2.0, 1.0])
        self.fases = {
            fase: Volume("DEMONSTRACAO_SINTETICA_" + str(i),
                         np.maximum(0, base + i * mascara * 80).astype(np.float32),
                         affine, (.8, .8, 2.0), "mm", "SINTÉTICO")
            for i, fase in enumerate(FASES)
        }
        self.mascara = Volume("DEMONSTRACAO_MASCARA", mascara, affine,
                              (.8, .8, 2.0), "mm", "SINTÉTICO")
        self.fase_atual.set(FASES[0])
        self.eixo.set(2)
        self.seletor_eixo.set("2")
        self.corte.set(28)
        self.contraste_auto(False)
        self.atualizar_fase(False)
        self.estado.set("DEMONSTRAÇÃO SINTÉTICA: dados fictícios, sem informação médica real.")

    def fechar_volumes(self):
        self.animando = False
        self.fases.clear()
        self.mascara = None
        self.imagem_atual = None
        self.mascara_atual = None
        self.painel_imagem.atualizar(None)
        self.painel_mascara.atualizar(None)
        self.info_mascara.set("Máscara não carregada")
        self.identificacao.set("Nenhum paciente carregado")
        self.info_volume.set("Dimensões: —\nEspaçamento: —\nIntensidade: —")
        self.rotulo_corte.config(text="CORTE —")
        self.estado.set("Os volumes foram fechados.")

    def atualizar_fase(self, recalcular_janela=True):
        vol = self.volume
        if vol is None:
            self.estado.set("Essa fase ainda não foi carregada. Use 'Abrir volume NIfTI'.")
            self.atualizar_imagens()
            return
        eixo = self.eixo.get()
        self.corte.set(min(self.corte.get(), vol.shape[eixo] - 1))
        self.escala_corte.configure(to=max(0, vol.shape[eixo] - 1))
        self.escala_corte.set(self.corte.get())
        if recalcular_janela:
            self.contraste_auto(False)
        self.atualizar_imagens()

    def atualizar_eixo(self, novo):
        self.eixo.set(novo)
        if self.volume is not None:
            self.corte.set(self.volume.shape[novo] // 2)
            self.escala_corte.configure(to=self.volume.shape[novo] - 1)
            self.escala_corte.set(self.corte.get())
        self.atualizar_imagens()

    def _mover_escala(self, novo):
        if self.volume is None:
            return
        valor = int(round(float(novo)))
        if valor != self.corte.get():
            self.corte.set(valor)
            self.atualizar_imagens()

    def mudar_corte(self, delta):
        if self.volume is None:
            return
        novo = int(np.clip(self.corte.get() + delta, 0, self.volume.shape[self.eixo.get()] - 1))
        self.corte.set(novo)
        self.escala_corte.set(novo)
        self.atualizar_imagens()

    def _rolar(self, sinal, ctrl=False):
        if ctrl:
            for painel in (self.painel_imagem, self.painel_mascara):
                painel.zoom = max(.3, min(8.0, painel.zoom * (1.15 if sinal > 0 else 1/1.15)))
                painel.redesenhar()
        else:
            self.mudar_corte(sinal)

    def alternar_animacao(self):
        if not self.volume:
            return
        self.animando = not self.animando
        self.botao_animar.configure(text="Pausar" if self.animando else "Reproduzir")
        if self.animando:
            self._proximo_animacao()

    def _proximo_animacao(self):
        if not self.animando or self.volume is None:
            return
        proximo = (self.corte.get() + 1) % self.volume.shape[self.eixo.get()]
        self.corte.set(proximo)
        self.escala_corte.set(proximo)
        self.atualizar_imagens()
        self.after(130, self._proximo_animacao)

    def contraste_auto(self, renderizar=True):
        if self.volume is None:
            return
        self.centro, self.largura = janela_automatica(self.volume.dados)
        self._entradas_janela()
        if renderizar:
            self.atualizar_imagens()

    def _entradas_janela(self):
        self.centro_entrada.delete(0, "end")
        self.centro_entrada.insert(0, f"{self.centro:.2f}")
        self.largura_entrada.delete(0, "end")
        self.largura_entrada.insert(0, f"{self.largura:.2f}")

    def aplicar_janela(self):
        try:
            centro = float(self.centro_entrada.get().replace(",", "."))
            largura = float(self.largura_entrada.get().replace(",", "."))
            if not (np.isfinite(centro) and np.isfinite(largura) and largura > 0):
                raise ValueError("Centro finito e largura positiva são necessários.")
            self.centro, self.largura = centro, largura
            self.atualizar_imagens()
        except ValueError:
            messagebox.showwarning("Janela inválida", "Informe valores numéricos válidos. A largura deve ser positiva.")

    def ajustar_orientacao(self, acao):
        if acao == "rotar":
            self.rotacao = (self.rotacao + 1) % 4
        elif acao == "h":
            self.flip_h = not self.flip_h
        elif acao == "v":
            self.flip_v = not self.flip_v
        else:
            self.rotacao, self.flip_h, self.flip_v = 0, False, False
        self.atualizar_imagens()

    def _mascara_compativel(self):
        return (self.volume is not None and self.mascara is not None
                and self.mascara.shape == self.volume.shape
                and np.allclose(self.mascara.affine, self.volume.affine,
                                rtol=1e-4, atol=1e-3))

    def atualizar_imagens(self):
        vol = self.volume
        if vol is None:
            self.painel_imagem.atualizar(None)
            self.painel_mascara.atualizar(None)
            return
        eixo = self.eixo.get()
        indice = min(self.corte.get(), vol.shape[eixo] - 1)
        self.corte.set(indice)
        valores = extrair_plano(vol.dados, eixo, indice)
        modo = self.modo.get()
        if modo.startswith("Subtração"):
            pre = self.fases.get(FASES[0])
            if pre is not None and pre.shape == vol.shape and np.allclose(pre.affine, vol.affine, atol=1e-3):
                diferenca = valores - extrair_plano(pre.dados, eixo, indice)
                amplitude = np.percentile(np.abs(diferenca[np.isfinite(diferenca)]), 99) if np.isfinite(diferenca).any() else 1
                cinza = ajustar_janela(diferenca, 0, max(1.0, 2 * amplitude), self.gamma.get())
            else:
                cinza = ajustar_janela(valores, self.centro, self.largura, self.gamma.get())
                self.estado.set("Subtração requer a fase pré-contraste com dimensões e affine compatíveis.")
        else:
            cinza = ajustar_janela(valores, self.centro, self.largura, self.gamma.get())
        cinza = aplicar_orientacao(cinza, self.rotacao, self.flip_h, self.flip_v)
        mask = None
        if self._mascara_compativel():
            mask = aplicar_orientacao(extrair_plano(self.mascara.dados, eixo, indice),
                                     self.rotacao, self.flip_h, self.flip_v)
        sobreposicao = criar_imagem(cinza, mask,
                                   "Sobreposição" if modo == "Sobreposição" else "Original",
                                   self.opacidade.get())
        self.imagem_atual = sobreposicao
        self.mascara_atual = criar_imagem(cinza, mask, "Máscara") if mask is not None else None
        titulo = f"EIXO {eixo}  ·  CORTE {indice + 1}/{vol.shape[eixo]}"
        self.painel_imagem.atualizar(sobreposicao, titulo)
        self.painel_mascara.atualizar(self.mascara_atual, "MESMO CORTE · SEGMENTAÇÃO" if mask is not None else "SEM MÁSCARA")
        self.rotulo_corte.config(text=f"EIXO {eixo}   |   {indice + 1} / {vol.shape[eixo]}")
        self.info_volume.set(f"Dimensões: {vol.shape}\n"
                             f"Espaçamento: {tuple(round(v, 3) for v in vol.espacamento)} {vol.unidade}\n"
                             f"Intensidade do corte: {np.nanmin(valores):.1f} a {np.nanmax(valores):.1f}")
        self.info_mascara.set(self._resumo_mascara())
        self.identificacao.set(f"{self._pid()}  ·  {self.fase_atual.get()}  ·  {vol.nome}")

    def _pid(self):
        vol = self.volume
        if vol is None:
            return "Nenhum paciente"
        if vol.fonte_formato == "SINTÉTICO":
            return "DEMONSTRAÇÃO - DADOS SINTÉTICOS"
        pesquisa = re.search(r"ISPY1[_-]\d+", vol.nome, re.IGNORECASE)
        return pesquisa.group(0) if pesquisa else "Volume NIfTI"

    def _resumo_mascara(self):
        if not self._mascara_compativel():
            return "Máscara ausente ou incompatível (dimensões/affine)"
        voxels = int(np.count_nonzero(self.mascara.dados))
        vol = self.mascara
        volume_mm3 = voxels * float(np.prod(vol.espacamento))
        if vol.unidade == "mm":
            tamanho = f"{volume_mm3 / 1000:.2f} cm³"
        elif vol.unidade == "micron":
            tamanho = f"{volume_mm3 / 1_000_000_000_000:.2f} cm³"
        elif vol.unidade == "meter":
            tamanho = f"{volume_mm3 * 1_000_000:.2f} cm³"
        else:
            tamanho = "unidade espacial não identificada"
        return f"Voxels: {voxels:,}\nVolume estimado: {tamanho}"

    def clicar_imagem(self, col, lin):
        if self.volume is None:
            return
        vol = self.volume
        eixo = self.eixo.get()
        plano = extrair_plano(vol.dados, eixo, self.corte.get())
        plano = aplicar_orientacao(plano, self.rotacao, self.flip_h, self.flip_v)
        if 0 <= lin < plano.shape[0] and 0 <= col < plano.shape[1]:
            intensidade = plano[lin, col]
            self.info_coordenada.set(f"Tela: ({col}, {lin})   |   Intensidade: {intensidade:.2f}")
            self.painel_imagem.ponto = (col, lin)
            self.painel_mascara.ponto = (col, lin)
            self.painel_imagem._marcar_ponto()
            self.painel_mascara._marcar_ponto()

    def exportar_imagem(self):
        if self.imagem_atual is None:
            messagebox.showinfo("Sem imagem", "Primeiro carregue um volume.")
            return
        caminho = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("Imagem PNG", "*.png")],
                                                initialfile="corte_dce_mri.png")
        if not caminho:
            return
        imgs = [self.imagem_atual]
        if self.mascara_atual is not None:
            imgs.append(self.mascara_atual)
        largura = sum(img.width for img in imgs) + 22 * (len(imgs) + 1)
        altura = max(img.height for img in imgs) + 80
        saida = Image.new("RGB", (largura, altura), (17, 27, 43))
        desenho = ImageDraw.Draw(saida)
        x = 22
        for pos, img in enumerate(imgs):
            desenho.text((x, 16), "RESSONANCIA" if pos == 0 else "MASCARA TUMORAL", fill=(222, 238, 255))
            saida.paste(img, (x, 44))
            x += img.width + 22
        saida.save(caminho)
        self.estado.set(f"Corte exportado: {caminho}")

    def abrir_estatisticas(self):
        if self.volume is None:
            messagebox.showinfo("Sem dados", "Abra um volume para ver suas informações.")
            return
        vol = self.volume
        finitos = vol.dados[np.isfinite(vol.dados)]
        resumo = (f"Arquivo: {vol.nome}\nDimensões: {vol.shape}\n"
                  f"Espaçamento: {vol.espacamento} {vol.unidade}\n"
                  f"Mínimo: {np.min(finitos):.2f}\nMáximo: {np.max(finitos):.2f}\n"
                  f"Média: {np.mean(finitos):.2f}\nDesvio padrão: {np.std(finitos):.2f}\n\n"
                  + self._resumo_mascara())
        messagebox.showinfo("Informações do exame / ROI", resumo)

    def abrir_histograma(self):
        if self.volume is None:
            messagebox.showinfo("Sem imagem", "Abra um volume antes de calcular o histograma.")
            return
        janela = tk.Toplevel(self)
        janela.title("Histograma do volume | DCE-MRI")
        janela.configure(bg=FUNDO)
        janela.geometry("780x450")
        tela = tk.Canvas(janela, bg=PAINEL, highlightthickness=0)
        tela.pack(fill="both", expand=True, padx=16, pady=16)
        amostra = self.volume.dados.ravel()[::max(1, self.volume.dados.size // 300_000)]
        amostra = amostra[np.isfinite(amostra)]
        inferior, superior = np.percentile(amostra, (0.5, 99.5))
        if inferior == superior:
            superior = inferior + 1
        hist, bordas = np.histogram(amostra, bins=100, range=(inferior, superior))
        def desenhar(evento=None):
            tela.delete("all")
            w, h = max(1, tela.winfo_width()), max(1, tela.winfo_height())
            left, top, right, bottom = 65, 60, w - 26, h - 56
            tela.create_text(left, 25, text="DISTRIBUIÇÃO DAS INTENSIDADES", anchor="w",
                             fill=TEXTO, font=("Segoe UI", 13, "bold"))
            tela.create_line(left, bottom, right, bottom, fill=SUAVE)
            tela.create_line(left, top, left, bottom, fill=SUAVE)
            vmax = max(int(hist.max()), 1)
            for i, conta in enumerate(hist):
                x0 = left + i / len(hist) * (right - left)
                x1 = left + (i + 1) / len(hist) * (right - left)
                y0 = bottom - int(conta) / vmax * (bottom - top)
                tela.create_rectangle(x0, y0, x1, bottom, fill=DESTAQUE, outline="")
            tela.create_text(left, bottom + 17, text=f"{bordas[0]:.1f}", fill=SUAVE, anchor="w")
            tela.create_text(right, bottom + 17, text=f"{bordas[-1]:.1f}", fill=SUAVE, anchor="e")
            tela.create_text(left + 2, top - 12, text=f"Máxima frequência: {vmax}", fill=SUAVE, anchor="w")
        tela.bind("<Configure>", desenhar)

    def abrir_ortogonais(self):
        if self.volume is None:
            messagebox.showinfo("Sem imagem", "Abra um volume para visualizar os três planos.")
            return
        topo = tk.Toplevel(self)
        topo.title("Visualização ortogonal | Eixos 0, 1 e 2")
        topo.configure(bg=FUNDO)
        topo.geometry("1120x500")
        quadros = tk.Frame(topo, bg=FUNDO)
        quadros.pack(fill="both", expand=True, padx=12, pady=10)
        quadros.rowconfigure(0, weight=1)
        for axis in range(3):
            quadros.columnconfigure(axis, weight=1, uniform="v")
        indices = [tk.IntVar(value=self.volume.shape[i] // 2) for i in range(3)]
        paines = []
        def redesenhar():
            if len(paines) != 3:
                return
            for axis in range(3):
                matriz = extrair_plano(self.volume.dados, axis, indices[axis].get())
                cinza = ajustar_janela(matriz, self.centro, self.largura, self.gamma.get())
                mask = None
                if self._mascara_compativel():
                    mask = extrair_plano(self.mascara.dados, axis, indices[axis].get())
                paines[axis].atualizar(criar_imagem(cinza, mask, "Sobreposição" if mask is not None else "Original"),
                                      f"CORTE {indices[axis].get() + 1}/{self.volume.shape[axis]}")
        for axis in range(3):
            bloco = tk.Frame(quadros, bg=PAINEL)
            bloco.grid(row=0, column=axis, padx=5, sticky="nsew")
            painel = PainelImagem(bloco, f"PLANO PERPENDICULAR AO EIXO {axis}")
            painel.pack(fill="both", expand=True)
            barra = ttk.Scale(bloco, from_=0, to=self.volume.shape[axis] - 1,
                              command=lambda valor, i=axis: (indices[i].set(int(float(valor))), redesenhar()))
            barra.pack(fill="x", padx=15, pady=10)
            barra.set(indices[axis].get())
            paines.append(painel)
        redesenhar()

    def abrir_3d(self):
        if not self._mascara_compativel():
            messagebox.showinfo("3D da segmentação", "Carregue primeiro uma imagem e uma máscara compatíveis.")
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from skimage.measure import marching_cubes
        except ImportError:
            messagebox.showinfo("Bibliotecas opcionais", "Para a superfície 3D, instale: "
                                "py -m pip install matplotlib scikit-image")
            return
        dados = self.mascara.dados
        if np.count_nonzero(dados) < 8:
            messagebox.showinfo("Máscara vazia", "A máscara não contém voxels suficientes para gerar uma superfície.")
            return
        # A redução preserva a geometria aproximada e impede que a janela trave.
        passo = max(1, int(np.ceil(max(dados.shape) / 125)))
        reduzido = dados[::passo, ::passo, ::passo]
        try:
            vertices, faces, _, _ = marching_cubes(np.pad(reduzido, 1), level=.5)
        except ValueError as erro:
            messagebox.showerror("Falha na malha", str(erro))
            return
        vertices = (vertices - 1) * np.array(self.mascara.espacamento) * passo
        janela = tk.Toplevel(self)
        janela.title("Superfície tridimensional da máscara tumoral - visualização exploratória")
        janela.geometry("820x670")
        fig = Figure(figsize=(8, 6), dpi=90, facecolor="#0b1220")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0b1220")
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces,
                        color="#40d6c0", linewidth=.1, antialiased=True, alpha=.95)
        ax.set_xlabel("Eixo 0")
        ax.set_ylabel("Eixo 1")
        ax.set_zlabel("Eixo 2")
        ax.set_title("Máscara 3D - gire com o mouse", color="white")
        ax.tick_params(colors="white")
        ax.set_box_aspect(np.ptp(vertices, axis=0) + 1)
        fig.tight_layout()
        plot = FigureCanvasTkAgg(fig, master=janela)
        plot.draw()
        plot.get_tk_widget().pack(fill="both", expand=True)
        barra = NavigationToolbar2Tk(plot, janela)
        barra.update()
        janela._figura = fig
        janela._plot = plot

    def mostrar_ajuda(self):
        messagebox.showinfo("Instruções de uso",
            "1. Selecione uma fase no menu lateral e abra seu arquivo .nii ou .nii.gz.\n"
            "2. Carregue a máscara tumoral correspondente ao mesmo paciente.\n"
            "3. Use o controle de cortes, as setas ou a roda do mouse para navegar.\n"
            "4. Segure Ctrl e use a roda do mouse para ampliar ou reduzir.\n"
            "5. Ajuste centro e largura da janela para controlar contraste e brilho.\n"
            "6. Use o modo Sobreposição para verificar onde está o tumor.\n"
            "7. O eixo de cortes pode ser 0, 1 ou 2, conforme a organização do NIfTI.\n"
            "8. Use a demonstração sintética para explorar a interface sem exames reais.\n\n"
            "Atenção: esta ferramenta é educacional, não diagnóstica.")

    def mostrar_sobre(self):
        messagebox.showinfo("Sobre o protótipo",
            "PUC Minas - Campus Coração Eucarístico\n"
            "Ciência da Computação - Processamento e Análise de Imagens\n\n"
            "Felipe Pereira da Silva\nRikerson Antônio Freitas Silva\nMateus Resente Ottoni\n\n"
            "Etapa 1: visualização de volumes NIfTI e segmentações.\n\n"
            "PROTÓTIPO EDUCACIONAL / REFERÊNCIA COM USO DE IA.\n"
            "O trabalho oficial deve ser implementado originalmente pelo grupo e "
            "declarar o uso de inteligência artificial, conforme o enunciado.")


if __name__ == "__main__":
    Aplicacao().mainloop()


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Paciente utilizado para conferir a preparação.
pid = "ISPY1_1001"

arquivo = Path("dados/cache/central") / f"{pid}.npz"

with np.load(arquivo) as dados:
    fases = dados["fases"]
    mascara = dados["mascara"]
    z = int(dados["z"])

print("Paciente:", pid)
print("Dimensões das fases:", fases.shape)
print("Dimensões da máscara:", mascara.shape)
print("Corte utilizado:", z)
print("Voxels marcados na máscara:", np.count_nonzero(mascara))

# Usamos a mesma escala de intensidade para comparar
# visualmente as três fases.
valores = fases[np.isfinite(fases) & (fases != 0)]

if valores.size:
    minimo, maximo = np.percentile(valores, [1, 99])
else:
    minimo, maximo = 0, 1

if maximo <= minimo:
    maximo = minimo + 1

figura, eixos = plt.subplots(1, 4, figsize=(16, 5))

nomes = [
    "Pré-contraste",
    "Pós-contraste precoce",
    "Pós-contraste tardio"
]

for i in range(3):
    eixos[i].imshow(
        fases[i],
        cmap="gray",
        vmin=minimo,
        vmax=maximo
    )

    eixos[i].set_title(nomes[i])
    eixos[i].axis("off")

# Mostramos a máscara sobre a fase pós-precoce.
eixos[3].imshow(
    fases[1],
    cmap="gray",
    vmin=minimo,
    vmax=maximo
)

mascara_visivel = np.ma.masked_where(
    mascara == 0,
    mascara
)

eixos[3].imshow(
    mascara_visivel,
    cmap="autumn",
    alpha=0.5,
    vmin=0,
    vmax=1
)

eixos[3].set_title("Máscara sobreposta")
eixos[3].axis("off")

figura.suptitle(
    f"Paciente {pid} — corte {z}",
    fontsize=14
)

plt.tight_layout()

destino = Path("resultados/graficos")
destino.mkdir(parents=True, exist_ok=True)

plt.savefig(
    destino / f"verificacao_{pid}.png",
    dpi=160
)

plt.show()

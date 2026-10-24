
import ast
from pathlib import Path

arquivo = Path(__file__).with_name(
    "PAI_DCE_MRI_Visualizador.py"
)

if not arquivo.is_file():
    raise FileNotFoundError(arquivo)

texto = arquivo.read_text(encoding="utf-8-sig")

inicio = "\ndef abrir_painel_ia(self):"
fim = "\n    def _construir_menu(self):"

pos_inicio = texto.find(inicio)
pos_fim = texto.find(fim, pos_inicio + len(inicio))

if pos_inicio == -1 or pos_fim == -1:
    raise RuntimeError(
        "Não encontrei o trecho esperado. "
        "O arquivo não foi alterado. "
        "Confira se as funções já foram corrigidas."
    )

# Seleciona somente as duas funções mal indentadas.
bloco = texto[pos_inicio + 1:pos_fim]

# Acrescenta quatro espaços a cada linha não vazia.
bloco_corrigido = "".join(
    "    " + linha if linha.strip() else linha
    for linha in bloco.splitlines(keepends=True)
)

novo_texto = (
    texto[:pos_inicio + 1]
    + bloco_corrigido
    + texto[pos_fim:]
)

# Verifica sintaxe e estrutura antes de modificar o arquivo.
arvore = ast.parse(novo_texto, filename=str(arquivo))

classes = [
    item for item in arvore.body
    if isinstance(item, ast.ClassDef)
    and item.name == "Aplicacao"
]

if len(classes) != 1:
    raise RuntimeError(
        "Classe Aplicacao não encontrada corretamente."
    )

metodos = {
    item.name
    for item in classes[0].body
    if isinstance(item, ast.FunctionDef)
}

obrigatorios = {
    "__init__",
    "_estilo",
    "abrir_painel_ia",
    "carregar_paciente_validacao",
    "_construir_menu",
    "_construir_interface",
    "abrir_volume",
    "fechar_volumes",
    "_executar_em_thread"
}

faltando = obrigatorios - metodos

if faltando:
    raise RuntimeError(
        "Métodos ausentes da classe Aplicacao: "
        + ", ".join(sorted(faltando))
    )

backup = arquivo.with_name(
    "PAI_DCE_MRI_Visualizador_ANTES_CORRECAO.py"
)

if backup.exists():
    raise FileExistsError(
        "Já existe um backup com esse nome: "
        + str(backup)
    )

backup.write_text(texto, encoding="utf-8")
arquivo.write_text(novo_texto, encoding="utf-8")

print("\nCORREÇÃO CONCLUÍDA COM SUCESSO!")
print("Arquivo corrigido:", arquivo)
print("Backup criado:", backup)

print("\nMÉTODOS VERIFICADOS:")
for nome in sorted(obrigatorios):
    print(" OK:", nome)

print("\nSintaxe e estrutura da classe validadas.")

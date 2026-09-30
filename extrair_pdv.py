"""
extrair_pdv.py
==============
Projeto Integrador V-A — Big Data e Inteligência Artificial (CEAD / PUC Goiás)
Organização parceira: Pé Fresco Calçados

Pipeline de extração e tratamento dos relatórios reais do PDV da loja
("Relatório de Controle de Venda", um PDF por mês, fornecidos pela proprietária).

Entrada  : data/raw/*.pdf            (exportações mensais do PDV)
Saídas   : data/raw/pe_fresco_pdv_bruto.csv         -> linhas extraídas, sem tratamento
           data/processed/pe_fresco_vendas_limpo.csv -> base analítica do dashboard

Etapas:
  E1. Leitura posicional do PDF (colunas por coordenada x) e junção das linhas
      de descrição que quebram em duas.
  E2. Validação: soma extraída == "Total Geral" impresso no rodapé do relatório.
  T1. Descarte de exportações incompletas (sem Total Geral).
  T2. Anonimização dos vendedores (dado pessoal) -> "Vendedor 01..NN".
  T3. Derivação de atributos a partir da descrição: tipo, categoria, calçado?,
      gênero, marca, numeração.
  T4. Métricas: custo_total, lucro_bruto, margem_pct, ticket (valor por item).
  T5. Sinalização de inconsistências (estoque negativo, custo zerado).

Execução (a partir da raiz do projeto):
    python src/extrair_pdv.py
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    sys.exit("Instale a dependência: pip install pdfplumber")

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_RAW = os.path.join(RAIZ, "data", "raw")
DIR_PROC = os.path.join(RAIZ, "data", "processed")
ARQ_BRUTO = os.path.join(DIR_RAW, "pe_fresco_pdv_bruto.csv")
ARQ_LIMPO = os.path.join(DIR_PROC, "pe_fresco_vendas_limpo.csv")

# Colunas numéricas do relatório: (nome, x_min, x_max) da borda direita da palavra
COLUNAS = [
    ("qtde_estoque", 430, 470), ("qtde_vendida", 470, 500), ("custo_medio", 500, 540),
    ("valor_vitrine", 540, 580), ("pct_desconto", 580, 615), ("valor_desconto", 615, 660),
    ("valor_troca", 660, 730), ("valor_venda", 730, 780), ("valor_vendido", 780, 830),
]

# --------------------------------------------------------------------------- #
# Dicionários de negócio (validados com a proprietária)
# --------------------------------------------------------------------------- #
CATEGORIA_POR_TIPO = {
    # calçados
    "SANDALIA": "Sandálias", "SANDALIAS": "Sandálias", "SANADALIA": "Sandálias", "PAPETE": "Sandálias", "PAPATE": "Sandálias", "ANABELA": "Sandálias", "RASTEIRA": "Sandálias", "RASTEIRINHA": "Sandálias",
    "TAMANCO": "Tamancos e Sapatilhas", "SAPATILHA": "Tamancos e Sapatilhas", "MULE": "Tamancos e Sapatilhas",
    "TENIS": "Tênis", "SAPATENIS": "Tênis", "CHUTEIRA": "Chuteiras",
    "CHINELO": "Chinelos", "CHINELA": "Chinelos", "BABUCHE": "Chinelos", "SLIDE": "Chinelos",
    "BOTA": "Botas e Sapatos", "BOTINA": "Botas e Sapatos", "BOTINAS": "Botas e Sapatos", "SCARPIN": "Botas e Sapatos", "SAPATO": "Botas e Sapatos", "MOCASSIM": "Botas e Sapatos", "COTURNO": "Botas e Sapatos",
    # não calçados
    "MEIA": "Meias", "MEIAO": "Meias",
    "TOALHA": "Cama, Mesa e Banho", "LENCOL": "Cama, Mesa e Banho", "COLCHA": "Cama, Mesa e Banho",
    "JOGO": "Cama, Mesa e Banho", "TRAVESSEIRO": "Cama, Mesa e Banho", "ROUPAO": "Cama, Mesa e Banho",
    "EDREDOM": "Cama, Mesa e Banho", "COBERTOR": "Cama, Mesa e Banho", "FRONHA": "Cama, Mesa e Banho",
    "PROTETOR": "Cama, Mesa e Banho", "MANTA": "Cama, Mesa e Banho", "TAPETE": "Cama, Mesa e Banho",
    "VARAO": "Cama, Mesa e Banho", "CORTINA": "Cama, Mesa e Banho",
    "CAMISETA": "Vestuário", "CAMISA": "Vestuário", "CAMI": "Vestuário", "BERMUDA": "Vestuário", "CALCA": "Vestuário", "CALÇA": "Vestuário", "CALCAO": "Vestuário", "TOP": "Vestuário", "BODY": "Vestuário", "JAQUETA": "Vestuário",
    "SHORT": "Vestuário", "REGATA": "Vestuário", "BLUSA": "Vestuário", "CUECA": "Vestuário",
    "BOLA": "Acessórios", "BOTOM": "Acessórios", "COLAR": "Acessórios", "CAPA": "Acessórios", "BOLSA": "Acessórios", "CARTEIRA": "Acessórios", "CINTO": "Acessórios", "RELOGIO": "Acessórios",
    "OCULOS": "Acessórios", "MOCHILA": "Acessórios", "BONE": "Acessórios", "NECESSAIRE": "Acessórios",
    "CADARCO": "Acessórios", "PALMILHA": "Acessórios",
    "ODORIZADOR": "Casa e Perfumaria", "DIFUSOR": "Casa e Perfumaria", "AROMATIZADOR": "Casa e Perfumaria",
    "HOME": "Casa e Perfumaria", "VELA": "Casa e Perfumaria", "SABONETE": "Casa e Perfumaria",
}
CATEGORIAS_CALCADO = {"Sandálias", "Tamancos e Sapatilhas", "Tênis", "Chuteiras", "Chinelos", "Botas e Sapatos"}

GENERO_TOKENS = {
    "FEMININO": "Feminino", "FEMININA": "Feminino", "FEM": "Feminino",
    "MASCULINO": "Masculino", "MASCULIN": "Masculino", "MASC": "Masculino",
    "INFANTIL": "Infantil", "INF": "Infantil", "KIDS": "Infantil", "BABY": "Infantil", "BEBE": "Infantil",
    "MENINA": "Infantil", "MENINO": "Infantil", "JUVENIL": "Infantil",
    "ADULTO": "Unissex/Adulto", "UNISSEX": "Unissex/Adulto", "UNICO": "Unissex/Adulto",
}


def _num(s: str):
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# E1/E2 — Extração e validação
# --------------------------------------------------------------------------- #
def extrair_pdf(caminho: str) -> tuple[pd.DataFrame, dict]:
    """Lê um relatório mensal e devolve (linhas, metadados com Total Geral)."""
    linhas, meta = [], {"arquivo": os.path.basename(caminho), "periodo": None, "total_geral": None, "paginas": 0}
    filial = usuario = None
    ultimo = None
    with pdfplumber.open(caminho) as pdf:
        meta["paginas"] = len(pdf.pages)
        for pno, pg in enumerate(pdf.pages, 1):
            por_linha: dict[int, list] = {}
            for w in pg.extract_words():
                por_linha.setdefault(round(w["top"] / 3), []).append(w)
            for chave in sorted(por_linha):
                ws = sorted(por_linha[chave], key=lambda w: w["x0"])
                texto = " ".join(w["text"] for w in ws)
                if texto.startswith("Período:"):
                    m = re.search(r"(\d{2}/\d{2}/\d{4}) a (\d{2}/\d{2}/\d{4})", texto)
                    meta["periodo"] = (m.group(1), m.group(2)) if m else None
                    continue
                if texto.startswith("Filial:"):
                    filial = texto.replace("Filial:", "").strip()
                    continue
                if texto.startswith("Usuário:") and "Página" not in texto:
                    usuario = texto.replace("Usuário:", "").strip()
                    continue
                if texto.startswith("Total Geral"):
                    nums = [_num(t) for t in texto.split()[2:]]
                    meta["total_geral"] = {"qtde": nums[1], "desconto": nums[2], "vitrine": nums[3], "vendido": nums[4]}
                    continue
                if texto.startswith(("Relatório", "Qtde.", "_", "Total")) or "Página:" in texto:
                    continue
                primeiro = ws[0]
                eh_item = (re.fullmatch(r"\d{3,6}", primeiro["text"]) and primeiro["x0"] < 58
                           and any(w["x1"] > 780 for w in ws))
                if eh_item:
                    desc = [w["text"] for w in ws if 58 <= w["x0"] < 428]
                    numericos = [w for w in ws if w["x0"] >= 428]
                    vals = {}
                    if len(numericos) == len(COLUNAS):
                        # caso padrão: 9 números alinhados à direita, na ordem das colunas
                        for (nome, _, _), w in zip(COLUNAS, numericos):
                            vals[nome] = _num(w["text"])
                    else:
                        # fallback: classifica cada número pela faixa de x da coluna
                        for nome, lo, hi in COLUNAS:
                            cand = [w["text"] for w in numericos if lo < w["x1"] <= hi]
                            vals[nome] = _num(cand[-1]) if cand else None
                    ultimo = {"pagina": pno, "filial": filial, "vendedor_nome": usuario,
                              "codigo": primeiro["text"], "descricao": " ".join(desc), **vals}
                    linhas.append(ultimo)
                elif ultimo is not None and all(58 <= w["x0"] < 430 for w in ws):
                    ultimo["descricao"] += " " + texto  # continuação da descrição
    df = pd.DataFrame(linhas)
    if meta["periodo"]:
        d, m, a = meta["periodo"][0].split("/")
        df.insert(0, "mes", f"{a}-{m}")
    df.insert(1, "arquivo", meta["arquivo"])
    return df, meta


def validar(df: pd.DataFrame, meta: dict) -> bool:
    """E2: confere a soma extraída com o Total Geral impresso no relatório."""
    tg = meta["total_geral"]
    if not tg:
        print(f"  [AVISO] {meta['arquivo']}: sem 'Total Geral' -> exportação incompleta ({meta['paginas']} páginas)")
        return False
    ok = (abs(df["valor_vendido"].sum() - tg["vendido"]) < 0.01
          and abs(df["valor_venda"].sum() - tg["vitrine"]) < 0.01
          and abs(df["valor_desconto"].sum() - tg["desconto"]) < 0.01)
    print(f"  {meta['arquivo']}: {len(df)} linhas | vendido R$ {df['valor_vendido'].sum():,.2f} "
          f"vs Total Geral R$ {tg['vendido']:,.2f} -> {'OK' if ok else 'DIVERGENTE'}")
    return ok


# --------------------------------------------------------------------------- #
# T2–T5 — Tratamento
# --------------------------------------------------------------------------- #
def _tamanho(desc: str):
    """Numeração: último token numérico (34..46) ou faixa (33/34, 37/40) da descrição."""
    tokens = desc.split()
    for t in reversed(tokens[-4:]):
        if re.fullmatch(r"\d{2}", t) and 16 <= int(t) <= 46:
            return t
        if re.fullmatch(r"\d{2}/\d{2}", t):
            return t
    return None


def _genero(desc: str) -> str:
    for t in desc.split():
        if t in GENERO_TOKENS:
            return GENERO_TOKENS[t]
    return "Não informado"


MARCA_NORMALIZA = {"Olimpikus": "Olympikus", "Olympicus": "Olympikus", "Via": "Via Marte/Vizzano",
                   "Picadilly": "Piccadilly", "Molekinho": "Moleca", "Molekinha": "Moleca",
                   "Sintentico": "Não informada", "Sintetico": "Não informada", "Borracha": "Não informada",
                   "Tecido": "Não informada", "Algodao": "Não informada", "Napa": "Não informada"}
_NAO_MARCA = {"COD", "REF", "COM", "SEM", "CANO", "SALTO", "BAIX", "BAIXO", "ALTO", "CASUAL", "ESPORTIVO",
              "CONFORTO", "CONFORTAVE", "SOCIETY", "INDOOR", "CAMPO", "DEDO", "SLIDE", "PAPETE", "BLOC",
              "SPORT", "ANA", "RASTEIRA", "PLATAFORMA", "TRATORADA"}


def _marca(desc: str) -> str:
    """Marca = token logo após o gênero (padrão do cadastro); sem gênero, primeiro
    token alfabético após o tipo que não seja descritor genérico."""
    tokens = desc.split()
    cand = None
    for i, t in enumerate(tokens):
        if t in GENERO_TOKENS and i + 1 < len(tokens):
            cand = tokens[i + 1]
            break
    if cand is None:
        for t in tokens[1:5]:
            if t.isalpha() and len(t) > 2 and t not in _NAO_MARCA and t not in GENERO_TOKENS:
                cand = t
                break
    if not cand or not cand.isalpha() or len(cand) <= 2 or cand in _NAO_MARCA:
        return "Não informada"
    m = cand.title()
    return MARCA_NORMALIZA.get(m, m)


def tratar(df_bruto: pd.DataFrame, meses_validos: list[str]) -> pd.DataFrame:
    df = df_bruto.copy()

    # T1 — descarta exportações incompletas
    df["mes_completo"] = df["mes"].isin(meses_validos)
    df = df[df["mes_completo"]].drop(columns=["mes_completo"]).copy()

    # T2 — anonimização dos vendedores (ordem alfabética estável)
    nomes = sorted(df["vendedor_nome"].dropna().unique())
    mapa = {n: f"Vendedor {i + 1:02d}" for i, n in enumerate(nomes)}
    df["vendedor"] = df["vendedor_nome"].map(mapa)
    df = df.drop(columns=["vendedor_nome"])
    df["filial"] = df["filial"].str.replace("PE FRESCO - ", "", regex=False).str.strip()

    # T3 — atributos derivados da descrição
    df["tipo"] = df["descricao"].str.split().str[0].str.upper()
    df["categoria"] = df["tipo"].map(CATEGORIA_POR_TIPO).fillna("Outros")
    df["eh_calcado"] = df["categoria"].isin(CATEGORIAS_CALCADO)
    df["genero"] = df["descricao"].map(_genero)
    df["marca"] = df["descricao"].map(_marca)
    df["tamanho"] = df["descricao"].map(_tamanho)
    df.loc[~df["eh_calcado"], "tamanho"] = None

    # T4 — métricas
    # Quantidade fracionada/absurda no PDV (ex.: 119,9 unidades para uma venda de
    # R$ 116): corrige para o número de unidades compatível com o valor recebido
    # e sinaliza a linha. É o único valor alterado em relação ao relatório.
    qtde_bruta = df["qtde_vendida"].astype(float)
    inconsistente = (qtde_bruta * df["valor_vitrine"] > 3 * df["valor_vendido"]) & (df["valor_vendido"] > 0)
    df["alerta_qtde_corrigida"] = inconsistente
    df.loc[inconsistente, "qtde_vendida"] = (df.loc[inconsistente, "valor_vendido"] / df.loc[inconsistente, "valor_vitrine"]).round().clip(lower=1)
    df["qtde_vendida"] = df["qtde_vendida"].round(0).astype(int)
    df["faturamento"] = df["valor_vendido"].round(2)
    df["custo_total"] = (df["custo_medio"] * df["qtde_vendida"]).round(2)
    df["lucro_bruto"] = (df["faturamento"] - df["custo_total"]).round(2)
    df["margem_pct"] = (df["lucro_bruto"] / df["faturamento"].where(df["faturamento"] != 0) * 100).round(2)
    df["preco_medio_item"] = (df["faturamento"] / df["qtde_vendida"].where(df["qtde_vendida"] != 0)).round(2)

    # T5 — sinalização de inconsistências (não altera os valores)
    df["alerta_estoque_negativo"] = df["qtde_estoque"] < 0
    df["alerta_custo_zerado"] = df["custo_medio"].fillna(0) <= 0
    # Venda atípica: valor recebido inferior a 30% do preço de vitrine sem desconto
    # registrado (trocas, brindes, ajustes de caixa). Mantida no faturamento;
    # excluída das análises de margem e desconto.
    # Cadastro genérico: SKU "curinga" com preço de vitrine 10.000 e custo 1,00
    # (chinelos de praia vendidos a preço variável). Faturamento real; custo e
    # preço de vitrine não confiáveis -> fora das análises de margem e desconto.
    df["cadastro_generico"] = (df["valor_vitrine"] >= 9999) | (df["custo_medio"].fillna(0) <= 1.0)
    df["venda_atipica"] = ((df["preco_medio_item"] < 0.30 * df["valor_vitrine"])
                           & (df["valor_desconto"].fillna(0) == 0) & ~df["cadastro_generico"])
    # Base confiável para margem/desconto
    df["margem_confiavel"] = ~(df["venda_atipica"] | df["cadastro_generico"])

    colunas = ["mes", "filial", "vendedor", "codigo", "descricao", "tipo", "categoria", "eh_calcado",
               "genero", "marca", "tamanho", "qtde_vendida", "qtde_estoque", "custo_medio",
               "valor_vitrine", "pct_desconto", "valor_desconto", "valor_troca", "faturamento",
               "custo_total", "lucro_bruto", "margem_pct", "preco_medio_item",
               "alerta_estoque_negativo", "alerta_custo_zerado", "alerta_qtde_corrigida", "venda_atipica", "cadastro_generico", "margem_confiavel", "arquivo", "pagina"]
    return df[colunas].sort_values(["mes", "filial", "vendedor", "codigo"]).reset_index(drop=True)


def resumo(df: pd.DataFrame) -> None:
    L = 70
    print("=" * L)
    print(" PÉ FRESCO CALÇADOS — BASE ANALÍTICA (PDV REAL) ".center(L, "="))
    print("=" * L)
    print(f"Meses analisados ............: {', '.join(sorted(df['mes'].unique()))}")
    print(f"Linhas (produto x mês x vend): {len(df):>8}")
    print(f"Itens vendidos ..............: {df['qtde_vendida'].sum():>8}")
    print(f"Faturamento .................: R$ {df['faturamento'].sum():>12,.2f}")
    print(f"Lucro bruto .................: R$ {df['lucro_bruto'].sum():>12,.2f}")
    print(f"Margem bruta (ponderada) ....: {df['lucro_bruto'].sum() / df['faturamento'].sum() * 100:>11.2f} %")
    print(f"Desconto concedido ..........: R$ {df['valor_desconto'].sum():>12,.2f} "
          f"({df['valor_desconto'].sum() / (df['faturamento'].sum() + df['valor_desconto'].sum()) * 100:.2f}% do preço de vitrine)")
    print("-" * L)
    print(df.groupby("categoria")["faturamento"].sum().sort_values(ascending=False).round(2).to_string())
    print("-" * L)
    print(f"Linhas com estoque negativo .: {int(df['alerta_estoque_negativo'].sum())} ({df['alerta_estoque_negativo'].mean() * 100:.1f}%)")
    print(f"Linhas com custo zerado .....: {int(df['alerta_custo_zerado'].sum())}")
    print(f"Quantidades corrigidas ......: {int(df['alerta_qtde_corrigida'].sum())}")
    print(f"Vendas atípicas (<30% vitr.) : {int(df['venda_atipica'].sum())} linhas, R$ {df.loc[df['venda_atipica'], 'faturamento'].sum():,.2f}")
    print(f"Cadastro genérico (curinga) .: {int(df['cadastro_generico'].sum())} linhas, R$ {df.loc[df['cadastro_generico'], 'faturamento'].sum():,.2f}")
    c = df[df["margem_confiavel"]]
    print(f"Margem bruta (base confiável): {c['lucro_bruto'].sum() / c['faturamento'].sum() * 100:>11.2f} %  ({len(c)} linhas)")
    print("=" * L)


def main() -> None:
    os.makedirs(DIR_PROC, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(DIR_RAW, "*.pdf")))
    if not pdfs:
        sys.exit(f"Nenhum PDF em {DIR_RAW}")
    print("Extraindo relatórios do PDV:")
    partes, validos = [], []
    for p in pdfs:
        df, meta = extrair_pdf(p)
        if validar(df, meta):
            validos.append(df["mes"].iloc[0])
        partes.append(df)
    bruto = pd.concat(partes, ignore_index=True)
    bruto.to_csv(ARQ_BRUTO, index=False, encoding="utf-8")

    limpo = tratar(bruto, validos)
    limpo.to_csv(ARQ_LIMPO, index=False, encoding="utf-8")
    resumo(limpo)
    print(f"Base bruta ..: {ARQ_BRUTO}\nBase tratada : {ARQ_LIMPO}")


if __name__ == "__main__":
    main()

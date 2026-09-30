"""
gerador_dados.py
================
Projeto Integrador V-A — Big Data e Inteligência Artificial (CEAD / PUC Goiás)
Organização parceira: Pé Fresco Calçados

Gera uma base sintética, porém realista, de ~2.000 transações de venda de
calçados, cobrindo 12 meses, e salva:

    data/raw/pe_fresco_vendas.csv           -> base bruta (com ruído proposital)
    data/processed/pe_fresco_vendas_limpo.csv -> base tratada, pronta para o dashboard

A base bruta contém "sujeiras" típicas de um sistema de PDV de pequeno varejo
(espaços extras, variação de caixa em categorias, alguns registros duplicados,
valores nulos de desconto), para que a etapa de tratamento seja demonstrável
no relatório técnico. O script é totalmente determinístico (seed = 42).

Execução (a partir da raiz do projeto):
    python src/gerador_dados.py
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configurações globais
# --------------------------------------------------------------------------- #
SEED = 42
N_TRANSACOES = 2000
DATA_INICIO = datetime(2025, 9, 1)   # período de 1 ano: set/2025 a ago/2026
DATA_FIM = datetime(2026, 8, 31)

# Caminhos relativos à raiz do projeto (o script pode ser chamado de qualquer pasta)
RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_RAW = os.path.join(RAIZ_PROJETO, "data", "raw")
DIR_PROCESSED = os.path.join(RAIZ_PROJETO, "data", "processed")
ARQ_RAW = os.path.join(DIR_RAW, "pe_fresco_vendas.csv")
ARQ_PROCESSED = os.path.join(DIR_PROCESSED, "pe_fresco_vendas_limpo.csv")

random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- #
# Parâmetros de negócio (calibrados para um varejo de calçados popular em Goiás)
# --------------------------------------------------------------------------- #
CATEGORIAS = ["Rasteirinhas", "Sandálias", "Tênis Casual", "Botas", "Chinelos"]
# Participação esperada de cada categoria no volume de vendas (soma = 1)
PESO_CATEGORIA = [0.24, 0.22, 0.26, 0.10, 0.18]

# Faixa de preço de venda (mínimo, máximo) e razão custo/preço por categoria.
# Botas têm margem apertada (custo alto e giro lento); chinelos têm margem
# alta em percentual, mas ticket baixo.
PARAM_CATEGORIA = {
    #  categoria      preço_min  preço_max  custo/preço (média, desvio)
    "Rasteirinhas":   (49.90,    139.90,    (0.52, 0.05)),
    "Sandálias":      (69.90,    199.90,    (0.55, 0.05)),
    "Tênis Casual":   (119.90,   349.90,    (0.58, 0.06)),
    "Botas":          (179.90,   449.90,    (0.64, 0.06)),
    "Chinelos":       (29.90,    89.90,     (0.45, 0.05)),
}

GENEROS = ["Feminino", "Masculino", "Infantil"]
# Probabilidade de gênero condicionada à categoria (loja com público majoritariamente feminino)
PESO_GENERO_POR_CATEGORIA = {
    "Rasteirinhas": [0.90, 0.00, 0.10],
    "Sandálias":    [0.72, 0.13, 0.15],
    "Tênis Casual": [0.48, 0.40, 0.12],
    "Botas":        [0.60, 0.35, 0.05],
    "Chinelos":     [0.45, 0.35, 0.20],
}

CANAIS = ["Loja Física", "WhatsApp", "Instagram"]
PESO_CANAL = [0.62, 0.25, 0.13]

# Grade de numeração: 34 a 44. Curva concentrada entre 36 e 39 (adulto),
# deslocada para baixo no infantil e para cima no masculino.
TAMANHOS = list(range(34, 45))


def _pesos_tamanho(genero: str) -> np.ndarray:
    """Retorna a distribuição de probabilidade da grade para um gênero."""
    if genero == "Feminino":
        centro, desvio = 37.0, 1.4
    elif genero == "Masculino":
        centro, desvio = 40.5, 1.6
    else:  # Infantil (a loja trabalha infantil a partir do 34, "juvenil")
        centro, desvio = 35.0, 1.2
    pesos = np.exp(-0.5 * ((np.array(TAMANHOS) - centro) / desvio) ** 2)
    return pesos / pesos.sum()


# --------------------------------------------------------------------------- #
# Sazonalidade mensal (multiplicador de volume). Picos em dezembro (Natal),
# maio (Dia das Mães) e novembro (Black Friday); vale em fevereiro/março.
# --------------------------------------------------------------------------- #
SAZONALIDADE = {
    1: 0.85, 2: 0.70, 3: 0.78, 4: 0.90, 5: 1.20, 6: 0.95,
    7: 0.92, 8: 0.98, 9: 0.95, 10: 1.05, 11: 1.25, 12: 1.55,
}


def _gerar_datas(n: int) -> list[datetime]:
    """Sorteia datas ponderadas pela sazonalidade mensal e pelo dia da semana."""
    total_dias = (DATA_FIM - DATA_INICIO).days + 1
    dias = [DATA_INICIO + timedelta(days=i) for i in range(total_dias)]
    pesos = []
    for d in dias:
        p = SAZONALIDADE[d.month]
        # Sábado vende mais; domingo a loja física fecha (só canais digitais)
        if d.weekday() == 5:
            p *= 1.45
        elif d.weekday() == 6:
            p *= 0.35
        pesos.append(p)
    pesos = np.array(pesos) / np.sum(pesos)
    idx = rng.choice(len(dias), size=n, p=pesos)
    return [dias[i] for i in idx]


def _politica_desconto(categoria: str, data: datetime, canal: str) -> float:
    """
    Reproduz a política (informal) de descontos da loja:
      - Sem desconto na maioria das vendas.
      - Descontos maiores em Botas fora da estação (queima de grade).
      - Black Friday (novembro) e liquidação de janeiro/fevereiro.
      - WhatsApp costuma negociar 5% a 10%.
    Retorna o desconto em percentual (0 a 40).
    """
    base = 0.0
    sorteio = rng.random()

    if data.month == 11 and data.day >= 20:
        base = rng.choice([10, 15, 20, 25, 30], p=[0.15, 0.25, 0.30, 0.20, 0.10])
    elif data.month in (1, 2):
        base = rng.choice([0, 10, 15, 20, 30], p=[0.35, 0.20, 0.20, 0.15, 0.10])
    elif categoria == "Botas" and data.month in (9, 10, 11, 12, 1, 2, 3):
        base = rng.choice([0, 15, 20, 30, 40], p=[0.30, 0.20, 0.20, 0.20, 0.10])
    elif sorteio < 0.62:
        base = 0.0
    else:
        base = rng.choice([5, 10, 15, 20], p=[0.45, 0.35, 0.15, 0.05])

    if canal == "WhatsApp" and base == 0 and rng.random() < 0.40:
        base = rng.choice([5, 10])

    return float(base)


def gerar_base_bruta(n: int = N_TRANSACOES) -> pd.DataFrame:
    """Gera o DataFrame bruto de transações."""
    registros = []
    datas = _gerar_datas(n)

    for i in range(n):
        data_venda = datas[i]
        categoria = rng.choice(CATEGORIAS, p=PESO_CATEGORIA)
        genero = rng.choice(GENEROS, p=PESO_GENERO_POR_CATEGORIA[categoria])
        tamanho = int(rng.choice(TAMANHOS, p=_pesos_tamanho(genero)))
        canal = rng.choice(CANAIS, p=PESO_CANAL)

        # Domingo: loja física fechada -> venda migra para canal digital
        if data_venda.weekday() == 6 and canal == "Loja Física":
            canal = rng.choice(["WhatsApp", "Instagram"], p=[0.65, 0.35])

        preco_min, preco_max, (razao_media, razao_desvio) = PARAM_CATEGORIA[categoria]
        # Preço arredondado para terminar em ,90 (prática comum no varejo)
        preco_unitario = round(rng.uniform(preco_min, preco_max), 0) - 0.10
        razao_custo = float(np.clip(rng.normal(razao_media, razao_desvio), 0.35, 0.85))
        custo_unitario = round(preco_unitario * razao_custo, 2)

        desconto = _politica_desconto(categoria, data_venda, canal)

        registros.append(
            {
                "id_venda": f"PF-{i + 1:05d}",
                "data_venda": data_venda.strftime("%Y-%m-%d"),
                "categoria": categoria,
                "genero": genero,
                "tamanho": tamanho,
                "preco_unitario": preco_unitario,
                "desconto_percentual": desconto,
                "custo_unitario": custo_unitario,
                "canal_venda": canal,
            }
        )

    df = pd.DataFrame(registros)
    return df


def injetar_ruido(df: pd.DataFrame) -> pd.DataFrame:
    """
    Injeta imperfeições típicas de exportação de PDV, para que a limpeza
    seja demonstrada no relatório:
      1. ~1,5% de linhas duplicadas.
      2. ~3% de categorias com caixa/espaços inconsistentes ("tênis casual ", "BOTAS").
      3. ~2% de desconto nulo (vazio no sistema = sem desconto).
      4. ~0,5% de custo unitário negativo (erro de digitação).
    """
    df = df.copy()
    n = len(df)

    # 1. Duplicatas
    idx_dup = rng.choice(n, size=int(n * 0.015), replace=False)
    df = pd.concat([df, df.iloc[idx_dup]], ignore_index=True)

    # 2. Inconsistência de texto na categoria
    idx_txt = rng.choice(len(df), size=int(len(df) * 0.03), replace=False)
    for j, i in enumerate(idx_txt):
        valor = df.at[i, "categoria"]
        if j % 3 == 0:
            df.at[i, "categoria"] = valor.upper()
        elif j % 3 == 1:
            df.at[i, "categoria"] = f" {valor.lower()} "
        else:
            df.at[i, "categoria"] = f"{valor}  "

    # 3. Desconto nulo
    idx_nulo = rng.choice(len(df), size=int(len(df) * 0.02), replace=False)
    df.loc[idx_nulo, "desconto_percentual"] = np.nan

    # 4. Custo negativo (erro de digitação)
    idx_neg = rng.choice(len(df), size=int(len(df) * 0.005), replace=False)
    df.loc[idx_neg, "custo_unitario"] = -df.loc[idx_neg, "custo_unitario"]

    # Embaralha as linhas para parecer uma exportação real
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def tratar_dados(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de tratamento (documentado no relatório técnico, seção 4):
      T2. Padronização textual da categoria (strip + lower + mapa canônico).
      T1. Remoção de duplicatas pela chave id_venda (mantém a 1ª ocorrência).
      T3. Desconto nulo -> 0 (regra de negócio confirmada com a gerência).
      T4. Custo negativo -> valor absoluto (erro de digitação confirmado).
      T5. Conversão de tipos (data, numéricos).
      T6. Cálculo das métricas derivadas:
            faturamento_liquido = preco_unitario * (1 - desconto/100)
            lucro_bruto         = faturamento_liquido - custo_unitario
            margem_lucro_pct    = lucro_bruto / faturamento_liquido * 100
      T7. Colunas auxiliares de tempo (ano_mes, mes, dia_semana).
      T8. Ordenação por data e id.
    """
    df = df_raw.copy()

    # T2 (executado antes de T1 para que duplicatas com grafia diferente
    #     da categoria também sejam reconhecidas)
    mapa_categoria = {c.lower(): c for c in CATEGORIAS}
    df["categoria"] = (
        df["categoria"].astype(str).str.strip().str.lower().map(mapa_categoria)
    )
    assert df["categoria"].notna().all(), "Categoria fora do domínio esperado."

    # T1 — id_venda é a chave primária do PDV: um id, uma venda
    df = df.drop_duplicates(subset=["id_venda"], keep="first")

    # T3
    df["desconto_percentual"] = df["desconto_percentual"].fillna(0.0)

    # T4
    df["custo_unitario"] = df["custo_unitario"].abs()

    # T5
    df["data_venda"] = pd.to_datetime(df["data_venda"])
    df["tamanho"] = df["tamanho"].astype(int)
    for col in ["preco_unitario", "desconto_percentual", "custo_unitario"]:
        df[col] = df[col].astype(float)

    # T6
    df["faturamento_liquido"] = (
        df["preco_unitario"] * (1 - df["desconto_percentual"] / 100)
    ).round(2)
    df["lucro_bruto"] = (df["faturamento_liquido"] - df["custo_unitario"]).round(2)
    df["margem_lucro_pct"] = (
        df["lucro_bruto"] / df["faturamento_liquido"] * 100
    ).round(2)

    # T7
    df["ano_mes"] = df["data_venda"].dt.to_period("M").astype(str)
    df["mes"] = df["data_venda"].dt.month
    nomes_dia = {
        0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta",
        4: "Sexta", 5: "Sábado", 6: "Domingo",
    }
    df["dia_semana"] = df["data_venda"].dt.weekday.map(nomes_dia)

    # T8
    colunas = [
        "id_venda", "data_venda", "ano_mes", "mes", "dia_semana",
        "categoria", "genero", "tamanho", "canal_venda",
        "preco_unitario", "desconto_percentual", "faturamento_liquido",
        "custo_unitario", "lucro_bruto", "margem_lucro_pct",
    ]
    df = df[colunas].sort_values(["data_venda", "id_venda"]).reset_index(drop=True)
    return df


def imprimir_resumo(df_raw: pd.DataFrame, df_limpo: pd.DataFrame) -> None:
    """Imprime um resumo de auditoria no terminal."""
    largura = 66
    print("=" * largura)
    print(" PÉ FRESCO CALÇADOS — GERAÇÃO DA BASE SINTÉTICA ".center(largura, "="))
    print("=" * largura)
    print(f"Linhas na base bruta ........: {len(df_raw):>6}")
    print(f"Linhas na base tratada ......: {len(df_limpo):>6}")
    print(f"Duplicatas removidas ........: {len(df_raw) - len(df_limpo):>6}")
    print(f"Período ......................: {df_limpo['data_venda'].min().date()} a "
          f"{df_limpo['data_venda'].max().date()}")
    print("-" * largura)
    fat = df_limpo["faturamento_liquido"].sum()
    lucro = df_limpo["lucro_bruto"].sum()
    print(f"Faturamento líquido total ...: R$ {fat:>12,.2f}")
    print(f"Lucro bruto total ...........: R$ {lucro:>12,.2f}")
    print(f"Ticket médio ................: R$ {fat / len(df_limpo):>12,.2f}")
    print(f"Margem bruta média ..........: {df_limpo['margem_lucro_pct'].mean():>11.2f} %")
    print("-" * largura)
    print("Vendas por categoria:")
    print(df_limpo["categoria"].value_counts().to_string())
    print("-" * largura)
    print("Vendas por canal:")
    print(df_limpo["canal_venda"].value_counts().to_string())
    print("=" * largura)
    print(f"Arquivo bruto ....: {ARQ_RAW}")
    print(f"Arquivo tratado ..: {ARQ_PROCESSED}")
    print("=" * largura)


def main() -> None:
    os.makedirs(DIR_RAW, exist_ok=True)
    os.makedirs(DIR_PROCESSED, exist_ok=True)

    df_bruto = injetar_ruido(gerar_base_bruta())
    df_bruto.to_csv(ARQ_RAW, index=False, encoding="utf-8")

    df_limpo = tratar_dados(df_bruto)
    df_limpo.to_csv(ARQ_PROCESSED, index=False, encoding="utf-8")

    imprimir_resumo(df_bruto, df_limpo)


if __name__ == "__main__":
    main()

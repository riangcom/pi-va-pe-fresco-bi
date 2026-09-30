"""
app.py
======
Dashboard analítico — Pé Fresco Calçados
Projeto Integrador V-A — Big Data e Inteligência Artificial (CEAD / PUC Goiás)

Princípios de design aplicados (Cole Nussbaumer Knaflic, "Storytelling com Dados"):
  * Um destaque por gráfico: cor de ênfase (laranja terracota) apenas no que
    importa; todo o resto em cinza neutro.
  * Redução de ruído: sem gridlines pesadas, sem bordas, sem legendas
    redundantes, eixos limpos, rótulos diretos nas barras.
  * Títulos que afirmam o achado ("Botas têm a menor margem"), não apenas o
    nome da métrica.
  * Ordenação intencional (barras ordenadas por valor, grade em ordem numérica).

Execução (a partir da raiz do projeto):
    streamlit run src/app.py
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuração da página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Pé Fresco Calçados — BI",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Paleta e tema visual (poucas cores, um único destaque)
# --------------------------------------------------------------------------- #
COR_DESTAQUE = "#D9622B"      # terracota — usada só para o que queremos enfatizar
COR_DESTAQUE_2 = "#1F5F8B"    # azul petróleo — segunda série quando indispensável
COR_NEUTRA = "#B8B8B8"        # cinza para contexto
COR_NEUTRA_ESCURA = "#6E6E6E"
COR_TEXTO = "#333333"
COR_FUNDO = "#FFFFFF"

TEMPLATE_PLOTLY = dict(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
                  size=13, color=COR_TEXTO),
        paper_bgcolor=COR_FUNDO,
        plot_bgcolor=COR_FUNDO,
        margin=dict(l=20, r=20, t=60, b=40),
        title=dict(font=dict(size=16, color=COR_TEXTO), x=0.0, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#DDDDDD"),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False,
                   linecolor="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                    x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        h1 {font-weight: 700; letter-spacing: -0.5px;}
        h3 {font-weight: 600; margin-top: 1.2rem;}
        div[data-testid="stMetric"] {
            background: #FAFAFA;
            border: 1px solid #EEEEEE;
            border-radius: 10px;
            padding: 14px 18px;
        }
        div[data-testid="stMetricLabel"] p {color: #6E6E6E; font-size: 0.85rem;}
        div[data-testid="stMetricValue"] {font-weight: 700; font-size: 1.65rem !important;}
        .nota {color:#6E6E6E; font-size:0.85rem; margin-top:-8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Carga de dados
# --------------------------------------------------------------------------- #
RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Procura a base tratada na estrutura do projeto e, como alternativa (deploy em
# nuvem com arquivos na mesma pasta), ao lado deste script.
_CANDIDATOS = [
    os.path.join(RAIZ_PROJETO, "data", "processed", "pe_fresco_vendas_limpo.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe_fresco_vendas_limpo.csv"),
]
ARQ_DADOS = next((c for c in _CANDIDATOS if os.path.exists(c)), _CANDIDATOS[0])


@st.cache_data(show_spinner=False)
def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["data_venda"])
    df["tamanho"] = df["tamanho"].astype(int)
    return df


@st.cache_data(show_spinner="Gerando a base de dados (seed 42)...")
def gerar_dados_em_memoria() -> pd.DataFrame:
    """Fallback: gera a base com o mesmo gerador determinístico, sem gravar em disco.
    Usado quando o CSV não está disponível (por exemplo, no deploy em nuvem)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gerador_dados as g
    df = g.tratar_dados(g.injetar_ruido(g.gerar_base_bruta()))
    df["tamanho"] = df["tamanho"].astype(int)
    return df


if os.path.exists(ARQ_DADOS):
    df_total = carregar_dados(ARQ_DADOS)
else:
    try:
        df_total = gerar_dados_em_memoria()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Base tratada não encontrada e não foi possível gerá-la em memória "
            f"({exc}). Execute `python src/gerador_dados.py` na raiz do projeto."
        )
        st.stop()


# --------------------------------------------------------------------------- #
# Funções utilitárias
# --------------------------------------------------------------------------- #
def brl(valor: float) -> str:
    """Formata número no padrão monetário brasileiro (R$ 1.234,56)."""
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(valor: float, casas: int = 1) -> str:
    return f"{valor:.{casas}f}".replace(".", ",") + "%"


def delta_pct(atual: float, anterior: float) -> str | None:
    if anterior in (0, None) or np.isnan(anterior):
        return None
    return f"{(atual / anterior - 1) * 100:+.1f}% vs. período anterior".replace(".", ",")


# --------------------------------------------------------------------------- #
# Barra lateral — filtros
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 👟 Pé Fresco Calçados")
    st.caption("Painel de Business Intelligence — Projeto Integrador V-A · PUC Goiás")
    st.markdown("---")
    st.markdown("### Filtros")

    data_min: date = df_total["data_venda"].min().date()
    data_max: date = df_total["data_venda"].max().date()
    periodo = st.date_input(
        "Período",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max,
        format="DD/MM/YYYY",
    )
    if isinstance(periodo, tuple) and len(periodo) == 2:
        data_ini, data_fim = periodo
    else:
        data_ini, data_fim = data_min, data_max

    categorias_opcoes = sorted(df_total["categoria"].unique())
    categorias_sel = st.multiselect("Categoria", categorias_opcoes, default=categorias_opcoes)

    generos_opcoes = sorted(df_total["genero"].unique())
    generos_sel = st.multiselect("Gênero", generos_opcoes, default=generos_opcoes)

    canais_opcoes = sorted(df_total["canal_venda"].unique())
    canais_sel = st.multiselect("Canal de venda", canais_opcoes, default=canais_opcoes)

    st.markdown("---")
    st.caption(
        "Fonte: base sintética gerada por `src/gerador_dados.py` (seed 42), "
        "calibrada com as regras de negócio levantadas junto à gerência da loja."
    )

# Aplicação dos filtros
mascara = (
    (df_total["data_venda"].dt.date >= data_ini)
    & (df_total["data_venda"].dt.date <= data_fim)
    & (df_total["categoria"].isin(categorias_sel))
    & (df_total["genero"].isin(generos_sel))
    & (df_total["canal_venda"].isin(canais_sel))
)
df = df_total.loc[mascara].copy()

# Período anterior de mesma duração (para o delta dos KPIs)
duracao = (pd.Timestamp(data_fim) - pd.Timestamp(data_ini)).days + 1
ini_ant = pd.Timestamp(data_ini) - pd.Timedelta(days=duracao)
fim_ant = pd.Timestamp(data_ini) - pd.Timedelta(days=1)
mascara_ant = (
    (df_total["data_venda"] >= ini_ant)
    & (df_total["data_venda"] <= fim_ant)
    & (df_total["categoria"].isin(categorias_sel))
    & (df_total["genero"].isin(generos_sel))
    & (df_total["canal_venda"].isin(canais_sel))
)
df_ant = df_total.loc[mascara_ant]

# --------------------------------------------------------------------------- #
# Cabeçalho
# --------------------------------------------------------------------------- #
st.title("Pé Fresco Calçados — Painel de Rentabilidade e Giro de Grade")
st.markdown(
    f"<p class='nota'>Período selecionado: <b>{data_ini.strftime('%d/%m/%Y')}</b> a "
    f"<b>{data_fim.strftime('%d/%m/%Y')}</b> · {len(df):,} transações"
    .replace(",", ".") + "</p>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("Nenhuma transação atende aos filtros selecionados. Amplie o período ou as categorias.")
    st.stop()

# --------------------------------------------------------------------------- #
# KPIs (Big Numbers)
# --------------------------------------------------------------------------- #
faturamento = df["faturamento_liquido"].sum()
lucro = df["lucro_bruto"].sum()
ticket = df["faturamento_liquido"].mean()
margem = df["margem_lucro_pct"].mean()

faturamento_ant = df_ant["faturamento_liquido"].sum() if not df_ant.empty else np.nan
lucro_ant = df_ant["lucro_bruto"].sum() if not df_ant.empty else np.nan
ticket_ant = df_ant["faturamento_liquido"].mean() if not df_ant.empty else np.nan
margem_ant = df_ant["margem_lucro_pct"].mean() if not df_ant.empty else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Faturamento Total", brl(faturamento), delta_pct(faturamento, faturamento_ant))
k2.metric("Lucro Bruto Total", brl(lucro), delta_pct(lucro, lucro_ant))
k3.metric("Ticket Médio", brl(ticket), delta_pct(ticket, ticket_ant))
k4.metric(
    "Margem Bruta Média",
    pct(margem),
    None if np.isnan(margem_ant) else f"{margem - margem_ant:+.1f} p.p. vs. período anterior".replace(".", ","),
)
st.caption(
    "Os deltas comparam com o período imediatamente anterior de mesma duração, "
    "quando há dados disponíveis."
)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Linha 1 — Evolução temporal | Faturamento por categoria
# --------------------------------------------------------------------------- #
c1, c2 = st.columns([3, 2])

with c1:
    # Granularidade adaptativa: mensal para períodos longos, semanal para curtos
    if duracao > 120:
        df["periodo"] = df["data_venda"].dt.to_period("M").dt.to_timestamp()
        rotulo_periodo = "mês"
    else:
        df["periodo"] = df["data_venda"].dt.to_period("W").dt.to_timestamp()
        rotulo_periodo = "semana"

    serie = (
        df.groupby("periodo", as_index=False)
        .agg(faturamento=("faturamento_liquido", "sum"), lucro=("lucro_bruto", "sum"))
        .sort_values("periodo")
    )
    pico = serie.loc[serie["faturamento"].idxmax()]

    fig_tempo = go.Figure()
    fig_tempo.add_trace(
        go.Scatter(
            x=serie["periodo"], y=serie["faturamento"], name="Faturamento líquido",
            mode="lines+markers", line=dict(color=COR_DESTAQUE, width=3),
            marker=dict(size=7),
            hovertemplate="%{x|%b/%Y}<br>Faturamento: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig_tempo.add_trace(
        go.Scatter(
            x=serie["periodo"], y=serie["lucro"], name="Lucro bruto",
            mode="lines+markers", line=dict(color=COR_NEUTRA_ESCURA, width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="%{x|%b/%Y}<br>Lucro bruto: R$ %{y:,.2f}<extra></extra>",
        )
    )
    fig_tempo.add_annotation(
        x=pico["periodo"], y=pico["faturamento"],
        text=f"Pico: {brl(pico['faturamento'])}",
        showarrow=True, arrowhead=0, ax=0, ay=-35,
        font=dict(color=COR_DESTAQUE, size=12),
    )
    fig_tempo.update_layout(
        template=TEMPLATE_PLOTLY,
        title=f"Faturamento e lucro por {rotulo_periodo}: a sazonalidade dita o caixa",
        yaxis_title=None, xaxis_title=None, height=400,
        yaxis=dict(tickprefix="R$ ", tickformat=",.0f"),
    )
    st.plotly_chart(fig_tempo, use_container_width=True)

with c2:
    por_cat = (
        df.groupby("categoria", as_index=False)
        .agg(faturamento=("faturamento_liquido", "sum"),
             lucro=("lucro_bruto", "sum"),
             margem=("margem_lucro_pct", "mean"))
        .sort_values("faturamento", ascending=True)
    )
    lider = por_cat.iloc[-1]["categoria"]
    cores_cat = [COR_DESTAQUE if c == lider else COR_NEUTRA for c in por_cat["categoria"]]

    fig_cat = go.Figure(
        go.Bar(
            x=por_cat["faturamento"], y=por_cat["categoria"], orientation="h",
            marker_color=cores_cat,
            text=[brl(v) for v in por_cat["faturamento"]],
            textposition="outside", cliponaxis=False,
            customdata=np.stack([por_cat["margem"], por_cat["lucro"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Faturamento: R$ %{x:,.2f}"
                          "<br>Lucro bruto: R$ %{customdata[1]:,.2f}"
                          "<br>Margem média: %{customdata[0]:.1f}%<extra></extra>",
        )
    )
    fig_cat.update_layout(
        template=TEMPLATE_PLOTLY,
        title=f"{lider} lidera o faturamento",
        height=400, xaxis=dict(visible=False), yaxis=dict(showgrid=False),
        margin=dict(l=20, r=90, t=60, b=20),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# --------------------------------------------------------------------------- #
# Linha 2 — Grade de tamanhos (curva ABC) | Desconto x Margem
# --------------------------------------------------------------------------- #
c3, c4 = st.columns(2)

with c3:
    grade = (
        df.groupby("tamanho", as_index=False)
        .agg(pares=("id_venda", "count"))
        .sort_values("tamanho")
    )
    # Curva ABC calculada sobre o ranking de volume
    ranking = grade.sort_values("pares", ascending=False).copy()
    ranking["acum_pct"] = ranking["pares"].cumsum() / ranking["pares"].sum() * 100
    ranking["classe"] = np.where(
        ranking["acum_pct"] <= 80, "A",
        np.where(ranking["acum_pct"] <= 95, "B", "C"),
    )
    grade = grade.merge(ranking[["tamanho", "classe", "acum_pct"]], on="tamanho")
    mapa_cor = {"A": COR_DESTAQUE, "B": COR_NEUTRA_ESCURA, "C": COR_NEUTRA}
    grade["cor"] = grade["classe"].map(mapa_cor)
    tamanhos_a = ", ".join(str(t) for t in sorted(grade.loc[grade["classe"] == "A", "tamanho"]))
    pct_a = grade.loc[grade["classe"] == "A", "pares"].sum() / grade["pares"].sum() * 100

    fig_grade = go.Figure(
        go.Bar(
            x=grade["tamanho"].astype(str), y=grade["pares"],
            marker_color=grade["cor"], showlegend=False,
            text=grade["pares"], textposition="outside", cliponaxis=False,
            customdata=np.stack([grade["classe"], grade["acum_pct"]], axis=-1),
            hovertemplate="Tamanho <b>%{x}</b><br>Pares: %{y}"
                          "<br>Classe ABC: %{customdata[0]}"
                          "<br>Acumulado: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    # Legenda manual da curva ABC (traços invisíveis apenas para a legenda)
    for classe, cor, rotulo in [
        ("A", COR_DESTAQUE, "Classe A — até 80% do volume"),
        ("B", COR_NEUTRA_ESCURA, "Classe B — 80% a 95%"),
        ("C", COR_NEUTRA, "Classe C — cauda"),
    ]:
        fig_grade.add_trace(go.Bar(x=[None], y=[None], marker_color=cor, name=rotulo))
    fig_grade.update_layout(
        template=TEMPLATE_PLOTLY, barmode="overlay", showlegend=True,
        title=f"Grade: numerações {tamanhos_a} concentram {pct_a:.0f}% dos pares",
        xaxis_title="Numeração", yaxis_title=None, height=440,
        xaxis=dict(type="category"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=90),
    )
    st.plotly_chart(fig_grade, use_container_width=True)

with c4:
    amostra = df.copy()
    corr = amostra["desconto_percentual"].corr(amostra["margem_lucro_pct"])
    margem_sem = amostra.loc[amostra["desconto_percentual"] == 0, "margem_lucro_pct"].mean()
    margem_20 = amostra.loc[amostra["desconto_percentual"] >= 20, "margem_lucro_pct"].mean()

    fig_disp = px.scatter(
        amostra, x="desconto_percentual", y="margem_lucro_pct",
        color="categoria",
        color_discrete_sequence=[COR_DESTAQUE, COR_DESTAQUE_2, "#8C8C8C", "#C4A484", "#5B8C5A"],
        opacity=0.55,
        hover_data={"id_venda": True, "preco_unitario": ":.2f",
                    "desconto_percentual": ":.0f", "margem_lucro_pct": ":.1f",
                    "categoria": False},
        labels={"desconto_percentual": "Desconto (%)",
                "margem_lucro_pct": "Margem bruta (%)", "categoria": "Categoria"},
    )
    # Linha de tendência global (regressão linear simples)
    if amostra["desconto_percentual"].nunique() > 1:
        coef = np.polyfit(amostra["desconto_percentual"], amostra["margem_lucro_pct"], 1)
        xs = np.linspace(0, amostra["desconto_percentual"].max(), 50)
        fig_disp.add_trace(
            go.Scatter(
                x=xs, y=np.polyval(coef, xs), mode="lines", name="Tendência",
                line=dict(color=COR_TEXTO, width=2, dash="dash"),
                hovertemplate="Desconto %{x:.0f}% → margem esperada %{y:.1f}%<extra></extra>",
            )
        )
    fig_disp.add_hline(y=0, line_color="#999999", line_width=1, line_dash="dot",
                       annotation_text="margem zero", annotation_position="bottom right",
                       annotation_font_color="#999999")
    titulo_disp = (
        f"Cada ponto de desconto custa margem "
        f"(r = {corr:.2f})".replace(".", ",")
        if not np.isnan(corr) else "Desconto × margem"
    )
    fig_disp.update_layout(
        template=TEMPLATE_PLOTLY, title=titulo_disp, height=440,
        yaxis=dict(ticksuffix="%"), xaxis=dict(ticksuffix="%"),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0, title=None),
        margin=dict(l=20, r=20, t=60, b=90),
    )
    fig_disp.update_traces(marker=dict(size=7))
    st.plotly_chart(fig_disp, use_container_width=True)
    if not np.isnan(margem_sem) and not np.isnan(margem_20):
        diferenca = f"{margem_sem - margem_20:.1f}".replace(".", ",")
        st.caption(
            f"Margem média sem desconto: **{pct(margem_sem)}** · "
            f"com desconto ≥ 20%: **{pct(margem_20)}** "
            f"(diferença de {diferenca} p.p.)."
        )

st.markdown("---")

# --------------------------------------------------------------------------- #
# Tabela resumo por categoria (apoio ao diagnóstico de mix)
# --------------------------------------------------------------------------- #
st.markdown("### Rentabilidade por categoria")
resumo_cat = (
    df.groupby("categoria")
    .agg(
        Pares=("id_venda", "count"),
        Faturamento=("faturamento_liquido", "sum"),
        Lucro_Bruto=("lucro_bruto", "sum"),
        Margem_Media=("margem_lucro_pct", "mean"),
        Desconto_Medio=("desconto_percentual", "mean"),
        Ticket_Medio=("faturamento_liquido", "mean"),
    )
    .sort_values("Lucro_Bruto", ascending=False)
    .reset_index()
)
resumo_cat["Part_Faturamento"] = resumo_cat["Faturamento"] / resumo_cat["Faturamento"].sum() * 100
resumo_cat["Part_Lucro"] = resumo_cat["Lucro_Bruto"] / resumo_cat["Lucro_Bruto"].sum() * 100
resumo_cat = resumo_cat.rename(columns={
    "categoria": "Categoria", "Lucro_Bruto": "Lucro bruto",
    "Margem_Media": "Margem média", "Desconto_Medio": "Desconto médio",
    "Ticket_Medio": "Ticket médio", "Part_Faturamento": "% do faturamento",
    "Part_Lucro": "% do lucro",
})
st.dataframe(
    resumo_cat,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Faturamento": st.column_config.NumberColumn(format="R$ %.2f"),
        "Lucro bruto": st.column_config.NumberColumn(format="R$ %.2f"),
        "Ticket médio": st.column_config.NumberColumn(format="R$ %.2f"),
        "Margem média": st.column_config.NumberColumn(format="%.1f%%"),
        "Desconto médio": st.column_config.NumberColumn(format="%.1f%%"),
        "% do faturamento": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "% do lucro": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    },
)

# --------------------------------------------------------------------------- #
# Seção inferior — Tabela detalhada com busca e download
# --------------------------------------------------------------------------- #
st.markdown("### Transações detalhadas")

col_busca, col_ordem, col_n = st.columns([2, 2, 1])
with col_busca:
    termo = st.text_input("Buscar (id da venda, categoria, canal, gênero)", value="")
with col_ordem:
    ordenar_por = st.selectbox(
        "Ordenar por",
        ["data_venda", "faturamento_liquido", "lucro_bruto", "margem_lucro_pct",
         "desconto_percentual", "preco_unitario"],
        index=0,
        format_func=lambda c: {
            "data_venda": "Data", "faturamento_liquido": "Faturamento líquido",
            "lucro_bruto": "Lucro bruto", "margem_lucro_pct": "Margem (%)",
            "desconto_percentual": "Desconto (%)", "preco_unitario": "Preço unitário",
        }[c],
    )
with col_n:
    decrescente = st.toggle("Decrescente", value=True)

df_tabela = df.drop(columns=["periodo"], errors="ignore")
if termo.strip():
    t = termo.strip().lower()
    df_tabela = df_tabela[
        df_tabela["id_venda"].str.lower().str.contains(t)
        | df_tabela["categoria"].str.lower().str.contains(t)
        | df_tabela["canal_venda"].str.lower().str.contains(t)
        | df_tabela["genero"].str.lower().str.contains(t)
    ]
df_tabela = df_tabela.sort_values(ordenar_por, ascending=not decrescente)

st.caption(f"{len(df_tabela):,} transações listadas".replace(",", "."))
st.dataframe(
    df_tabela,
    use_container_width=True,
    hide_index=True,
    height=380,
    column_config={
        "id_venda": "ID",
        "data_venda": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
        "ano_mes": "Ano-mês",
        "mes": None,
        "dia_semana": "Dia",
        "categoria": "Categoria",
        "genero": "Gênero",
        "tamanho": "Nº",
        "canal_venda": "Canal",
        "preco_unitario": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
        "desconto_percentual": st.column_config.NumberColumn("Desc.", format="%.0f%%"),
        "faturamento_liquido": st.column_config.NumberColumn("Fat. líquido", format="R$ %.2f"),
        "custo_unitario": st.column_config.NumberColumn("Custo", format="R$ %.2f"),
        "lucro_bruto": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
        "margem_lucro_pct": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    },
)

csv_bytes = df_tabela.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="⬇️ Baixar transações filtradas (CSV)",
    data=csv_bytes,
    file_name=f"pe_fresco_vendas_{data_ini:%Y%m%d}_{data_fim:%Y%m%d}.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Projeto Integrador V-A · Big Data e Inteligência Artificial · CEAD / Escola Politécnica "
    "e de Artes · PUC Goiás · Parceria: Pé Fresco Calçados."
)

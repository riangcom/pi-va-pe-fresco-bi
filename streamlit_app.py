"""
app.py
======
Dashboard analítico — Pé Fresco Calçados
Projeto Integrador V-A — Big Data e Inteligência Artificial (CEAD / PUC Goiás)

Fonte: relatórios mensais reais do PDV da loja ("Relatório de Controle de Venda"),
extraídos e tratados por src/extrair_pdv.py.

Princípios de design aplicados (Cole Nussbaumer Knaflic, "Storytelling com Dados"):
  * Uma cor de destaque (terracota) apenas no que o título afirma; resto em cinza.
  * Redução de ruído: sem bordas, grid leve, rótulos diretos, legendas só quando
    há mais de uma série.
  * Títulos afirmativos e dinâmicos (recalculados a cada filtro).

Execução (a partir da raiz do projeto):
    streamlit run src/app.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pé Fresco Calçados — BI", page_icon="👟", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Paleta e tema
# --------------------------------------------------------------------------- #
COR_DESTAQUE = "#D9622B"
COR_DESTAQUE_2 = "#1F5F8B"
COR_NEUTRA = "#B8B8B8"
COR_NEUTRA_ESCURA = "#6E6E6E"
COR_TEXTO = "#333333"
COR_FUNDO = "#FFFFFF"

TEMPLATE_PLOTLY = dict(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=COR_TEXTO),
    paper_bgcolor=COR_FUNDO, plot_bgcolor=COR_FUNDO, margin=dict(l=20, r=20, t=60, b=40),
    title=dict(font=dict(size=16, color=COR_TEXTO), x=0.0, xanchor="left"),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="#DDDDDD"),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False, linecolor="#FFFFFF"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", font_size=12),
))

st.markdown("""
<style>
  .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
  h1 {font-weight: 700; letter-spacing: -0.5px;}
  h3 {font-weight: 600; margin-top: 1.2rem;}
  div[data-testid="stMetric"] {background:#FAFAFA; border:1px solid #EEEEEE; border-radius:10px; padding:14px 18px;}
  div[data-testid="stMetricLabel"] p {color:#6E6E6E; font-size:0.85rem;}
  div[data-testid="stMetricValue"] {font-weight:700; font-size:1.65rem !important;}
  .nota {color:#6E6E6E; font-size:0.85rem; margin-top:-8px;}
</style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CANDIDATOS = [
    os.path.join(RAIZ, "data", "processed", "pe_fresco_vendas_limpo.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe_fresco_vendas_limpo.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe_fresco_vendas_limpo.csv.gz"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe_fresco_vendas_limpo.csv.xz"),
]
ARQ_DADOS = next((c for c in _CANDIDATOS if os.path.exists(c)), _CANDIDATOS[0])

MESES_PT = {"2026-07": "Jul/2026", "2026-08": "Ago/2026", "2026-09": "Set/2026", "2026-10": "Out/2026",
            "2026-11": "Nov/2026", "2026-12": "Dez/2026"}


@st.cache_data(show_spinner=False)
def carregar(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, dtype={"tamanho": "string", "codigo": "string"})
    for c in ["eh_calcado", "venda_atipica", "cadastro_generico", "margem_confiavel",
              "alerta_estoque_negativo", "alerta_qtde_corrigida"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    df["mes_rotulo"] = df["mes"].map(MESES_PT).fillna(df["mes"])
    return df


if not os.path.exists(ARQ_DADOS):
    st.error("Base tratada não encontrada. Execute `python src/extrair_pdv.py` na raiz do projeto.")
    st.stop()
df_total = carregar(ARQ_DADOS)


def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",") + "%"


def margem_de(d: pd.DataFrame) -> float:
    c = d[d["margem_confiavel"]]
    return float(c["lucro_bruto"].sum() / c["faturamento"].sum() * 100) if c["faturamento"].sum() else np.nan


# --------------------------------------------------------------------------- #
# Filtros
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 👟 Pé Fresco Calçados")
    st.caption("Painel de Business Intelligence — Projeto Integrador V-A · PUC Goiás")
    st.markdown("---")
    st.markdown("### Filtros")
    meses = sorted(df_total["mes"].unique())
    meses_sel = st.multiselect("Mês", meses, default=meses, format_func=lambda m: MESES_PT.get(m, m))
    filiais = sorted(df_total["filial"].unique())
    filiais_sel = st.multiselect("Filial", filiais, default=filiais)
    categorias = sorted(df_total["categoria"].unique())
    categorias_sel = st.multiselect("Categoria", categorias, default=categorias)
    generos = sorted(df_total["genero"].unique())
    generos_sel = st.multiselect("Gênero", generos, default=generos)
    so_calcados = st.toggle("Somente calçados", value=False)
    st.markdown("---")
    st.caption("Fonte: Relatório de Controle de Venda do PDV da Pé Fresco Calçados "
               "(exportações mensais fornecidas pela proprietária), tratado por `src/extrair_pdv.py`. "
               "Vendedores anonimizados.")

mask = (df_total["mes"].isin(meses_sel) & df_total["filial"].isin(filiais_sel)
        & df_total["categoria"].isin(categorias_sel) & df_total["genero"].isin(generos_sel))
if so_calcados:
    mask &= df_total["eh_calcado"]
df = df_total.loc[mask].copy()

# --------------------------------------------------------------------------- #
# Cabeçalho + KPIs
# --------------------------------------------------------------------------- #
st.title("Pé Fresco Calçados — Painel de Mix, Grade e Rentabilidade")
periodo = " a ".join([MESES_PT.get(m, m) for m in (min(meses_sel), max(meses_sel))]) if meses_sel else "—"
st.markdown(f"<p class='nota'>Período: <b>{periodo}</b> · {len(filiais_sel)} filial(is) · "
            f"{int(df['qtde_vendida'].sum()):,} itens vendidos em {len(df):,} linhas de venda</p>".replace(",", "."),
            unsafe_allow_html=True)
if df.empty:
    st.warning("Nenhuma venda atende aos filtros selecionados.")
    st.stop()

fat = df["faturamento"].sum()
lucro = df["lucro_bruto"].sum()
itens = int(df["qtde_vendida"].sum())
preco_medio = fat / itens if itens else np.nan
margem = margem_de(df)

# comparação com o outro mês, quando exatamente um mês está selecionado
delta_fat = delta_lucro = delta_preco = delta_margem = None
if len(meses_sel) == 1:
    idx = meses.index(meses_sel[0])
    if idx > 0:
        ant = df_total[(df_total["mes"] == meses[idx - 1]) & df_total["filial"].isin(filiais_sel)
                       & df_total["categoria"].isin(categorias_sel) & df_total["genero"].isin(generos_sel)]
        if so_calcados:
            ant = ant[ant["eh_calcado"]]
        if not ant.empty:
            fa, la, ia = ant["faturamento"].sum(), ant["lucro_bruto"].sum(), int(ant["qtde_vendida"].sum())
            rot = f"vs. {MESES_PT.get(meses[idx - 1], meses[idx - 1])}"
            delta_fat = f"{(fat / fa - 1) * 100:+.1f}% {rot}".replace(".", ",")
            delta_lucro = f"{(lucro / la - 1) * 100:+.1f}% {rot}".replace(".", ",")
            delta_preco = f"{(preco_medio / (fa / ia) - 1) * 100:+.1f}% {rot}".replace(".", ",")
            delta_margem = f"{margem - margem_de(ant):+.1f} p.p. {rot}".replace(".", ",")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Faturamento Total", brl(fat), delta_fat)
k2.metric("Lucro Bruto Total", brl(lucro), delta_lucro)
k3.metric("Preço Médio por Item", brl(preco_medio), delta_preco)
k4.metric("Margem Bruta", pct(margem), delta_margem)
st.caption("Margem bruta = lucro bruto / faturamento, calculada sobre a base confiável (exclui vendas atípicas e "
           "SKUs de cadastro genérico). Selecione um único mês para ver a variação frente ao mês anterior.")
st.markdown("---")

# --------------------------------------------------------------------------- #
# Linha 1 — Mês × filial | Categoria
# --------------------------------------------------------------------------- #
c1, c2 = st.columns([3, 2])
with c1:
    mf = df.groupby(["mes", "mes_rotulo", "filial"], as_index=False).agg(fat=("faturamento", "sum"), lucro=("lucro_bruto", "sum"))
    lider_fil = mf.groupby("filial")["fat"].sum().idxmax()
    fig = go.Figure()
    cores_fil = {f: (COR_DESTAQUE if f == lider_fil else c) for f, c in zip(sorted(mf["filial"].unique()), [COR_NEUTRA, COR_NEUTRA_ESCURA, "#8C8C8C"])}
    for f in sorted(mf["filial"].unique()):
        sub = mf[mf["filial"] == f].sort_values("mes")
        fig.add_trace(go.Bar(x=sub["mes_rotulo"], y=sub["fat"], name=f, marker_color=cores_fil[f],
                             text=[brl(v) for v in sub["fat"]], textposition="outside", cliponaxis=False,
                             customdata=sub["lucro"],
                             hovertemplate="<b>%{fullData.name}</b> · %{x}<br>Faturamento: R$ %{y:,.2f}<br>Lucro bruto: R$ %{customdata:,.2f}<extra></extra>"))
    tot = mf.groupby("mes_rotulo")["fat"].sum()
    fig.update_layout(template=TEMPLATE_PLOTLY, barmode="group", height=400,
                      title=f"{lider_fil} lidera o faturamento no período" + (f" — total {brl(tot.iloc[-1])} em {tot.index[-1]}" if len(tot) else ""),
                      yaxis=dict(tickprefix="R$ ", tickformat=",.0f", title=None), xaxis=dict(title=None))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    pc = df.groupby("categoria", as_index=False).agg(fat=("faturamento", "sum"), lucro=("lucro_bruto", "sum"), itens=("qtde_vendida", "sum")).sort_values("fat")
    lider = pc.iloc[-1]["categoria"]
    fig = go.Figure(go.Bar(x=pc["fat"], y=pc["categoria"], orientation="h",
                           marker_color=[COR_DESTAQUE if c == lider else COR_NEUTRA for c in pc["categoria"]],
                           text=[brl(v) for v in pc["fat"]], textposition="outside", cliponaxis=False,
                           customdata=np.stack([pc["itens"], pc["lucro"]], axis=-1),
                           hovertemplate="<b>%{y}</b><br>Faturamento: R$ %{x:,.2f}<br>Itens: %{customdata[0]}<br>Lucro bruto: R$ %{customdata[1]:,.2f}<extra></extra>"))
    fig.update_layout(template=TEMPLATE_PLOTLY, height=400, xaxis=dict(visible=False), yaxis=dict(showgrid=False),
                      margin=dict(l=20, r=100, t=60, b=20),
                      title=f"{lider} lidera: {pct(pc.iloc[-1]['fat'] / fat * 100, 0)} do faturamento")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Linha 2 — Grade ABC (calçados) | Margem por categoria
# --------------------------------------------------------------------------- #
c3, c4 = st.columns(2)
with c3:
    g = df[df["eh_calcado"] & df["tamanho"].notna()].copy()
    g = g[g["tamanho"].str.fullmatch(r"\d{2}")]
    if g.empty:
        st.info("Sem calçados com numeração no recorte selecionado.")
    else:
        g["tamanho"] = g["tamanho"].astype(int)
        grade = g.groupby("tamanho", as_index=False)["qtde_vendida"].sum().sort_values("tamanho")
        rk = grade.sort_values("qtde_vendida", ascending=False).copy()
        rk["acum"] = rk["qtde_vendida"].cumsum() / rk["qtde_vendida"].sum() * 100
        rk["classe"] = np.where(rk["acum"] <= 80, "A", np.where(rk["acum"] <= 95, "B", "C"))
        grade = grade.merge(rk[["tamanho", "classe", "acum"]], on="tamanho")
        cores = {"A": COR_DESTAQUE, "B": COR_NEUTRA_ESCURA, "C": COR_NEUTRA}
        a = sorted(grade.loc[grade["classe"] == "A", "tamanho"])
        share_a = grade.loc[grade["classe"] == "A", "qtde_vendida"].sum() / grade["qtde_vendida"].sum() * 100
        fig = go.Figure(go.Bar(x=grade["tamanho"].astype(str), y=grade["qtde_vendida"], marker_color=grade["classe"].map(cores),
                               showlegend=False, text=grade["qtde_vendida"], textposition="outside", cliponaxis=False,
                               customdata=np.stack([grade["classe"], grade["acum"]], axis=-1),
                               hovertemplate="Numeração <b>%{x}</b><br>Pares: %{y}<br>Classe: %{customdata[0]}<br>Acumulado: %{customdata[1]:.1f}%<extra></extra>"))
        for cl, rot in [("A", "Classe A — até 80% dos pares"), ("B", "Classe B — 80% a 95%"), ("C", "Classe C — cauda")]:
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color=cores[cl], name=rot))
        faixa = f"{a[0]} a {a[-1]}" if a and a[-1] - a[0] == len(a) - 1 else ", ".join(map(str, a))
        fig.update_layout(template=TEMPLATE_PLOTLY, barmode="overlay", showlegend=True, height=440,
                          title=f"Grade: numerações {faixa} concentram {share_a:.0f}% dos pares",
                          xaxis=dict(type="category", title="Numeração"), yaxis=dict(title=None),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
                          margin=dict(l=20, r=20, t=60, b=90))
        st.plotly_chart(fig, use_container_width=True)

with c4:
    conf = df[df["margem_confiavel"]]
    mc = conf.groupby("categoria").apply(lambda x: pd.Series({"margem": x["lucro_bruto"].sum() / x["faturamento"].sum() * 100, "fat": x["faturamento"].sum()})).reset_index().sort_values("margem")
    menor, maior = mc.iloc[0], mc.iloc[-1]
    fig = go.Figure(go.Bar(x=mc["margem"], y=mc["categoria"], orientation="h",
                           marker_color=[COR_DESTAQUE if c == menor["categoria"] else (COR_DESTAQUE_2 if c == maior["categoria"] else COR_NEUTRA) for c in mc["categoria"]],
                           text=[pct(v) for v in mc["margem"]], textposition="outside", cliponaxis=False,
                           customdata=mc["fat"], hovertemplate="<b>%{y}</b><br>Margem bruta: %{x:.1f}%<br>Faturamento: R$ %{customdata:,.2f}<extra></extra>"))
    fig.add_vline(x=margem, line_color=COR_NEUTRA_ESCURA, line_dash="dot", annotation_text=f"média {pct(margem)}", annotation_position="top")
    fig.update_layout(template=TEMPLATE_PLOTLY, height=440, xaxis=dict(ticksuffix="%", range=[0, max(75, mc["margem"].max() + 12)], title=None),
                      yaxis=dict(showgrid=False), margin=dict(l=20, r=70, t=60, b=40),
                      title=f"Margem: {menor['categoria']} tem a menor ({pct(menor['margem'])}); {maior['categoria']}, a maior ({pct(maior['margem'])})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Desconto concedido no recorte: **{brl(df['valor_desconto'].sum())}** "
               f"({pct(df['valor_desconto'].sum() / (fat + df['valor_desconto'].sum()) * 100, 2)} do preço de vitrine) "
               f"em {int((df['valor_desconto'] > 0).sum())} linhas de venda.")

# --------------------------------------------------------------------------- #
# Linha 3 — Marcas (calçados) | Vendedores por filial
# --------------------------------------------------------------------------- #
c5, c6 = st.columns(2)
with c5:
    mk = conf[conf["eh_calcado"] & (conf["marca"] != "Não informada")]
    if mk.empty:
        st.info("Sem calçados com marca identificada no recorte.")
    else:
        mb = mk.groupby("marca").apply(lambda x: pd.Series({"fat": x["faturamento"].sum(), "margem": x["lucro_bruto"].sum() / x["faturamento"].sum() * 100, "itens": x["qtde_vendida"].sum()})).reset_index()
        mb = mb.sort_values("fat", ascending=False).head(10).sort_values("fat")
        top = mb.iloc[-1]["marca"]
        fig = go.Figure(go.Bar(x=mb["fat"], y=mb["marca"], orientation="h",
                               marker_color=[COR_DESTAQUE if m == top else COR_NEUTRA for m in mb["marca"]],
                               text=[f"{brl(f)} · {pct(m)}" for f, m in zip(mb["fat"], mb["margem"])], textposition="outside", cliponaxis=False,
                               customdata=np.stack([mb["margem"], mb["itens"]], axis=-1),
                               hovertemplate="<b>%{y}</b><br>Faturamento: R$ %{x:,.2f}<br>Margem: %{customdata[0]:.1f}%<br>Itens: %{customdata[1]}<extra></extra>"))
        fig.update_layout(template=TEMPLATE_PLOTLY, height=420, xaxis=dict(visible=False), yaxis=dict(showgrid=False),
                          margin=dict(l=20, r=150, t=60, b=20),
                          title=f"Marcas de calçado: {top} lidera (faturamento · margem)")
        st.plotly_chart(fig, use_container_width=True)

with c6:
    vf = df.groupby(["filial", "vendedor"], as_index=False)["faturamento"].sum()
    vf = vf[vf["faturamento"] > 0]
    top_v = vf.sort_values("faturamento", ascending=False).iloc[0]
    fig = go.Figure()
    vend_ordem = vf.groupby("vendedor")["faturamento"].sum().sort_values(ascending=False).index
    for i, v in enumerate(vend_ordem):
        sub = vf[vf["vendedor"] == v]
        fig.add_trace(go.Bar(x=sub["filial"], y=sub["faturamento"], name=v,
                             marker_color=COR_DESTAQUE if v == top_v["vendedor"] else ([COR_NEUTRA_ESCURA, COR_NEUTRA, "#D0D0D0", "#E4E4E4"][i % 4]),
                             hovertemplate="<b>%{fullData.name}</b> · %{x}<br>R$ %{y:,.2f}<extra></extra>"))
    fig.update_layout(template=TEMPLATE_PLOTLY, barmode="stack", height=420, showlegend=False,
                      title=f"{top_v['vendedor']} ({top_v['filial']}) responde por {pct(top_v['faturamento'] / fat * 100, 0)} do faturamento",
                      yaxis=dict(tickprefix="R$ ", tickformat=",.0f", title=None), xaxis=dict(title=None))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada bloco é um vendedor (anonimizado). Passe o mouse para ver o valor.")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
st.markdown("### Rentabilidade por categoria")
res = df.groupby("categoria").agg(Itens=("qtde_vendida", "sum"), Faturamento=("faturamento", "sum"), Lucro=("lucro_bruto", "sum"),
                                  Desconto=("valor_desconto", "sum")).reset_index()
res["Margem"] = res["categoria"].map(conf.groupby("categoria").apply(lambda x: x["lucro_bruto"].sum() / x["faturamento"].sum() * 100))
res["Preço médio"] = res["Faturamento"] / res["Itens"]
res["% do faturamento"] = res["Faturamento"] / fat * 100
res["% do lucro"] = res["Lucro"] / lucro * 100
res = res.rename(columns={"categoria": "Categoria", "Lucro": "Lucro bruto"}).sort_values("Faturamento", ascending=False)
st.dataframe(res, use_container_width=True, hide_index=True, column_config={
    "Faturamento": st.column_config.NumberColumn(format="R$ %.2f"),
    "Lucro bruto": st.column_config.NumberColumn(format="R$ %.2f"),
    "Desconto": st.column_config.NumberColumn(format="R$ %.2f"),
    "Preço médio": st.column_config.NumberColumn(format="R$ %.2f"),
    "Margem": st.column_config.NumberColumn(format="%.1f%%"),
    "% do faturamento": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    "% do lucro": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

with st.expander("Qualidade dos dados do PDV neste recorte"):
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Linhas com estoque negativo", f"{int(df['alerta_estoque_negativo'].sum())} ({pct(df['alerta_estoque_negativo'].mean() * 100, 0)})")
    q2.metric("Quantidades corrigidas", int(df["alerta_qtde_corrigida"].sum()))
    q3.metric("Vendas atípicas (< 30% da vitrine)", int(df["venda_atipica"].sum()))
    q4.metric("SKUs de cadastro genérico", int(df["cadastro_generico"].sum()))
    st.caption("Sinalizações produzidas na etapa de tratamento (ver relatório técnico, seção 2.5). Nenhum valor de "
               "faturamento foi alterado; apenas quantidades inconsistentes foram corrigidas e sinalizadas.")

st.markdown("### Linhas de venda detalhadas")
cb, co, cd = st.columns([2, 2, 1])
with cb:
    termo = st.text_input("Buscar (descrição, marca, código, vendedor)", value="")
with co:
    ordenar = st.selectbox("Ordenar por", ["faturamento", "lucro_bruto", "margem_pct", "qtde_vendida", "valor_desconto"], index=0,
                           format_func=lambda c: {"faturamento": "Faturamento", "lucro_bruto": "Lucro bruto", "margem_pct": "Margem (%)",
                                                  "qtde_vendida": "Itens", "valor_desconto": "Desconto (R$)"}[c])
with cd:
    desc_ = st.toggle("Decrescente", value=True)

cols = ["mes_rotulo", "filial", "vendedor", "codigo", "descricao", "categoria", "genero", "marca", "tamanho", "qtde_vendida",
        "custo_medio", "valor_vitrine", "valor_desconto", "faturamento", "lucro_bruto", "margem_pct"]
tab = df[cols].copy()
if termo.strip():
    t = termo.strip().lower()
    tab = tab[tab["descricao"].str.lower().str.contains(t, regex=False) | tab["marca"].str.lower().str.contains(t, regex=False)
              | tab["codigo"].str.lower().str.contains(t, regex=False) | tab["vendedor"].str.lower().str.contains(t, regex=False)]
tab = tab.sort_values(ordenar, ascending=not desc_)
st.caption(f"{len(tab):,} linhas listadas".replace(",", "."))
st.dataframe(tab, use_container_width=True, hide_index=True, height=380, column_config={
    "mes_rotulo": "Mês", "filial": "Filial", "vendedor": "Vendedor", "codigo": "Cód.", "descricao": "Descrição",
    "categoria": "Categoria", "genero": "Gênero", "marca": "Marca", "tamanho": "Nº", "qtde_vendida": "Itens",
    "custo_medio": st.column_config.NumberColumn("Custo méd.", format="R$ %.2f"),
    "valor_vitrine": st.column_config.NumberColumn("Vitrine", format="R$ %.2f"),
    "valor_desconto": st.column_config.NumberColumn("Desc.", format="R$ %.2f"),
    "faturamento": st.column_config.NumberColumn("Faturamento", format="R$ %.2f"),
    "lucro_bruto": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
    "margem_pct": st.column_config.NumberColumn("Margem", format="%.1f%%")})

csv_bytes = tab.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("⬇️ Baixar linhas filtradas (CSV)", data=csv_bytes, file_name="pe_fresco_vendas_filtro.csv", mime="text/csv")

st.markdown("---")
st.caption("Projeto Integrador V-A · Big Data e Inteligência Artificial · CEAD / Escola Politécnica e de Artes · PUC Goiás · "
           "Parceria: Pé Fresco Calçados. Dados reais do PDV (ago–set/2026), vendedores anonimizados.")

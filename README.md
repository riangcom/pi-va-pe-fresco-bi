# Pé Fresco Calçados — Painel de BI (Projeto Integrador V-A · PUC Goiás)

Dashboard interativo (Streamlit + Plotly) de mix, grade de numeração e rentabilidade da Pé Fresco Calçados, construído sobre os **relatórios reais do PDV** da loja (Relatório de Controle de Venda, agosto e setembro de 2026, 3 filiais).

**Aluno:** Rian Godinho Caixeta Oliveira Martins · **Professor:** Thalles Santos · **Disciplina:** 11304921015_20262_02
**App:** https://pe-fresco-bi.streamlit.app

## Arquivos

| Arquivo | Função |
|---|---|
| `streamlit_app.py` | Dashboard (KPIs, filtros, 6 visualizações, tabelas com download, painel de qualidade dos dados) |
| `extrair_pdv.py` | Lê os PDFs mensais do PDV, valida contra o Total Geral impresso, anonimiza vendedores, deriva categoria/gênero/marca/numeração e grava a base analítica |
| `pe_fresco_vendas_limpo.csv.gz` | Base analítica tratada e anonimizada (1.645 linhas de venda, ago–set/2026) |
| `requirements.txt` | streamlit, plotly, pandas, numpy, pdfplumber |

Os PDFs originais do PDV não estão neste repositório (contêm nomes de funcionários); acompanham apenas o pacote de entrega da disciplina.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Para atualizar com um novo mês: salve o PDF do PDV em `data/raw/` na estrutura completa do projeto e rode `python src/extrair_pdv.py`.

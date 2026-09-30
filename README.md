# Pé Fresco Calçados — Painel de BI (Projeto Integrador V-A · PUC Goiás)

Dashboard interativo (Streamlit + Plotly) de rentabilidade por categoria, giro de grade e política de descontos da Pé Fresco Calçados.

**Aluno:** Rian Godinho Caixeta Oliveira Martins · **Professor:** Thalles Santos · **Disciplina:** 11304921015_20262_02

## Arquivos

| Arquivo | Função |
|---|---|
| `streamlit_app.py` | Dashboard (KPIs, filtros, 4 visualizações, tabelas com download) |
| `gerador_dados.py` | Gera e trata a base sintética (seed 42) |
| `requirements.txt` | streamlit, plotly, pandas, numpy |

A base tratada (2.000 transações, seed 42) é gerada em memória na primeira execução; rodar `python gerador_dados.py` grava os CSVs em disco.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

O projeto completo (relatório técnico, slides, roteiro do vídeo e estrutura `src/`, `data/`, `docs/`) é entregue no pacote `projeto_integrador_pe_fresco.zip`.

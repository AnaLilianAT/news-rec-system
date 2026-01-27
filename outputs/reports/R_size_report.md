# Auditoria: Tamanho de R_user e Correlação com GH

## Objetivo

Verificar se a construção de R_user_global (acumulando todas as sessões)
está causando alta variância e valores GH>1.

## Dados

- **Usuários processados**: 215
- **Algoritmos**: 9
- **Total de pares de avaliação**: 3550

## Distribuição de |R| por Algoritmo

| Algoritmo | N usuários | p25 | p50 | p75 | p90 | Max | GH Média | GH DP |
|-----------|-----------|-----|-----|-----|-----|-----|----------|-------|
| knn_item_item                       |         33 |    7 |   10 |   20 |   29 |   45 |    0.729 |  0.629 |
| knn_item_item-mmr-diversification   |         21 |    5 |    7 |   12 |   13 |   17 |    0.193 |  0.166 |
| knn_item_item-topic-diversification |         27 |    7 |    9 |   17 |   30 |   34 |    0.668 |  0.593 |
| knn_user_user                       |         24 |    9 |   19 |   39 |   74 |   95 |    1.531 |  1.589 |
| knn_user_user-mmr-diversification   |         19 |    3 |    5 |    7 |   13 |   31 |    0.241 |  0.391 |
| knn_user_user-topic-diversification |         25 |    6 |   13 |   16 |   25 |  230 |    1.215 |  2.489 |
| svd                                 |         25 |   16 |   20 |   39 |   43 |   58 |    1.354 |  0.812 |
| svd-mmr-diversification             |         19 |    6 |    7 |    8 |   10 |   16 |    0.234 |  0.187 |
| svd-topic-diversification           |         22 |    5 |    8 |   13 |   19 |   28 |    0.514 |  0.400 |


## Correlação entre GH e |R|

Correlação de Spearman entre GH_user_global e n_R_items:

| Algoritmo | Correlação | p-valor | Interpretação |
|-----------|-----------|---------|---------------|
| knn_item_item                       |      0.973 |   0.0000 | ⚠️ FORTE |
| knn_item_item-mmr-diversification   |      0.850 |   0.0000 | ⚠️ FORTE |
| knn_item_item-topic-diversification |      0.957 |   0.0000 | ⚠️ FORTE |
| knn_user_user                       |      0.979 |   0.0000 | ⚠️ FORTE |
| knn_user_user-mmr-diversification   |      0.832 |   0.0000 | ⚠️ FORTE |
| knn_user_user-topic-diversification |      0.872 |   0.0000 | ⚠️ FORTE |
| svd                                 |      0.961 |   0.0000 | ⚠️ FORTE |
| svd-mmr-diversification             |      0.766 |   0.0001 | ⚠️ FORTE |
| svd-topic-diversification           |      0.792 |   0.0000 | ⚠️ FORTE |


## Análise de Sessões

| Algoritmo | Sessões Médias | Sessões Mediana | Sessões Max |
|-----------|---------------|----------------|-------------|
| knn_item_item                       |            1.6 |             1.0 |           7 |
| svd                                 |            1.7 |             1.0 |           5 |
| knn_user_user                       |            2.0 |             1.0 |           9 |
| knn_user_user-topic-diversification |            2.4 |             1.0 |          21 |
| svd-mmr-diversification             |            1.4 |             1.0 |           4 |
| knn_item_item-topic-diversification |            1.5 |             1.0 |           3 |
| knn_user_user-mmr-diversification   |            1.5 |             1.0 |           6 |
| svd-topic-diversification           |            1.9 |             1.5 |           7 |
| knn_item_item-mmr-diversification   |            1.8 |             2.0 |           3 |


## Diagnóstico

### ⚠️ PROBLEMA IDENTIFICADO

**9 algoritmos** apresentam correlação |ρ| > 0.5 entre GH e |R|:
- **knn_item_item**: ρ = 0.973 (p=0.0000)
- **knn_item_item-mmr-diversification**: ρ = 0.850 (p=0.0000)
- **knn_item_item-topic-diversification**: ρ = 0.957 (p=0.0000)
- **knn_user_user**: ρ = 0.979 (p=0.0000)
- **knn_user_user-mmr-diversification**: ρ = 0.832 (p=0.0000)
- **knn_user_user-topic-diversification**: ρ = 0.872 (p=0.0000)
- **svd**: ρ = 0.961 (p=0.0000)
- **svd-mmr-diversification**: ρ = 0.766 (p=0.0001)
- **svd-topic-diversification**: ρ = 0.792 (p=0.0000)

**Implicação**: GH está crescendo com |R|, confirmando que
acumular itens de múltiplas sessões distorce a métrica.

### ⚠️ ALTA VARIÂNCIA DETECTADA

**2 algoritmos** têm DP(GH) > 1.0:
- **knn_user_user**: DP = 1.589
- **knn_user_user-topic-diversification**: DP = 2.489

**Implicação**: Variância excessiva, provavelmente devido a |R| variando muito.

### Valores GH > 1.0

**53 usuários (24.7%)** têm GH > 1.0
 ⚠️ **PROBLEMÁTICO**


## Recomendação

✅ **CALCULAR GH POR SESSÃO**

A abordagem atual (R_user_global) acumula itens de múltiplas sessões,
causando:
1. |R| muito grande → GH > 1 em muitos casos
2. Alta variância (DP > 1.0)
3. Correlação entre GH e |R|

**Solução**: Calcular GH_session para cada (user_id, t_rec),
depois agregar: GH_user = média(GH_session).

## Top 20 Usuários com Maior |R|

| user_id | algorithm | |R| | #sessões | GH |
|---------|-----------|-----|----------|-----|
|     251 | knn_user_user-topic-diversification | 230 |       21 | 12.638 |
|     347 | knn_user_user                       |  95 |        9 | 5.571 |
|     185 | knn_user_user                       |  80 |        4 | 4.721 |
|     221 | knn_user_user                       |  80 |        4 | 4.396 |
|     130 | knn_user_user                       |  60 |        3 | 3.607 |
|     358 | svd                                 |  58 |        3 | 3.404 |
|     170 | knn_user_user-topic-diversification |  56 |        5 | 3.804 |
|     140 | knn_user_user                       |  54 |        3 | 2.881 |
|     268 | svd                                 |  47 |        4 | 2.619 |
|      11 | knn_item_item                       |  45 |        3 | 2.648 |
|     277 | svd                                 |  45 |        5 | 2.508 |
|     213 | knn_item_item                       |  40 |        2 | 2.011 |
|     257 | knn_user_user                       |  40 |        2 | 2.037 |
|     286 | svd                                 |  40 |        2 | 2.135 |
|     313 | svd                                 |  40 |        2 | 2.204 |
|     331 | svd                                 |  40 |        2 | 2.093 |
|     196 | svd                                 |  39 |        2 | 2.054 |
|     230 | knn_user_user                       |  39 |        2 | 2.063 |
|     304 | svd                                 |  35 |        2 | 2.074 |
|     261 | knn_item_item-topic-diversification |  34 |        3 | 1.611 |


# Relatório de Avaliação - Replay Temporal (ALL-BETWEEN)

**Data de geração**: 2026-01-20 12:25:51

## Exposição

- **Total de interações no teste**: 5,828
- **Interações expostas (na top-20)**: 1,811
- **Taxa de exposição**: 31.07%

## RMSE por Algoritmo

RMSE mede o erro de predição entre score_pred e rating_real para itens expostos.

| Algoritmo | RMSE | N Pares | Média Rating Real | Média Score Pred |
|-----------|------|---------|-------------------|------------------|
| svd | 0.6297 | 64 | 0.750 | 0.551 |
| svd-topic-diversification | 0.7508 | 220 | 0.695 | 0.709 |
| svd-mmr-diversification | 0.8104 | 155 | 0.581 | 0.661 |
| knn_user_user | 0.8341 | 23 | 0.652 | 1.000 |
| knn_item_item | 0.9889 | 232 | 0.388 | 0.359 |
| knn_item_item-topic-diversification | 1.0043 | 334 | 0.497 | 0.711 |
| knn_item_item-mmr-diversification | 1.0429 | 168 | 0.429 | 0.863 |
| knn_user_user-mmr-diversification | 1.1203 | 109 | 0.385 | 0.899 |
| knn_user_user-topic-diversification | 1.4001 | 506 | 0.026 | 0.902 |

### Estatísticas RMSE por Usuário

| Algoritmo | Média RMSE | Mediana RMSE | Std RMSE | N Usuários |
|-----------|------------|--------------|----------|------------|
| knn_item_item | 0.9451 | 1.0343 | 0.3304 | 12 |
| knn_item_item-mmr-diversification | 0.9956 | 0.9572 | 0.4183 | 20 |
| knn_item_item-topic-diversification | 0.9807 | 0.9686 | 0.2610 | 25 |
| knn_user_user | 0.6870 | 0.7416 | 0.5238 | 4 |
| knn_user_user-mmr-diversification | 1.0286 | 1.0000 | 0.4161 | 17 |
| knn_user_user-topic-diversification | 1.1061 | 1.0690 | 0.4157 | 25 |
| svd | 0.5363 | 0.4716 | 0.3072 | 4 |
| svd-mmr-diversification | 0.7089 | 0.7763 | 0.3065 | 20 |
| svd-topic-diversification | 0.6501 | 0.6876 | 0.3283 | 22 |

## GH Cosine (Homogeneidade por Features)

GH_COSINE mede a similaridade média (cosseno) entre itens da lista top-20.

| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |
|-----------|----------|------------|--------|----------|
| knn_item_item | 0.2454 | 0.2448 | 0.0117 | 161 |
| knn_item_item-mmr-diversification | 0.1592 | 0.1571 | 0.0343 | 71 |
| knn_item_item-topic-diversification | 0.2403 | 0.2385 | 0.0324 | 70 |
| knn_user_user | 0.2754 | 0.2924 | 0.0584 | 12 |
| knn_user_user-mmr-diversification | 0.1553 | 0.1504 | 0.0272 | 48 |
| knn_user_user-topic-diversification | 0.2358 | 0.2343 | 0.0300 | 99 |
| svd | 0.2436 | 0.2554 | 0.0369 | 7 |
| svd-mmr-diversification | 0.2130 | 0.2111 | 0.0210 | 56 |
| svd-topic-diversification | 0.2545 | 0.2495 | 0.0296 | 78 |

## GH Jaccard (Homogeneidade por Tópicos)

GH_JACCARD mede a similaridade Jaccard média entre tópicos dominantes dos itens.

| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |
|-----------|----------|------------|--------|----------|
| knn_item_item | 0.1422 | 0.1452 | 0.0156 | 161 |
| knn_item_item-mmr-diversification | 0.0888 | 0.0828 | 0.0174 | 71 |
| knn_item_item-topic-diversification | 0.1086 | 0.1051 | 0.0164 | 70 |
| knn_user_user | 0.1412 | 0.1337 | 0.0411 | 12 |
| knn_user_user-mmr-diversification | 0.0908 | 0.0921 | 0.0177 | 48 |
| knn_user_user-topic-diversification | 0.1159 | 0.1166 | 0.0191 | 99 |
| svd | 0.1350 | 0.1359 | 0.0150 | 7 |
| svd-mmr-diversification | 0.1004 | 0.1005 | 0.0117 | 56 |
| svd-topic-diversification | 0.1080 | 0.1077 | 0.0150 | 78 |

## Resumo Geral

- **Algoritmos avaliados**: 9
- **Usuários com métricas**: 198
- **Total de eval_pairs**: 1,811
- **Listas avaliadas (GH)**: 602

### Top 5 Algoritmos (Menor RMSE)

2. **svd**: RMSE = 0.6297 (64 pares)
5. **svd-topic-diversification**: RMSE = 0.7508 (220 pares)
6. **svd-mmr-diversification**: RMSE = 0.8104 (155 pares)
3. **knn_user_user**: RMSE = 0.8341 (23 pares)
1. **knn_item_item**: RMSE = 0.9889 (232 pares)

### Top 5 Algoritmos (Menor GH Cosine = Mais Diversos)

1. **knn_user_user-mmr-diversification**: GH = 0.1553 (48 listas)
2. **knn_item_item-mmr-diversification**: GH = 0.1592 (71 listas)
3. **svd-mmr-diversification**: GH = 0.2130 (56 listas)
4. **knn_user_user-topic-diversification**: GH = 0.2358 (99 listas)
5. **knn_item_item-topic-diversification**: GH = 0.2403 (70 listas)

## Interpretação

- **RMSE baixo**: Predições mais precisas
- **GH alto**: Lista mais homogênea (menos diversa)
- **GH baixo**: Lista mais diversificada
- **Taxa de exposição**: Percentual de avaliações no teste que foram recomendadas

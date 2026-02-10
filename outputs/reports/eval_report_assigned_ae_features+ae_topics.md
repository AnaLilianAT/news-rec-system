# Relatório de Avaliação - Replay Temporal (ALL-BETWEEN)

**Data de geração**: 2026-02-09 11:29:41

## Exposição

- **Total de interações no teste**: 5,828
- **Interações expostas (na top-20)**: 3,537
- **Taxa de exposição**: 60.69%

## RMSE por Algoritmo

RMSE mede o erro de predição entre score_pred e rating_real para itens expostos.

| Algoritmo | RMSE | N Pares | Média Rating Real | Média Score Pred |
|-----------|------|---------|-------------------|------------------|
| svd | 0.7490 | 683 | 0.649 | 0.664 |
| svd-topic-diversification | 0.7643 | 221 | 0.679 | 0.710 |
| svd-mmr-diversification | 0.8404 | 145 | 0.524 | 0.672 |
| knn_item_item | 0.9538 | 524 | 0.508 | 0.698 |
| knn_item_item-mmr-diversification | 1.0247 | 181 | 0.486 | 0.962 |
| knn_item_item-topic-diversification | 1.0335 | 377 | 0.454 | 0.894 |
| knn_user_user-mmr-diversification | 1.0357 | 132 | 0.477 | 0.940 |
| knn_user_user | 1.0559 | 683 | 0.416 | 0.820 |
| knn_user_user-topic-diversification | 1.3979 | 591 | -0.027 | 0.943 |

### Estatísticas RMSE por Usuário

| Algoritmo | Média RMSE | Mediana RMSE | Std RMSE | N Usuários |
|-----------|------------|--------------|----------|------------|
| knn_item_item | 0.7872 | 0.8339 | 0.4143 | 33 |
| knn_item_item-mmr-diversification | 0.7516 | 0.9134 | 0.5415 | 22 |
| knn_item_item-topic-diversification | 1.0099 | 1.0690 | 0.3487 | 27 |
| knn_user_user | 0.8891 | 0.9570 | 0.4394 | 24 |
| knn_user_user-mmr-diversification | 0.8623 | 0.9082 | 0.5598 | 18 |
| knn_user_user-topic-diversification | 1.0213 | 1.0690 | 0.4770 | 25 |
| svd | 0.6946 | 0.7651 | 0.2851 | 25 |
| svd-mmr-diversification | 0.7338 | 0.8182 | 0.3330 | 20 |
| svd-topic-diversification | 0.6565 | 0.6876 | 0.3294 | 22 |

## GH Cosine (Homogeneidade por Features)

GH_COSINE mede a similaridade média (cosseno) entre itens da lista top-20.

| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |
|-----------|----------|------------|--------|----------|
| knn_item_item | 0.2511 | 0.2436 | 0.0294 | 206 |
| knn_item_item-mmr-diversification | 0.1647 | 0.1643 | 0.0177 | 71 |
| knn_item_item-topic-diversification | 0.2343 | 0.2360 | 0.0252 | 70 |
| knn_user_user | 0.2328 | 0.2306 | 0.0259 | 81 |
| knn_user_user-mmr-diversification | 0.1693 | 0.1660 | 0.0216 | 48 |
| knn_user_user-topic-diversification | 0.2371 | 0.2378 | 0.0316 | 99 |
| svd | 0.2385 | 0.2380 | 0.0280 | 66 |
| svd-mmr-diversification | 0.2523 | 0.2524 | 0.0263 | 56 |
| svd-topic-diversification | 0.2554 | 0.2508 | 0.0283 | 78 |

## GH Jaccard (Homogeneidade por Tópicos)

GH_JACCARD mede a similaridade Jaccard média entre tópicos dominantes dos itens.

| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |
|-----------|----------|------------|--------|----------|
| knn_item_item | 0.1217 | 0.1278 | 0.0166 | 206 |
| knn_item_item-mmr-diversification | 0.0917 | 0.0907 | 0.0144 | 71 |
| knn_item_item-topic-diversification | 0.1089 | 0.1074 | 0.0148 | 70 |
| knn_user_user | 0.1188 | 0.1155 | 0.0232 | 81 |
| knn_user_user-mmr-diversification | 0.0939 | 0.0960 | 0.0099 | 48 |
| knn_user_user-topic-diversification | 0.1134 | 0.1115 | 0.0179 | 99 |
| svd | 0.1092 | 0.1050 | 0.0168 | 66 |
| svd-mmr-diversification | 0.1105 | 0.1114 | 0.0134 | 56 |
| svd-topic-diversification | 0.1065 | 0.1056 | 0.0157 | 78 |

## Resumo Geral

- **Algoritmos avaliados**: 9
- **Usuários com métricas**: 262
- **Total de eval_pairs**: 3,537
- **Listas avaliadas (GH)**: 775

### Top 5 Algoritmos (Menor RMSE)

2. **svd**: RMSE = 0.7490 (683 pares)
5. **svd-topic-diversification**: RMSE = 0.7643 (221 pares)
6. **svd-mmr-diversification**: RMSE = 0.8404 (145 pares)
1. **knn_item_item**: RMSE = 0.9538 (524 pares)
9. **knn_item_item-mmr-diversification**: RMSE = 1.0247 (181 pares)

### Top 5 Algoritmos (Menor GH Cosine = Mais Diversos)

1. **knn_item_item-mmr-diversification**: GH = 0.1647 (71 listas)
2. **knn_user_user-mmr-diversification**: GH = 0.1693 (48 listas)
3. **knn_user_user**: GH = 0.2328 (81 listas)
4. **knn_item_item-topic-diversification**: GH = 0.2343 (70 listas)
5. **knn_user_user-topic-diversification**: GH = 0.2371 (99 listas)

## Interpretação

- **RMSE baixo**: Predições mais precisas
- **GH alto**: Lista mais homogênea (menos diversa)
- **GH baixo**: Lista mais diversificada
- **Taxa de exposição**: Percentual de avaliações no teste que foram recomendadas

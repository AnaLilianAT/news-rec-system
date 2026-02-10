# Relatório de Avaliação - Replay Temporal (ALL-BETWEEN)

**Data de geração**: 2026-02-09 11:27:14

## Exposição

- **Total de interações no teste**: 5,828
- **Interações expostas (na top-20)**: 3,533
- **Taxa de exposição**: 60.62%

## RMSE por Algoritmo

RMSE mede o erro de predição entre score_pred e rating_real para itens expostos.

| Algoritmo | RMSE | N Pares | Média Rating Real | Média Score Pred |
|-----------|------|---------|-------------------|------------------|
| svd | 0.7490 | 683 | 0.649 | 0.664 |
| svd-topic-diversification | 0.7508 | 220 | 0.695 | 0.709 |
| svd-mmr-diversification | 0.8404 | 145 | 0.524 | 0.672 |
| knn_item_item | 0.9538 | 524 | 0.508 | 0.698 |
| knn_item_item-mmr-diversification | 1.0247 | 181 | 0.486 | 0.962 |
| knn_item_item-topic-diversification | 1.0313 | 376 | 0.457 | 0.889 |
| knn_user_user-mmr-diversification | 1.0357 | 132 | 0.477 | 0.940 |
| knn_user_user | 1.0559 | 683 | 0.416 | 0.820 |
| knn_user_user-topic-diversification | 1.3942 | 589 | -0.017 | 0.946 |

### Estatísticas RMSE por Usuário

| Algoritmo | Média RMSE | Mediana RMSE | Std RMSE | N Usuários |
|-----------|------------|--------------|----------|------------|
| knn_item_item | 0.7872 | 0.8339 | 0.4143 | 33 |
| knn_item_item-mmr-diversification | 0.7516 | 0.9134 | 0.5415 | 22 |
| knn_item_item-topic-diversification | 1.0110 | 1.0690 | 0.3497 | 27 |
| knn_user_user | 0.8891 | 0.9570 | 0.4394 | 24 |
| knn_user_user-mmr-diversification | 0.8623 | 0.9082 | 0.5598 | 18 |
| knn_user_user-topic-diversification | 1.0231 | 1.0000 | 0.4838 | 25 |
| svd | 0.6946 | 0.7651 | 0.2851 | 25 |
| svd-mmr-diversification | 0.7338 | 0.8182 | 0.3330 | 20 |
| svd-topic-diversification | 0.6501 | 0.6876 | 0.3283 | 22 |

## GH Cosine (Homogeneidade por Features)

GH_COSINE mede a similaridade média (cosseno) entre itens da lista top-20.

| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |
|-----------|----------|------------|--------|----------|
| knn_item_item | 0.2511 | 0.2436 | 0.0294 | 206 |
| knn_item_item-mmr-diversification | 0.1647 | 0.1643 | 0.0177 | 71 |
| knn_item_item-topic-diversification | 0.2381 | 0.2366 | 0.0286 | 70 |
| knn_user_user | 0.2328 | 0.2306 | 0.0259 | 81 |
| knn_user_user-mmr-diversification | 0.1693 | 0.1660 | 0.0216 | 48 |
| knn_user_user-topic-diversification | 0.2354 | 0.2376 | 0.0304 | 99 |
| svd | 0.2385 | 0.2380 | 0.0280 | 66 |
| svd-mmr-diversification | 0.2523 | 0.2524 | 0.0263 | 56 |
| svd-topic-diversification | 0.2545 | 0.2495 | 0.0296 | 78 |

## GH Jaccard (Homogeneidade por Tópicos)

GH_JACCARD mede a similaridade Jaccard média entre tópicos dominantes dos itens.

| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |
|-----------|----------|------------|--------|----------|
| knn_item_item | 0.1217 | 0.1278 | 0.0166 | 206 |
| knn_item_item-mmr-diversification | 0.0917 | 0.0907 | 0.0144 | 71 |
| knn_item_item-topic-diversification | 0.1082 | 0.1064 | 0.0154 | 70 |
| knn_user_user | 0.1188 | 0.1155 | 0.0232 | 81 |
| knn_user_user-mmr-diversification | 0.0939 | 0.0960 | 0.0099 | 48 |
| knn_user_user-topic-diversification | 0.1140 | 0.1142 | 0.0193 | 99 |
| svd | 0.1092 | 0.1050 | 0.0168 | 66 |
| svd-mmr-diversification | 0.1105 | 0.1114 | 0.0134 | 56 |
| svd-topic-diversification | 0.1080 | 0.1077 | 0.0150 | 78 |

## Resumo Geral

- **Algoritmos avaliados**: 9
- **Usuários com métricas**: 262
- **Total de eval_pairs**: 3,533
- **Listas avaliadas (GH)**: 775

### Top 5 Algoritmos (Menor RMSE)

2. **svd**: RMSE = 0.7490 (683 pares)
5. **svd-topic-diversification**: RMSE = 0.7508 (220 pares)
6. **svd-mmr-diversification**: RMSE = 0.8404 (145 pares)
1. **knn_item_item**: RMSE = 0.9538 (524 pares)
9. **knn_item_item-mmr-diversification**: RMSE = 1.0247 (181 pares)

### Top 5 Algoritmos (Menor GH Cosine = Mais Diversos)

1. **knn_item_item-mmr-diversification**: GH = 0.1647 (71 listas)
2. **knn_user_user-mmr-diversification**: GH = 0.1693 (48 listas)
3. **knn_user_user**: GH = 0.2328 (81 listas)
4. **knn_user_user-topic-diversification**: GH = 0.2354 (99 listas)
5. **knn_item_item-topic-diversification**: GH = 0.2381 (70 listas)

## Interpretação

- **RMSE baixo**: Predições mais precisas
- **GH alto**: Lista mais homogênea (menos diversa)
- **GH baixo**: Lista mais diversificada
- **Taxa de exposição**: Percentual de avaliações no teste que foram recomendadas

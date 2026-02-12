# Embedding Dimension × Seed Sweep - Visualizações

Este diretório contém os gráficos gerados a partir do sweep de dimensão × seed.

## Conteúdo

Cada arquivo PNG representa um algoritmo de recomendação:

- **knni.png** - K-Nearest Neighbors (Item-based)
- **knni_mmr.png** - KNN Item + MMR (Maximal Marginal Relevance)
- **knni_td.png** - KNN Item + Topic Diversification
- **knnu.png** - K-Nearest Neighbors (User-based)
- **knnu_mmr.png** - KNN User + MMR
- **knnu_td.png** - KNN User + Topic Diversification
- **svd.png** - Singular Value Decomposition
- **svd_mmr.png** - SVD + MMR
- **svd_td.png** - SVD + Topic Diversification

## Estrutura dos Gráficos

Cada gráfico mostra:

### Eixo X
- Dimensão do embedding (d): valores testados (e.g., 13, 18, 23, 28)

### Eixo Y Esquerdo (Vermelho)
- **RMSE** (Root Mean Squared Error): quanto menor, melhor
- Linha sólida com círculos: média
- Área preenchida: intervalo de confiança 95% (IC95)

### Eixo Y Direito (Verde)
- **GH** (Gini-Simpson - diversidade): quanto maior, melhor
- Linha tracejada com quadrados: média
- Área preenchida: intervalo de confiança 95% (IC95)

## Interpretação

### Banda (área preenchida)
- **Banda estreita**: resultados consistentes entre diferentes seeds
- **Banda larga**: alta variabilidade, resultados dependem da seed

### Trade-off RMSE vs GH
- **RMSE baixo + GH alto**: ideal (acurácia + diversidade)
- **RMSE baixo + GH baixo**: preciso mas pouco diverso
- **RMSE alto + GH alto**: diverso mas impreciso

### Escolha de Dimensão
1. Identificar "knee point" na curva de RMSE
2. Verificar estabilidade do GH
3. Considerar largura das bandas
4. Balancear acurácia, diversidade e estabilidade

## Como Foram Gerados

```bash
# Gerar todos os gráficos com IC95
python -m src.experiments.plot_embedding_dim_seed_sweep

# Customizar tipo de banda
python -m src.experiments.plot_embedding_dim_seed_sweep --band std

# Gerar apenas alguns algoritmos
python -m src.experiments.plot_embedding_dim_seed_sweep \
  --algorithms svd "svd td" knnu
```

## Dados de Origem

Os gráficos são gerados a partir de:
- **Input**: `outputs/experiments/embedding_dim_seed_sweep_agg.parquet`
- **Colunas**: d, algorithm, rmse_mean, rmse_ci95_low/high, gh_mean, gh_ci95_low/high

## Especificações Técnicas

- **Formato**: PNG
- **Resolução**: 150 DPI
- **Tamanho**: 10×6 polegadas (1500×900 pixels)
- **Biblioteca**: matplotlib 3.x

## Exemplo de Análise

**Comparar SVD puro vs SVD com diversificação:**
1. Abrir `svd.png` e `svd_td.png`
2. Observar:
   - SVD puro: RMSE mais baixo, GH moderado
   - SVD TD: RMSE ligeiramente maior, GH mais alto
3. Conclusão: SVD TD oferece melhor diversidade com pequeno custo em acurácia

**Identificar dimensão ótima:**
1. Abrir plot do melhor algoritmo (geralmente `svd_td.png`)
2. Localizar ponto onde RMSE para de diminuir significativamente
3. Verificar se GH permanece estável
4. Exemplo: d=18 pode ser ótimo se RMSE(18) ≈ RMSE(28) mas d=18 < d=28

---

**Última atualização**: 2026-02-12
**Script**: `src/experiments/plot_embedding_dim_seed_sweep.py`

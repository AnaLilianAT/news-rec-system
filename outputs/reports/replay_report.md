# Relatório de Replay Temporal

**Data de geração:** 2026-02-06 11:19:04

---

## 1. Visão Geral dos Checkpoints

- **Total de checkpoints:** 775
- **Checkpoints com candidate pool:** 775 (100.0%)
- **Checkpoints sem candidate pool:** 0 (0.0%)
- **Usuários únicos:** 262

**Período temporal:**
- Início: 2019-05-07 17:34:13.279216+00:00
- Fim: 2019-08-26 20:46:49.124349+00:00
- Duração: 111 dias

## 2. Distribuição de Candidate Pools

### 2.1 Distribuição de candidate_size

| Candidate Size | Checkpoints | Percentual |
|----------------|-------------|------------|
| 100 itens |   444 |  57.3% ████████████████████████████ |
|  99 itens |    43 |   5.5% ██ |
|  98 itens |     5 |   0.6%  |
|  97 itens |   110 |  14.2% ███████ |
|  20 itens |   173 |  22.3% ███████████ |

**Estatísticas:**
- Média: 81.6 itens
- Mediana: 100 itens
- Mínimo: 20 itens
- Máximo: 100 itens

## 3. Checkpoints por Usuário

**Total de checkpoints por usuário:**
- Média: 3.0 checkpoints/usuário
- Mediana: 2 checkpoints/usuário
- Mínimo: 1 checkpoints
- Máximo: 111 checkpoints

**Checkpoints com pool por usuário:**
- Usuários com ≥1 checkpoint com pool: 262 (100.0%)
- Média: 3.0 checkpoints/usuário
- Mediana: 2 checkpoints/usuário

**Distribuição (top 10):**
- 111 checkpoints:   1 usuários (  0.4%)
-  25 checkpoints:   1 usuários (  0.4%)
-  23 checkpoints:   2 usuários (  0.8%)
-  12 checkpoints:   1 usuários (  0.4%)
-  10 checkpoints:   2 usuários (  0.8%)
-   8 checkpoints:   1 usuários (  0.4%)
-   7 checkpoints:   1 usuários (  0.4%)
-   6 checkpoints:   7 usuários (  2.7%)
-   5 checkpoints:  13 usuários (  5.0%)
-   4 checkpoints:  17 usuários (  6.5%)

## 4. Cobertura de Candidate Pools

- **Usuários com ≥1 candidate pool:** 262 (100.0%)
- **Usuários sem candidate pool:** 0 (0.0%)

## 5. Janelas de Avaliação (t_rec → t_next_rec)

**Duração das janelas:**
- Média: 28.2 dias
- Mediana: 0.0 dias
- Mínimo: 0.0 dias
- Máximo: 125.9 dias

**Distribuição de durações:**
- <1 dia      :   498 checkpoints ( 64.3%) ████████████████████████████████
- 1-7 dias    :     4 checkpoints (  0.5%) 
- 7-14 dias   :     3 checkpoints (  0.4%) 
- 14-30 dias  :    10 checkpoints (  1.3%) 
- 30-60 dias  :     5 checkpoints (  0.6%) 
- >60 dias    :   255 checkpoints ( 32.9%) ████████████████

## 6. Validações

**Ordenação temporal:** Todos os checkpoints têm t_next_rec > t_rec

✓ **Consistência de candidate_size:** Todos os tamanhos estão corretos

**Duplicatas:** Nenhuma duplicata encontrada

## 7. Recomendações para Uso

### 7.1 Regra de Treino/Teste

Para cada checkpoint (user_id, t_rec):

**TREINO:**
```python
train_data = df_interactions[df_interactions['rating_when'] < t_rec]
```
- Todas as interações de TODOS os usuários antes de t_rec
- Representa o conhecimento global do sistema naquele momento

**TESTE:**
```python
test_data = df_interactions[
    (df_interactions['user_id'] == user_id) &
    (df_interactions['rating_when'] > t_rec) &
    (df_interactions['rating_when'] <= t_next_rec)
]
```
- Avaliações do usuário no intervalo (t_rec, t_next_rec]

**EXPOSIÇÃO (para RMSE):**
```python
exposed_news = checkpoint['candidate_news_ids'][:20]  # top-20
observable_test = test_data[test_data['news_id'].isin(exposed_news)]
```
- Apenas avaliações de itens que estavam na lista recomendada

### 7.2 Filtros Recomendados

**Atenção:** 179 usuários (68.3%) têm <3 checkpoints.
- Considere filtrar usuários com ≥3 checkpoints para treino/validação consistente

### 7.3 Próximos Passos

1. **Implementar baseline de recomendação** (popularity, item-based CF)
2. **Calcular métricas por checkpoint** (RMSE, NDCG@20, etc.)
3. **Agregar métricas** (média, mediana) por usuário e global
4. **Analisar evolução temporal** das métricas ao longo do tempo

---

*Relatório gerado automaticamente pelo pipeline de replay temporal*

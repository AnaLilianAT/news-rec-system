# Relatório de Qualidade de Dados

**Data de geração:** 2026-02-06 11:18:55

---

## 1. Validação de Consistência

### 1.1 Cobertura de Notícias

- **Notícias no catálogo (news.csv):** 1,938
- **Notícias com features:** 1,938
- **Notícias com interações:** 1,833

**Análise de sobreposição:**
- Features sem correspondência no catálogo: 0
- Interações sem correspondência no catálogo: 0
- Interações sem features: 0

## 2. Distribuições de Dados

### 2.1 Interações (Avaliações Explícitas)

- **Total de interações:** 12,349
- **Usuários únicos:** 298
- **Notícias únicas:** 1,833
- **Período:** 2019-05-07 17:18:42.187700+00:00 a 2019-09-10 14:54:26.998826+00:00

**Distribuição de ratings:**
- Rating -1.0:  4,689 ( 38.0%) ██████████████████
- Rating 0.0:    150 (  1.2%) 
- Rating 1.0:  7,510 ( 60.8%) ██████████████████████████████

**Estatísticas de rating:**
- Média: 0.23
- Mediana: 1.00
- Desvio padrão: 0.97

**Interações por usuário:**
- Média: 41.4
- Mediana: 36
- Mínimo: 1
- Máximo: 470

### 2.2 Sessões de Recomendação

- **Total de sessões:** 1,139
- **Usuários únicos:** 262
- **Período:** 2019-05-07 17:34:13.279216+00:00 a 2019-08-26 20:46:49.124349+00:00

**Distribuição de list_size:**
- 100 itens:   460 sessões ( 40.4%) ████████████████████
-  99 itens:    46 sessões (  4.0%) ██
-  98 itens:     5 sessões (  0.4%) 
-  97 itens:   110 sessões (  9.7%) ████
-  89 itens:     1 sessões (  0.1%) 
-  20 itens:   517 sessões ( 45.4%) ██████████████████████

**Sessões por flag diversified:**
- diversifed=False: 775 (68.0%)
- diversifed=True: 364 (32.0%)

**Sessões por usuário:**
- Média: 4.3
- Mediana: 3
- Mínimo: 1
- Máximo: 111

### 2.3 Features e Topics

- **Notícias com features:** 1,938
- **Número de features:** 83
- **Notícias com topics:** 1,938

**Distribuição de topics por notícia:**
- Média: 2.50
- Mediana: 2
- Mínimo: 1
- Máximo: 10

**Frequência de cada topic:**
- Topic0: 376 notícias (19.4%)
- Topic1: 251 notícias (13.0%)
- Topic2: 339 notícias (17.5%)
- Topic3: 518 notícias (26.7%)
- Topic4: 268 notícias (13.8%)
- Topic5: 783 notícias (40.4%)
- Topic6: 157 notícias (8.1%)
- Topic7: 112 notícias (5.8%)
- Topic8: 284 notícias (14.7%)
- Topic9: 166 notícias (8.6%)
- Topic10: 309 notícias (15.9%)
- Topic11: 142 notícias (7.3%)
- Topic12: 536 notícias (27.7%)
- Topic13: 46 notícias (2.4%)
- Topic14: 377 notícias (19.5%)
- Topic15: 172 notícias (8.9%)

**Features de sentimento:**
- Polaridade: média=0.072, std=0.086, min=-0.500, max=0.526
- Subjetividade: média=0.391, std=0.100, min=0.000, max=1.000

## 3. Qualidade de Dados

### 3.1 Valores Ausentes

**df_interactions:** ✓ Nenhum valor ausente

**df_rec_sessions:** ✓ Nenhum valor ausente

**df_features:** ✓ Nenhum valor ausente

**df_topics:** ✓ Nenhum valor ausente

### 3.2 Duplicatas

- **df_interactions:** 0 duplicatas
- **df_rec_sessions:** 0 duplicatas
- **df_features:** 0 duplicatas
- **df_topics:** 0 duplicatas

## 4. Recomendações

✓ **Nenhum problema crítico detectado!**

Os dados estão consistentes e prontos para uso no pipeline de recomendação.

---

*Relatório gerado automaticamente pelo pipeline de validação*

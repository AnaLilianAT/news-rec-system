# Tabela 6.1 - GH por Sessão (Abordagem da Tese)

## Metodologia

1. **GH por sessão**: Para cada (user_id, t_rec):
   - R_session = itens expostos e avaliados naquela sessão
   - GH_session = (1/|R_session|) × Σ_{i<j} Jaccard(i,j)
   - Exigir |R_session| >= 2

2. **GH por usuário**: GH_user = média(GH_session)
   - Exigir pelo menos 1 sessão válida

3. **GH por algoritmo**: Agregação por usuário (média, mediana, DP, p-valor)

## Dados Processados

- **Sessões válidas**: 362
- **Usuários válidos**: 215
- **Algoritmos**: 9

## Distribuição de Sessões por Usuário

| Algoritmo | N usuários | Sessões Médias | Sessões Mediana | Sessões Max |
|-----------|-----------|---------------|----------------|-------------|
| knnu      |         24 |            1.9 |             1.0 |           7 |
| knnu td   |         25 |            2.3 |             1.0 |          21 |
| knnu mmr  |         18 |            1.6 |             1.0 |           6 |
| knni      |         33 |            1.5 |             1.0 |           5 |
| knni td   |         27 |            1.5 |             1.0 |           3 |
| knni mmr  |         22 |            1.7 |             1.5 |           3 |
| svd       |         25 |            1.7 |             1.0 |           5 |
| svd td    |         22 |            1.7 |             1.0 |           5 |
| svd mmr   |         19 |            1.3 |             1.0 |           4 |


## Tabela 6.1 - GH por Interação (Baseado em Sessões)

```
Algoritmo  Usuários Média Mediana Desvio Padrão p-valor
     knnu        24 0.692   0.740         0.422  0.1099
  knnu td        25 0.439   0.486         0.267  0.1075
 knnu mmr        18 0.103   0.096         0.086  0.1610
     knni        33 0.456   0.408         0.330  0.0558
  knni td        27 0.382   0.365         0.230  0.7483
 knni mmr        22 0.084   0.090         0.053  0.3501
      svd        25 0.821   0.910         0.287  0.2250
   svd td        22 0.274   0.260         0.148  0.5937
  svd mmr        19 0.168   0.153         0.121  0.1777
```

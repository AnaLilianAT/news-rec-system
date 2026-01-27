# Comparação de Definições de R_session - Tabela 6.1

## Objetivo

Comparar 4 definições de R_session para identificar qual reproduz melhor
os valores e a baixa variância da Tabela 6.1 da tese (intervalo ~0.72-0.76).

## Definições Testadas

1. **MODE 1** (atual): R_session = itens expostos + avaliados naquela sessão
2. **MODE 2** (exposição pura): R_session = itens expostos na top-20
3. **MODE 3a** (K interagidos): R_session = primeiros 5 itens avaliados (na top-20)
4. **MODE 3b** (K interagidos): R_session = primeiros 10 itens avaliados (na top-20)

## Estatísticas de Sessões

- **MODE 1**: 362 sessões válidas, 215 usuários
- **MODE 2**: 381 sessões válidas, 220 usuários
- **MODE 3a** (K=5): 362 sessões válidas, 215 usuários
- **MODE 3b** (K=10): 362 sessões válidas, 215 usuários

## Distribuição de |R_session| por Modo

| Mode | n sessões | |R| médio | |R| mediano | |R| min | |R| max |
|------|----------|----------|------------|---------|---------|
| MODE1    |       362 |       9.2 |         8.0 |        2 |       20 |
| MODE2    |       381 |      20.0 |        20.0 |       20 |       20 |
| MODE3_K5 |       362 |       4.5 |         5.0 |        2 |        5 |
| MODE3_K10 |       362 |       7.1 |         8.0 |        2 |       10 |


## Tabela 6.1 - MODE 1 (Expostos + Avaliados)

```
Algoritmo  Usuários Média Mediana Desvio Padrão   Min   Max p-valor
     knnu        24 0.692   0.740         0.422 0.000 1.486  0.1099
  knnu td        25 0.457   0.490         0.289 0.000 1.134  0.4332
 knnu mmr        19 0.107   0.091         0.103 0.000 0.319  0.0345
     knni        33 0.456   0.408         0.330 0.000 1.332  0.0558
  knni td        27 0.378   0.365         0.221 0.000 0.905  0.8947
 knni mmr        21 0.067   0.067         0.046 0.000 0.156  0.5479
      svd        25 0.821   0.910         0.287 0.208 1.291  0.2250
   svd td        22 0.272   0.253         0.147 0.061 0.620  0.5208
  svd mmr        19 0.161   0.167         0.108 0.000 0.384  0.5728
```

## Tabela 6.1 - MODE 2 (Exposição Pura)

```
Algoritmo  Usuários Média Mediana Desvio Padrão   Min   Max p-valor
     knnu        24 1.069   1.074         0.169 0.777 1.486  0.3943
  knnu td        25 1.067   1.085         0.192 0.662 1.574  0.5633
 knnu mmr        19 0.823   0.807         0.129 0.669 1.036  0.0398
     knni        33 0.999   0.983         0.147 0.710 1.332  0.9252
  knni td        28 0.999   0.999         0.114 0.819 1.258  0.3976
 knni mmr        22 0.757   0.754         0.095 0.596 0.936  0.5699
      svd        26 1.070   1.031         0.139 0.894 1.398  0.0032
   svd td        23 1.018   0.992         0.118 0.816 1.209  0.4015
  svd mmr        20 0.938   0.935         0.061 0.815 1.064  0.4335
```

## Tabela 6.1 - MODE 3a (K=5 Interagidos)

```
Algoritmo  Usuários Média Mediana Desvio Padrão   Min   Max p-valor
     knnu        24 0.183   0.171         0.087 0.000 0.362  0.3740
  knnu td        25 0.173   0.153         0.113 0.000 0.496  0.3035
 knnu mmr        19 0.086   0.091         0.077 0.000 0.257  0.0961
     knni        33 0.172   0.157         0.104 0.000 0.530  0.0196
  knni td        27 0.187   0.158         0.108 0.000 0.420  0.2411
 knni mmr        21 0.066   0.054         0.047 0.000 0.139  0.0933
      svd        25 0.236   0.215         0.095 0.073 0.450  0.3954
   svd td        22 0.197   0.168         0.110 0.061 0.500  0.0685
  svd mmr        19 0.101   0.083         0.081 0.000 0.240  0.0695
```

## Tabela 6.1 - MODE 3b (K=10 Interagidos)

```
Algoritmo  Usuários Média Mediana Desvio Padrão   Min   Max p-valor
     knnu        24 0.398   0.459         0.201 0.000 0.733  0.1932
  knnu td        25 0.371   0.392         0.228 0.000 0.805  0.5577
 knnu mmr        19 0.107   0.091         0.103 0.000 0.319  0.0345
     knni        33 0.308   0.327         0.158 0.000 0.598  0.2086
  knni td        27 0.331   0.346         0.176 0.000 0.760  0.8179
 knni mmr        21 0.067   0.067         0.046 0.000 0.156  0.5479
      svd        25 0.463   0.475         0.117 0.208 0.683  0.6147
   svd td        22 0.272   0.253         0.147 0.061 0.620  0.5208
  svd mmr        19 0.161   0.167         0.108 0.000 0.384  0.5728
```

## Comparação de Médias Globais

| Mode | Média Global | DP Médio | Min Global | Max Global | Distância de 0.74 |
|------|-------------|----------|------------|------------|-------------------|
| MODE1    |        0.379 ❌ |     0.217 |       0.000 |       1.486 |              0.361 |
| MODE2    |        0.971 ❌ |     0.130 |       0.596 |       1.574 |              0.231 |
| MODE3_K5 |        0.156 ❌ |     0.091 |       0.000 |       0.530 |              0.584 |
| MODE3_K10 |        0.275 ❌ |     0.143 |       0.000 |       0.805 |              0.465 |

**Intervalo da tese**: [0.72, 0.76]

## Análise de Variabilidade

Desvio padrão médio por modo:

- **MODE1**: 0.217
- **MODE2**: 0.130
- **MODE3_K5**: 0.091
- **MODE3_K10**: 0.143


## Recomendação

✅ **Melhor modo**: MODE2
- Média global: 0.971 (distância: 0.231)
- DP médio: 0.130

**Razão**: Este modo apresenta a menor distância do intervalo da tese
e mantém variabilidade controlada.

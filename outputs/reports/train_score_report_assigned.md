# Relatório de Treino e Score (Assigned)

**Data de geração:** 2026-02-06 11:19:41

---

## 1. Resumo do Processamento

- **Total de checkpoints processados:** 775
- **Timestamps únicos (t_rec):** 775
- **Checkpoints pulados:** 0
- **Taxa de sucesso:** 100.0%

## 2. Treinamentos Executados

- **Total de treinamentos:** 775
- **Tamanho médio do treino:** 5528 interações

**Treinamentos por algoritmo:**
- **KNNU:** 228 treinos (29.4%)
- **KNNI:** 347 treinos (44.8%)
- **SVD:** 200 treinos (25.8%)

## 3. Otimização

Este pipeline implementa otimização inteligente:

- Agrupa checkpoints por `t_rec`
- Treina apenas algoritmos necessários por `t_rec`
- Cada usuário recebe predições do algoritmo atribuído
- Evita treinos desnecessários

**Economia de treinamentos:**
- Máximo possível (sem otimização): 2,325 treinos
- Executados (com otimização): 775 treinos
- **Economia: 1,550 treinos (66.7%)**

---

*Relatório gerado automaticamente pelo pipeline de treino/score*

# Relatório de Treino e Score (Assigned)

**Data de geração:** 2026-01-22 10:25:09

---

## 1. Resumo do Processamento

- **Total de checkpoints processados:** 602
- **Timestamps únicos (t_rec):** 602
- **Checkpoints pulados:** 0
- **Taxa de sucesso:** 100.0%

## 2. Treinamentos Executados

- **Total de treinamentos:** 602
- **Tamanho médio do treino:** 4888 interações

**Treinamentos por algoritmo:**
- **KNNU:** 159 treinos (26.4%)
- **KNNI:** 302 treinos (50.2%)
- **SVD:** 141 treinos (23.4%)

## 3. Otimização

Este pipeline implementa otimização inteligente:

- ✅ Agrupa checkpoints por `t_rec`
- ✅ Treina apenas algoritmos necessários por `t_rec`
- ✅ Cada usuário recebe predições do algoritmo atribuído
- ✅ Evita treinos desnecessários

**Economia de treinamentos:**
- Máximo possível (sem otimização): 1,806 treinos
- Executados (com otimização): 602 treinos
- **Economia: 1,204 treinos (66.7%)**

---

*Relatório gerado automaticamente pelo pipeline de treino/score*

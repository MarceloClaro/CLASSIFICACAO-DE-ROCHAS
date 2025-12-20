# Guia de Análise Estatística Avançada

## 📊 Visão Geral

O módulo de análise estatística fornece uma avaliação abrangente das predições do modelo de classificação de rochas, incluindo 10 componentes principais conforme especificado nos requisitos do projeto.

## 🎯 Componentes Implementados

### 1. Intervalos de Confiança (Margem de Variação Possível)
- Calcula intervalos de confiança de 95% para as principais classes
- Usa distribuição t de Student para estimativas precisas
- Fornece margem de erro para cada predição

**Exemplo de Saída:**
```
Basalto:
  - Probabilidade Média: 85.3%
  - Intervalo: [82.1%, 88.5%]
  - Margem de Erro: ±3.2%
```

### 2. Testes de Significância Estatística
- Testes t pareados para comparar probabilidades
- Verifica se diferenças são estatisticamente significativas (p < 0.05)
- Distingue diferenças reais de variações aleatórias

**Exemplo de Saída:**
```
Basalto vs Granito:
  - Diferença de Probabilidade: 35.2%
  - p-valor: 0.0001
  - Resultado: Diferença significativa
```

### 3. Validação Bootstrap
- Executa múltiplas predições com dropout ativado
- Configurável de 50 a 500 iterações
- Quantifica variabilidade da predição

**Configurações:**
- Padrão: 100 iterações
- Rápido: 50 iterações
- Preciso: 200-500 iterações

### 4. Lista de Alternativas Principais
- Ordenadas por probabilidade decrescente
- Até 5 principais diagnósticos diferenciais
- Inclui nível de confiança interpretado

**Níveis de Confiança:**
- Muito Alto: ≥ 90%
- Alto: 75-90%
- Moderado: 50-75%
- Baixo: 30-50%
- Muito Baixo: < 30%

### 5. Critérios de Exclusão
- Remove automaticamente opções improváveis
- Threshold padrão: < 5% de probabilidade
- Fornece razão para exclusão

**Exemplo:**
```
Classes Excluídas: 8
Classes Consideradas: 4
Opções Descartadas:
  - Quartzito: Probabilidade muito baixa (< 5%)
  - Arenito: Probabilidade muito baixa (< 5%)
```

### 6. Identificação de Características Distintivas
- Analisa mapas de ativação Grad-CAM
- Identifica regiões de alta importância
- Classifica padrão de ativação

**Padrões Identificados:**
- Dispersas: > 30% da imagem (múltiplas regiões)
- Moderadamente focadas: 15-30%
- Altamente focadas: 5-15% (região específica)
- Muito concentradas: < 5% (atenção localizada)

### 7. Identificação de Fontes de Incerteza
- **Variação do Modelo (Aleatória):** Variabilidade entre predições
- **Ambiguidade da Predição (Epistêmica):** Entropia da distribuição
- **Incerteza Total:** Combinação ponderada das duas

**Níveis de Incerteza:**
- Muito Baixa: < 0.1
- Baixa: 0.1-0.2
- Moderada: 0.2-0.4
- Alta: 0.4-0.6
- Muito Alta: > 0.6

### 8. Avaliação de Impacto de Erro
- Calcula probabilidade de erro (1 - confiança)
- Avalia nível de risco da categoria
- Gera recomendações específicas

**Níveis de Risco:**
- Low: Baixo impacto
- Medium: Impacto moderado
- High: Alto impacto
- Critical: Impacto crítico

**Recomendações Automáticas:**
- ⚠️ Alta probabilidade de erro em categoria de alto risco → Validação adicional
- ⚠️ Confiança baixa → Análise complementar
- ℹ️ Confiança moderada → Monitoramento
- ✅ Confiança adequada → Resultado confiável

### 9. Margem de Segurança
- Define thresholds configuráveis
- Mínimo aceitável (padrão: 70%)
- Alvo desejado (padrão: 90%)
- Calcula distância até cada threshold

**Status:**
- 🔴 Abaixo do mínimo → Não recomendado
- 🟡 Margem crítica → Usar com cautela
- 🟢 Margem aceitável → Dentro dos parâmetros
- 🟢 Margem adequada → Alvo atingido

### 10. Impacto Prático e Consequências
- Ambiguidade diagnóstica
- Ação recomendada
- Nível de prioridade
- Necessidade de especialista

**Ações Recomendadas:**
1. Proceder com diagnóstico primário (alta confiança)
2. Considerar com monitoramento (confiança moderada)
3. Investigar diferenciais (múltiplas possibilidades)
4. Análise complementar necessária (baixa confiança)

## 🚀 Como Usar

### Passo 1: Treinar o Modelo
```bash
streamlit run app5.py
```

1. Upload do dataset de rochas em formato ZIP
2. Configure parâmetros de treinamento
3. Aguarde conclusão do treinamento

### Passo 2: Avaliar Imagem
1. Selecione "Sim" para "Deseja avaliar uma imagem?"
2. Faça upload de uma imagem de rocha
3. Visualize predição básica e Grad-CAM

### Passo 3: Ativar Análise Estatística
1. Marque checkbox "Ativar Análise Estatística Completa"
2. (Opcional) Configure parâmetros:
   - Número de iterações bootstrap (50-500)
   - Confiança mínima aceitável (50-90%)
   - Confiança alvo (70-99%)

### Passo 4: Executar Análise
1. Clique em "🔬 Executar Análise Estatística Completa"
2. Aguarde processamento (10-60 segundos dependendo das iterações)
3. Revise relatório completo com 10 seções

### Passo 5: Interpretar Resultados

#### Visualizações Disponíveis:
1. **Distribuição Bootstrap:** Histograma de probabilidades para top 3 classes
2. **Intervalos de Confiança:** Barras horizontais com margens de erro
3. **Decomposição de Incerteza:** Métricas de variação e ambiguidade
4. **Margem de Segurança:** Visualização de thresholds

## 📈 Exemplo de Relatório Completo

```markdown
# 📊 Relatório de Análise Estatística Completa

## 1️⃣ Resultado Principal
**Classe Predita:** Basalto
**Confiança:** 87.5%
**Confiança Bootstrap (média):** 85.3%
**Incerteza:** 0.0823

## 2️⃣ Intervalos de Confiança (95%)
**Basalto:**
  - Probabilidade Média: 85.3%
  - Intervalo: [82.1%, 88.5%]
  - Margem de Erro: ±3.2%

**Granito:**
  - Probabilidade Média: 10.2%
  - Intervalo: [8.5%, 11.9%]
  - Margem de Erro: ±1.7%

## 3️⃣ Testes de Significância Estatística
**Basalto vs Granito:**
  - Diferença de Probabilidade: 75.1%
  - p-valor: 0.0000
  - Resultado: Diferença significativa

## 4️⃣ Validação Bootstrap
Resultado validado através de 100 análises independentes.

**Estatísticas de Variação:**
  - Basalto: Desvio padrão = 0.0312
  - Granito: Desvio padrão = 0.0187

## 5️⃣ Principais Alternativas
1. **Basalto**
   - Probabilidade: 85.3%
   - Nível de Confiança: Muito Alto

2. **Granito**
   - Probabilidade: 10.2%
   - Nível de Confiança: Baixo

## 6️⃣ Critérios de Exclusão
**Classes Excluídas:** 8
**Classes Consideradas:** 4

## 7️⃣ Fontes de Incerteza
**Nível de Incerteza Total:** Baixa

**Fontes:**
  - Variação do Modelo: 0.0312
  - Ambiguidade da Predição: 0.1234

## 8️⃣ Impacto de Possível Erro
**Probabilidade de Erro:** 14.7%
**Nível de Risco:** MEDIUM
**Recomendação:** ✅ Confiança adequada. Resultado confiável.

## 9️⃣ Margem de Segurança
**Confiança Atual:** 85.3%
**Mínimo Aceitável:** 70.0%
**Alvo Desejado:** 90.0%
**Status:** SAFE
**Interpretação:** 🟢 MARGEM ACEITÁVEL - Dentro dos parâmetros seguros

## 🔟 Impacto Prático
**Diagnóstico Primário:** Basalto
**Ação Recomendada:** Proceder com diagnóstico primário
**Nível de Prioridade:** Normal
**Requer Especialista:** Não
```

## 🔧 Configurações Avançadas

### Ajuste de Parâmetros Bootstrap
- **50 iterações:** Análise rápida (5-10 seg)
- **100 iterações:** Balanceado (15-20 seg) ⭐ Recomendado
- **200 iterações:** Preciso (30-40 seg)
- **500 iterações:** Muito preciso (60-90 seg)

### Ajuste de Thresholds
- **Confiança Mínima:** Para aplicações críticas, aumente para 80-85%
- **Confiança Alvo:** Para aplicações de pesquisa, pode ser 85-90%
- **Threshold de Exclusão:** Padrão 5%, ajuste para 10% se houver muitas classes

## 📚 Referências Científicas

Este módulo implementa metodologias baseadas em:

1. **Bootstrap Validation:**
   - Efron, B. (1979). "Bootstrap methods: another look at the jackknife"
   - DiCiccio, T. J., & Efron, B. (1996). "Bootstrap confidence intervals"

2. **Uncertainty Quantification:**
   - Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian approximation"
   - Kendall, A., & Gal, Y. (2017). "What uncertainties do we need in Bayesian deep learning?"

3. **Statistical Significance:**
   - Student (1908). "The probable error of a mean"
   - Welch, B. L. (1947). "The generalization of 'Student's' problem"

4. **Clinical Decision Support:**
   - Mongan, J., et al. (2020). "Checklist for AI in Medical Imaging (CLAIM)"
   - Liu, X., et al. (2019). "Reporting guidelines for clinical trials with AI"

## 🐛 Troubleshooting

### Problema: "ModuleNotFoundError"
**Solução:** Certifique-se de que todas as dependências estão instaladas:
```bash
pip install -r requirements.txt
```

### Problema: Bootstrap muito lento
**Solução:** Reduza o número de iterações para 50 ou use GPU:
```python
# Verificar se GPU está disponível
import torch
print(torch.cuda.is_available())
```

### Problema: Memória insuficiente
**Solução:** 
- Reduza iterações bootstrap para 50
- Use batch size menor no treinamento
- Feche outras aplicações

### Problema: Resultados inconsistentes
**Solução:**
- Aumente iterações bootstrap para 200+
- Verifique qualidade da imagem de entrada
- Considere retreinar modelo com mais dados

## 💡 Dicas de Uso

1. **Para Análise Rápida:** Use 50 iterações bootstrap
2. **Para Publicações:** Use 200+ iterações e documente parâmetros
3. **Para Aplicações Críticas:** Configure thresholds mais altos (85-90%)
4. **Para Pesquisa Exploratória:** Mantenha thresholds padrão (70-90%)

## 📞 Suporte

Para questões ou sugestões sobre a análise estatística:

- **Email:** marceloclaro@gmail.com
- **WhatsApp:** +55 88 98158-7145
- **Instagram:** [@marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
- **DOI:** https://doi.org/10.5281/zenodo.13910277

---

**Projeto Geomaker + IA**  
*Laboratório de Educação e Inteligência Artificial*

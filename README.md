# CLASSIFICAÇÃO DE IMAGENS COM DEEP LEARNING

## 🎯 Visão Geral

Sistema avançado de classificação de imagens utilizando Deep Learning com análise de eficiência e desempenho para qualidade científica **Qualis A1**.

### 🌟 Características Principais

- 🧠 **Múltiplos Modelos**: ResNet18, ResNet50, DenseNet121
- 📊 **Análise de Performance Completa**: Métricas detalhadas de classificação e eficiência
- ⚡ **Otimização de Recursos**: Monitoramento de tempo e memória
- 📈 **Visualizações Científicas**: Gráficos de alta qualidade para publicações
- 💾 **Exportação de Resultados**: Relatórios em CSV para análise posterior
- 🎓 **Qualidade Acadêmica**: Metodologia rigorosa alinhada com Qualis A1

### ✨ Novas Funcionalidades (v3.0)

- 🎨 **Pré-processamento Avançado**: Melhoria automática de qualidade das imagens
- 🔄 **Técnicas de Augmentation**: None, Standard, Mixup, CutMix
- 📉 **LR Schedulers**: None, CosineAnnealingLR, OneCycleLR
- ⚙️ **Múltiplos Otimizadores**: Adam, AdamW, SGD, Ranger, Lion
- 🎯 **Regularização L1 e L2**: Controle fino de overfitting
- 🔍 **4 Tipos de Grad-CAM**: GradCAM, GradCAM++, SmoothGradCAM++, LayerCAM

### 🚀 **NOVO! Funcionalidades v5.0**

- 🌐 **Visualização 3D Interativa**: PCA e Grad-CAM em 3D com Plotly
- 🤖 **Chat com IA**: Análise diagnóstica PhD-level com Gemini e Groq
- 🧬 **Algoritmos Genéticos**: Interpretação multi-angular automatizada
- 📚 **Referências Acadêmicas**: Integração com PubMed, arXiv e Google Scholar
- 🔬 **Análise Forense**: Diagnóstico detalhado como residência médica/perícia
- 📋 **Relatórios Automáticos**: Geração de laudos técnicos completos

👉 **[Ver documentação completa v5.0](FEATURES_V5.md)**

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- CUDA (opcional, para GPU)

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

## 📱 Como Usar

### 1. Aplicação Avançada v5.0 (app5.py) 🆕

```bash
streamlit run app5.py
```

**Funcionalidades Completas v5.0**:
- ✅ Todas as funcionalidades do app3.py
- ✅ **Visualização 3D Interativa** (PCA e Grad-CAM)
- ✅ **Chat com IA** (Gemini e Groq para análise diagnóstica)
- ✅ **Algoritmos Genéticos** (interpretação multi-angular)
- ✅ **Referências Acadêmicas** (PubMed, arXiv)
- ✅ **Análise PhD-level** (diagnóstico forense/médico)
- ✅ **Relatórios Automáticos** (laudos técnicos completos)

**Requer API Keys** (opcionais):
- Google Gemini API: https://ai.google.dev/
- Groq API: https://console.groq.com/

### 2. Aplicação Principal (app3.py / app4.py)

```bash
streamlit run app3.py
# ou
streamlit run app4.py
```

**Funcionalidades**:
- ✅ Treinamento de modelos com aumento de dados avançado
- ✅ Múltiplas técnicas de augmentation (Mixup, CutMix)
- ✅ 5 otimizadores diferentes para experimentação
- ✅ Learning Rate Schedulers para melhor convergência
- ✅ Regularização L1 e L2 configuráveis
- ✅ Análise automática de eficiência e desempenho
- ✅ Visualização de métricas detalhadas
- ✅ Exportação de relatórios
- ✅ 4 variantes de Grad-CAM para interpretabilidade
- ✅ Clustering não supervisionado

### 3. Guia de Comparação de Modelos

```bash
streamlit run model_comparison_guide.py
```

**Conteúdo**:
- 📚 Documentação completa de métricas
- 🔬 Guia de seleção de modelos
- 💡 Dicas de otimização
- 📊 Exemplos de análises comparativas

## 📊 Métricas Implementadas

### Classificação
- **Acurácia Global**
- **Precisão, Recall e F1-Score** (por classe e macro/weighted)
- **Matriz de Confusão Normalizada**
- **AUC-ROC** (multiclasse)
- **Curvas ROC** (quando aplicável)

### Eficiência
- **Tempo de Inferência** (média e desvio padrão)
- **Throughput** (amostras/segundo)
- **Uso de Memória** (modelo, sistema, GPU)
- **Score de Eficiência Composto** (0-1)

## 🎓 Qualidade Científica (Qualis A1)

### Elementos Implementados

✅ **Metodologia Rigorosa**
- Métricas padronizadas internacionalmente
- Validação em conjunto de teste independente
- Seed fixo para reprodutibilidade

✅ **Análise Estatística**
- Métricas detalhadas por classe
- Intervalos de confiança (desvio padrão)
- Análise de erros

✅ **Visualizações Científicas**
- Gráficos de alta qualidade
- Comparações entre modelos
- Curvas de aprendizado

✅ **Documentação Completa**
- Código bem comentado
- Explicações teóricas
- Referências bibliográficas

✅ **Exportação de Resultados**
- Relatórios em CSV
- Dados prontos para LaTeX/Excel
- Gráficos em alta resolução

## 📖 Documentação Adicional

- [📊 Análise de Performance](PERFORMANCE_ANALYSIS.md) - Documentação completa do sistema de análise
- [🔬 Guia de Comparação](model_comparison_guide.py) - Interface interativa para comparação de modelos
- [✨ Melhorias no Treinamento](TRAINING_IMPROVEMENTS.md) - Documentação detalhada das novas funcionalidades v3.0
- [🚀 **NOVO! Funcionalidades v5.0**](FEATURES_V5.md) - Documentação completa da versão 5.0

## 🛠️ Estrutura do Projeto

```
CLASSIFICACAO-DE-ROCHAS/
├── app.py                      # Aplicação básica
├── app2.py                     # Aplicação intermediária
├── app3.py                     # Aplicação completa com análise
├── app4.py                     # Variante app3
├── app5.py                     # 🆕 Aplicação v5.0 com IA e 3D
├── performance_analyzer.py     # Módulo de análise de performance
├── model_comparison_guide.py   # Guia interativo de comparação
├── visualization_3d.py         # 🆕 Módulo de visualização 3D
├── ai_chat_module.py           # 🆕 Chat com IA (Gemini/Groq)
├── academic_references.py      # 🆕 Busca de referências acadêmicas
├── genetic_interpreter.py      # 🆕 Algoritmos genéticos
├── PERFORMANCE_ANALYSIS.md     # Documentação técnica
├── FEATURES_V5.md              # 🆕 Documentação v5.0
├── requirements.txt            # Dependências (atualizado)
└── dataset/                    # Dados de treinamento
```

## 🎯 Casos de Uso

### 🏥 Diagnóstico Médico
- Classificação de lesões de pele
- Detecção de tumores em imagens médicas
- Análise de retina

**Modelo Recomendado**: DenseNet121 ou ResNet50 (acurácia prioritária)

### 🏭 Controle de Qualidade Industrial
- Detecção de defeitos em produtos
- Classificação de matérias-primas
- Inspeção automatizada

**Modelo Recomendado**: ResNet50 (balanceamento tempo/acurácia)

### 🌍 Sensoriamento Remoto
- Classificação de uso do solo
- Detecção de mudanças
- Análise de cobertura vegetal

**Modelo Recomendado**: ResNet50 ou DenseNet121

### 📱 Aplicações Mobile
- Reconhecimento de objetos
- Realidade aumentada
- Assistentes visuais

**Modelo Recomendado**: ResNet18 (velocidade e leveza)

## 📈 Exemplo de Resultados

```
=== Relatório de Performance ===

Métricas de Classificação:
  Acurácia: 0.9450
  Precisão Macro: 0.9420
  Recall Macro: 0.9380
  F1-Score Macro: 0.9400
  AUC-ROC: 0.9520

Métricas de Eficiência:
  Tempo de Inferência: 18.50 ms
  Throughput: 54.05 amostras/s
  Memória do Modelo: 45.23 MB
  Memória GPU: 512.00 MB

Score de Eficiência Geral: 0.8650
✅ Excelente - Qualidade Qualis A1
```

## 🔧 Configurações Avançadas

### Hiperparâmetros Principais

- **Número de Épocas**: 1-500 (padrão: 200)
- **Taxa de Aprendizagem**: 0.0001-0.1 (padrão: 0.0001)
- **Batch Size**: 4-64 (padrão: 16)
- **Fine-Tuning**: Habilitar para ajustar todas as camadas
- **L1 Regularization**: 0.0-0.01 (padrão: 0.0) - Promove esparsidade
- **L2 Regularization**: 0.0-0.1 (padrão: 0.01) - Weight decay
- **Early Stopping Patience**: 1-10 (padrão: 3)

### Técnicas de Aumento de Dados (Novas!)

- ✅ **None**: Sem augmentation, apenas normalização
- ✅ **Standard**: Rotação, flip, color jitter, crop, affine
- ✅ **Mixup**: Mistura linear de imagens e labels
- ✅ **CutMix**: Recorta e cola regiões entre imagens

### Otimizadores Disponíveis (Novos!)

- ✅ **Adam**: Adaptativo padrão (recomendado)
- ✅ **AdamW**: Adam com weight decay melhorado
- ✅ **SGD**: Gradiente descendente com momentum Nesterov
- ✅ **Ranger**: Lookahead + RAdam (avançado)
- ✅ **Lion**: Otimizador eficiente do Google (2023)

### Learning Rate Schedulers (Novos!)

- ✅ **None**: LR constante
- ✅ **CosineAnnealingLR**: Redução suave em formato cosseno
- ✅ **OneCycleLR**: Super-convergência (aumenta depois reduz)

### Variantes de Grad-CAM (Expandido!)

- ✅ **GradCAM**: Implementação básica
- ✅ **GradCAM++**: Pesos melhorados
- ✅ **SmoothGradCAM++**: Mapas suavizados
- ✅ **LayerCAM**: Análise por camada

### Técnicas de Regularização

- ✅ **Data Augmentation**: Standard, Mixup, CutMix
- ✅ **Dropout**: p=0.5 na camada final
- ✅ **L1 Regularization**: Promove esparsidade nos pesos
- ✅ **L2 Regularization**: Weight decay para pesos menores
- ✅ **Early Stopping**: Para evitar overfitting
- ✅ **Weighted Loss**: Para classes desbalanceadas

## 📚 Referências Bibliográficas

As técnicas e métricas implementadas são baseadas em:

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Huang, G., et al. (2017). "Densely Connected Convolutional Networks"
3. Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC"
4. Strubell, E., et al. (2019). "Energy and Policy Considerations for Deep Learning"
5. Zhang, H., et al. (2018). "mixup: Beyond Empirical Risk Minimization"
6. Yun, S., et al. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers"
7. Smith, L. N., & Topin, N. (2019). "Super-Convergence: Very Fast Training of Neural Networks"
8. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization"
9. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks"
10. Chattopadhay, A., et al. (2018). "Grad-CAM++: Generalized Gradient-Based Visual Explanations"

## 👥 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença especificada no arquivo LICENSE.

## 📧 Contato

**Projeto Geomaker + IA**
- Email: marceloclaro@gmail.com
- WhatsApp: (88) 981587145
- Instagram: [@marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
- DOI: https://doi.org/10.5281/zenodo.13910277

## 🙏 Agradecimentos

Desenvolvido no contexto do Laboratório de Educação e Inteligência Artificial - Geomaker.

> "A melhor forma de prever o futuro é inventá-lo." - Alan Kay

---

**Última atualização**: 2024  
**Versão**: 5.0 (com visualização 3D, IA, e algoritmos genéticos)  
**Versões anteriores**: v3.0 (melhorias de treinamento), v4.0 (otimizações)

CLASSIFICAÇÃO DE PELE: https://g.co/gemini/share/6c65af20056b

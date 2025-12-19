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

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- CUDA (opcional, para GPU)

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

## 📱 Como Usar

### 1. Aplicação Principal (app3.py)

```bash
streamlit run app3.py
```

**Funcionalidades**:
- ✅ Treinamento de modelos com aumento de dados
- ✅ Análise automática de eficiência e desempenho
- ✅ Visualização de métricas detalhadas
- ✅ Exportação de relatórios
- ✅ Grad-CAM para interpretabilidade
- ✅ Clustering não supervisionado

### 2. Guia de Comparação de Modelos

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

## 🛠️ Estrutura do Projeto

```
CLASSIFICACAO-DE-ROCHAS/
├── app.py                      # Aplicação básica
├── app2.py                     # Aplicação intermediária
├── app3.py                     # Aplicação completa com análise
├── app4.py                     # Variante app3
├── performance_analyzer.py     # Módulo de análise de performance
├── model_comparison_guide.py   # Guia interativo de comparação
├── PERFORMANCE_ANALYSIS.md     # Documentação técnica
├── requirements.txt            # Dependências
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
- **L2 Regularization**: 0.0-0.1 (padrão: 0.01)
- **Early Stopping Patience**: 1-10 (padrão: 3)

### Técnicas de Regularização

- ✅ **Data Augmentation**: Rotação, flip, color jitter
- ✅ **Dropout**: p=0.5 na camada final
- ✅ **L2 Regularization**: Weight decay
- ✅ **Early Stopping**: Para evitar overfitting
- ✅ **Weighted Loss**: Para classes desbalanceadas

## 📚 Referências Bibliográficas

As técnicas e métricas implementadas são baseadas em:

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Huang, G., et al. (2017). "Densely Connected Convolutional Networks"
3. Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC"
4. Strubell, E., et al. (2019). "Energy and Policy Considerations for Deep Learning"

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
**Versão**: 2.0 (com análise de performance)

CLASSIFICAÇÃO DE PELE: https://g.co/gemini/share/6c65af20056b

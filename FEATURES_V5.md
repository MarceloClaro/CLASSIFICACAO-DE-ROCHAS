# Novas Funcionalidades - Versão 5.0

## 🎯 Visão Geral

A versão 5.0 do sistema de classificação de imagens introduz recursos avançados de visualização 3D, análise diagnóstica com IA, interpretação multi-angular com algoritmos genéticos e integração com referências acadêmicas.

---

## ✨ Novas Funcionalidades

### 1. 🌐 Visualização 3D Interativa

#### Visualização PCA 3D
- **Descrição**: Visualização interativa em 3D das features extraídas usando PCA
- **Tecnologia**: Plotly para gráficos interativos
- **Recursos**:
  - Rotação 3D interativa
  - Zoom e pan
  - Hover com informações detalhadas
  - Visualização por classe com cores distintas
  - Exibição da variância explicada por componente

#### Visualização Grad-CAM 3D
- **Descrição**: Mapa de ativação em superfície 3D
- **Recursos**:
  - Visualização de superfície 3D do heatmap
  - Colormap "Hot" para melhor interpretação
  - Interatividade completa com Plotly
  - Identificação de regiões de alta ativação

**Como usar:**
```python
# No app5.py, selecione 3 componentes no dropdown de PCA
n_components = st.selectbox("Escolha o número de componentes principais", [2, 3])

# Para Grad-CAM 3D, marque a checkbox após a avaliação
show_3d_gradcam = st.checkbox("Mostrar Grad-CAM em 3D")
```

---

### 2. 🤖 Chat com IA para Análise Diagnóstica

#### Suporte para Múltiplos Provedores
- **Google Gemini**: modelos gemini-1.5-pro-latest (recomendado), gemini-1.5-flash-latest, gemini-1.0-pro-latest, gemini-pro, gemini-pro-vision
- **Groq**: modelos mixtral-8x7b-32768, llama-3.1-70b-versatile, llama-3.1-8b-instant

#### Análise Diagnóstica Completa
O sistema gera análises de nível PhD incluindo:

1. **Análise Clínica/Forense Detalhada**
   - Interpretação minuciosa dos resultados
   - Significado clínico/científico da classificação
   - Fatores que influenciaram a predição
   - Implicações da confiança do modelo

2. **Correlação com Padrões Conhecidos**
   - Comparação com casos similares na literatura
   - Padrões característicos observados
   - Desvios ou peculiaridades notáveis

3. **Interpretação Multi-Angular**
   - Visão morfológica
   - Análise textural
   - Considerações contextuais
   - Implicações práticas

4. **Diagnóstico Diferencial**
   - Classes alternativas consideradas
   - Razões para descarte de outras hipóteses
   - Casos limítrofes ou ambíguos

5. **Recomendações e Considerações**
   - Sugestões para confirmação diagnóstica
   - Limitações da análise atual
   - Necessidade de exames complementares
   - Considerações éticas

6. **Embasamento Científico**
   - Citações e referências relevantes
   - Metodologias estabelecidas
   - Evidências científicas de suporte

**Como usar:**
```python
# 1. Marque a checkbox "Ativar Análise Diagnóstica Avançada com IA"
# 2. Selecione o provedor (Gemini ou Groq)
# 3. Escolha o modelo
# 4. Insira sua API key
# 5. Clique em "Gerar Análise Diagnóstica Completa"
```

**Exemplo de Prompt Gerado:**
```
Como especialista em diagnóstico de imagens com nível de PhD...
Classe Predita: Melanoma
Confiança: 0.9450 (94.50%)

Informações de Treinamento:
- Épocas: 200
- Taxa de Aprendizagem: 0.0001
- Modelo: ResNet50
...

Análise Grad-CAM:
- Porcentagem de ativação alta: 45.20%
- Localização principal: região central direita
...
```

---

### 3. 🧬 Algoritmos Genéticos para Interpretação Multi-Angular

#### Funcionalidade
Utiliza algoritmos evolutivos (DEAP) para gerar múltiplas perspectivas diagnósticas, explorando diferentes ângulos de interpretação.

#### Perspectivas Geradas
1. **Análise Morfológica Dominante**
   - Foco em características estruturais
   - Peso: 50% morfologia

2. **Análise Textural Focada**
   - Ênfase em propriedades texturais
   - Peso: 50% textura

3. **Análise Cromática Prioritária**
   - Prioriza distribuição de cores
   - Peso: 50% cor

4. **Análise Espacial Contextual**
   - Considera arranjo espacial
   - Peso: 50% espacial

5. **Análise Estatística Integrada**
   - Foco em parâmetros estatísticos
   - Peso: 30% estatística

#### Algoritmo Genético
- **População**: 20 indivíduos
- **Gerações**: 10
- **Operadores**:
  - Crossover: Two-point (70% probabilidade)
  - Mutação: Gaussian (20% probabilidade)
  - Seleção: Tournament (tamanho 3)

#### Fitness Function
```python
fitness = diversity_score + balance_score - conf_penalty
```

**Como usar:**
```python
# Após gerar a análise com IA, marque:
use_genetic = st.checkbox("Gerar Análise Multi-Perspectiva")
```

**Exemplo de Saída:**
```
### Análise Morfológica Dominante
Confiança Ajustada: 0.9450 (94.50%)
Foco Principal: Análise morfológica (peso: 0.50)

Sob esta perspectiva, que prioriza características morfológicas,
a classificação como 'Melanoma' apresenta 94.5% de confiança.
A morfologia estrutural da amostra revela padrões característicos
que corroboram o diagnóstico...
```

---

### 4. 📚 Integração com Referências Acadêmicas

#### Fontes Integradas
1. **PubMed** (NCBI)
   - Artigos biomédicos revisados por pares
   - API pública do NIH
   
2. **arXiv**
   - Preprints de computação e IA
   - API aberta

3. **Google Scholar** (opcional)
   - Ampla cobertura acadêmica
   - Requer biblioteca scholarly

#### Estratégia de Busca
```python
queries = [
    f"{domain} {class_name} deep learning",
    f"{class_name} classification neural network",
    f"{class_name} diagnosis machine learning"
]
```

#### Informações Coletadas
- Título do artigo
- Autores (primeiros 3 + et al.)
- Fonte/Periódico
- Ano de publicação
- URL/DOI

**Exemplo de Referência:**
```
1. Deep Learning for Skin Lesion Classification
   - Autores: Esteva A., Kuprel B., Novoa R. et al.
   - Fonte: PubMed (PMID: 28117445)
   - Ano: 2017
   - Periódico: Nature
   - URL: https://pubmed.ncbi.nlm.nih.gov/28117445/
```

---

## 📦 Módulos Criados

### 1. `visualization_3d.py`
Funções para visualizações 3D interativas:
- `visualize_pca_3d()`: PCA em 3D
- `visualize_activation_heatmap_3d()`: Grad-CAM 3D
- `visualize_confusion_matrix_3d()`: Matriz de confusão 3D
- `visualize_feature_importance_3d()`: Importância de features 3D

### 2. `ai_chat_module.py`
Sistema de chat com IA:
- `AIAnalyzer`: Classe principal para análise
- `generate_diagnostic_prompt()`: Gera prompts estruturados
- `analyze()`: Envia prompt e recebe resposta
- `describe_gradcam_regions()`: Análise textual do Grad-CAM

### 3. `academic_references.py`
Sistema de busca de referências:
- `AcademicReferenceFetcher`: Classe principal
- `search_pubmed()`: Busca no PubMed
- `search_arxiv()`: Busca no arXiv
- `search_google_scholar()`: Busca no Google Scholar
- `format_references_for_display()`: Formatação para UI

### 4. `genetic_interpreter.py`
Interpretação com algoritmos genéticos:
- `GeneticDiagnosticInterpreter`: Classe principal
- `generate_perspectives()`: Gera perspectivas com AG
- `interpret_from_perspective()`: Interpreta de um ângulo
- `generate_multi_angle_report()`: Relatório completo

---

## 🔧 Dependências Adicionadas

```txt
plotly                    # Visualizações 3D interativas
google-generativeai       # API do Google Gemini
groq                      # API Groq
requests                  # HTTP requests
beautifulsoup4           # Web scraping
scholarly                # Google Scholar (opcional)
deap                     # Algoritmos genéticos
```

**Instalação:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar o app5.py

### 1. Preparação
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run app5.py
```

### 2. Fluxo de Trabalho

#### Etapa 1: Treinamento
1. Upload do dataset (ZIP)
2. Configure parâmetros de treinamento
3. Selecione modelo e otimizador
4. Inicie o treinamento

#### Etapa 2: Avaliação Básica
1. Upload da imagem para avaliar
2. Visualize predição e confiança
3. Analise Grad-CAM 2D

#### Etapa 3: Visualização 3D
1. **PCA 3D**: Selecione 3 componentes no dropdown
2. **Grad-CAM 3D**: Marque a checkbox "Mostrar Grad-CAM em 3D"
3. Interaja com os gráficos (rotação, zoom, hover)

#### Etapa 4: Análise com IA
1. Marque "Ativar Análise Diagnóstica Avançada com IA"
2. Selecione provedor (Gemini ou Groq)
3. Escolha o modelo
4. Insira API key
5. Clique em "Gerar Análise Diagnóstica Completa"
6. Aguarde busca de referências acadêmicas
7. Revise a análise PhD-level gerada

#### Etapa 5: Interpretação Multi-Angular
1. Após análise com IA, marque "Gerar Análise Multi-Perspectiva"
2. Aguarde execução do algoritmo genético
3. Revise as 5 perspectivas diferentes geradas
4. Analise o consenso das perspectivas

---

## 📊 Exemplos de Uso

### Exemplo 1: Diagnóstico Médico (Classificação de Pele)
```python
# 1. Treinar com dataset de lesões de pele
# 2. Avaliar imagem de paciente
# 3. Usar Gemini com gemini-1.5-pro-latest (recomendado)
# 4. API key do Google AI Studio
# 5. Obter análise detalhada como residência médica
```

### Exemplo 2: Análise Geológica (Classificação de Rochas)
```python
# 1. Treinar com dataset de rochas
# 2. Avaliar amostra geológica
# 3. Usar Groq com mixtral-8x7b-32768
# 4. Visualizar PCA 3D das features
# 5. Obter interpretação multi-angular
```

### Exemplo 3: Controle de Qualidade Industrial
```python
# 1. Treinar com imagens de produtos
# 2. Avaliar peça para inspeção
# 3. Visualizar Grad-CAM 3D
# 4. Gerar relatório técnico com IA
# 5. Exportar análise para documentação
```

---

## ⚙️ Configurações Avançadas

### Ajustes de API
```python
# Timeout para requisições
timeout = 10  # segundos

# Número de referências por fonte
max_per_source = 3

# Temperatura do modelo (criatividade)
temperature = 0.7  # 0.0 = determinístico, 1.0 = criativo
```

### Parâmetros do Algoritmo Genético
```python
population_size = 20      # Tamanho da população
generations = 10          # Número de gerações
cxpb = 0.7               # Probabilidade de crossover
mutpb = 0.2              # Probabilidade de mutação
```

### Personalização de Visualizações 3D
```python
# Camera position
eye = dict(x=1.5, y=1.5, z=1.5)

# Tamanho dos marcadores
marker_size = 6

# Opacidade
opacity = 0.8
```

---

## 🔒 Segurança e Privacidade

### Gerenciamento de API Keys
- **Nunca** commit API keys no código
- Use `st.text_input(type="password")` para entrada segura
- Keys são armazenadas apenas na sessão do Streamlit
- Não são salvas em disco

### Dados do Paciente/Amostra
- Imagens processadas localmente
- Apenas metadados são enviados para APIs
- Sem compartilhamento de imagens com serviços externos
- Conformidade com LGPD/GDPR

### Rate Limiting
```python
# PubMed: Máximo 3 requests/segundo
# arXiv: Sem limite oficial, mas use com moderação
# Google Scholar: Use delays (time.sleep) para evitar bloqueio
time.sleep(1)  # Entre requisições
```

---

## 📈 Métricas de Performance

### Tempo de Execução Típico
- **Visualização 3D PCA**: ~1-2 segundos
- **Grad-CAM 3D**: ~2-3 segundos
- **Busca de Referências**: ~5-10 segundos
- **Análise com IA**: ~10-30 segundos (depende do modelo)
- **Algoritmo Genético**: ~3-5 segundos

### Uso de Recursos
- **Memória adicional**: ~200-500 MB (módulos novos)
- **CPU**: Baixo impacto (exceto AG)
- **GPU**: Não necessária para novos módulos
- **Rede**: ~1-5 MB por análise completa

---

## 🐛 Troubleshooting

### Problema: Visualização 3D não aparece
**Solução:**
```bash
pip install --upgrade plotly streamlit
```

### Problema: Erro na API Gemini/Groq
**Causas comuns:**
1. API key inválida → Verifique no console do provedor
2. Sem créditos → Adicione créditos na conta
3. Rate limit → Aguarde alguns segundos

### Problema: Referências não encontradas
**Causas:**
1. Termo de busca muito específico
2. Timeout de rede
3. API PubMed/arXiv temporariamente indisponível

**Solução:**
```python
# Aumentar timeout
self.timeout = 30  # em academic_references.py
```

### Problema: Algoritmo Genético muito lento
**Solução:**
```python
# Reduzir parâmetros
population_size = 10
generations = 5
```

---

## 🔄 Comparação de Versões

| Funcionalidade | v4.0 | v5.0 |
|---------------|------|------|
| Visualização 2D | ✅ | ✅ |
| Visualização 3D | ❌ | ✅ |
| Grad-CAM 2D | ✅ | ✅ |
| Grad-CAM 3D | ❌ | ✅ |
| Chat com IA | ❌ | ✅ |
| Multi-Provedor IA | ❌ | ✅ |
| Referências Acadêmicas | ❌ | ✅ |
| Algoritmos Genéticos | ❌ | ✅ |
| Análise PhD-level | ❌ | ✅ |
| PubMed Integration | ❌ | ✅ |
| arXiv Integration | ❌ | ✅ |

---

## 📚 Referências Técnicas

### Visualização 3D
1. Plotly Documentation: https://plotly.com/python/
2. Interactive 3D Plots in Python: https://plotly.com/python/3d-charts/

### Análise com IA
1. Google Gemini API: https://ai.google.dev/docs
2. Groq API: https://console.groq.com/docs

### Algoritmos Genéticos
1. DEAP Documentation: https://deap.readthedocs.io/
2. Goldberg, D. E. (1989). "Genetic Algorithms in Search"

### Web Scraping Acadêmico
1. PubMed API: https://www.ncbi.nlm.nih.gov/home/develop/api/
2. arXiv API: https://arxiv.org/help/api/

---

## 🎓 Casos de Uso Recomendados

### 1. Pesquisa Acadêmica
- Use análise multi-angular para explorar diferentes hipóteses
- Integre referências automaticamente em publicações
- Gere visualizações 3D para apresentações

### 2. Diagnóstico Clínico
- Obtenha segunda opinião com IA
- Analise Grad-CAM para explicabilidade
- Gere laudos técnicos detalhados

### 3. Análise Forense
- Documente múltiplas perspectivas de evidência
- Gere relatórios periciais completos
- Embase conclusões com referências científicas

### 4. Controle de Qualidade
- Visualize padrões em 3D para detecção de anomalias
- Automatize relatórios de inspeção
- Mantenha documentação rastreável

---

## 🚧 Desenvolvimento Futuro

### Recursos Planejados
- [ ] Integração com mais APIs de IA (Claude, GPT-4)
- [ ] Exportação de relatórios em PDF
- [ ] Dashboard de métricas em tempo real
- [ ] Suporte para vídeos e séries temporais
- [ ] Análise colaborativa multi-usuário
- [ ] Fine-tuning automático com feedback do usuário

### Melhorias Técnicas
- [ ] Cache de referências acadêmicas
- [ ] Otimização de algoritmos genéticos com paralelismo
- [ ] WebGL para visualizações 3D mais rápidas
- [ ] Suporte offline para análise sem internet

---

## 📧 Suporte e Contribuições

Para dúvidas, sugestões ou reportar bugs:
- Email: marceloclaro@gmail.com
- WhatsApp: (88) 981587145
- Instagram: @marceloclaro.geomaker

Contribuições são bem-vindas! Por favor, abra um Pull Request com suas melhorias.

---

**Desenvolvido por:** Laboratório de Educação e Inteligência Artificial - Geomaker  
**Versão:** 5.0  
**Data:** 2024  
**Licença:** Conforme especificado no arquivo LICENSE

> "A melhor forma de prever o futuro é inventá-lo." - Alan Kay

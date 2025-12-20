# Guia Rápido - Versão 5.0

## 🚀 Início Rápido

Este guia ajudará você a começar rapidamente com as novas funcionalidades da versão 5.0.

---

## 📋 Pré-requisitos

### 1. Instalação Básica

```bash
# Clone o repositório
git clone https://github.com/MarceloClaro/CLASSIFICACAO-DE-ROCHAS.git
cd CLASSIFICACAO-DE-ROCHAS

# Instale as dependências
pip install -r requirements.txt
```

### 2. API Keys (Opcional - Para Chat com IA)

Você precisará de pelo menos uma destas API keys:

#### **Opção A: Google Gemini (Recomendado)**
1. Acesse: https://ai.google.dev/
2. Clique em "Get API Key"
3. Crie um projeto no Google AI Studio
4. Copie sua API key

#### **Opção B: Groq**
1. Acesse: https://console.groq.com/
2. Faça login ou crie uma conta
3. Vá para "API Keys"
4. Crie uma nova API key
5. Copie sua API key

---

## 🎯 Exemplo Completo em 5 Minutos

### Passo 1: Preparar o Dataset

```bash
# Estrutura esperada do ZIP:
dataset.zip
├── classe1/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
├── classe2/
│   ├── imagem1.jpg
│   └── ...
└── classe3/
    └── ...
```

### Passo 2: Executar o App

```bash
streamlit run app5.py
```

O navegador abrirá automaticamente em `http://localhost:8501`

### Passo 3: Treinar o Modelo

1. **Upload do Dataset**
   - Clique em "Browse files"
   - Selecione seu arquivo ZIP
   - Aguarde o upload

2. **Configurar Parâmetros** (valores recomendados para teste rápido)
   ```
   Modelo: ResNet18 (mais rápido)
   Épocas: 10
   Taxa de Aprendizagem: 0.001
   Batch Size: 16
   Augmentação: Standard
   Otimizador: Adam
   ```

3. **Iniciar Treinamento**
   - Clique no botão de treinar
   - Aguarde a conclusão (2-10 minutos dependendo do dataset)

### Passo 4: Avaliar uma Imagem

1. **Upload da Imagem**
   - Marque "Sim" em "Deseja avaliar uma imagem?"
   - Faça upload da imagem de teste
   
2. **Visualizar Resultados Básicos**
   - Classe predita
   - Confiança
   - Grad-CAM 2D

### Passo 5: Explorar Visualizações 3D

1. **PCA 3D**
   - Role até "Análise PCA das Features"
   - Marque a checkbox
   - Selecione "3 componentes"
   - Interaja com o gráfico:
     - Clique e arraste para rotacionar
     - Scroll para zoom
     - Hover para ver detalhes

2. **Grad-CAM 3D**
   - Após avaliar imagem
   - Marque "Mostrar Grad-CAM em 3D"
   - Explore a superfície 3D do mapa de ativação

### Passo 6: Análise com IA (Opcional)

1. **Ativar Chat**
   - Marque "Ativar Análise Diagnóstica Avançada com IA"

2. **Configurar**
   ```
   Provedor: gemini (ou groq)
   Modelo: gemini-1.0-pro (ou mixtral-8x7b-32768)
   API Key: [cole sua API key]
   ```

3. **Gerar Análise**
   - Clique em "Gerar Análise Diagnóstica Completa"
   - Aguarde 10-30 segundos
   - Revise o relatório PhD-level gerado

4. **Análise Multi-Angular**
   - Marque "Gerar Análise Multi-Perspectiva"
   - Aguarde a execução do algoritmo genético (3-5 segundos)
   - Explore as 5 perspectivas diferentes

---

## 💡 Exemplos de Uso por Caso

### Caso 1: Diagnóstico Médico Rápido

```python
# Configuração otimizada para diagnóstico médico
Modelo: ResNet50 ou DenseNet121
Épocas: 50-100
Taxa de Aprendizagem: 0.0001
Augmentação: Standard
Fine-Tuning: Habilitado
```

**Workflow:**
1. Treinar com dataset de lesões
2. Avaliar imagem de paciente
3. Visualizar Grad-CAM para explicabilidade
4. Gerar análise com Gemini para laudo técnico
5. Usar análise multi-angular para segunda opinião

### Caso 2: Pesquisa Acadêmica

```python
# Configuração para pesquisa científica
Modelo: DenseNet121
Épocas: 200
Taxa de Aprendizagem: 0.0001
Augmentação: CutMix
Otimizador: AdamW
Early Stopping: Habilitado
```

**Workflow:**
1. Treinar com dataset experimental
2. Exportar métricas detalhadas
3. Gerar visualizações 3D para paper
4. Coletar referências acadêmicas automaticamente
5. Usar análise multi-angular para discussão

### Caso 3: Análise Geológica

```python
# Configuração para classificação de rochas
Modelo: ResNet50
Épocas: 100
Taxa de Aprendizagem: 0.001
Augmentação: Standard
```

**Workflow:**
1. Treinar com imagens de rochas
2. Avaliar amostra desconhecida
3. Visualizar PCA 3D para análise de agrupamento
4. Gerar relatório técnico com IA
5. Incluir referências geológicas

---

## 🔍 Troubleshooting Rápido

### Problema: "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### Problema: "API key inválida"
- Verifique se copiou corretamente (sem espaços)
- Confirme se tem créditos na conta
- Tente regenerar a key

### Problema: Visualização 3D não aparece
```bash
pip install --upgrade plotly streamlit
# Reinicie o app
```

### Problema: Out of memory durante treinamento
```python
# Reduza o batch size
Batch Size: 8 ou 4

# Ou use modelo menor
Modelo: ResNet18
```

### Problema: Treinamento muito lento
```python
# Reduza épocas para teste
Épocas: 10-20

# Ou use dataset menor para protótipo
```

---

## 📊 Métricas e Interpretação

### Confiança do Modelo

| Confiança | Interpretação | Ação Recomendada |
|-----------|---------------|------------------|
| > 0.95 | Muito Alta | Aceitar classificação |
| 0.85 - 0.95 | Alta | Provável correto, verificar Grad-CAM |
| 0.70 - 0.85 | Moderada | Revisar manualmente, análise multi-angular |
| < 0.70 | Baixa | Considerar classe inconclusiva |

### Grad-CAM

**Ativação Alta (> 50%)**
- Modelo identificou características claras
- Maior confiabilidade

**Ativação Dispersa (< 30%)**
- Características ambíguas
- Revisar diagnóstico

### Análise Multi-Angular

**Concordância Alta entre Perspectivas**
- Diagnóstico robusto
- Diferentes aspectos convergem

**Discordância entre Perspectivas**
- Caso complexo
- Considerar exames adicionais

---

## 🎓 Dicas de Boas Práticas

### 1. Preparação de Dados
✅ **Fazer:**
- Usar imagens balanceadas entre classes
- Aplicar pré-processamento básico
- Verificar qualidade das imagens

❌ **Evitar:**
- Classes com < 50 imagens
- Imagens com ruído excessivo
- Desbalanceamento extremo (> 10:1)

### 2. Treinamento
✅ **Fazer:**
- Começar com poucos épocas (10-20) para teste
- Usar early stopping para evitar overfitting
- Monitorar acurácia de validação

❌ **Evitar:**
- Learning rate muito alta (> 0.01)
- Treinar sem validação
- Ignorar sinais de overfitting

### 3. Avaliação
✅ **Fazer:**
- Testar com múltiplas imagens
- Analisar Grad-CAM para entender decisões
- Comparar diferentes perspectivas

❌ **Evitar:**
- Confiar apenas na confiança numérica
- Ignorar contexto clínico/científico
- Usar como única ferramenta diagnóstica

### 4. Uso da IA
✅ **Fazer:**
- Revisar análises geradas
- Verificar referências acadêmicas
- Usar como suporte à decisão

❌ **Evitar:**
- Aceitar cegamente sem revisão
- Usar sem API key própria (compartilhada)
- Desconsiderar limitações do modelo

---

## 🚀 Próximos Passos

Após dominar o básico:

1. **Experimentar Configurações Avançadas**
   - Diferentes augmentações (Mixup, CutMix)
   - Múltiplos otimizadores
   - Learning rate schedulers

2. **Comparar Modelos**
   - Treinar ResNet18, ResNet50 e DenseNet121
   - Comparar métricas de performance
   - Analisar trade-off velocidade vs. acurácia

3. **Explorar Clustering**
   - Analisar agrupamento não supervisionado
   - Identificar padrões ocultos
   - Validar estrutura do dataset

4. **Documentar Resultados**
   - Exportar métricas para CSV
   - Salvar visualizações 3D
   - Gerar relatórios para publicação

---

## 📚 Recursos Adicionais

### Documentação
- [FEATURES_V5.md](FEATURES_V5.md) - Documentação completa
- [README.md](README.md) - Visão geral do projeto
- [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) - Análise de performance

### Tutoriais Online
- Google Gemini API: https://ai.google.dev/docs
- Groq API: https://console.groq.com/docs
- Plotly 3D: https://plotly.com/python/3d-charts/
- Streamlit: https://docs.streamlit.io/

### Comunidade
- Issues GitHub: https://github.com/MarceloClaro/CLASSIFICACAO-DE-ROCHAS/issues
- Email: marceloclaro@gmail.com
- WhatsApp: (88) 981587145

---

## ⚡ Atalhos Úteis

### Comandos Rápidos
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar app v5.0
streamlit run app5.py

# Executar app v3.0/v4.0
streamlit run app3.py

# Atualizar todas as libs
pip install --upgrade -r requirements.txt

# Verificar versões
pip list | grep -E "streamlit|plotly|torch"
```

### Atalhos do Streamlit
- `Ctrl + C` - Parar o servidor
- `R` - Recarregar app (no navegador)
- `C` - Limpar cache
- `?` - Mostrar atalhos

---

## 🎉 Conclusão

Você está pronto para usar todas as funcionalidades da versão 5.0!

**Resumo:**
1. ✅ Instalar dependências
2. ✅ Executar `streamlit run app5.py`
3. ✅ Treinar modelo
4. ✅ Avaliar imagens
5. ✅ Explorar visualizações 3D
6. ✅ Usar análise com IA (opcional)
7. ✅ Gerar interpretações multi-angulares

**Dúvidas?** Consulte [FEATURES_V5.md](FEATURES_V5.md) para documentação detalhada.

---

**Boa sorte com suas análises!** 🚀

> "A melhor forma de prever o futuro é inventá-lo." - Alan Kay

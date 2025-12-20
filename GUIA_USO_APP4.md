# Guia de Uso - App4.py com Visão Computacional e Vision Transformers

## 🚀 Início Rápido

### 1. Configurar API para Análise com IA

#### Passo 1: Obter API Key
- **Para Gemini**: https://makersuite.google.com/app/apikey
- **Para Groq**: https://console.groq.com/keys

#### Passo 2: Configurar na Sidebar
1. Na sidebar, expanda "🔑 Configuração de API para Análise IA"
2. Selecione o provedor (Gemini ou Groq)
3. Escolha o modelo:
   - **Gemini**: gemini-1.5-pro (recomendado para visão)
   - **Groq**: llama-3.1-70b-versatile
4. Cole sua API Key
5. Aguarde confirmação "✅ API Key configurada!"

---

## 🏗️ Escolher Arquitetura do Modelo

### CNNs (Redes Convolucionais)
**Quando usar:**
- Datasets pequenos (<5000 imagens)
- Recursos limitados de GPU
- Treinamento rápido necessário

**Modelos disponíveis:**
- **ResNet18**: Mais rápido, bom para testes
- **ResNet50**: Equilibrado, recomendado
- **DenseNet121**: Eficiente em memória

### Vision Transformers (ViT)
**Quando usar:**
- Datasets médios/grandes (>5000 imagens)
- GPU com 6GB+ VRAM
- Máxima precisão desejada

**Modelos disponíveis:**
- **ViT-B/16**: Recomendado para maioria dos casos
- **ViT-B/32**: Mais rápido, menos preciso
- **ViT-L/16**: Máxima precisão (requer 8GB+ VRAM)

**⚠️ Importante para ViT:**
- Use batch size menor (8-16 vs 32)
- Treinamento mais lento
- Pode dar OOM (Out of Memory) em GPUs pequenas

---

## 📊 Exportar Resultados em CSV

### 1. Histórico de Treinamento
**O que contém:**
- Número da época
- Loss de treino e validação
- Accuracy de treino e validação

**Quando baixar:**
- Aparece automaticamente após treinamento completar
- Botão: "📥 Baixar CSV - Histórico de Treinamento"

**Uso:**
- Analisar evolução do modelo
- Identificar overfitting
- Comparar diferentes treinamentos

### 2. Resultados de Clustering
**O que contém:**
- ID de cada amostra
- Label verdadeiro e nome da classe
- Cluster hierárquico atribuído
- Cluster K-Means atribuído
- Métricas ARI e NMI

**Quando baixar:**
- Após análise de clustering completar
- Botão: "📥 Baixar CSV - Resultados de Clustering"

**Uso:**
- Validar agrupamento automático
- Identificar confusões entre classes
- Análise estatística externa

### 3. Resultado de Classificação Individual
**O que contém:**
- Nome da imagem
- Classe predita e confiança
- Modelo usado
- Hiperparâmetros de treinamento
- Tipo de Grad-CAM usado

**Quando baixar:**
- Após classificar uma imagem
- Botão: "📥 Baixar CSV - Resultado da Classificação"

**Uso:**
- Documentar classificações
- Comparar diferentes modelos
- Relatórios técnicos

### 4. Análise Completa com IA
**O que contém:**
- Todos os dados da classificação
- Descrição detalhada do Grad-CAM
- Análise técnica completa da IA
- Interpretação forense
- Recomendações

**Quando baixar:**
- Após gerar análise com IA (requer API configurada)
- Botão: "📥 Baixar CSV - Análise Completa com IA"

**Uso:**
- Laudos técnicos
- Análise forense
- Documentação completa

---

## 🤖 Usar Análise com IA (Visão Computacional)

### Pré-requisitos:
1. ✅ API configurada na sidebar
2. ✅ Modelo treinado
3. ✅ Imagem classificada

### Passo a Passo:

1. **Treinar o Modelo**
   - Upload do ZIP com imagens
   - Selecione arquitetura e modelo
   - Configure hiperparâmetros
   - Aguarde treinamento

2. **Classificar Imagem**
   - Selecione "Sim" em "Deseja avaliar uma imagem?"
   - Faça upload da imagem
   - Veja classificação e Grad-CAM

3. **Gerar Análise com IA**
   - Se API configurada, verá seção "🤖 Análise Diagnóstica com IA"
   - Clique em "🔬 Gerar Análise Completa com IA + Visão"
   - Aguarde processamento (10-30 segundos)

4. **Interpretar Resultados**
   - **Descrição Visual**: O que a IA "vê" na imagem
   - **Interpretação Técnica**: Avaliação da classificação
   - **Análise Forense**: Detecção de anomalias
   - **Recomendações**: Ações sugeridas

5. **Exportar**
   - Baixe o CSV com análise completa

### Exemplo de Análise Gemini:
```
🔬 ANÁLISE DETALHADA:

1. DESCRIÇÃO VISUAL:
   - Observo uma rocha com textura granular
   - Coloração predominante cinza-escuro
   - Grãos minerais visíveis de aproximadamente 2-5mm
   - Superfície irregular com pequenas cavidades

2. INTERPRETAÇÃO TÉCNICA:
   - A classificação como "Granito" é COMPATÍVEL
   - Características observadas: textura fanerítica, 
     minerais quartzo e feldspato visíveis
   - Confiança de 94.3% é ADEQUADA

3. ANÁLISE FORENSE:
   - Sem artefatos de processamento detectados
   - Iluminação uniforme
   - Imagem autêntica

4. RECOMENDAÇÕES:
   - Classificação pode ser ACEITA
   - Análise petrográfica confirmatória sugerida
```

---

## 🎯 Fluxo de Trabalho Completo

### Cenário 1: Classificação de Rochas com IA

```mermaid
1. Preparar Dataset
   └─> Organizar em pastas por classe
   └─> Zipar pasta raiz
   
2. Configurar Aplicação
   └─> Sidebar: Selecionar ViT-B/16
   └─> Configurar API Gemini
   └─> Definir hiperparâmetros
   
3. Treinar Modelo
   └─> Upload ZIP
   └─> Aguardar treinamento
   └─> Baixar CSV histórico
   
4. Analisar Clustering
   └─> Ver visualizações PCA
   └─> Baixar CSV clustering
   
5. Classificar Amostras
   └─> Upload de imagem individual
   └─> Gerar Grad-CAM
   └─> Baixar CSV classificação
   
6. Análise com IA
   └─> Gerar análise completa
   └─> Revisar interpretação
   └─> Baixar CSV análise IA
   
7. Documentar
   └─> Consolidar todos os CSVs
   └─> Gerar relatório final
```

---

## 💡 Dicas e Boas Práticas

### Treinamento
1. **Comece com CNN** para validar dataset
2. **Use ViT** quando CNN já funciona bem
3. **Monitore overfitting** via histórico CSV
4. **Ajuste batch size** se houver OOM

### Análise com IA
1. **Gemini é melhor** para análise visual detalhada
2. **Groq é mais rápido** mas visão limitada
3. **Compare múltiplas análises** para validar
4. **Use Grad-CAM** para guiar interpretação

### CSV Export
1. **Baixe TODOS os CSVs** para documentação completa
2. **Use Excel/Python** para análise consolidada
3. **Mantenha organizado** por data/experimento
4. **Versionamento** dos resultados importante

### Performance
1. **ViT requer mais tempo** - seja paciente
2. **Monitore GPU** com nvidia-smi
3. **Reduza batch size** se necessário
4. **Use cache de API** quando possível

---

## ⚠️ Resolução de Problemas

### Erro: "Out of Memory"
**Solução:**
- Reduza batch size (ex: 32 → 16 → 8)
- Use modelo menor (ViT-L → ViT-B, ResNet50 → ResNet18)
- Feche outros processos GPU

### Erro: "API Key inválida"
**Solução:**
- Verifique API key copiada corretamente
- Confirme que chave está ativa na plataforma
- Tente regenerar a chave

### ViT muito lento
**Solução:**
- Normal: ViT é 2-3x mais lento que CNN
- Reduza épocas inicialmente
- Use ViT-B/32 para testes rápidos

### Grad-CAM não aparece
**Solução:**
- Normal para alguns modelos ViT
- Tente com CNN primeiro
- Verifique console para erros

### Análise IA genérica
**Solução:**
- Use Gemini ao invés de Groq
- Verifique que modelo suporta visão
- Grad-CAM deve estar ativo

---

## 📞 Suporte

Para problemas técnicos:
- Email: marceloclaro@gmail.com
- WhatsApp: (88) 981587145
- Instagram: @marceloclaro.geomaker

---

**Versão:** 4.0.0  
**Última atualização:** 2025-12-20

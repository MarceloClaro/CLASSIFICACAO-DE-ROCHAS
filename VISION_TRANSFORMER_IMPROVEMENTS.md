# Vision Transformer e Melhorias Avançadas - Documentação Completa

## 🎯 Visão Geral

Este documento descreve todas as melhorias implementadas no sistema de classificação de imagens, focando em:
1. **Adição de Vision Transformers** (ViT e Swin)
2. **Reinforcement Learning** para ajuste dinâmico
3. **Agentes CrewAI** para pesquisa inteligente
4. **Técnicas avançadas de regularização**

---

## 🤖 Novos Modelos: Vision Transformers

### Por que Vision Transformers?

Os modelos CNN tradicionais (ResNet, DenseNet) usam convoluções que capturam padrões **locais**. Vision Transformers usam mecanismos de **atenção** para capturar relações **globais** na imagem, o que pode resultar em melhor desempenho, especialmente para:
- Imagens com padrões complexos e distribuídos
- Datasets onde contexto global é importante
- Tarefas que requerem entendimento de relações entre regiões distantes

### Modelos Disponíveis

#### 1. **ViT-B/16** (torchvision)
- **Arquitetura**: Vision Transformer Base, patches 16x16
- **Parâmetros**: ~86M
- **Uso**: Boa performance geral, versão oficial do PyTorch
- **Recomendação**: Dataset médio a grande (>1000 imagens)

#### 2. **ViT-B/32** (torchvision)
- **Arquitetura**: Vision Transformer Base, patches 32x32
- **Parâmetros**: ~88M
- **Uso**: Mais rápido que ViT-B/16, menos preciso
- **Recomendação**: Quando velocidade é prioridade

#### 3. **ViT-L/16** (torchvision)
- **Arquitetura**: Vision Transformer Large, patches 16x16
- **Parâmetros**: ~307M
- **Uso**: Máxima precisão, requer muita memória GPU
- **Recomendação**: Dataset grande (>5000 imagens), GPU potente

#### 4. **ViT-B/16-timm** (timm - NOVO! 🆕)
- **Arquitetura**: Vision Transformer Base da biblioteca timm
- **Parâmetros**: ~86M
- **Uso**: **Versão mais robusta e melhor treinada que torchvision**
- **Vantagem**: Pré-treinamento superior, melhor generalização
- **Recomendação**: **PRIMEIRA ESCOLHA para ViT Base**

#### 5. **ViT-L/16-timm** (timm - NOVO! 🆕)
- **Arquitetura**: Vision Transformer Large da biblioteca timm
- **Parâmetros**: ~307M
- **Uso**: **Versão melhorada do ViT Large**
- **Vantagem**: Melhor desempenho que versão torchvision
- **Recomendação**: Para máxima precisão com dataset grande

#### 6. **Swin-T** (Swin Transformer Tiny - NOVO! 🆕)
- **Arquitetura**: Swin Transformer com arquitetura hierárquica
- **Parâmetros**: ~28M
- **Uso**: **Melhor eficiência que ViT, performance superior em muitos casos**
- **Vantagem**: Atenção em janelas (window-based), computacionalmente eficiente
- **Recomendação**: **EXCELENTE ESCOLHA para datasets médios**

#### 7. **Swin-B** (Swin Transformer Base - NOVO! 🆕)
- **Arquitetura**: Swin Transformer Base
- **Parâmetros**: ~88M
- **Uso**: **State-of-the-art performance, arquitetura hierárquica**
- **Vantagem**: Melhor que ViT-B em muitos benchmarks
- **Recomendação**: **PRIMEIRA ESCOLHA para máxima performance**

### Comparação: CNN vs Vision Transformer

| Característica | CNN (ResNet, DenseNet) | Vision Transformer (ViT, Swin) |
|----------------|------------------------|--------------------------------|
| **Padrões** | Locais e hierárquicos | Globais e relações de longo alcance |
| **Inductive Bias** | Forte (convoluções) | Fraco (atenção) |
| **Dados Necessários** | Menor (500+ imagens) | Maior (1000+ imagens) |
| **Memória GPU** | Menor | Maior |
| **Velocidade** | Mais rápido | Mais lento |
| **Performance** | Boa | Potencialmente melhor com dados suficientes |
| **Robustez a Oclusão** | Boa | Excelente (atenção global) |

---

## 🎯 Reinforcement Learning para Ajuste Dinâmico

### O que é?

Um sistema de **Q-Learning** que ajusta automaticamente a **learning rate** durante o treinamento baseado no desempenho de validação.

### Como Funciona?

1. **Estado**: Definido por tendências de perda e acurácia
   - `improving_improving`: Perda e acurácia melhorando
   - `improving_degrading`: Perda melhorando, acurácia piorando
   - `degrading_improving`: Perda piorando, acurácia melhorando
   - `degrading_degrading`: Ambos piorando

2. **Ações**: 3 possíveis ações
   - `increase_lr`: Aumenta LR em 20%
   - `decrease_lr`: Diminui LR em 20%
   - `keep_lr`: Mantém LR atual

3. **Recompensa**: Calculada baseada em melhoria de desempenho
   ```python
   reward = (prev_loss - current_loss) * 10 + (current_acc - prev_acc) * 100
   ```

4. **Q-Learning**: Atualiza Q-values para aprender política ótima
   ```python
   Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
   ```

### Vantagens

- ✅ **Adaptativo**: Ajusta LR automaticamente sem intervenção manual
- ✅ **Inteligente**: Aprende a melhor estratégia durante o treinamento
- ✅ **Robusto**: Pode recuperar de platôs de treinamento
- ✅ **Transparente**: Mostra ações e recompensas em tempo real

### Quando Usar?

- 🎯 Datasets desafiadores onde LR fixo não funciona bem
- 🎯 Treinamentos longos (>20 épocas)
- 🎯 Quando você não tem certeza da melhor LR
- ⚠️ **NÃO usar** com OneCycleLR (conflito de estratégias)

### Exemplo de Output

```
Época 10/50
Perda de Treino: 0.4521 | Acurácia de Treino: 0.8234
Perda de Validação: 0.4892 | Acurácia de Validação: 0.8156
🎯 RL Action: decrease_lr | New LR: 0.000080 | Reward: 0.1245
```

---

## 🤖 Agente CrewAI para Pesquisa Inteligente

### O que é?

Um **agente inteligente** que pesquisa na web as melhores estratégias de treinamento para seu modelo e tipo de dataset.

### Como Funciona?

1. **Agente ML Researcher**: Especialista em otimização de deep learning
2. **Pesquisa Web**: Busca papers, artigos e best practices
3. **Recomendações**: Fornece insights sobre:
   - Learning rate ótima
   - Melhores técnicas de augmentation
   - Batch size recomendado
   - Scheduler mais adequado
   - Armadilhas comuns a evitar

### Configuração

⚠️ **Requer API Keys** (opcional):
- OpenAI API Key para o agente
- Serper API Key para busca web

### Vantagens

- 📚 Acesso a conhecimento atualizado
- 🎓 Recomendações baseadas em pesquisa científica
- 💡 Insights que você pode não conhecer
- 🚀 Acelera experimentação

### Quando Usar?

- 🆕 Novo tipo de dataset que você nunca trabalhou
- 🎯 Quer maximizar performance
- 📖 Quer aprender melhores práticas
- ⚠️ **EXPERIMENTAL** - ainda em desenvolvimento

---

## ✨ Label Smoothing

### O que é?

Técnica que **suaviza** os rótulos para prevenir **overconfidence** do modelo.

### Como Funciona?

Em vez de usar rótulos one-hot (0, 1), usa distribuição suavizada:
```
Original: [0, 1, 0, 0]  (100% confiante)
Suavizado: [0.025, 0.925, 0.025, 0.025]  (92.5% confiante)
```

### Fórmula

```
y_smooth = (1 - ε) * y_true + ε / K
```
Onde:
- ε = smoothing factor (geralmente 0.1)
- K = número de classes

### Vantagens

- ✅ **Melhora Generalização**: Modelo menos overconfident
- ✅ **Reduz Overfitting**: Especialmente com poucos dados
- ✅ **Essencial para ViT**: Vision Transformers beneficiam muito
- ✅ **Calibração**: Probabilidades mais calibradas

### Valores Recomendados

| Dataset | Smoothing |
|---------|-----------|
| Pequeno (<500) | 0.15-0.2 |
| Médio (500-2000) | 0.1-0.15 |
| Grande (>2000) | 0.05-0.1 |
| Vision Transformer | 0.1-0.2 |
| CNN | 0.05-0.1 |

---

## 📊 Exponential Moving Average (EMA)

### O que é?

Mantém uma **média móvel exponencial** dos pesos do modelo durante o treinamento.

### Como Funciona?

```python
shadow_weight = decay * shadow_weight + (1 - decay) * current_weight
```

A cada passo de otimização, os pesos "shadow" são atualizados suavemente. No final do treinamento, usamos os pesos EMA em vez dos pesos finais.

### Vantagens

- ✅ **Estabilidade**: Pesos finais mais estáveis
- ✅ **Generalização**: Geralmente melhor performance no teste
- ✅ **Robustez**: Menos sensível a flutuações de treinamento
- ✅ **State-of-the-art**: Usado em modelos SOTA

### Parâmetros

- **Decay**: 0.999 (padrão)
  - Maior = mais suavização
  - Menor = mais reativo

### Quando Usar?

- ✅ **SEMPRE** - especialmente com ViT
- ✅ Treinamentos longos
- ✅ Learning rates altas
- ✅ Quando busca máxima performance

---

## ⚡ Gradient Clipping

### O que é?

Limita a **norma dos gradientes** para prevenir **exploding gradients**.

### Como Funciona?

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Se a norma dos gradientes exceder 1.0, eles são escalonados proporcionalmente.

### Por que é Importante?

- **Vision Transformers**: Especialmente sensíveis a gradientes explosivos
- **Estabilidade**: Treinamento mais estável
- **Convergência**: Melhor convergência em modelos grandes

### Vantagens

- ✅ **Essencial para ViT**: Quase obrigatório
- ✅ **Estabilidade**: Previne NaN e divergência
- ✅ **Sem custo**: Overhead computacional mínimo
- ✅ **Best Practice**: Usado em todos os modelos modernos

### Quando Usar?

- ✅ **SEMPRE com Vision Transformers**
- ✅ Modelos grandes (>50M parâmetros)
- ✅ Learning rates altas
- ✅ Training instável

---

## 🎨 Melhorias em Data Augmentation (app5.py)

### Transformações Aprimoradas

#### Antes (Standard Simples)
```python
RandomApply([
    RandomHorizontalFlip(),
    RandomRotation(90),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
    ...
], p=0.5)
```

#### Depois (Standard Robusto)
```python
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.2)
RandomRotation(degrees=30)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15)
RandomResizedCrop(224, scale=(0.7, 1.0))
RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # NOVO
RandomErasing(p=0.2, scale=(0.02, 0.15))  # NOVO
```

### Novas Transformações

1. **GaussianBlur**: Simula variações de foco
2. **RandomErasing**: Simula oclusões parciais
3. **Parâmetros Mais Agressivos**: Maior variabilidade

### Vantagens

- ✅ **Mais Robustez**: Modelo lida melhor com variações
- ✅ **Menos Overfitting**: Mais diversidade de dados
- ✅ **Melhor Generalização**: Performance em imagens reais

---

## 📋 Guia de Uso Prático

### Cenário 1: Dataset Pequeno (<500 imagens)

**Recomendações:**
- **Modelo**: DenseNet121 ou Swin-T
- **Augmentation**: mixup ou cutmix
- **Label Smoothing**: 0.15-0.2
- **EMA**: ✅ Ativado
- **Gradient Clipping**: ✅ Ativado
- **RL**: ⚠️ Opcional
- **Learning Rate**: 0.0001
- **Batch Size**: 8-16
- **Epochs**: 100-200

### Cenário 2: Dataset Médio (500-2000 imagens)

**Recomendações:**
- **Modelo**: ResNet50 ou Swin-T
- **Augmentation**: standard
- **Label Smoothing**: 0.1
- **EMA**: ✅ Ativado
- **Gradient Clipping**: ✅ Ativado
- **RL**: ✅ Ativado
- **Learning Rate**: 0.0001
- **Batch Size**: 16-32
- **Epochs**: 50-100

### Cenário 3: Dataset Grande (>2000 imagens) com GPU Potente

**Recomendações:**
- **Modelo**: ViT-B/16-timm ou Swin-B
- **Augmentation**: standard
- **Label Smoothing**: 0.1
- **EMA**: ✅ Ativado
- **Gradient Clipping**: ✅ Ativado
- **RL**: ✅ Ativado
- **Scheduler**: OneCycleLR
- **Learning Rate**: 0.0001
- **Batch Size**: 32-64
- **Epochs**: 30-50

### Cenário 4: Máxima Performance (Competição)

**Recomendações:**
- **Modelo**: Ensemble de Swin-B + ViT-L/16-timm
- **Augmentation**: mixup + cutmix alternado
- **Label Smoothing**: 0.1
- **EMA**: ✅ Ativado (decay=0.9999)
- **Gradient Clipping**: ✅ Ativado
- **RL**: ✅ Ativado
- **Scheduler**: CosineAnnealingLR com warmup
- **Fine-tuning**: ✅ Completo
- **Learning Rate**: 0.00005
- **Batch Size**: Máximo possível
- **Epochs**: 100+
- **TTA**: Test-Time Augmentation na inferência

---

## 🔧 Solução de Problemas

### Problema: Out of Memory (OOM)

**Soluções:**
1. Reduzir batch size (16 → 8 → 4)
2. Usar modelo menor (ViT-L → ViT-B → Swin-T)
3. Usar gradient accumulation (não implementado ainda)
4. Reduzir resolução de imagem

### Problema: Treinamento Instável (Loss = NaN)

**Soluções:**
1. ✅ Ativar Gradient Clipping
2. Reduzir learning rate (0.001 → 0.0001)
3. Aumentar warmup steps
4. Verificar dados (valores inválidos, dimensões)

### Problema: Overfitting Severo

**Soluções:**
1. Aumentar label smoothing (0.1 → 0.15)
2. Usar mixup ou cutmix
3. Aumentar L2 regularization
4. Aumentar dropout
5. Mais data augmentation
6. Early stopping mais agressivo

### Problema: Underfitting

**Soluções:**
1. Modelo maior (ResNet18 → ResNet50 → ViT)
2. Mais épocas
3. Learning rate maior
4. Menos regularização
5. Fine-tuning completo

### Problema: Convergência Lenta

**Soluções:**
1. ✅ Ativar RL para ajuste dinâmico
2. Usar OneCycleLR scheduler
3. Aumentar learning rate inicial
4. Verificar gradient flow
5. Usar otimizador AdamW

---

## 📚 Referências Científicas

### Vision Transformers
- Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Liu et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- Touvron et al. (2021). "Training data-efficient image transformers"

### Label Smoothing
- Szegedy et al. (2016). "Rethinking the Inception Architecture for Computer Vision"
- Müller et al. (2019). "When Does Label Smoothing Help?"

### EMA
- Polyak & Juditsky (1992). "Acceleration of Stochastic Approximation by Averaging"
- Tarvainen & Valpola (2017). "Mean teachers are better role models"

### Reinforcement Learning for Hyperparameters
- Li et al. (2017). "Learning to Optimize"
- Baker et al. (2017). "Designing Neural Network Architectures using Reinforcement Learning"

---

## 🚀 Próximas Melhorias Planejadas

### Fase 5: Métricas Avançadas
- [ ] Expected Calibration Error (ECE)
- [ ] Maximum Calibration Error (MCE)
- [ ] Ensemble predictions
- [ ] Per-class análise detalhada

### Fase 6: Augmentation Avançado
- [ ] RandAugment
- [ ] TrivialAugment
- [ ] AutoAugment
- [ ] Test-Time Augmentation (TTA)

### Fase 7: Eficiência
- [ ] Gradient Accumulation
- [ ] Mixed Precision Training (FP16)
- [ ] Model pruning
- [ ] Knowledge distillation

### Fase 8: Scheduler Avançado
- [ ] Warmup para todos os schedulers
- [ ] Polynomial decay
- [ ] Plateau-based adjustment
- [ ] Cyclical learning rates

---

## ✅ Como Testar as Melhorias

### Experimento Baseline
1. Treinar ResNet50 sem melhorias
2. Configuração: standard augmentation, sem label smoothing, sem EMA
3. Registrar: acurácia final, loss, tempo

### Experimento com Melhorias
1. Treinar ResNet50 com todas as melhorias
2. Configuração: standard augmentation, label smoothing=0.1, EMA ativado, gradient clipping
3. Comparar com baseline

### Experimento Vision Transformer
1. Treinar Swin-T com melhorias
2. Comparar com ResNet50 baseline
3. Avaliar se ViT melhora resultados

### Métricas de Comparação
- **Acurácia**: Final test accuracy
- **Convergência**: Épocas para atingir 90% de acurácia
- **Estabilidade**: Variância da loss nas últimas 10 épocas
- **Calibração**: Diferença entre confiança e acurácia

---

## 💡 Dicas e Best Practices

### ✅ DO (Faça)
- Sempre use Gradient Clipping com ViT
- Sempre use EMA - quase sem custo, grande benefício
- Use Label Smoothing para melhorar generalização
- Experimente RL em treinamentos longos
- Use Swin Transformers em vez de ViT quando possível
- Monitore overfitting com gráficos train/val
- Salve múltiplos checkpoints

### ❌ DON'T (Não Faça)
- Não use RL + OneCycleLR juntos
- Não use ViT em datasets muito pequenos (<500)
- Não ignore warnings de OOM - ajuste batch size
- Não use Label Smoothing > 0.3
- Não desative Gradient Clipping com ViT
- Não use fine-tuning sem dados suficientes

---

## 📞 Suporte e Contato

**Projeto Geomaker + IA**
- Email: marceloclaro@gmail.com
- WhatsApp: (88) 981587145
- Instagram: [@marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
- DOI: https://doi.org/10.5281/zenodo.13910277

---

**Última atualização**: 2024  
**Versão**: 5.0 (com Vision Transformers, RL, CrewAI, EMA, Label Smoothing)  

> "The best way to predict the future is to invent it." - Alan Kay

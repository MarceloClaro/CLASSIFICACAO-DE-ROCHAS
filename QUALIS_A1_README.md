# Geomaker v2.0 - Melhorias Qualis A1

## 📋 Visão Geral

Este documento descreve as melhorias implementadas no Geomaker v2.0 para atingir o nível de qualidade exigido por periódicos científicos Qualis A1.

## 🎯 Principais Melhorias

### 1. Auditoria Completa de Experimentos

- **ExperimentAuditor**: Sistema de logging estruturado
  - Rastreamento de todas as operações
  - Versionamento de modelos e checkpoints
  - Configuração completa do experimento
  - Geração de relatórios de reprodutibilidade

```python
from qualis_a1_improvements import ExperimentAuditor, ExperimentConfig

# Criar configuração
config = ExperimentConfig(
    data_dir='./dataset',
    num_classes=10,
    model_name='ResNet50',
    epochs=100,
    learning_rate=0.0001,
    # ... outros parâmetros
)

# Inicializar auditor
auditor = ExperimentAuditor(log_dir='./experiments')
auditor.log_experiment_start(config)

# Durante treinamento
auditor.log_epoch(epoch, train_metrics, valid_metrics)
auditor.log_checkpoint(model, epoch, metrics, is_best=True)
```

### 2. Análise de Curvas de Aprendizado

- Detecta automaticamente problemas de under/overfitting
- Fornece recomendações específicas
- Visualizações detalhadas com análise

```python
from qualis_a1_improvements import LearningCurveAnalyzer

analyzer = LearningCurveAnalyzer()

# Analisar curvas
analysis = analyzer.analyze_learning_curve(
    train_losses, valid_losses,
    train_accs, valid_accs
)

# Gerar gráfico
fig = analyzer.plot_learning_curves(
    train_losses, valid_losses,
    train_accs, valid_accs,
    analysis=analysis
)
```

### 3. Calibração de Probabilidades

- **Temperature Scaling**: Ajusta confiança do modelo
- **ECE (Expected Calibration Error)**: Métrica de calibração
- Curvas de calibração para visualização

```python
from qualis_a1_improvements import ProbabilityCalibrator

calibrator = ProbabilityCalibrator()

# Calcular ECE
ece = calibrator.calculate_ece(y_true, y_prob)

# Plotar curva de calibração
fig = calibrator.plot_calibration_curve(y_true, y_prob)
```

### 4. Validação Estatística Rigorosa

- **Teste de McNemar**: Comparação entre modelos
- **Bootstrap**: Intervalos de confiança robustos
- Testes de significância estatística

```python
from qualis_a1_improvements import StatisticalValidator

validator = StatisticalValidator()

# Comparar dois modelos
result = validator.mcnemar_test(y_true, y_pred_model1, y_pred_model2)
print(f"P-value: {result['p_value']:.4f}")
print(f"Significativo: {result['significant']}")

# Intervalo de confiança via bootstrap
ci = validator.bootstrap_confidence_interval(
    accuracy_score, y_true, y_prob, n_bootstrap=1000
)
print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
```

### 5. Métricas Avançadas

Conjunto completo de métricas para publicação:

- Acurácia e Acurácia Balanceada
- Precision, Recall, F1-Score (macro e weighted)
- **Cohen's Kappa**: Concordância ajustada ao acaso
- **Matthews Correlation Coefficient**: Métrica balanceada
- **ROC-AUC**: One-vs-Rest e One-vs-One
- **ECE**: Calibração de probabilidades
- **Log Loss**: Perda probabilística

```python
from qualis_a1_improvements import AdvancedMetrics

# Calcular todas as métricas
metrics = AdvancedMetrics.calculate_all_metrics(
    y_true, y_pred, y_prob, class_names
)

# Gerar relatório
report = AdvancedMetrics.generate_report(metrics, class_names)
print(report)
```

## 📊 Estrutura de Logs e Artefatos

```
experiments_logs/
├── exp_20251230_143052_a1b2c3d4.log          # Log detalhado
├── exp_20251230_143052_a1b2c3d4_config.json  # Configuração
├── exp_20251230_143052_a1b2c3d4_results.json # Resultados finais
├── checkpoints/
│   ├── exp_20251230_143052_a1b2c3d4_best.pt  # Melhor modelo
│   ├── exp_20251230_143052_a1b2c3d4_epoch_50.pt
│   └── exp_20251230_143052_a1b2c3d4_epoch_100.pt
```

## 🔬 Níveis de Qualidade

### Excelente (Qualis A1)
- Acurácia > 95%
- Cohen's Kappa > 0.80
- ECE < 0.10
- Intervalos de confiança estreitos
- Diferença estatisticamente significativa vs baseline

### Muito Bom (Qualis A2)
- Acurácia > 90%
- Cohen's Kappa > 0.60
- ECE < 0.15

### Bom (Qualis B1)
- Acurácia > 85%
- Cohen's Kappa > 0.40
- ECE < 0.20

## 📖 Referências Implementadas

1. **Temperature Scaling** - Guo et al., ICML 2017
2. **Expected Calibration Error** - Naeini et al., AAAI 2015
3. **Cohen's Kappa** - Cohen, Educational and Psychological Measurement 1960
4. **Matthews Correlation** - Matthews, Biochimica et Biophysica Acta 1975
5. **McNemar's Test** - McNemar, Psychometrika 1947
6. **Bootstrap Methods** - Efron & Tibshirani, Statistical Science 1986

## 🚀 Exemplo Completo

```python
import torch
from qualis_a1_improvements import (
    ExperimentAuditor, ExperimentConfig,
    LearningCurveAnalyzer, ProbabilityCalibrator,
    StatisticalValidator, AdvancedMetrics
)

# 1. Configurar experimento
config = ExperimentConfig(
    data_dir='./dataset',
    num_classes=10,
    train_split=0.7,
    valid_split=0.15,
    test_split=0.15,
    model_name='ResNet50',
    fine_tune=True,
    dropout_p=0.5,
    epochs=100,
    learning_rate=0.0001,
    batch_size=32,
    optimizer_name='Adam',
    scheduler_name='CosineAnnealingLR',
    l2_lambda=0.01,
    l1_lambda=0.0,
    use_weighted_loss=True,
    label_smoothing=0.1,
    timestamp=datetime.now().isoformat(),
    random_seed=42,
    device=str(device),
    python_version=sys.version,
    torch_version=torch.__version__,
    experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# 2. Inicializar auditor
auditor = ExperimentAuditor()
auditor.log_experiment_start(config)

# 3. Treinar modelo
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []

for epoch in range(config.epochs):
    # ... treinamento ...
    
    train_metrics = {'loss': train_loss, 'acc': train_acc}
    valid_metrics = {'loss': valid_loss, 'acc': valid_acc}
    
    auditor.log_epoch(epoch, train_metrics, valid_metrics)
    
    if is_best:
        auditor.log_checkpoint(model, epoch, valid_metrics, is_best=True)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

# 4. Analisar curvas de aprendizado
analyzer = LearningCurveAnalyzer()
analysis = analyzer.analyze_learning_curve(
    train_losses, valid_losses,
    train_accs, valid_accs
)
fig = analyzer.plot_learning_curves(
    train_losses, valid_losses,
    train_accs, valid_accs,
    analysis=analysis
)
fig.savefig('learning_curves.png', dpi=300)

# 5. Avaliar no conjunto de teste
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# 6. Calcular métricas avançadas
metrics = AdvancedMetrics.calculate_all_metrics(
    y_true, y_pred, y_prob, class_names
)
report = AdvancedMetrics.generate_report(metrics, class_names)
print(report)

# 7. Validação estatística
calibrator = ProbabilityCalibrator()
ece = calibrator.calculate_ece(y_true, y_prob)
print(f"\nECE: {ece:.4f}")

# Plot calibração
fig_cal = calibrator.plot_calibration_curve(y_true, y_prob)
fig_cal.savefig('calibration_curve.png', dpi=300)

# 8. Intervalo de confiança
validator = StatisticalValidator()
ci = validator.bootstrap_confidence_interval(
    accuracy_score, y_true, y_pred, n_bootstrap=1000
)
print(f"Acurácia: {ci['mean']:.4f} ± {ci['std']:.4f}")
print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
```

## 💡 Dicas para Publicação

### 1. Reprodutibilidade
- Sempre defina `random_seed` fixo
- Registre todas as versões de bibliotecas
- Salve configuração completa do experimento
- Use o `ExperimentAuditor` para logging automático

### 2. Validação Rigorosa
- Use validação cruzada estratificada (k-fold)
- Calcule intervalos de confiança via bootstrap
- Compare com baseline usando teste de McNemar
- Reporte múltiplas métricas, não apenas acurácia

### 3. Calibração
- Sempre verifique ECE
- Use temperature scaling se ECE > 0.15
- Plote curvas de calibração
- Reporte confiança calibrada

### 4. Análise de Erros
- Inclua matriz de confusão normalizada
- Analise erros por classe
- Identifique padrões de falha
- Use Grad-CAM para explicabilidade

### 5. Comparação com Estado-da-Arte
- Compare com pelo menos 3 baselines
- Use mesma divisão de dados
- Reporte significância estatística
- Inclua tabela comparativa

## 📝 Template para Paper

### Abstract
```
We propose [method name] for [task]. Our approach achieves [X]% accuracy
with Cohen's Kappa of [Y], significantly outperforming baselines (p < 0.001,
McNemar's test). The model demonstrates excellent calibration (ECE = [Z])
and robust generalization. Code and models are publicly available.
```

### Results Section
```
Table 1: Performance comparison on [dataset]

Model          | Acc.  | Kappa | F1    | ECE   | 95% CI
---------------|-------|-------|-------|-------|----------------
Baseline-1     | 0.850 | 0.650 | 0.847 | 0.120 | [0.840, 0.860]
Baseline-2     | 0.880 | 0.720 | 0.876 | 0.105 | [0.870, 0.890]
Ours           | 0.952 | 0.850 | 0.951 | 0.082 | [0.945, 0.959]

Statistical significance: p < 0.001 (McNemar's test vs all baselines)
```

## 🔧 Solução de Problemas

### ECE Alto (> 0.20)
```python
# Aplicar temperature scaling
optimal_temp = 1.5  # Determinar via validação
probs_calibrated = calibrator.temperature_scaling(logits, optimal_temp)
```

### Overfitting Detectado
```python
# Seguir recomendações do analyzer
analysis = analyzer.analyze_learning_curve(...)
for rec in analysis['recommendations']:
    print(f"- {rec}")
```

### Baixa Significância Estatística
```python
# Aumentar número de bootstrap samples
ci = validator.bootstrap_confidence_interval(
    ..., n_bootstrap=10000  # Aumentar de 1000 para 10000
)
```

## 📧 Contato

- **Autor**: Prof. Marcelo Claro
- **Email**: marceloclaro@gmail.com
- **WhatsApp**: (88) 981587145
- **DOI**: https://doi.org/10.5281/zenodo.13910277

---

© 2025 Geomaker + IA - Todos os direitos reservados

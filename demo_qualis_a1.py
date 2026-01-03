"""
Demo: Uso do Módulo de Melhorias Qualis A1
===========================================

Este script demonstra como usar as melhorias implementadas para
publicação em periódicos Qualis A1.
"""

import sys
import numpy as np
from datetime import datetime

# Importar melhorias
from qualis_a1_improvements import (
    ExperimentAuditor, ExperimentConfig,
    LearningCurveAnalyzer, ProbabilityCalibrator,
    StatisticalValidator, AdvancedMetrics
)

print("="*70)
print("DEMO: Melhorias Qualis A1 - Geomaker v2.0")
print("="*70)
print()

# =============================================================================
# 1. CONFIGURAÇÃO E AUDITORIA DE EXPERIMENTO
# =============================================================================

print("1. Configuração e Auditoria de Experimento")
print("-"*70)

# Criar configuração do experimento
config = ExperimentConfig(
    data_dir='./dataset',
    num_classes=5,
    train_split=0.7,
    valid_split=0.15,
    test_split=0.15,
    model_name='ResNet50',
    fine_tune=True,
    dropout_p=0.5,
    epochs=50,
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
    device='cpu',
    python_version=sys.version,
    torch_version='2.0.0',
    experiment_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# Inicializar auditor
auditor = ExperimentAuditor(log_dir='./demo_experiments')
auditor.log_experiment_start(config)

print(f"✓ Experimento ID: {config.experiment_id}")
print(f"✓ Configuração salva em: {auditor.artifacts['config']}")
print()

# =============================================================================
# 2. SIMULAÇÃO DE TREINAMENTO E LOGGING
# =============================================================================

print("2. Simulação de Treinamento")
print("-"*70)

# Simular treinamento (dados fictícios)
np.random.seed(42)

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

print("Simulando 50 épocas de treinamento...")
for epoch in range(1, 51):
    # Simular melhoria gradual
    base_train_loss = 2.0 * np.exp(-epoch/10) + 0.1
    base_valid_loss = 2.0 * np.exp(-epoch/10) + 0.15
    
    train_loss = base_train_loss + np.random.normal(0, 0.02)
    valid_loss = base_valid_loss + np.random.normal(0, 0.03)
    
    train_acc = 1 - (train_loss / 2.0)
    valid_acc = 1 - (valid_loss / 2.0)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    
    # Log a cada 10 épocas
    if epoch % 10 == 0:
        train_metrics = {'loss': train_loss, 'acc': train_acc}
        valid_metrics = {'loss': valid_loss, 'acc': valid_acc}
        auditor.log_epoch(epoch, train_metrics, valid_metrics)

print(f"✓ {len(train_losses)} épocas simuladas")
print(f"✓ Métricas registradas no log")
print()

# =============================================================================
# 3. ANÁLISE DE CURVAS DE APRENDIZADO
# =============================================================================

print("3. Análise de Curvas de Aprendizado")
print("-"*70)

analyzer = LearningCurveAnalyzer()

# Analisar curvas
analysis = analyzer.analyze_learning_curve(
    train_losses, valid_losses,
    train_accs, valid_accs
)

print(f"Status: {analysis['status'].upper()}")
print(f"Gap de Perda: {analysis['gap_loss']:.4f}")
print(f"Gap de Acurácia: {analysis['gap_acc']:.4f}")
print(f"Tendência: {'Aumentando' if analysis['trend'] > 0 else 'Diminuindo'}")
print()
print("Recomendações:")
for i, rec in enumerate(analysis['recommendations'], 1):
    print(f"  {i}. {rec}")
print()

# Gerar gráfico
try:
    fig = analyzer.plot_learning_curves(
        train_losses, valid_losses,
        train_accs, valid_accs,
        analysis=analysis
    )
    fig.savefig('./demo_experiments/learning_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: ./demo_experiments/learning_curves.png")
except Exception as e:
    print(f"⚠ Não foi possível salvar o gráfico: {e}")
print()

# =============================================================================
# 4. GERAÇÃO DE DADOS DE TESTE SINTÉTICOS
# =============================================================================

print("4. Simulação de Resultados de Teste")
print("-"*70)

# Gerar dados sintéticos para demonstração
n_samples = 500
n_classes = 5

# Labels verdadeiros
y_true = np.random.randint(0, n_classes, n_samples)

# Simular probabilidades com boa acurácia (~92%)
y_prob = np.zeros((n_samples, n_classes))
for i in range(n_samples):
    true_class = y_true[i]
    # Alta probabilidade para classe correta
    y_prob[i, true_class] = np.random.uniform(0.6, 0.95)
    # Distribuir resto
    remaining = 1.0 - y_prob[i, true_class]
    other_classes = [c for c in range(n_classes) if c != true_class]
    random_probs = np.random.dirichlet(np.ones(len(other_classes)))
    for j, c in enumerate(other_classes):
        y_prob[i, c] = remaining * random_probs[j]

# Predições
y_pred = np.argmax(y_prob, axis=1)

# Introduzir alguns erros (~8%)
n_errors = int(0.08 * n_samples)
error_indices = np.random.choice(n_samples, n_errors, replace=False)
for idx in error_indices:
    true_class = y_true[idx]
    wrong_class = (true_class + 1) % n_classes
    y_pred[idx] = wrong_class

print(f"✓ Gerados {n_samples} exemplos de teste")
print(f"✓ {n_classes} classes")
print()

# =============================================================================
# 5. MÉTRICAS AVANÇADAS
# =============================================================================

print("5. Cálculo de Métricas Avançadas")
print("-"*70)

class_names = [f"Classe_{i}" for i in range(n_classes)]

metrics = AdvancedMetrics.calculate_all_metrics(
    y_true, y_pred, y_prob, class_names
)

print(f"Acurácia: {metrics['accuracy']:.4f}")
print(f"Acurácia Balanceada: {metrics['balanced_accuracy']:.4f}")
print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
print(f"Matthews Corrcoef: {metrics['matthews_corrcoef']:.4f}")
print(f"Log Loss: {metrics['log_loss']:.4f}")
print(f"ECE (Calibração): {metrics['ece']:.4f}")
if 'roc_auc_ovr' in metrics:
    print(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
print()

# Gerar relatório completo
report = AdvancedMetrics.generate_report(metrics, class_names)
print(report)
print()

# =============================================================================
# 6. CALIBRAÇÃO DE PROBABILIDADES
# =============================================================================

print("6. Análise de Calibração")
print("-"*70)

calibrator = ProbabilityCalibrator()

# Calcular ECE
ece = calibrator.calculate_ece(y_true, y_prob)
print(f"ECE: {ece:.4f}")

if ece < 0.10:
    print("✓ Probabilidades bem calibradas")
elif ece < 0.20:
    print("⚠ Calibração moderada (considerar temperature scaling)")
else:
    print("⚠ Probabilidades mal calibradas (requer calibração)")
print()

# Gerar curva de calibração
try:
    fig_cal = calibrator.plot_calibration_curve(y_true, y_prob)
    fig_cal.savefig('./demo_experiments/calibration_curve.png', dpi=150, bbox_inches='tight')
    print("✓ Curva de calibração salva: ./demo_experiments/calibration_curve.png")
except Exception as e:
    print(f"⚠ Não foi possível salvar curva de calibração: {e}")
print()

# =============================================================================
# 7. VALIDAÇÃO ESTATÍSTICA
# =============================================================================

print("7. Validação Estatística")
print("-"*70)

validator = StatisticalValidator()

# Intervalo de confiança via bootstrap
from sklearn.metrics import accuracy_score

ci = validator.bootstrap_confidence_interval(
    accuracy_score, y_true, y_pred, n_bootstrap=1000, ci_level=0.95
)

print(f"Acurácia Média (Bootstrap): {ci['mean']:.4f} ± {ci['std']:.4f}")
print(f"Intervalo de Confiança 95%: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
print(f"Margem de Erro: ±{ci['margin_error']:.4f}")
print()

# Teste de McNemar (comparar com modelo fictício)
print("Teste de McNemar (comparação com baseline):")
# Simular predições de baseline com ~85% de acurácia
y_pred_baseline = y_pred.copy()
n_baseline_errors = int(0.15 * n_samples)
baseline_error_indices = np.random.choice(n_samples, n_baseline_errors, replace=False)
for idx in baseline_error_indices:
    true_class = y_true[idx]
    wrong_class = (true_class + 2) % n_classes
    y_pred_baseline[idx] = wrong_class

mcnemar_result = validator.mcnemar_test(y_true, y_pred, y_pred_baseline)
print(f"  Estatística: {mcnemar_result['statistic']:.4f}")
print(f"  P-value: {mcnemar_result['p_value']:.4f}")
print(f"  Significativo (α=0.05): {mcnemar_result['significant']}")
print(f"  Interpretação: {mcnemar_result['interpretation']}")
print()

# =============================================================================
# RESUMO FINAL
# =============================================================================

print("="*70)
print("RESUMO DA DEMONSTRAÇÃO")
print("="*70)
print()
print("✓ Auditoria completa configurada e funcionando")
print(f"✓ {len(train_losses)} épocas simuladas e registradas")
print("✓ Análise de curvas de aprendizado realizada")
print("✓ Métricas avançadas calculadas")
print("✓ Calibração de probabilidades analisada")
print("✓ Validação estatística realizada")
print()
print("Arquivos gerados:")
print(f"  - {auditor.artifacts.get('config', 'N/A')}")
print("  - ./demo_experiments/learning_curves.png")
print("  - ./demo_experiments/calibration_curve.png")
print()
print("Este pipeline está pronto para publicação Qualis A1!")
print()
print("="*70)

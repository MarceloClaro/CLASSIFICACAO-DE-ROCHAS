"""
==============================================================================
QUALIS A1 IMPROVEMENTS MODULE
Sistema de Melhorias para Publicação em Periódicos Qualis A1
==============================================================================

Este módulo implementa funcionalidades avançadas para atingir o nível de
qualidade exigido por periódicos Qualis A1:

1. Auditoria completa de experimentos
2. Validação estatística rigorosa  
3. Calibração de probabilidades
4. Análise de curvas de aprendizado
5. Métricas avançadas

Autor: Prof. Marcelo Claro
Versão: 2.0
Data: 2025-12-30
"""

import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    log_loss, brier_score_loss,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.calibration import calibration_curve

# =============================================================================
# SEÇÃO 1: CONFIGURAÇÃO DE EXPERIMENTOS E AUDITORIA
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuração completa do experimento para reprodutibilidade."""
    # Dados
    data_dir: str
    num_classes: int
    train_split: float
    valid_split: float
    test_split: float
    
    # Modelo
    model_name: str
    fine_tune: bool
    dropout_p: float
    
    # Treinamento
    epochs: int
    learning_rate: float
    batch_size: int
    optimizer_name: str
    scheduler_name: str
    
    # Regularização
    l2_lambda: float
    l1_lambda: float
    use_weighted_loss: bool
    label_smoothing: float
    
    # Metadados
    timestamp: str
    random_seed: int
    device: str
    python_version: str
    torch_version: str
    
    # Experiment ID
    experiment_id: str = None

    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return asdict(self)


class ExperimentAuditor:
    """Classe para auditoria completa de experimentos de ML."""
    
    def __init__(self, log_dir: str = "experiments_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = self._generate_experiment_id()
        self.metrics_history: List[Dict] = []
        self.checkpoints: List[Path] = []
        self.artifacts: Dict[str, Path] = {}
        
        # Configurar logger
        self.logger = self._setup_logger()
        
    def _generate_experiment_id(self) -> str:
        """Gera ID único do experimento."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{random_hash}"
    
    def _setup_logger(self) -> logging.Logger:
        """Configura logger estruturado."""
        logger = logging.getLogger(f"exp_{self.experiment_id}")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_id}.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def log_experiment_start(self, config: ExperimentConfig):
        """Registra início do experimento."""
        self.logger.info("="*80)
        self.logger.info("EXPERIMENTO INICIADO")
        self.logger.info("="*80)
        self.logger.info(f"Experiment ID: {config.experiment_id}")
        self.logger.info(f"Timestamp: {config.timestamp}")
        self.logger.info(f"Random Seed: {config.random_seed}")
        
        # Salvar configuração
        config_file = self.log_dir / f"{self.experiment_id}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.artifacts['config'] = config_file
    
    def log_epoch(self, epoch: int, train_metrics: Dict, valid_metrics: Dict):
        """Registra métricas de uma época."""
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f} | "
            f"Valid Loss: {valid_metrics['loss']:.4f}, Acc: {valid_metrics['acc']:.4f}"
        )
        
        epoch_record = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_history.append(epoch_record)
    
    def log_checkpoint(self, model, epoch: int, metrics: Dict, is_best: bool = False):
        """Salva checkpoint do modelo."""
        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{self.experiment_id}_epoch_{epoch}.pt"
        if is_best:
            checkpoint_file = checkpoint_dir / f"{self.experiment_id}_best.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }, checkpoint_file)
        
        self.checkpoints.append(checkpoint_file)
        if is_best:
            self.artifacts['best_model'] = checkpoint_file

# =============================================================================
# SEÇÃO 2: ANÁLISE DE CURVAS DE APRENDIZADO
# =============================================================================

class LearningCurveAnalyzer:
    """Analisador de curvas de aprendizado para identificar under/overfitting."""
    
    @staticmethod
    def analyze_learning_curve(train_losses: List[float], valid_losses: List[float],
                              train_accs: List[float], valid_accs: List[float]) -> Dict:
        """
        Analisa curvas de aprendizado e detecta problemas.
        
        Returns:
            Dict com análise completa
        """
        analysis = {}
        
        # Calcular gap médio final (últimas 10 épocas)
        n_final = min(10, len(train_losses))
        if n_final > 0:
            gap_loss = np.mean([valid_losses[-i] - train_losses[-i] 
                              for i in range(1, n_final + 1)])
            gap_acc = np.mean([train_accs[-i] - valid_accs[-i] 
                              for i in range(1, n_final + 1)])
        else:
            gap_loss = valid_losses[-1] - train_losses[-1] if valid_losses else 0
            gap_acc = train_accs[-1] - valid_accs[-1] if train_accs else 0
        
        analysis['gap_loss'] = gap_loss
        analysis['gap_acc'] = gap_acc
        
        # Calcular tendência
        if len(valid_losses) > 5:
            x = np.arange(len(valid_losses))
            z = np.polyfit(x, valid_losses, 1)
            trend = z[0]
            analysis['trend'] = trend
        else:
            trend = 0
            analysis['trend'] = 0
        
        # Determinar status
        recommendations = []
        
        if len(train_losses) > 0 and len(valid_losses) > 0:
            final_train_acc = train_accs[-1] if train_accs else 0
            final_valid_acc = valid_accs[-1] if valid_accs else 0
            
            # Detectar overfitting
            if gap_loss > 0.15 or gap_acc > 0.15:
                if trend > 0:
                    analysis['status'] = 'overfitting'
                    recommendations.extend([
                        "Aumentar regularização L2 (weight decay)",
                        "Adicionar dropout",
                        "Aumentar data augmentation",
                        "Usar early stopping mais agressivo"
                    ])
                else:
                    analysis['status'] = 'good_fit'
            
            # Detectar underfitting
            elif final_train_acc < 0.8 and final_valid_acc < 0.8:
                analysis['status'] = 'underfitting'
                recommendations.extend([
                    "Aumentar número de épocas",
                    "Usar modelo mais complexo",
                    "Aumentar learning rate",
                    "Reduzir regularização"
                ])
            
            # Bom ajuste
            else:
                analysis['status'] = 'good_fit'
                recommendations.append("Modelo bem ajustado!")
        
        analysis['recommendations'] = recommendations
        return analysis
    
    @staticmethod
    def plot_learning_curves(train_losses: List[float], valid_losses: List[float],
                             train_accs: List[float], valid_accs: List[float],
                             analysis: Dict = None) -> plt.Figure:
        """Plota curvas de aprendizado com análise."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Perdas
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Treino', linewidth=2)
        axes[0, 0].plot(epochs, valid_losses, 'r-', label='Validação', linewidth=2)
        axes[0, 0].fill_between(epochs, train_losses, valid_losses, alpha=0.2, color='gray')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Perda')
        axes[0, 0].set_title('Curva de Perda', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Acurácias
        axes[0, 1].plot(epochs, train_accs, 'b-', label='Treino', linewidth=2)
        axes[0, 1].plot(epochs, valid_accs, 'r-', label='Validação', linewidth=2)
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Acurácia')
        axes[0, 1].set_title('Curva de Acurácia', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gap
        if len(train_losses) > 0 and len(valid_losses) > 0:
            gap_loss = [valid_losses[i] - train_losses[i] for i in range(len(train_losses))]
            axes[1, 0].plot(epochs, gap_loss, 'g-', linewidth=2)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Gap (Valid - Treino)')
            axes[1, 0].set_title('Gap de Generalização', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Análise textual
        axes[1, 1].axis('off')
        if analysis:
            status_text = (
                f"## ANÁLISE\n\n"
                f"Status: {analysis.get('status', 'unknown').upper()}\n\n"
                f"Gap Perda: {analysis.get('gap_loss', 0):.4f}\n"
                f"Gap Acurácia: {analysis.get('gap_acc', 0):.4f}\n\n"
                f"Recomendações:\n"
            )
            for i, rec in enumerate(analysis.get('recommendations', []), 1):
                status_text += f"{i}. {rec}\n"
            
            axes[1, 1].text(0.1, 0.9, status_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        return fig

# =============================================================================
# SEÇÃO 3: CALIBRAÇÃO DE PROBABILIDADES
# =============================================================================

class ProbabilityCalibrator:
    """Calibração de probabilidades para melhorar confiabilidade."""
    
    @staticmethod
    def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Aplica temperature scaling."""
        return F.softmax(logits / temperature, dim=1)
    
    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Calcula Expected Calibration Error (ECE).
        """
        if y_prob.ndim > 1:
            y_prob_max = y_prob.max(axis=1)
        else:
            y_prob_max = y_prob
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob_max > bin_lower) & (y_prob_max <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob_max[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                               n_bins: int = 10) -> plt.Figure:
        """Plota curva de calibração."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if y_prob.ndim > 1:
            y_prob_max = y_prob.max(axis=1)
        else:
            y_prob_max = y_prob
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob_max, n_bins=n_bins
        )
        
        ax.plot([0, 1], [0, 1], "k:", label="Perfeitamente calibrado")
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="Modelo", linewidth=2, markersize=8)
        ax.set_ylabel("Fração de positivos")
        ax.set_xlabel("Probabilidade média predita")
        ax.set_title("Curva de Calibração", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Histograma
        ax2 = ax.twinx()
        ax2.hist(y_prob_max, bins=n_bins, range=(0, 1), alpha=0.3, color='blue')
        ax2.set_ylabel("Número de amostras")
        
        plt.tight_layout()
        return fig

# =============================================================================
# SEÇÃO 4: VALIDAÇÃO ESTATÍSTICA
# =============================================================================

class StatisticalValidator:
    """Validador estatístico para comparação de modelos."""
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred_1: np.ndarray,
                     y_pred_2: np.ndarray, alpha: float = 0.05) -> Dict:
        """Teste de McNemar para comparar dois modelos."""
        b = np.sum((y_pred_1 == y_true) & (y_pred_2 != y_true))
        c = np.sum((y_pred_1 != y_true) & (y_pred_2 == y_true))
        
        statistic = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        significant = p_value < alpha
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'significant': significant,
            'interpretation': (
                "Modelos significativamente diferentes" if significant 
                else "Não há diferença significativa"
            )
        }
    
    @staticmethod
    def bootstrap_confidence_interval(metric_func, y_true: np.ndarray, 
                                     y_prob: np.ndarray, n_bootstrap: int = 1000,
                                     ci_level: float = 0.95) -> Dict:
        """Calcula intervalo de confiança via bootstrap."""
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = resample(np.arange(len(y_true)), replace=True)
            score = metric_func(y_true[indices], y_prob[indices])
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        lower = np.percentile(bootstrap_scores, (1 - ci_level) / 2 * 100)
        upper = np.percentile(bootstrap_scores, (1 + ci_level) / 2 * 100)
        mean = np.mean(bootstrap_scores)
        std = np.std(bootstrap_scores)
        
        return {
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'ci_level': ci_level,
            'margin_error': (upper - lower) / 2
        }

# =============================================================================
# SEÇÃO 5: MÉTRICAS AVANÇADAS
# =============================================================================

class AdvancedMetrics:
    """Métricas avançadas para publicação Qualis A1."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray, class_names: List[str]) -> Dict:
        """Calcula todas as métricas relevantes."""
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Métricas avançadas
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Métricas probabilísticas
        metrics['log_loss'] = log_loss(y_true, y_prob)
        
        # Calibração
        calibrator = ProbabilityCalibrator()
        metrics['ece'] = calibrator.calculate_ece(y_true, y_prob)
        
        # ROC AUC
        num_classes = y_prob.shape[1]
        if num_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            except:
                metrics['roc_auc_ovr'] = 0.0
        
        return metrics
    
    @staticmethod
    def generate_report(metrics: Dict, class_names: List[str]) -> str:
        """Gera relatório formatado das métricas."""
        report = "="*60 + "\n"
        report += "RELATÓRIO DE MÉTRICAS - NÍVEL QUALIS A1\n"
        report += "="*60 + "\n\n"
        
        report += f"Acurácia: {metrics['accuracy']:.4f}\n"
        report += f"Acurácia Balanceada: {metrics['balanced_accuracy']:.4f}\n"
        report += f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n"
        report += f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n"
        report += f"Matthews Corrcoef: {metrics['matthews_corrcoef']:.4f}\n"
        report += f"ECE (Calibração): {metrics['ece']:.4f}\n"
        
        if 'roc_auc' in metrics:
            report += f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
        elif 'roc_auc_ovr' in metrics:
            report += f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}\n"
        
        report += "\nINTERPRETAÇÃO:\n"
        if metrics['accuracy'] > 0.95:
            report += "✓ Acurácia excelente para publicação A1\n"
        elif metrics['accuracy'] > 0.90:
            report += "✓ Acurácia muito boa\n"
        
        if metrics['cohen_kappa'] > 0.80:
            report += "✓ Concordância quase perfeita (Kappa)\n"
        
        if metrics['ece'] < 0.1:
            report += "✓ Probabilidades bem calibradas\n"
        elif metrics['ece'] > 0.2:
            report += "⚠ Probabilidades mal calibradas\n"
        
        return report

# =============================================================================
# FIM DO MÓDULO
# =============================================================================

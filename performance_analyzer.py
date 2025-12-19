"""
Módulo de Análise de Eficiência e Desempenho de Classificação
Desenvolvido para análise científica de qualidade Qualis A1
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

class PerformanceAnalyzer:
    """
    Classe para análise detalhada de eficiência e desempenho de modelos de classificação
    """
    
    def __init__(self, device):
        self.device = device
        self.metrics = {}
        self.timing_data = {}
        
    def measure_inference_time(self, model, dataloader, num_samples=100):
        """
        Mede o tempo de inferência médio do modelo
        """
        model.eval()
        times = []
        samples_processed = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                if samples_processed >= num_samples:
                    break
                    
                inputs = inputs.to(self.device)
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                
                times.append(end_time - start_time)
                samples_processed += inputs.size(0)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        self.timing_data['inference_avg_ms'] = avg_time * 1000
        self.timing_data['inference_std_ms'] = std_time * 1000
        self.timing_data['samples_per_second'] = 1.0 / avg_time if avg_time > 0 else 0
        
        return avg_time, std_time
    
    def measure_memory_usage(self, model):
        """
        Mede o uso de memória do modelo
        """
        # Memória do modelo
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_memory_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Memória do sistema
        process = psutil.Process()
        system_memory_mb = process.memory_info().rss / (1024 ** 2)
        
        # Memória GPU se disponível
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        
        self.metrics['model_memory_mb'] = model_memory_mb
        self.metrics['system_memory_mb'] = system_memory_mb
        self.metrics['gpu_memory_mb'] = gpu_memory_mb
        
        return model_memory_mb, system_memory_mb, gpu_memory_mb
    
    def compute_detailed_metrics(self, model, dataloader, classes):
        """
        Calcula métricas detalhadas de classificação
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # Relatório de classificação
        report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        
        # Métricas por classe
        class_metrics = {}
        for class_name in classes:
            if class_name in report:
                class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        # Métricas globais
        self.metrics['accuracy'] = report['accuracy']
        self.metrics['macro_avg_precision'] = report['macro avg']['precision']
        self.metrics['macro_avg_recall'] = report['macro avg']['recall']
        self.metrics['macro_avg_f1'] = report['macro avg']['f1-score']
        self.metrics['weighted_avg_precision'] = report['weighted avg']['precision']
        self.metrics['weighted_avg_recall'] = report['weighted avg']['recall']
        self.metrics['weighted_avg_f1'] = report['weighted avg']['f1-score']
        
        # AUC-ROC para multiclasse
        if len(classes) > 2:
            binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
            try:
                roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
                self.metrics['roc_auc_weighted'] = roc_auc
            except (ValueError, IndexError) as e:
                # Não foi possível calcular AUC-ROC (ex: apenas uma classe presente)
                self.metrics['roc_auc_weighted'] = 0.0
        
        return class_metrics, all_labels, all_preds, all_probs
    
    def compute_efficiency_score(self):
        """
        Calcula um score de eficiência baseado em múltiplos fatores
        """
        # Constantes de normalização
        TIME_NORMALIZATION_FACTOR = 100  # ms
        MEMORY_NORMALIZATION_FACTOR = 100  # MB
        
        # Normalizar métricas (valores entre 0 e 1)
        accuracy_score = self.metrics.get('accuracy', 0)
        
        # Inferência rápida é melhor (normalizar para 0-1)
        inference_time = self.timing_data.get('inference_avg_ms', 1000)
        time_score = 1.0 / (1.0 + inference_time / TIME_NORMALIZATION_FACTOR)
        
        # Memória baixa é melhor
        memory_mb = self.metrics.get('model_memory_mb', 1000)
        memory_score = 1.0 / (1.0 + memory_mb / MEMORY_NORMALIZATION_FACTOR)
        
        # Score ponderado
        efficiency_score = (
            0.5 * accuracy_score +  # 50% peso na acurácia
            0.3 * time_score +       # 30% peso no tempo
            0.2 * memory_score       # 20% peso na memória
        )
        
        self.metrics['efficiency_score'] = efficiency_score
        return efficiency_score
    
    def generate_performance_report(self):
        """
        Gera relatório estruturado de performance
        """
        report = {
            'Métricas de Classificação': {
                'Acurácia': f"{self.metrics.get('accuracy', 0):.4f}",
                'Precisão Macro': f"{self.metrics.get('macro_avg_precision', 0):.4f}",
                'Recall Macro': f"{self.metrics.get('macro_avg_recall', 0):.4f}",
                'F1-Score Macro': f"{self.metrics.get('macro_avg_f1', 0):.4f}",
                'AUC-ROC': f"{self.metrics.get('roc_auc_weighted', 0):.4f}",
            },
            'Métricas de Eficiência': {
                'Tempo de Inferência Médio (ms)': f"{self.timing_data.get('inference_avg_ms', 0):.2f}",
                'Desvio Padrão Tempo (ms)': f"{self.timing_data.get('inference_std_ms', 0):.2f}",
                'Amostras por Segundo': f"{self.timing_data.get('samples_per_second', 0):.2f}",
            },
            'Uso de Memória': {
                'Memória do Modelo (MB)': f"{self.metrics.get('model_memory_mb', 0):.2f}",
                'Memória do Sistema (MB)': f"{self.metrics.get('system_memory_mb', 0):.2f}",
                'Memória GPU (MB)': f"{self.metrics.get('gpu_memory_mb', 0):.2f}",
            },
            'Score de Eficiência Geral': f"{self.metrics.get('efficiency_score', 0):.4f}"
        }
        
        return report
    
    def plot_performance_comparison(self, model_results: Dict[str, Dict]):
        """
        Cria gráficos comparativos entre diferentes modelos
        
        Args:
            model_results: Dicionário com resultados de múltiplos modelos
                          {'ResNet18': metrics_dict, 'ResNet50': metrics_dict, ...}
        """
        models = list(model_results.keys())
        
        # Preparar dados
        accuracies = [model_results[m].get('accuracy', 0) for m in models]
        inference_times = [model_results[m].get('inference_avg_ms', 0) for m in models]
        memory_usage = [model_results[m].get('model_memory_mb', 0) for m in models]
        efficiency_scores = [model_results[m].get('efficiency_score', 0) for m in models]
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparação de Performance entre Modelos', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Acurácia
        axes[0, 0].bar(models, accuracies, color='steelblue')
        axes[0, 0].set_ylabel('Acurácia')
        axes[0, 0].set_title('Acurácia por Modelo')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Gráfico 2: Tempo de Inferência
        axes[0, 1].bar(models, inference_times, color='coral')
        axes[0, 1].set_ylabel('Tempo (ms)')
        axes[0, 1].set_title('Tempo de Inferência Médio')
        for i, v in enumerate(inference_times):
            axes[0, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        # Gráfico 3: Uso de Memória
        axes[1, 0].bar(models, memory_usage, color='lightgreen')
        axes[1, 0].set_ylabel('Memória (MB)')
        axes[1, 0].set_title('Uso de Memória do Modelo')
        for i, v in enumerate(memory_usage):
            axes[1, 0].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        # Gráfico 4: Score de Eficiência
        axes[1, 1].bar(models, efficiency_scores, color='gold')
        axes[1, 1].set_ylabel('Score de Eficiência')
        axes[1, 1].set_title('Score de Eficiência Geral')
        axes[1, 1].set_ylim([0, 1])
        for i, v in enumerate(efficiency_scores):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_detailed_metrics(self, class_metrics, classes):
        """
        Cria visualizações detalhadas por classe
        """
        metrics_df = pd.DataFrame(class_metrics).T
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Métricas Detalhadas por Classe', fontsize=14, fontweight='bold')
        
        # Precision
        metrics_df['precision'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Precisão por Classe')
        axes[0].set_ylabel('Precisão')
        axes[0].set_ylim([0, 1])
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        
        # Recall
        metrics_df['recall'].plot(kind='bar', ax=axes[1], color='lightcoral')
        axes[1].set_title('Recall por Classe')
        axes[1].set_ylabel('Recall')
        axes[1].set_ylim([0, 1])
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        
        # F1-Score
        metrics_df['f1-score'].plot(kind='bar', ax=axes[2], color='lightgreen')
        axes[2].set_title('F1-Score por Classe')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_ylim([0, 1])
        axes[2].set_xticklabels(classes, rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def export_report_to_csv(self, filename='performance_report.csv'):
        """
        Exporta relatório para CSV
        """
        report_data = []
        
        # Adicionar métricas de classificação
        report_data.append(['Métrica', 'Valor'])
        report_data.append(['Acurácia', self.metrics.get('accuracy', 0)])
        report_data.append(['Precisão Macro', self.metrics.get('macro_avg_precision', 0)])
        report_data.append(['Recall Macro', self.metrics.get('macro_avg_recall', 0)])
        report_data.append(['F1-Score Macro', self.metrics.get('macro_avg_f1', 0)])
        report_data.append(['AUC-ROC', self.metrics.get('roc_auc_weighted', 0)])
        report_data.append(['', ''])
        
        # Adicionar métricas de tempo
        report_data.append(['Tempo Inferência Médio (ms)', self.timing_data.get('inference_avg_ms', 0)])
        report_data.append(['Amostras/Segundo', self.timing_data.get('samples_per_second', 0)])
        report_data.append(['', ''])
        
        # Adicionar métricas de memória
        report_data.append(['Memória Modelo (MB)', self.metrics.get('model_memory_mb', 0)])
        report_data.append(['Memória Sistema (MB)', self.metrics.get('system_memory_mb', 0)])
        report_data.append(['', ''])
        
        # Score de eficiência
        report_data.append(['Score de Eficiência', self.metrics.get('efficiency_score', 0)])
        
        df = pd.DataFrame(report_data)
        df.to_csv(filename, index=False, header=False)
        return filename

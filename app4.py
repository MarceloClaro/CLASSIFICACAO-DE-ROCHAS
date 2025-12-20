import os

# Disable CrewAI telemetry to avoid signal handler errors in non-main threads (Streamlit)
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'

import zipfile
import shutil
import tempfile
import random
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import streamlit as st
import gc
import logging
import base64
import io
# Importações adicionais para Grad-CAM
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, LayerCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
import cv2
# Importar otimizadores avançados
try:
    import torch_optimizer as optim_advanced
    ADVANCED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZERS_AVAILABLE = False

# Importar timm para Vision Transformers
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Importar CrewAI para agentes inteligentes
try:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import WebsiteSearchTool, SerperDevTool
    from langchain.llms import OpenAI
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Importar APIs com suporte de visão
try:
    # Prioritize stable google-generativeai package (recommended)
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    GEMINI_NEW_API = False  # Stable google-generativeai package
except ImportError:
    # Fallback to new beta package if stable not available
    try:
        import google.genai as genai
        GEMINI_AVAILABLE = True
        GEMINI_NEW_API = True  # Beta google-genai package
    except ImportError:
        GEMINI_AVAILABLE = False
        GEMINI_NEW_API = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Import multi-agent system
try:
    from multi_agent_system import ManagerAgent
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False

# Import new advanced modules from app5
try:
    import sys
    # Force module reload to avoid Streamlit caching issues
    # Note: This is necessary because Streamlit's module system can cache imports
    # incorrectly, leading to KeyError exceptions. This is different from function
    # caching and cannot be solved with @st.cache decorators.
    if 'visualization_3d' in sys.modules:
        del sys.modules['visualization_3d']
    from visualization_3d import visualize_pca_3d, visualize_activation_heatmap_3d, create_interactive_3d_visualization
    VISUALIZATION_3D_AVAILABLE = True
except (ImportError, KeyError, ModuleNotFoundError) as e:
    VISUALIZATION_3D_AVAILABLE = False
    # print(f"Warning: visualization_3d not available: {e}")

try:
    # Force module reload to avoid Streamlit caching issues
    if 'ai_chat_module' in sys.modules:
        del sys.modules['ai_chat_module']
    from ai_chat_module import AIAnalyzer, describe_gradcam_regions, get_gemini_model_path
    AI_CHAT_AVAILABLE = True
except (ImportError, KeyError, ModuleNotFoundError) as e:
    AI_CHAT_AVAILABLE = False
    # print(f"Warning: ai_chat_module not available: {e}")
    # Define fallback function if module not available
    def get_gemini_model_path(model_name: str, use_new_api: bool = False) -> str:
        """Fallback: Get the correct model path for Gemini API calls."""
        clean_name = model_name.replace('models/', '')
        if use_new_api:
            return f'models/{clean_name}'
        else:
            return clean_name

try:
    # Force module reload to avoid Streamlit caching issues
    if 'academic_references' in sys.modules:
        del sys.modules['academic_references']
    from academic_references import AcademicReferenceFetcher, format_references_for_display
    ACADEMIC_REF_AVAILABLE = True
except (ImportError, KeyError, ModuleNotFoundError) as e:
    ACADEMIC_REF_AVAILABLE = False
    # print(f"Warning: academic_references not available: {e}")

try:
    from genetic_interpreter import GeneticDiagnosticInterpreter
    GENETIC_INTERP_AVAILABLE = True
except ImportError:
    GENETIC_INTERP_AVAILABLE = False

# Constants
CONVERGENCE_CHECK_EPOCHS = 5  # Number of recent epochs to check for convergence stability

# Valid model lists (based on official Google Gemini Cookbook)
# Reference: https://github.com/google-gemini/cookbook
VALID_GEMINI_MODELS = [
    'gemini-2.5-flash',  # ⭐ Recommended - fast and efficient with multimodal support
    'gemini-2.5-flash-lite',  # Even faster version
    'gemini-2.5-pro',  # Advanced model with thinking capabilities
    'gemini-3-flash-preview',  # Preview of next generation
    'gemini-3-pro-preview',  # Advanced preview
    # Legacy models (for backwards compatibility, but not recommended)
    'gemini-1.5-pro-latest',
    'gemini-1.5-flash-latest',
    'gemini-1.0-pro-latest',
    'gemini-pro',
    'gemini-1.0-pro-vision-latest'
]

VALID_GROQ_MODELS = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'mixtral-8x7b-32768',
    'llama-3.1-70b-versatile',
    'llama-3.1-8b-instant'
]

def validate_model_name(model_name, provider):
    """
    Validate and sanitize model name to ensure deprecated models are not used.
    Based on official Google Gemini Cookbook: https://github.com/google-gemini/cookbook
    
    Args:
        model_name: The model name from session state or user input
        provider: The AI provider ('Gemini', 'gemini', 'Groq', 'groq')
    
    Returns:
        str: Valid model name or default model for the provider
    """
    if not model_name:
        # Return default model if none specified
        # Normalize provider to lowercase for comparison
        provider_lower = provider.lower() if provider else ''
        return 'gemini-2.5-flash' if provider_lower == 'gemini' else 'mixtral-8x7b-32768'
    
    # Normalize provider to lowercase for case-insensitive comparison
    provider_lower = provider.lower() if provider else ''
    
    # Check if model is in valid list
    if provider_lower == 'gemini':
        if model_name in VALID_GEMINI_MODELS:
            return model_name
        else:
            # Model is deprecated or invalid, return recommended default
            return 'gemini-2.5-flash'
    elif provider_lower == 'groq':
        if model_name in VALID_GROQ_MODELS:
            return model_name
        else:
            # Model is invalid, return default
            return 'mixtral-8x7b-32768'
    else:
        # Unknown provider, return the model as-is
        return model_name

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para tornar os gráficos mais bonitos
sns.set_style('whitegrid')

def set_seed(seed):
    """
    Define a seed para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed para reprodutibilidade

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormaliza um tensor de imagem normalizado com valores ImageNet.
    
    Args:
        tensor: Tensor de imagem (C, H, W) ou array numpy (H, W, C)
        mean: Média usada na normalização
        std: Desvio padrão usado na normalização
    
    Returns:
        Array numpy (H, W, C) com valores no intervalo [0, 1]
    """
    if isinstance(tensor, torch.Tensor):
        # Convert tensor to numpy
        image = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image = tensor
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image

# ==================== STATISTICAL ANALYSIS MODULE ====================

class StatisticalAnalyzer:
    """
    Classe para análise estatística avançada de predições do modelo.
    Inclui intervalos de confiança, testes de significância, bootstrap, etc.
    """
    
    @staticmethod
    def calculate_confidence_interval(probabilities, confidence_level=0.95):
        """
        Calcula intervalos de confiança para as probabilidades usando método normal.
        
        Args:
            probabilities: Array de probabilidades
            confidence_level: Nível de confiança (padrão 0.95 para 95%)
        
        Returns:
            Dict com intervalo inferior e superior
        """
        from scipy import stats
        
        mean = np.mean(probabilities)
        std_error = stats.sem(probabilities)
        margin_error = std_error * stats.t.ppf((1 + confidence_level) / 2, len(probabilities) - 1)
        
        return {
            'mean': mean,
            'lower': max(0, mean - margin_error),
            'upper': min(1, mean + margin_error),
            'margin_error': margin_error
        }
    
    @staticmethod
    def bootstrap_validation(model, image_tensor, n_iterations=100, dropout_rate=0.1):
        """
        Realiza validação bootstrap através de múltiplas predições com dropout.
        
        Args:
            model: Modelo treinado
            image_tensor: Tensor da imagem
            n_iterations: Número de iterações bootstrap
            dropout_rate: Taxa de dropout para variação
        
        Returns:
            Dict com estatísticas bootstrap
        """
        model.train()  # Ativa dropout
        predictions = []
        probabilities_per_class = []
        
        with torch.no_grad():
            for _ in range(n_iterations):
                output = model(image_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy()[0])
        
        predictions = np.array(predictions)
        mean_probs = np.mean(predictions, axis=0)
        std_probs = np.std(predictions, axis=0)
        
        model.eval()  # Volta para modo de avaliação
        
        return {
            'mean_probabilities': mean_probs,
            'std_probabilities': std_probs,
            'predictions_distribution': predictions,
            'confidence_bootstrap': np.max(mean_probs),
            'uncertainty': np.max(std_probs)
        }
    
    @staticmethod
    def significance_test(prob1, prob2, predictions_dist):
        """
        Testa se há diferença significativa entre duas probabilidades.
        
        Args:
            prob1: Probabilidade da classe 1
            prob2: Probabilidade da classe 2
            predictions_dist: Distribuição de predições do bootstrap
        
        Returns:
            Dict com resultado do teste
        """
        from scipy import stats
        
        # Teste t pareado
        diff = predictions_dist[:, 0] - predictions_dist[:, 1] if predictions_dist.shape[1] > 1 else None
        
        if diff is not None:
            t_stat, p_value = stats.ttest_1samp(diff, 0)
            significant = p_value < 0.05
        else:
            t_stat, p_value, significant = None, None, None
        
        return {
            'probability_diff': abs(prob1 - prob2),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'interpretation': 'Diferença significativa' if significant else 'Diferença não significativa'
        }

class DiagnosticAnalyzer:
    """
    Classe para análise diagnóstica diferencial e critérios de exclusão.
    """
    
    @staticmethod
    def differential_diagnosis(probabilities, classes, top_k=3, threshold=0.1):
        """
        Lista diagnósticos diferenciais principais baseado nas probabilidades.
        
        Args:
            probabilities: Array de probabilidades para cada classe
            classes: Lista de nomes das classes
            top_k: Número de diagnósticos principais a retornar
            threshold: Limiar mínimo de probabilidade para considerar
        
        Returns:
            Lista de diagnósticos diferenciais
        """
        # Ordenar por probabilidade
        sorted_indices = np.argsort(probabilities)[::-1]
        
        differentials = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = probabilities[idx]
            
            if prob >= threshold:
                differentials.append({
                    'rank': i + 1,
                    'class': classes[idx],
                    'probability': prob,
                    'confidence_level': DiagnosticAnalyzer._interpret_confidence(prob)
                })
        
        return differentials
    
    @staticmethod
    def _interpret_confidence(prob):
        """Interpreta o nível de confiança"""
        if prob >= 0.9:
            return 'Muito Alto'
        elif prob >= 0.75:
            return 'Alto'
        elif prob >= 0.5:
            return 'Moderado'
        elif prob >= 0.3:
            return 'Baixo'
        else:
            return 'Muito Baixo'
    
    @staticmethod
    def exclusion_criteria(probabilities, classes, exclusion_threshold=0.05):
        """
        Aplica critérios de exclusão baseados em probabilidades muito baixas.
        
        Args:
            probabilities: Array de probabilidades
            classes: Lista de classes
            exclusion_threshold: Limiar abaixo do qual classes são excluídas
        
        Returns:
            Dict com classes excluídas e razões
        """
        excluded = []
        
        for i, (prob, class_name) in enumerate(zip(probabilities, classes)):
            if prob < exclusion_threshold:
                excluded.append({
                    'class': class_name,
                    'probability': prob,
                    'reason': f'Probabilidade muito baixa (< {exclusion_threshold:.1%})'
                })
        
        return {
            'excluded_count': len(excluded),
            'excluded_classes': excluded,
            'remaining_count': len(classes) - len(excluded)
        }
    
    @staticmethod
    def distinctive_features(activation_map, threshold_percentile=75):
        """
        Identifica características distintivas baseadas no mapa de ativação.
        
        Args:
            activation_map: Mapa de ativação do Grad-CAM
            threshold_percentile: Percentil para identificar regiões importantes
        
        Returns:
            Dict com informações sobre características distintivas
        """
        if activation_map is None:
            return None
        
        threshold = np.percentile(activation_map, threshold_percentile)
        important_regions = activation_map > threshold
        
        return {
            'high_activation_percentage': (np.sum(important_regions) / important_regions.size) * 100,
            'max_activation': np.max(activation_map),
            'mean_activation': np.mean(activation_map),
            'activation_concentration': np.std(activation_map),
            'interpretation': DiagnosticAnalyzer._interpret_activation_pattern(
                (np.sum(important_regions) / important_regions.size) * 100
            )
        }
    
    @staticmethod
    def _interpret_activation_pattern(percentage):
        """Interpreta o padrão de ativação"""
        if percentage > 30:
            return 'Características dispersas - múltiplas regiões importantes'
        elif percentage > 15:
            return 'Características moderadamente focadas'
        elif percentage > 5:
            return 'Características altamente focadas - região específica'
        else:
            return 'Características muito concentradas - atenção localizada'

class UncertaintyAnalyzer:
    """
    Classe para quantificação de incerteza e análise de risco.
    """
    
    @staticmethod
    def quantify_uncertainty(bootstrap_results, entropy_weight=0.5):
        """
        Quantifica fontes de incerteza na predição.
        
        Args:
            bootstrap_results: Resultados do bootstrap
            entropy_weight: Peso para a entropia na incerteza total
        
        Returns:
            Dict com análise de incerteza
        """
        mean_probs = bootstrap_results['mean_probabilities']
        std_probs = bootstrap_results['std_probabilities']
        
        # Incerteza aleatória (epistêmica) - variação das predições
        aleatoric_uncertainty = np.mean(std_probs)
        
        # Incerteza do modelo (entropia)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = np.log(len(mean_probs))
        normalized_entropy = entropy / max_entropy
        
        # Incerteza total
        total_uncertainty = (1 - entropy_weight) * aleatoric_uncertainty + entropy_weight * normalized_entropy
        
        return {
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'model_entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'total_uncertainty': total_uncertainty,
            'uncertainty_level': UncertaintyAnalyzer._classify_uncertainty(total_uncertainty),
            'sources': {
                'model_variation': aleatoric_uncertainty,
                'prediction_ambiguity': normalized_entropy
            }
        }
    
    @staticmethod
    def _classify_uncertainty(uncertainty):
        """Classifica o nível de incerteza"""
        if uncertainty < 0.1:
            return 'Muito Baixa'
        elif uncertainty < 0.2:
            return 'Baixa'
        elif uncertainty < 0.4:
            return 'Moderada'
        elif uncertainty < 0.6:
            return 'Alta'
        else:
            return 'Muito Alta'
    
    @staticmethod
    def assess_error_impact(top_probabilities, classes, risk_categories=None):
        """
        Avalia o impacto de possíveis erros de classificação.
        
        Args:
            top_probabilities: Probabilidades das top classes
            classes: Nomes das classes
            risk_categories: Dict mapeando classes para níveis de risco (opcional)
        
        Returns:
            Dict com avaliação de impacto
        """
        if risk_categories is None:
            # Risco padrão baseado apenas na confiança
            risk_categories = {cls: 'medium' for cls in classes}
        
        # Calcular probabilidade de erro
        error_probability = 1 - np.max(top_probabilities)
        
        # Avaliar impacto
        predicted_class = classes[np.argmax(top_probabilities)]
        risk_level = risk_categories.get(predicted_class, 'medium')
        
        return {
            'error_probability': error_probability,
            'predicted_class_risk': risk_level,
            'impact_score': error_probability * UncertaintyAnalyzer._risk_weight(risk_level),
            'recommendation': UncertaintyAnalyzer._generate_recommendation(
                error_probability, risk_level
            )
        }
    
    @staticmethod
    def _risk_weight(risk_level):
        """Retorna peso do nível de risco"""
        weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 5}
        return weights.get(risk_level.lower(), 2)
    
    @staticmethod
    def _generate_recommendation(error_prob, risk_level):
        """Gera recomendação baseada em erro e risco"""
        if error_prob > 0.3 and risk_level in ['high', 'critical']:
            return '⚠️ ATENÇÃO: Alta probabilidade de erro em categoria de alto risco. Recomenda-se validação adicional.'
        elif error_prob > 0.5:
            return '⚠️ Confiança baixa. Considere análise complementar ou consulta especializada.'
        elif error_prob > 0.3:
            return 'ℹ️ Confiança moderada. Monitoramento recomendado.'
        else:
            return '✅ Confiança adequada. Resultado confiável.'
    
    @staticmethod
    def safety_margin(confidence, min_acceptable=0.7, target=0.9):
        """
        Estabelece margem de segurança para a predição.
        
        Args:
            confidence: Confiança da predição
            min_acceptable: Confiança mínima aceitável
            target: Confiança alvo desejada
        
        Returns:
            Dict com análise de margem de segurança
        """
        margin_to_minimum = confidence - min_acceptable
        margin_to_target = target - confidence
        
        status = 'safe' if confidence >= min_acceptable else 'unsafe'
        meets_target = confidence >= target
        
        return {
            'confidence': confidence,
            'min_acceptable': min_acceptable,
            'target': target,
            'margin_to_minimum': margin_to_minimum,
            'margin_to_target': margin_to_target,
            'status': status,
            'meets_target': meets_target,
            'safety_score': min(1.0, confidence / target),
            'interpretation': UncertaintyAnalyzer._interpret_safety(
                margin_to_minimum, meets_target
            )
        }
    
    @staticmethod
    def _interpret_safety(margin, meets_target):
        """Interpreta a margem de segurança"""
        if margin < 0:
            return '🔴 ABAIXO DO MÍNIMO ACEITÁVEL - Não recomendado para uso'
        elif margin < 0.1:
            return '🟡 MARGEM CRÍTICA - Usar com extrema cautela'
        elif meets_target:
            return '🟢 MARGEM ADEQUADA - Confiança alvo atingida'
        else:
            return '🟢 MARGEM ACEITÁVEL - Dentro dos parâmetros seguros'
    
    @staticmethod
    def clinical_impact_assessment(confidence, class_name, differential_diagnoses):
        """
        Avalia o impacto clínico/prático da predição.
        
        Args:
            confidence: Confiança da predição principal
            class_name: Nome da classe predita
            differential_diagnoses: Lista de diagnósticos diferenciais
        
        Returns:
            Dict com avaliação de impacto clínico
        """
        # Calcular ambiguidade diagnóstica
        if len(differential_diagnoses) > 1:
            top2_diff = differential_diagnoses[0]['probability'] - differential_diagnoses[1]['probability']
        else:
            top2_diff = 1.0
        
        # Determinar ação recomendada
        if confidence >= 0.9 and top2_diff > 0.3:
            action = 'Proceder com diagnóstico primário'
            priority = 'Normal'
        elif confidence >= 0.75 and top2_diff > 0.2:
            action = 'Considerar diagnóstico primário com monitoramento'
            priority = 'Média'
        elif len(differential_diagnoses) > 1 and differential_diagnoses[1]['probability'] > 0.3:
            action = 'Investigar diagnósticos diferenciais - múltiplas possibilidades'
            priority = 'Alta'
        else:
            action = 'Análise complementar necessária'
            priority = 'Alta'
        
        return {
            'primary_diagnosis': class_name,
            'diagnostic_confidence': confidence,
            'differential_count': len(differential_diagnoses),
            'diagnostic_ambiguity': 1 - top2_diff if len(differential_diagnoses) > 1 else 0,
            'recommended_action': action,
            'priority_level': priority,
            'requires_specialist': confidence < 0.75 or (
                len(differential_diagnoses) > 1 and differential_diagnoses[1]['probability'] > 0.3
            )
        }

# Enhanced image preprocessing class
class EnhancedImagePreprocessor:
    """Classe para melhorar o tratamento de imagens antes do treinamento"""
    
    @staticmethod
    def enhance_image_quality(image):
        """Aplica melhorias de qualidade na imagem"""
        # Ajustar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Ajustar nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Ajustar brilho levemente
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)
        
        return image

def get_augmentation_transforms(augmentation_type='standard'):
    """
    Retorna transformações de acordo com o tipo de aumento de dados
    
    Args:
        augmentation_type: 'none', 'standard', 'mixup', 'cutmix'
    """
    if augmentation_type == 'none':
        # Sem aumento de dados - apenas transformações básicas
        train_transform = transforms.Compose([
            transforms.Lambda(EnhancedImagePreprocessor.enhance_image_quality),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Standard ou base para mixup/cutmix
        train_transform = transforms.Compose([
            transforms.Lambda(EnhancedImagePreprocessor.enhance_image_quality),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=0, shear=10),
            ], p=0.5),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    return train_transform

# Transformações para validação e teste com normalização ImageNet
test_transforms = transforms.Compose([
    transforms.Lambda(EnhancedImagePreprocessor.enhance_image_quality),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Implementação de Mixup
def mixup_data(x, y, alpha=1.0):
    """Aplica Mixup ao batch de dados"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calcula a loss para Mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Implementação de CutMix
def cutmix_data(x, y, alpha=1.0):
    """Aplica CutMix ao batch de dados"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Gerar bbox
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Centro do box
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Ajustar lambda com a área real
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam

# Definir as transformações padrão para compatibilidade com código existente
train_transforms = get_augmentation_transforms('standard')

# Dataset personalizado
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def seed_worker(worker_id):
    """
    Função para definir a seed em cada worker do DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def visualize_data(dataset, classes):
    """
    Exibe algumas imagens do conjunto de dados com suas classes.
    """
    st.write("### 📊 Visualização de algumas imagens do conjunto de dados original:")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)  # Converter a imagem PIL em array NumPy
        axes[i].imshow(image)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    st.pyplot(fig)
    plt.close(fig)

def plot_class_distribution(dataset, classes, title="Distribuição das Classes"):
    """
    Exibe a distribuição das classes no conjunto de dados e mostra os valores quantitativos.
    """
    # Extrair os rótulos das classes para todas as imagens no dataset
    labels = [label for _, label in dataset]
    
    # Contagem de cada classe
    class_counts = np.bincount(labels)
    
    # Plotar o gráfico com as contagens
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=labels, hue=labels, ax=ax, palette="Set2", legend=False)
    
    # Adicionar os nomes das classes no eixo X
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    
    # Adicionar as contagens acima das barras
    for i, count in enumerate(class_counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Número de Imagens")
    
    st.pyplot(fig)
    plt.close(fig)
    
    return class_counts

def show_augmented_images(dataset, transform, classes, num_augmentations=5):
    """
    Mostra imagens originais e suas versões aumentadas.
    """
    st.write("### 🔄 Exemplos de Imagens Aumentadas (Data Augmentation)")
    st.write("Cada linha mostra uma imagem original seguida de suas versões aumentadas:")
    
    # Selecionar 3 imagens aleatórias
    num_samples = 3
    for sample_idx in range(num_samples):
        idx = np.random.randint(len(dataset))
        original_image, label = dataset[idx]
        
        # Criar figura com 1 original + num_augmentations aumentadas
        fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 3))
        
        # Mostrar imagem original
        axes[0].imshow(np.array(original_image))
        axes[0].set_title(f'Original\n{classes[label]}')
        axes[0].axis('off')
        axes[0].set_facecolor('#e6f2ff')
        
        # Mostrar imagens aumentadas
        for i in range(1, num_augmentations + 1):
            augmented_image = transform(original_image)
            # Desnormalizar para visualização usando a função helper
            augmented_np = denormalize_image(augmented_image)
            
            axes[i].imshow(augmented_np)
            axes[i].set_title(f'Aumentada {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def calculate_dataset_statistics(dataset, classes):
    """
    Calcula estatísticas do dataset incluindo média, desvio padrão, etc.
    """
    st.write("### 📈 Estatísticas do Dataset")
    
    # Contagem por classe
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    
    # Criar dataframe com estatísticas
    stats_data = {
        'Classe': classes,
        'Quantidade': class_counts,
        'Percentual (%)': [f"{(count/len(dataset)*100):.2f}" for count in class_counts]
    }
    
    df_stats = pd.DataFrame(stats_data)
    
    st.write("#### Distribuição de Classes:")
    # Fixed: Removed width=None parameter as it's no longer supported in Streamlit
    st.dataframe(df_stats)
    
    # Estatísticas gerais
    st.write("#### Estatísticas Gerais:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Imagens", len(dataset))
    
    with col2:
        st.metric("Número de Classes", len(classes))
    
    with col3:
        st.metric("Imagens por Classe (Média)", f"{np.mean(class_counts):.1f}")
    
    with col4:
        st.metric("Desvio Padrão", f"{np.std(class_counts):.1f}")
    
    # Verificar balanceamento
    min_count = np.min(class_counts)
    max_count = np.max(class_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 1.5:
        st.warning(f"⚠️ Dataset desbalanceado detectado! Razão: {imbalance_ratio:.2f}x (Classe mais frequente / Classe menos frequente)")
        st.info("💡 Recomendação: Considere usar 'Perda Ponderada para Classes Desbalanceadas' nas configurações.")
    else:
        st.success(f"✅ Dataset relativamente balanceado. Razão: {imbalance_ratio:.2f}x")
    
    return df_stats

def visualize_pca_features(features, labels, classes, n_components=2):
    """
    Visualiza features usando PCA.
    """
    st.write(f"### 🔬 Análise PCA ({n_components} Componentes)")
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    # Mostrar variância explicada
    explained_var = pca.explained_variance_ratio_
    st.write(f"**Variância Explicada:** {explained_var[0]*100:.2f}% (PC1), {explained_var[1]*100:.2f}% (PC2)")
    st.write(f"**Variância Total Explicada:** {sum(explained_var)*100:.2f}%")
    
    # Criar visualização
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Mapear labels para nomes de classes
    labels_named = [classes[label] for label in labels]
    
    # Criar scatter plot
    scatter = sns.scatterplot(
        x=features_pca[:, 0], 
        y=features_pca[:, 1], 
        hue=labels_named,
        palette="tab10",
        ax=ax,
        s=100,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xlabel(f'Componente Principal 1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'Componente Principal 2 ({explained_var[1]*100:.1f}%)')
    ax.set_title('Visualização PCA das Features Extraídas')
    ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    return features_pca, explained_var

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False):
    """
    Retorna o modelo pré-treinado selecionado, incluindo CNNs e Vision Transformers.
    
    Args:
        model_name: Nome do modelo (ResNet18, ResNet50, DenseNet121, ViT-B/16, ViT-B/32, ViT-L/16, Swin-T, Swin-B)
        num_classes: Número de classes
        dropout_p: Taxa de dropout
        fine_tune: Se deve fazer fine-tuning completo
    
    Returns:
        model: Modelo PyTorch configurado
    """
    # CNNs Tradicionais
    if model_name == 'ResNet18':
        model = models.resnet18(weights='DEFAULT')
    elif model_name == 'ResNet50':
        model = models.resnet50(weights='DEFAULT')
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights='DEFAULT')
    # Vision Transformers (torchvision)
    elif model_name == 'ViT-B/16':
        model = models.vit_b_16(weights='DEFAULT')
    elif model_name == 'ViT-B/32':
        model = models.vit_b_32(weights='DEFAULT')
    elif model_name == 'ViT-L/16':
        model = models.vit_l_16(weights='DEFAULT')
    # Vision Transformers e Swin (timm - mais robustos)
    elif model_name == 'ViT-B/16-timm' and TIMM_AVAILABLE:
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes, drop_rate=dropout_p)
        if not fine_tune:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        return model.to(device)
    elif model_name == 'ViT-L/16-timm' and TIMM_AVAILABLE:
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes, drop_rate=dropout_p)
        if not fine_tune:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        return model.to(device)
    elif model_name == 'Swin-T' and TIMM_AVAILABLE:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes, drop_rate=dropout_p)
        if not fine_tune:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        return model.to(device)
    elif model_name == 'Swin-B' and TIMM_AVAILABLE:
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes, drop_rate=dropout_p)
        if not fine_tune:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        return model.to(device)
    else:
        st.error(f"Modelo '{model_name}' não suportado ou timm não está disponível.")
        return None

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Configurar camada de saída baseada no tipo de modelo
    if model_name.startswith('ResNet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
        # Ensure final layer requires grad
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
        # Ensure final layer requires grad
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name.startswith('ViT'):
        # Vision Transformers usam 'heads' ao invés de 'fc' ou 'classifier'
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
        # Ensure final layer requires grad
        for param in model.heads.head.parameters():
            param.requires_grad = True
    else:
        st.error("Tipo de modelo não suportado para configuração.")
        return None

    model = model.to(device)
    return model

# ============= MELHORIAS AVANÇADAS =============

# 1. Label Smoothing Loss
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss.
    Previne overconfident predictions e melhora generalização.
    """
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)
        
        # Create smoothed distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply class weights if provided
        if self.weight is not None:
            # Weight each sample based on its target class
            sample_weights = self.weight[target]
            loss = torch.sum(-true_dist * log_preds, dim=-1) * sample_weights
            return loss.mean()
        else:
            return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))

# 2. Exponential Moving Average (EMA) para pesos do modelo
class ModelEMA:
    """
    Mantém média móvel exponencial dos pesos do modelo.
    Melhora generalização e estabilidade.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# 3. Agente CrewAI para buscar informações de treinamento
class TrainingResearchAgent:
    """
    Agente inteligente que busca informações na web para melhorar o treinamento.
    """
    def __init__(self, use_crewai=False):
        self.use_crewai = use_crewai and CREWAI_AVAILABLE
        self.insights = []
        
    def research_training_strategies(self, model_name, dataset_type):
        """
        Pesquisa estratégias de treinamento para o modelo e tipo de dataset.
        """
        if not self.use_crewai:
            # Retornar insights padrão
            return {
                'learning_rate': 0.0001,
                'batch_size': 16,
                'augmentation': 'standard',
                'scheduler': 'CosineAnnealingLR',
                'insights': ['Using default configuration']
            }
        
        try:
            # Definir agentes CrewAI
            researcher = Agent(
                role='ML Training Researcher',
                goal=f'Find optimal training strategies for {model_name} on {dataset_type} images',
                backstory='Expert in deep learning optimization and hyperparameter tuning',
                verbose=True,
                allow_delegation=False
            )
            
            # Definir tarefa
            research_task = Task(
                description=f'''
                Research and recommend:
                1. Optimal learning rate for {model_name}
                2. Best data augmentation strategies for {dataset_type}
                3. Recommended batch size and scheduler
                4. Common pitfalls to avoid
                ''',
                agent=researcher,
                expected_output='Training recommendations in JSON format'
            )
            
            # Executar crew
            crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                verbose=2,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            self.insights.append(result)
            
            return {
                'research': result,
                'insights': self.insights
            }
            
        except Exception as e:
            st.warning(f"CrewAI research failed: {e}. Using default config.")
            return {
                'learning_rate': 0.0001,
                'batch_size': 16,
                'augmentation': 'standard',
                'scheduler': 'CosineAnnealingLR',
                'insights': [f'Error: {e}']
            }

# 4. Reinforcement Learning para ajuste dinâmico de hiperparâmetros
class ReinforcementLearningTrainer:
    """
    Usa RL para ajustar hiperparâmetros durante o treinamento.
    Implementa Q-Learning simples para ajustar learning rate e batch size.
    """
    def __init__(self, initial_lr=0.001, lr_range=(0.00001, 0.01)):
        self.lr = initial_lr
        self.lr_range = lr_range
        self.q_table = {}  # Estado -> Ação -> Q-value
        self.alpha = 0.1  # Learning rate do RL
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.actions = ['increase_lr', 'decrease_lr', 'keep_lr']
        self.performance_history = []
        
    def get_state(self, val_loss, val_acc, epoch):
        """
        Define o estado baseado em perda e acurácia de validação.
        """
        if len(self.performance_history) == 0:
            # First epoch - use neutral state
            return "initial_epoch_0"
        
        prev = self.performance_history[-1]
        loss_trend = 'improving' if val_loss < prev['loss'] else 'degrading'
        acc_trend = 'improving' if val_acc > prev['acc'] else 'degrading'
        
        return f"{loss_trend}_{acc_trend}_epoch{epoch % 10}"
    
    def choose_action(self, state):
        """
        Escolhe ação usando epsilon-greedy.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Atualiza Q-value usando Q-learning.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}
        
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state][action] = new_q
    
    def adjust_learning_rate(self, action):
        """
        Ajusta learning rate baseado na ação.
        """
        if action == 'increase_lr':
            self.lr = min(self.lr * 1.2, self.lr_range[1])
        elif action == 'decrease_lr':
            self.lr = max(self.lr * 0.8, self.lr_range[0])
        # 'keep_lr' não faz nada
        
        return self.lr
    
    def compute_reward(self, val_loss, val_acc):
        """
        Computa recompensa baseado no desempenho.
        """
        if len(self.performance_history) == 0:
            reward = 0.0
        else:
            prev = self.performance_history[-1]
            loss_improvement = prev['loss'] - val_loss
            acc_improvement = val_acc - prev['acc']
            reward = loss_improvement * 10 + acc_improvement * 100
        
        self.performance_history.append({'loss': val_loss, 'acc': val_acc})
        return reward
    
    def step(self, val_loss, val_acc, epoch, optimizer):
        """
        Executa um passo de RL: escolhe ação, ajusta LR, computa recompensa.
        """
        state = self.get_state(val_loss, val_acc, epoch)
        action = self.choose_action(state)
        new_lr = self.adjust_learning_rate(action)
        
        # Atualizar learning rate no otimizador
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        reward = self.compute_reward(val_loss, val_acc)
        next_state = self.get_state(val_loss, val_acc, epoch + 1)
        self.update_q_value(state, action, reward, next_state)
        
        return {
            'action': action,
            'new_lr': new_lr,
            'reward': reward,
            'state': state
        }

def train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, l1_lambda, patience, optimizer_name='Adam', scheduler_name='None', augmentation_type='standard', label_smoothing=0.1, use_gradient_clipping=True, use_ema=True, use_rl=False, use_crewai=False):
    """
    Função principal para treinamento do modelo com melhorias avançadas.
    
    Args:
        data_dir: Diretório com os dados
        num_classes: Número de classes
        model_name: Nome do modelo
        fine_tune: Se deve fazer fine-tuning completo
        epochs: Número de épocas
        learning_rate: Taxa de aprendizagem
        batch_size: Tamanho do lote
        train_split: Proporção de treino
        valid_split: Proporção de validação
        use_weighted_loss: Se deve usar perda ponderada
        l2_lambda: Regularização L2 (weight decay)
        l1_lambda: Regularização L1
        patience: Paciência para early stopping
        optimizer_name: Nome do otimizador (Adam, AdamW, SGD, Ranger, Lion)
        scheduler_name: Nome do scheduler (None, CosineAnnealingLR, OneCycleLR)
        augmentation_type: Tipo de aumento de dados (none, standard, mixup, cutmix)
        label_smoothing: Suavização de labels (0.0-0.3, padrão 0.1)
        use_gradient_clipping: Se deve usar gradient clipping
        use_ema: Se deve usar Exponential Moving Average nos pesos
        use_rl: Se deve usar Reinforcement Learning para ajuste dinâmico
        use_crewai: Se deve usar CrewAI para pesquisa de estratégias
    
    Returns:
        tuple: (model, classes) ou None em caso de erro
    """
    set_seed(42)

    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # ========== CONTAGEM INICIAL DOS DADOS ==========
    st.write("## 📊 ANÁLISE INICIAL DO DATASET")
    st.write(f"### 🔢 **Contagem Inicial: {len(full_dataset)} imagens**")
    
    # Exibir estatísticas detalhadas
    stats_df = calculate_dataset_statistics(full_dataset, full_dataset.classes)
    
    # Exibir algumas imagens do dataset original
    visualize_data(full_dataset, full_dataset.classes)
    
    # Plotar distribuição inicial
    st.write("### 📊 Distribuição Inicial das Classes")
    initial_class_counts = plot_class_distribution(full_dataset, full_dataset.classes, 
                                                    title="Distribuição INICIAL das Classes (Sem Aumento de Dados)")

    # ========== TÉCNICA DE AUMENTO DE DADOS ==========
    st.write("---")
    st.write("## 🔄 APLICAÇÃO DA TÉCNICA DE AUMENTO DE DADOS")
    st.write(f"**Técnica Selecionada:** `{augmentation_type}`")
    
    if augmentation_type == 'none':
        st.info("ℹ️ Nenhuma técnica de aumento de dados foi selecionada. As imagens serão usadas como estão.")
    elif augmentation_type == 'standard':
        st.info("ℹ️ Técnica Standard: Aplicação de transformações aleatórias (rotação, flip, crop, jitter, etc.)")
    elif augmentation_type == 'mixup':
        st.info("ℹ️ Técnica Mixup: Mistura linear de pares de imagens e seus rótulos")
    elif augmentation_type == 'cutmix':
        st.info("ℹ️ Técnica CutMix: Recorte e colagem de regiões entre imagens diferentes")
    
    # Obter transformações baseadas no tipo de augmentação
    train_transform = get_augmentation_transforms(augmentation_type)
    
    # Mostrar exemplos de imagens aumentadas
    if augmentation_type != 'none':
        show_augmented_images(full_dataset, train_transform, full_dataset.classes, num_augmentations=4)
    
    # ========== ESTIMATIVA APÓS AUMENTO ==========
    st.write("---")
    st.write("## 📈 ESTIMATIVA APÓS AUMENTO DE DADOS")
    
    # Calcular estimativa de imagens após aumento
    # Durante o treinamento, cada época gera versões aumentadas
    if augmentation_type == 'none':
        augmentation_multiplier = 1
        st.write(f"### 🔢 **Total Estimado: {len(full_dataset)} imagens** (sem aumento)")
    else:
        # Com augmentation, cada época gera versões diferentes
        # Estimativa conservadora: cada imagem pode gerar de 3-5 variações por época
        augmentation_multiplier = 4  # Média estimada
        total_estimated = len(full_dataset) * augmentation_multiplier * epochs
        st.write(f"### 🔢 **Total de Imagens Original: {len(full_dataset)}**")
        st.write(f"### 🔢 **Multiplicador Estimado por Época: ~{augmentation_multiplier}x**")
        st.write(f"### 🔢 **Total Estimado Durante {epochs} Épocas: ~{total_estimated:,} imagens aumentadas**")
        st.info(f"💡 **Explicação:** Durante o treinamento, cada uma das {len(full_dataset)} imagens originais será " +
                f"transformada aleatoriamente a cada época, gerando aproximadamente {augmentation_multiplier}x variações únicas " +
                f"ao longo de {epochs} épocas, totalizando cerca de {total_estimated:,} imagens processadas.")
    
    st.write("---")
    
    # Criar o dataset personalizado com aumento de dados
    train_dataset = CustomDataset(full_dataset, transform=train_transform)
    valid_dataset = CustomDataset(full_dataset, transform=test_transforms)
    test_dataset = CustomDataset(full_dataset, transform=test_transforms)

    # Dividir os índices para treino, validação e teste
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_end = int(train_split * dataset_size)
    valid_end = int((train_split + valid_split) * dataset_size)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Dataloaders
    g = torch.Generator()
    g.manual_seed(42)

    # Inicializar agentes inteligentes se habilitados
    research_agent = None
    rl_trainer = None
    
    if use_crewai:
        st.write("🤖 **Inicializando Agente CrewAI para pesquisa de estratégias...**")
        research_agent = TrainingResearchAgent(use_crewai=True)
        research_results = research_agent.research_training_strategies(model_name, "rock classification")
        if research_results and 'insights' in research_results:
            st.info(f"📚 **Insights do Agente:** {research_results['insights']}")
    
    if use_rl:
        if scheduler_name != 'None':
            st.warning("⚠️ **Atenção:** RL está ativo junto com um scheduler. O RL pode entrar em conflito com o scheduler. Considere desativar um deles.")
        st.write("🎯 **Inicializando Reinforcement Learning para ajuste dinâmico...**")
        rl_trainer = ReinforcementLearningTrainer(initial_lr=learning_rate)
    
    if use_weighted_loss:
        targets = [full_dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets)
        class_counts = class_counts + 1e-6  # Para evitar divisão por zero
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)
            st.info(f"✨ **Usando Label Smoothing ({label_smoothing}) com pesos de classe**")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            st.info(f"✨ **Usando Label Smoothing ({label_smoothing})**")
        else:
            criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Carregar o modelo
    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune)
    if model is None:
        return None

    # Definir o otimizador com L2 regularization (weight_decay)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=learning_rate, weight_decay=l2_lambda, momentum=0.9, nesterov=True)
    elif optimizer_name == 'Ranger' and ADVANCED_OPTIMIZERS_AVAILABLE:
        optimizer = optim_advanced.Ranger(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name == 'Lion' and ADVANCED_OPTIMIZERS_AVAILABLE:
        optimizer = optim_advanced.Lion(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    else:
        st.warning(f"Otimizador {optimizer_name} não disponível. Usando Adam.")
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    
    # Configurar Learning Rate Scheduler
    scheduler = None
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/100)
    elif scheduler_name == 'OneCycleLR':
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate*10, 
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3
        )

    # Listas para armazenar as perdas e acurácias
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Early Stopping
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    
    # Parâmetros para Mixup e CutMix
    use_mixup = (augmentation_type == 'mixup')
    use_cutmix = (augmentation_type == 'cutmix')
    mixup_alpha = 1.0
    cutmix_alpha = 1.0
    
    # Cache de parâmetros para regularização L1 (otimização)
    trainable_params_list = list(filter(lambda p: p.requires_grad, model.parameters())) if l1_lambda > 0 else []
    
    # Inicializar EMA se habilitado
    model_ema = None
    if use_ema:
        model_ema = ModelEMA(model, decay=0.999)
        st.info("📊 **Usando Exponential Moving Average (EMA) para pesos do modelo**")

    # Treinamento
    for epoch in range(epochs):
        set_seed(42 + epoch)
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Aplicar Mixup ou CutMix se selecionado
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            elif use_cutmix:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
            
            try:
                outputs = model(inputs)
            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")
                return None

            # Calcular loss
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                _, preds = torch.max(outputs, 1)
            else:
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Adicionar regularização L1 se configurado
            if l1_lambda > 0:
                l1_reg = torch.tensor(0., device=device)
                for param in trainable_params_list:
                    l1_reg += torch.norm(param, 1)
                loss = loss + l1_lambda * l1_reg
            
            loss.backward()
            
            # Gradient clipping para estabilizar o treinamento
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Atualizar EMA se habilitado
            if model_ema is not None:
                model_ema.update()
            
            # Atualizar scheduler OneCycleLR a cada batch
            if scheduler_name == 'OneCycleLR' and scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data if not (use_mixup or use_cutmix) else preds == labels_a.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validação
        model.eval()
        valid_running_loss = 0.0
        valid_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                valid_running_loss += loss.item() * inputs.size(0)
                valid_running_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_running_loss / len(valid_dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_dataset)
        valid_losses.append(valid_epoch_loss)
        valid_accuracies.append(valid_epoch_acc.item())

        st.write(f'**Época {epoch+1}/{epochs}**')
        st.write(f'Perda de Treino: {epoch_loss:.4f} | Acurácia de Treino: {epoch_acc:.4f}')
        st.write(f'Perda de Validação: {valid_epoch_loss:.4f} | Acurácia de Validação: {valid_epoch_acc:.4f}')
        
        # Aplicar Reinforcement Learning se habilitado
        if rl_trainer is not None:
            rl_result = rl_trainer.step(valid_epoch_loss, valid_epoch_acc.item(), epoch, optimizer)
            st.write(f"🎯 **RL Action:** {rl_result['action']} | **New LR:** {rl_result['new_lr']:.6f} | **Reward:** {rl_result['reward']:.4f}")

        # Atualizar scheduler CosineAnnealingLR após cada época
        if scheduler_name == 'CosineAnnealingLR' and scheduler is not None:
            scheduler.step()

        # Early Stopping
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write('Early stopping!')
                model.load_state_dict(best_model_wts)
                break

    # Carregar os melhores pesos do modelo
    model.load_state_dict(best_model_wts)
    
    # Aplicar pesos EMA se habilitado
    if model_ema is not None:
        st.info("📊 **Aplicando pesos EMA ao modelo final...**")
        model_ema.apply_shadow()

    # Gráficos de Perda e Acurácia
    plot_metrics(epochs, train_losses, valid_losses, train_accuracies, valid_accuracies)

    # Avaliação Final no Conjunto de Teste
    st.write("**Avaliação no Conjunto de Teste**")
    compute_metrics(model, test_loader, full_dataset.classes)

    # Análise de Erros
    st.write("**Análise de Erros**")
    error_analysis(model, test_loader, full_dataset.classes)

    # Liberar memória
    del train_loader, valid_loader
    gc.collect()
    
    # Preparar histórico de treinamento para exportação
    training_history = {
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'train_accuracy': train_accuracies,
        'valid_accuracy': valid_accuracies
    }

    return model, full_dataset.classes, training_history

def plot_metrics(epochs, train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plota os gráficos de perda e acurácia.
    """
    epochs_range = range(1, len(train_losses)+1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Perda
    ax[0].plot(epochs_range, train_losses, label='Treino')
    ax[0].plot(epochs_range, valid_losses, label='Validação')
    ax[0].set_title('Perda por Época')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Perda')
    ax[0].legend()

    # Gráfico de Acurácia
    ax[1].plot(epochs_range, train_accuracies, label='Treino')
    ax[1].plot(epochs_range, valid_accuracies, label='Validação')
    ax[1].set_title('Acurácia por Época')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

def compute_metrics(model, dataloader, classes):
    """
    Calcula métricas detalhadas e exibe matriz de confusão e relatório de classificação.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Relatório de Classificação
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
    st.text("Relatório de Classificação:")
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão Normalizada
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão Normalizada')
    st.pyplot(fig)
    plt.close(fig)

    # Curva ROC
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, [p[1] for p in all_probs])
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curva ROC')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        plt.close(fig)
    else:
        # Multiclasse
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"AUC-ROC Média Ponderada: {roc_auc:.4f}")

def error_analysis(model, dataloader, classes):
    """
    Realiza análise de erros mostrando algumas imagens mal classificadas.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(inputs[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                if len(misclassified_images) >= 5:
                    break

    if misclassified_images:
        st.write("Algumas imagens mal classificadas:")
        num_images = min(5, len(misclassified_images))
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        
        # Handle case when only one image (axes is not an array)
        if num_images == 1:
            axes = [axes]
            
        for i in range(num_images):
            image = misclassified_images[i]
            # Denormalize the image for proper display
            image = denormalize_image(image)
            axes[i].imshow(image)
            axes[i].set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            axes[i].axis('off')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")

def create_export_csv(training_history, classification_results=None, clustering_results=None):
    """
    Cria um DataFrame consolidado com todos os resultados para exportação CSV.
    
    Args:
        training_history: Dict com histórico de treinamento (epoch, losses, accuracies)
        classification_results: Dict opcional com resultados de classificação de imagens
        clustering_results: Dict opcional com resultados de clustering
    
    Returns:
        pd.DataFrame: DataFrame consolidado para exportação
    """
    # Criar DataFrame do histórico de treinamento
    df_training = pd.DataFrame(training_history)
    
    # Se houver resultados de classificação, adicionar
    if classification_results:
        df_classification = pd.DataFrame([classification_results])
        # Adicionar colunas vazias de treinamento para manter consistência
        for col in df_training.columns:
            if col not in df_classification.columns:
                df_classification[col] = None
        df_combined = pd.concat([df_training, df_classification], ignore_index=True)
    else:
        df_combined = df_training
    
    # Se houver resultados de clustering, adicionar
    if clustering_results:
        df_clustering = pd.DataFrame([clustering_results])
        for col in df_combined.columns:
            if col not in df_clustering.columns:
                df_clustering[col] = None
        df_combined = pd.concat([df_combined, df_clustering], ignore_index=True)
    
    return df_combined

def export_to_csv(df, filename="resultados_treinamento.csv"):
    """
    Converte DataFrame para CSV e retorna para download.
    
    Args:
        df: DataFrame para exportar
        filename: Nome do arquivo CSV
    
    Returns:
        str: CSV em formato de string
    """
    return df.to_csv(index=False).encode('utf-8')

def encode_image_to_base64(image):
    """
    Codifica uma imagem PIL para base64.
    
    Args:
        image: PIL Image
    
    Returns:
        str: Imagem codificada em base64
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def optimize_image_for_api(image, max_size=(1024, 1024)):
    """
    Otimiza uma imagem PIL para envio à API, reduzindo tamanho se necessário.
    
    Args:
        image: PIL Image
        max_size: Tupla (largura, altura) do tamanho máximo
    
    Returns:
        PIL Image otimizada
    """
    if image is None:
        return None
        
    # Se a imagem já está dentro do tamanho máximo, retornar cópia
    if image.size[0] <= max_size[0] and image.size[1] <= max_size[1]:
        return image.copy()
    
    # Redimensionar mantendo proporção
    image_copy = image.copy()
    image_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image_copy

def retry_api_call(func, max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    Wrapper para tentar chamar uma função de API com retry exponencial.
    Tenta extrair o retry_delay sugerido pela API quando disponível.
    
    Args:
        func: Função a ser executada (deve retornar tupla (sucesso, resultado))
        max_retries: Número máximo de tentativas
        initial_delay: Atraso inicial em segundos
        backoff_factor: Fator multiplicativo para cada retry
    
    Returns:
        Resultado da função ou mensagem de erro
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            success, result = func()
            if success:
                return result
            # Se não teve sucesso mas também não teve exceção, não retry
            return result
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Se for erro de quota/rate limit, tentar fazer retry
            if "429" in str(e) or "quota" in error_str or "rate limit" in error_str:
                if attempt < max_retries - 1:
                    # Tentar extrair o retry_delay sugerido pela API
                    suggested_delay = None
                    try:
                        # Procurar por "retry in XXs" ou "retry_delay { seconds: XX }"
                        match = re.search(r'retry in (\d+(?:\.\d+)?)s', str(e))
                        if match:
                            suggested_delay = float(match.group(1))
                        else:
                            match = re.search(r'seconds:\s*(\d+)', str(e))
                            if match:
                                suggested_delay = float(match.group(1))
                    except (ValueError, AttributeError):
                        # If parsing fails, continue with default delay
                        pass
                    
                    # Usar o delay sugerido se for razoável (< 120s), senão usar exponencial backoff
                    if suggested_delay and suggested_delay < 120:
                        actual_delay = min(suggested_delay, 60)  # Cap at 60s for user experience
                    else:
                        actual_delay = delay
                    
                    # Se o delay sugerido for muito longo, não vale a pena retry
                    if suggested_delay and suggested_delay > 60:
                        # Não fazer retry, deixar o erro ser tratado pelo caller
                        raise e
                    
                    time.sleep(actual_delay)
                    delay *= backoff_factor
                    continue
            # Para outros erros, não fazer retry
            raise e
    
    # Se esgotou tentativas
    if last_error:
        raise last_error
    return "Erro desconhecido ao tentar chamada de API"

def analyze_image_with_gemini(image, api_key, model_name, class_name, confidence, gradcam_description="", gradcam_image=None, max_retries=2):
    """
    Analisa uma imagem usando Google Gemini com visão computacional.
    
    Args:
        image: PIL Image (imagem original)
        api_key: Chave API do Gemini
        model_name: Nome do modelo Gemini (deve suportar visão)
        class_name: Classe predita pelo modelo
        confidence: Confiança da predição
        gradcam_description: Descrição textual do Grad-CAM
        gradcam_image: PIL Image com Grad-CAM sobreposto (opcional)
        max_retries: Número máximo de tentativas em caso de rate limit
    
    Returns:
        str: Análise técnica e forense da imagem
    """
    if not GEMINI_AVAILABLE:
        return "Google Generative AI não está disponível. Instale com: pip install google-generativeai"
    
    # Otimizar imagens antes de enviar (reduz custos e melhora performance)
    optimized_image = optimize_image_for_api(image, max_size=(1024, 1024))
    optimized_gradcam = optimize_image_for_api(gradcam_image, max_size=(1024, 1024)) if gradcam_image is not None else None
    
    # Construir prompt baseado na disponibilidade de Grad-CAM
    if optimized_gradcam is not None:
        prompt = f"""
Você é um especialista em análise de imagens e interpretação técnica e forense.

**Contexto da Classificação:**
- Classe Predita: {class_name}
- Confiança: {confidence:.4f} ({confidence*100:.2f}%)
- Análise Grad-CAM: {gradcam_description if gradcam_description else 'Veja a segunda imagem'}

**IMPORTANTE:** Você receberá DUAS imagens:
1. **Primeira imagem**: A imagem ORIGINAL classificada
2. **Segunda imagem**: A mesma imagem com sobreposição de Grad-CAM (mapa de calor vermelho-amarelo)

O Grad-CAM (Gradient-weighted Class Activation Mapping) mostra onde a rede neural focou sua "atenção" 
para fazer a classificação. Regiões em vermelho/amarelo indicam áreas de alta importância para a decisão.

Por favor, realize uma análise COMPLETA e DETALHADA das DUAS imagens, incluindo:

1. **Descrição Visual da Imagem Original:**
   - Descreva todos os elementos visuais presentes na imagem original
   - Identifique padrões, texturas, cores e formas relevantes
   - Analise a qualidade e características da imagem

2. **Análise do Grad-CAM (Segunda Imagem):**
   - Identifique quais regiões da imagem têm maior ativação (vermelho/amarelo intenso)
   - Descreva O QUE está presente nessas regiões de alta ativação
   - Avalie se essas regiões fazem sentido para a classificação como "{class_name}"
   - Compare: O modelo está focando nas características corretas?

3. **Interpretação Técnica Integrada:**
   - Avalie se a classificação como "{class_name}" é compatível com o que você observa
   - Relacione as características visuais da imagem original com as regiões de ativação
   - Analise se a confiança de {confidence*100:.2f}% é justificada pelas regiões focadas
   - Identifique se há características importantes ignoradas pelo modelo

4. **Análise Forense:**
   - Identifique possíveis artefatos ou anomalias nas imagens
   - Avalie a integridade e autenticidade da imagem
   - Verifique se o Grad-CAM está focando em artefatos em vez de características reais
   - Destaque áreas de interesse ou preocupação

5. **Recomendações:**
   - Sugira se a classificação deve ser aceita ou revista
   - Baseie-se na correlação entre características visuais e regiões de ativação
   - Recomende análises adicionais se necessário
   - Forneça orientações para melhorar a confiança na classificação

Seja detalhado, técnico e preciso na sua análise. Relacione SEMPRE os dois aspectos: 
o que você vê na imagem original e onde o modelo está focando no Grad-CAM.
"""
    else:
        prompt = f"""
Você é um especialista em análise de imagens e interpretação técnica e forense.

**Contexto da Classificação:**
- Classe Predita: {class_name}
- Confiança: {confidence:.4f} ({confidence*100:.2f}%)
- Análise Grad-CAM: {gradcam_description if gradcam_description else 'Não disponível'}

Por favor, realize uma análise COMPLETA e DETALHADA da imagem fornecida, incluindo:

1. **Descrição Visual Detalhada:**
   - Descreva todos os elementos visuais presentes na imagem
   - Identifique padrões, texturas, cores e formas relevantes
   - Analise a qualidade e características da imagem

2. **Interpretação Técnica:**
   - Avalie se a classificação como "{class_name}" é compatível com o que você observa
   - Identifique características específicas que suportam ou contradizem a classificação
   - Analise a confiança de {confidence*100:.2f}% em relação aos padrões visuais

3. **Análise Forense:**
   - Identifique possíveis artefatos ou anomalias na imagem
   - Avalie a integridade e autenticidade da imagem
   - Destaque áreas de interesse ou preocupação

4. **Recomendações:**
   - Sugira se a classificação deve ser aceita ou revista
   - Recomende análises adicionais se necessário
   - Forneça orientações para melhorar a confiança na classificação

Seja detalhado, técnico e preciso na sua análise.
"""
    
    # Função interna para fazer a chamada da API
    def make_api_call():
        if GEMINI_NEW_API:
            # New beta google-genai package API
            client = genai.Client(api_key=api_key)
            
            # Convert PIL images to bytes
            img_byte_arr = io.BytesIO()
            optimized_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Get correct model path for beta API
            model_path = get_gemini_model_path(model_name, use_new_api=True)
            
            # Build content list
            content_parts = [prompt, {"mime_type": "image/png", "data": img_byte_arr}]
            
            # Add Grad-CAM image if available
            if optimized_gradcam is not None:
                gradcam_byte_arr = io.BytesIO()
                optimized_gradcam.save(gradcam_byte_arr, format='PNG')
                gradcam_byte_arr = gradcam_byte_arr.getvalue()
                content_parts.append({"mime_type": "image/png", "data": gradcam_byte_arr})
            
            response = client.models.generate_content(
                model=model_path,
                contents=content_parts
            )
            return (True, response.text)
        else:
            # Stable google-generativeai package API (recommended)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            # Build content list
            content_parts = [prompt, optimized_image]
            
            # Add Grad-CAM image if available
            if optimized_gradcam is not None:
                content_parts.append(optimized_gradcam)
            
            response = model.generate_content(content_parts)
            return (True, response.text)
    
    # Tentar fazer a chamada com retry
    try:
        return retry_api_call(make_api_call, max_retries=max_retries, initial_delay=2.0, backoff_factor=2.0)
    except Exception as e:
        error_msg = f"Erro ao analisar com Gemini: {str(e)}\n\n"
        error_type = str(e).lower()
        error_full = str(e)
        
        # Extract specific information from quota errors
        quota_info = {
            'retry_delay': None,
            'quota_metric': None,
            'model': model_name
        }
        
        # Try to parse retry delay
        match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_full)
        if match:
            quota_info['retry_delay'] = float(match.group(1))
        else:
            match = re.search(r'seconds:\s*(\d+)', error_full)
            if match:
                quota_info['retry_delay'] = float(match.group(1))
        
        # Try to parse quota metric
        match = re.search(r'quota_metric:\s*"([^"]+)"', error_full)
        if match:
            quota_info['quota_metric'] = match.group(1)
        
        # Provide helpful guidance based on error type
        if "configure" in error_type:
            error_msg += (
                "💡 Dica: Parece que há um problema de configuração da API.\n"
                "   Certifique-se de usar: pip install google-generativeai\n"
            )
        elif "404" in str(e) and "not found" in error_type:
            error_msg += (
                "🔍 Modelo não encontrado. Use os modelos atuais do Gemini API.\n"
                "   📚 Baseado no cookbook oficial: https://github.com/google-gemini/cookbook\n"
                "   \n"
                "   Modelos recomendados (todos com suporte multimodal/visão):\n"
                "   - gemini-2.0-flash-exp ⭐ RECOMENDADO (última versão, grátis)\n"
                "   - gemini-1.5-flash (rápido e eficiente)\n"
                "   - gemini-1.5-pro (avançado com capacidade de raciocínio)\n"
                "   \n"
                "   ⚠️ Modelos legados (1.0) não são mais recomendados\n"
            )
        elif "api key" in error_type or "401" in str(e) or "403" in str(e):
            error_msg += (
                "🔑 Verifique se a API key está correta e ativa.\n"
                "   Obtenha sua API key em: https://ai.google.dev/\n"
            )
        elif "quota" in error_type or "rate limit" in error_type or "429" in str(e):
            error_msg += "⏱️ **Limite de Quota Atingido**\n\n"
            
            # Specific information about the quota
            if "free_tier" in error_full:
                error_msg += "📊 **Tipo de Quota:** Free Tier (Gratuito)\n"
            
            if quota_info['quota_metric']:
                metric_name = quota_info['quota_metric'].split('/')[-1]
                error_msg += f"📈 **Métrica Excedida:** {metric_name}\n"
            
            error_msg += f"🔢 **Modelo Usado:** {quota_info['model']}\n"
            
            if quota_info['retry_delay']:
                minutes = int(quota_info['retry_delay'] / 60)
                seconds = int(quota_info['retry_delay'] % 60)
                if minutes > 0:
                    error_msg += f"⏳ **Tempo Sugerido de Espera:** {minutes}min {seconds}s\n"
                else:
                    error_msg += f"⏳ **Tempo Sugerido de Espera:** {seconds}s\n"
            
            error_msg += f"\n🔄 **Tentativas Realizadas:** {max_retries}\n"
            error_msg += "\n💡 **Soluções Recomendadas:**\n\n"
            error_msg += "**Opção 1 - Usar Análise Multi-Agente (RECOMENDADO)** ✨\n"
            error_msg += "   - Não requer API externa\n"
            error_msg += "   - Sistema com 15 especialistas virtuais\n"
            error_msg += "   - Análise completa e detalhada\n"
            error_msg += "   - Role para baixo e clique em 'Gerar Análise Multi-Especialista'\n\n"
            
            error_msg += "**Opção 2 - Mudar de Modelo Gemini**\n"
            if "gemini-2.5-pro" in model_name or "gemini-1.5-pro" in model_name:
                error_msg += "   - Tente usar 'gemini-1.5-flash' (mais leve, quota maior)\n"
                error_msg += "   - Ou 'gemini-2.0-flash-exp' (versão experimental gratuita)\n"
            else:
                error_msg += "   - Verifique modelos alternativos disponíveis\n"
            error_msg += "   - Reconfigure na barra lateral\n\n"
            
            error_msg += "**Opção 3 - Aguardar e Tentar Novamente**\n"
            if quota_info['retry_delay']:
                if quota_info['retry_delay'] < 120:
                    error_msg += f"   - Aguarde ~{int(quota_info['retry_delay'])}s e tente novamente\n"
                else:
                    error_msg += "   - Aguarde alguns minutos (quota diária pode estar esgotada)\n"
            else:
                error_msg += "   - Aguarde 1-2 minutos e tente novamente\n"
            error_msg += "   - Verifique sua quota em: https://ai.google.dev/\n\n"
            
            error_msg += "**Opção 4 - Upgrade do Plano**\n"
            error_msg += "   - Considere upgrade para aumentar limites\n"
            error_msg += "   - Veja detalhes em: https://ai.google.dev/pricing\n"
            
        elif "resource" in error_type and "exhausted" in error_type:
            error_msg += (
                "💳 Recursos/créditos esgotados. Verifique sua conta.\n"
            )
        else:
            error_msg += (
                "📖 Consulte o guia: API_SETUP_GUIDE.md para mais detalhes.\n"
            )
        
        return error_msg

def analyze_image_with_groq_vision(image, api_key, model_name, class_name, confidence, gradcam_description=""):
    """
    Analisa uma imagem usando Groq com visão computacional.
    Nota: Groq pode ter limitações de visão dependendo do modelo.
    
    Args:
        image: PIL Image
        api_key: Chave API do Groq
        model_name: Nome do modelo Groq
        class_name: Classe predita pelo modelo
        confidence: Confiança da predição
        gradcam_description: Descrição do Grad-CAM
    
    Returns:
        str: Análise técnica e forense da imagem
    """
    if not GROQ_AVAILABLE:
        return "Groq não está disponível. Instale com: pip install groq"
    
    try:
        # Codificar imagem para base64
        image_base64 = encode_image_to_base64(image)
        
        client = Groq(api_key=api_key)
        
        prompt = f"""
        Você é um especialista em análise de imagens e interpretação técnica e forense.
        
        **Contexto da Classificação:**
        - Classe Predita: {class_name}
        - Confiança: {confidence:.4f} ({confidence*100:.2f}%)
        - Análise Grad-CAM: {gradcam_description if gradcam_description else 'Não disponível'}
        
        IMPORTANTE: Com base nas informações fornecidas e na descrição visual que você pode inferir,
        realize uma análise COMPLETA e DETALHADA, incluindo:
        
        1. **Interpretação Técnica:**
           - Avalie se a classificação como "{class_name}" parece apropriada
           - Identifique características que você esperaria ver nesta classe
           - Analise a confiança de {confidence*100:.2f}%
        
        2. **Análise Forense:**
           - Discuta possíveis pontos de atenção na classificação
           - Sugira áreas que podem precisar de verificação adicional
        
        3. **Recomendações:**
           - Sugira se a classificação deve ser aceita ou revista
           - Recomende análises adicionais se necessário
           - Forneça orientações para melhorar a confiança
        
        Nota: Se o modelo não suporta visão direta, forneça análise baseada no contexto fornecido.
        """
        
        # Tentar com suporte de imagem (alguns modelos Groq podem não suportar)
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                },
                            },
                        ],
                    }
                ],
                model=model_name,
            )
        except:
            # Fallback: análise apenas com texto se visão não é suportada
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\n[NOTA: Análise baseada apenas em contexto textual, pois o modelo pode não suportar visão direta]"
                    }
                ],
                model=model_name,
            )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Erro ao analisar com Groq: {str(e)}"

def generate_gradcam_description(activation_map):
    """
    Gera uma descrição textual do mapa de ativação Grad-CAM.
    
    Args:
        activation_map: Mapa de ativação numpy array
    
    Returns:
        str: Descrição das regiões ativadas
    """
    if activation_map is None:
        return "Grad-CAM não disponível"
    
    # Calcular estatísticas do mapa de ativação
    mean_activation = np.mean(activation_map)
    max_activation = np.max(activation_map)
    
    # Encontrar regiões de alta ativação (acima de 70% do máximo)
    threshold = 0.7 * max_activation
    high_activation_regions = activation_map > threshold
    num_high_regions = np.sum(high_activation_regions)
    total_pixels = activation_map.size
    percentage_high = (num_high_regions / total_pixels) * 100
    
    description = f"""
    O mapa Grad-CAM mostra:
    - Ativação média: {mean_activation:.3f}
    - Ativação máxima: {max_activation:.3f}
    - Regiões de alta ativação: {percentage_high:.1f}% da imagem
    - O modelo focou em {num_high_regions} pixels específicos para tomar sua decisão
    """
    
    # Analisar distribuição espacial
    height, width = activation_map.shape
    center_activation = activation_map[height//4:3*height//4, width//4:3*width//4].mean()
    border_activation = (activation_map[:height//4, :].mean() + 
                        activation_map[3*height//4:, :].mean() + 
                        activation_map[:, :width//4].mean() + 
                        activation_map[:, 3*width//4:].mean()) / 4
    
    if center_activation > border_activation * 1.5:
        description += "\n    - O modelo focou principalmente no CENTRO da imagem"
    elif border_activation > center_activation * 1.5:
        description += "\n    - O modelo focou principalmente nas BORDAS da imagem"
    else:
        description += "\n    - O modelo analisou a imagem de forma DISTRIBUÍDA"
    
    return description

def extract_features(dataset, model, batch_size):
    """
    Extrai características de um conjunto de dados usando um modelo pré-treinado.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

def perform_clustering(features, num_clusters):
    """
    Aplica algoritmos de clustering às características.
    """
    # Clustering Hierárquico
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    hierarchical_labels = hierarchical.fit_predict(features)

    # K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(features)

    return hierarchical_labels, kmeans_labels

def evaluate_clustering(true_labels, cluster_labels, method_name):
    """
    Avalia os resultados do clustering comparando com as classes reais.
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    st.write(f"**Métricas para {method_name}:**")
    st.write(f"Adjusted Rand Index: {ari:.4f}")
    st.write(f"Normalized Mutual Information Score: {nmi:.4f}")

def visualize_clusters(features, true_labels, hierarchical_labels, kmeans_labels, classes):
    """
    Visualiza os clusters usando redução de dimensionalidade e inclui as classes verdadeiras com nomes de rótulos.
    """
    # Redução de dimensionalidade com PCA para visualizar os clusters em 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Mapear os rótulos verdadeiros para os nomes das classes
    true_labels_named = [classes[label] for label in true_labels]
    
    # Usar as cores distintas e visíveis para garantir que os clusters sejam claramente separados
    color_palette = sns.color_palette("tab10", len(set(true_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))  # Agora temos 3 gráficos: Hierarchical, K-Means e classes verdadeiras

    # Clustering Hierárquico
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=hierarchical_labels, palette="deep", ax=axes[0], legend='full')
    axes[0].set_title('Clustering Hierárquico')
    ari_hierarchical = adjusted_rand_score(true_labels, hierarchical_labels)
    nmi_hierarchical = normalized_mutual_info_score(true_labels, hierarchical_labels)
    axes[0].text(0.1, 0.9, f"ARI: {ari_hierarchical:.2f}\nNMI: {nmi_hierarchical:.2f}", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # K-Means Clustering
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=kmeans_labels, palette="deep", ax=axes[1], legend='full')
    axes[1].set_title('K-Means Clustering')
    ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
    nmi_kmeans = normalized_mutual_info_score(true_labels, kmeans_labels)
    axes[1].text(0.1, 0.9, f"ARI: {ari_kmeans:.2f}\nNMI: {nmi_kmeans:.2f}", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # Classes verdadeiras
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=true_labels_named, palette=color_palette, ax=axes[2], legend='full')
    axes[2].set_title('Classes Verdadeiras')

    # Exibir os gráficos
    st.pyplot(fig)
    plt.close(fig)

def evaluate_image(model, image, classes):
    """
    Avalia uma única imagem e retorna a classe predita e a confiança.
    """
    model.eval()
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        class_name = classes[class_idx]
        return class_name, confidence.item()

def evaluate_image_with_statistics(model, image, classes, activation_map=None, n_bootstrap=100):
    """
    Avalia uma imagem com análise estatística completa.
    
    Args:
        model: Modelo treinado
        image: Imagem PIL
        classes: Lista de nomes das classes
        activation_map: Mapa de ativação do Grad-CAM (opcional)
        n_bootstrap: Número de iterações para bootstrap
    
    Returns:
        Dict com análise completa incluindo estatísticas e diagnósticos
    """
    model.eval()
    
    # Ensure model parameters don't require gradients (in case Grad-CAM left them enabled)
    for param in model.parameters():
        param.requires_grad = False
    
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    
    # 1. Predição básica
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probs_array = probabilities.cpu().numpy()[0]
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        class_name = classes[class_idx]
    
    # 2. Bootstrap validation
    stat_analyzer = StatisticalAnalyzer()
    bootstrap_results = stat_analyzer.bootstrap_validation(model, image_tensor, n_bootstrap)
    
    # 3. Confidence intervals
    confidence_interval = stat_analyzer.calculate_confidence_interval(
        bootstrap_results['predictions_distribution'][:, class_idx]
    )
    
    # 4. Differential diagnostics
    diag_analyzer = DiagnosticAnalyzer()
    differential_diagnoses = diag_analyzer.differential_diagnosis(
        bootstrap_results['mean_probabilities'], 
        classes, 
        top_k=5
    )
    
    # 5. Exclusion criteria
    exclusion_analysis = diag_analyzer.exclusion_criteria(
        bootstrap_results['mean_probabilities'], 
        classes
    )
    
    # 6. Distinctive features (se houver activation map)
    distinctive_features = None
    if activation_map is not None:
        distinctive_features = diag_analyzer.distinctive_features(activation_map)
    
    # 7. Uncertainty quantification
    uncertainty_analyzer = UncertaintyAnalyzer()
    uncertainty_analysis = uncertainty_analyzer.quantify_uncertainty(bootstrap_results)
    
    # 8. Error impact assessment
    error_impact = uncertainty_analyzer.assess_error_impact(
        bootstrap_results['mean_probabilities'],
        classes
    )
    
    # 9. Safety margin
    safety_analysis = uncertainty_analyzer.safety_margin(
        confidence.item(),
        min_acceptable=0.7,
        target=0.9
    )
    
    # 10. Clinical/practical impact
    clinical_impact = uncertainty_analyzer.clinical_impact_assessment(
        confidence.item(),
        class_name,
        differential_diagnoses
    )
    
    # 11. Significance test (se houver diagnósticos diferenciais)
    significance_test = None
    if len(differential_diagnoses) >= 2:
        significance_test = stat_analyzer.significance_test(
            differential_diagnoses[0]['probability'],
            differential_diagnoses[1]['probability'],
            bootstrap_results['predictions_distribution']
        )
    
    return {
        # Básico
        'predicted_class': class_name,
        'predicted_index': class_idx,
        'confidence': confidence.item(),
        'all_probabilities': probs_array,
        
        # Estatísticas
        'confidence_interval': confidence_interval,
        'bootstrap_results': bootstrap_results,
        'significance_test': significance_test,
        
        # Diagnóstico
        'differential_diagnoses': differential_diagnoses,
        'exclusion_analysis': exclusion_analysis,
        'distinctive_features': distinctive_features,
        
        # Incerteza e Risco
        'uncertainty_analysis': uncertainty_analysis,
        'error_impact': error_impact,
        'safety_analysis': safety_analysis,
        'clinical_impact': clinical_impact
    }

#________________________________________________

#________________________________________________

def display_statistical_analysis(analysis_results):
    """
    Exibe análise estatística completa em formato organizado e acessível no Streamlit.
    Apresentação adaptada para público leigo com explicações contextualizadas.
    
    Args:
        analysis_results: Resultados da função evaluate_image_with_statistics
    """
    st.write("---")
    st.write("## 📊 ANÁLISE ESTATÍSTICA COMPLETA")
    st.info("💡 **O que é isso?** Esta análise mostra o resultado da classificação da imagem e " +
            "o grau de certeza do sistema sobre essa classificação. Quanto maior a certeza, " +
            "mais confiável é o resultado.")
    
    # ========== PREDIÇÃO PRINCIPAL ==========
    st.write("### 🎯 Resultado da Análise")
    st.write("**O que a inteligência artificial identificou na sua imagem:**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classificação Identificada", analysis_results['predicted_class'])
        st.caption("Esta é a categoria que o sistema identificou")
    with col2:
        confidence_pct = analysis_results['confidence']
        st.metric("Nível de Certeza", f"{confidence_pct:.2%}")
        if confidence_pct >= 0.90:
            st.caption("✅ Certeza muito alta")
        elif confidence_pct >= 0.70:
            st.caption("⚠️ Certeza moderada")
        else:
            st.caption("⚠️ Certeza baixa - requer atenção")
    with col3:
        safety_emoji = {
            'safe': '🟢',
            'unsafe': '🔴'
        }[analysis_results['safety_analysis']['status']]
        safety_status = 'SEGURO' if analysis_results['safety_analysis']['status'] == 'safe' else 'REQUER ATENÇÃO'
        st.metric("Avaliação de Confiabilidade", 
                 f"{safety_emoji} {safety_status}")
        st.caption("Indica se o resultado é confiável o suficiente")
    
    # ========== INTERVALOS DE CONFIANÇA ==========
    st.write("---")
    st.write("### 📈 Análise de Confiabilidade (Intervalo de Confiança)")
    st.write("**O que significa?** O sistema testou a classificação várias vezes para verificar " +
             "se o resultado é consistente. Isso nos dá uma faixa de valores onde a certeza real " +
             "provavelmente está.")
    
    ci = analysis_results['confidence_interval']
    st.write(f"**Certeza Média (após múltiplos testes):** {ci['mean']:.2%}")
    st.write(f"**Faixa de Confiança (95%):** entre {ci['lower']:.2%} e {ci['upper']:.2%}")
    st.write(f"**Margem de Variação:** ±{ci['margin_error']:.2%}")
    
    # Progress bar visual (convert to Python float for Streamlit compatibility)
    st.progress(float(ci['mean']))
    
    # Explicação adicional
    with st.expander("📖 Entenda melhor este resultado"):
        st.write("""
        **Como interpretar:**
        - Se a faixa é estreita (pequena diferença entre o menor e maior valor): o resultado é mais estável e confiável
        - Se a faixa é ampla: há mais incerteza e o resultado pode variar
        - A margem de variação mostra o quanto o resultado pode "oscilar" para mais ou para menos
        
        **Exemplo prático:** Se a certeza está em 65% com margem de ±4%, isso significa que 
        o resultado real provavelmente está entre 61% e 69%.
        """)
    
    # ========== DIAGNÓSTICOS DIFERENCIAIS ==========
    st.write("---")
    st.write("### 🔍 Possibilidades Alternativas (Diagnósticos Diferenciais)")
    st.write("**O que significa?** Além da classificação principal, o sistema identifica outras " +
             "possibilidades que a imagem poderia representar, ordenadas por probabilidade.")
    
    diff_data = []
    for diff in analysis_results['differential_diagnoses']:
        diff_data.append({
            'Posição': diff['rank'],
            'Categoria': diff['class'],
            'Probabilidade': f"{diff['probability']:.2%}",
            'Nível de Certeza': diff['confidence_level']
        })
    
    if diff_data:
        st.dataframe(pd.DataFrame(diff_data), width='stretch')
        st.caption("💡 A primeira linha é a classificação mais provável, as demais são alternativas em ordem decrescente de probabilidade")
    
    # Teste de significância
    if analysis_results['significance_test'] and analysis_results['significance_test']['p_value']:
        st.write("#### 📊 Comparação entre as Duas Principais Possibilidades")
        sig_test = analysis_results['significance_test']
        st.write(f"**Diferença de Probabilidade:** {sig_test['probability_diff']:.2%}")
        st.write(f"**Valor-p (teste estatístico):** {sig_test['p_value']:.4f}")
        
        with st.expander("📖 O que é o valor-p?"):
            st.write("""
            O **valor-p** é uma medida estatística que nos ajuda a determinar se a diferença 
            entre duas opções é significativa (importante) ou se pode ter ocorrido por acaso.
            
            **Regra prática:**
            - Valor-p < 0.05: A diferença é **significativa** - há uma diferença real entre as duas opções
            - Valor-p ≥ 0.05: A diferença **não é significativa** - as duas opções são muito parecidas
            
            **Neste caso:** {}
            """.format(
                "As duas principais possibilidades são **estatisticamente diferentes**, " +
                "ou seja, há uma clara vantagem da primeira opção sobre a segunda." 
                if sig_test['significant'] 
                else "As duas principais possibilidades são **muito semelhantes**, " +
                "o que indica que o sistema teve dificuldade em distinguir entre elas. " +
                "Recomenda-se cautela e possivelmente uma análise adicional."
            ))
        
        if sig_test['significant']:
            st.success(f"✅ {sig_test['interpretation']} (p < 0.05) - Há diferença clara entre as opções")
        else:
            st.warning(f"⚠️ {sig_test['interpretation']} (p ≥ 0.05) - As opções são muito similares, difícil distinguir")
    
    # ========== CRITÉRIOS DE EXCLUSÃO ==========
    st.write("---")
    st.write("### ❌ Categorias Descartadas (Critérios de Exclusão)")
    st.write("**O que significa?** O sistema identificou categorias que têm probabilidade muito baixa " +
             "de serem a resposta correta e as descartou da análise.")
    
    excl = analysis_results['exclusion_analysis']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Categorias Descartadas", excl['excluded_count'])
        st.caption("Opções com probabilidade muito baixa")
    with col2:
        st.metric("Categorias Consideradas", excl['remaining_count'])
        st.caption("Opções ainda em análise")
    
    if excl['excluded_classes']:
        with st.expander("Ver categorias descartadas (probabilidade muito baixa)"):
            st.write("Estas categorias foram descartadas porque a probabilidade era muito pequena:")
            for exc in excl['excluded_classes'][:5]:  # Mostrar até 5
                st.write(f"- **{exc['class']}**: {exc['reason']}")
    
    # ========== CARACTERÍSTICAS DISTINTIVAS ==========
    if analysis_results['distinctive_features']:
        st.write("---")
        st.write("### 🎨 Regiões Importantes da Imagem (Características Distintivas)")
        st.write("**O que significa?** O sistema analisa quais partes da imagem foram mais importantes " +
                 "para tomar a decisão de classificação.")
        
        feat = analysis_results['distinctive_features']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ativação Máxima", f"{feat['max_activation']:.3f}")
            st.caption("Intensidade máxima nas áreas analisadas")
        with col2:
            st.metric("Ativação Média", f"{feat['mean_activation']:.3f}")
            st.caption("Intensidade média geral")
        with col3:
            st.metric("Área de Alta Relevância", f"{feat['high_activation_percentage']:.1f}%")
            st.caption("Porcentagem da imagem considerada importante")
        
        st.info(f"**Interpretação:** {feat['interpretation']}")
        
        with st.expander("📖 Como interpretar estes valores"):
            st.write("""
            **Áreas de ativação** mostram onde o sistema "prestou mais atenção" na imagem:
            
            - **Alta ativação em área pequena** (< 15%): O sistema focou em detalhes específicos
            - **Alta ativação em área média** (15-30%): Análise equilibrada de várias características
            - **Alta ativação em área grande** (> 30%): O sistema considerou muitas partes da imagem
            
            Valores mais altos de ativação indicam regiões que tiveram maior peso na decisão.
            """)
    
    # ========== ANÁLISE DE INCERTEZA ==========
    st.write("---")
    st.write("### 🎲 Medição da Incerteza (Quantificação de Incerteza)")
    st.write("**O que significa?** Esta análise mostra o quanto o sistema está incerto sobre o resultado. " +
             "Maior incerteza significa que o resultado pode ser menos confiável.")
    
    uncert = analysis_results['uncertainty_analysis']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nível de Incerteza", uncert['uncertainty_level'])
        st.caption("Classificação geral da incerteza")
    with col2:
        st.metric("Incerteza Total", f"{uncert['total_uncertainty']:.3f}")
        st.caption("Valor combinado de todas as fontes")
    with col3:
        st.metric("Entropia Normalizada", f"{uncert['normalized_entropy']:.3f}")
        st.caption("Medida de dispersão das probabilidades")
    
    st.write("**Fontes de Incerteza (de onde vem a dúvida):**")
    st.write(f"- **Variação do Modelo:** {uncert['sources']['model_variation']:.3f} " +
             "(quanto o resultado varia entre múltiplas análises)")
    st.write(f"- **Ambiguidade da Predição:** {uncert['sources']['prediction_ambiguity']:.3f} " +
             "(quanto as probabilidades estão distribuídas entre várias opções)")
    
    with st.expander("📖 Entenda a incerteza"):
        st.write("""
        **Níveis de Incerteza:**
        - **Muito Baixa/Baixa**: O sistema está bastante confiante no resultado
        - **Moderada**: Há alguma dúvida, mas o resultado ainda é útil
        - **Alta/Muito Alta**: O sistema tem muita dúvida - cuidado ao usar este resultado
        
        **Fontes:**
        - **Variação do Modelo**: Se o modelo dá resultados diferentes ao analisar a mesma imagem várias vezes
        - **Ambiguidade**: Se várias categorias têm probabilidades semelhantes, criando dúvida
        """)
    
    # ========== IMPACTO DE ERROS ==========
    st.write("---")
    st.write("### ⚠️ Risco de Erro (Avaliação de Impacto de Erros)")
    st.write("**O que significa?** Esta análise estima a probabilidade de o resultado estar errado " +
             "e qual seria o impacto de um possível erro.")
    
    error_imp = analysis_results['error_impact']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidade de Erro", f"{error_imp['error_probability']:.2%}")
        st.caption("Chance de a classificação estar incorreta")
    with col2:
        st.metric("Índice de Impacto", f"{error_imp['impact_score']:.3f}")
        st.caption("Gravidade de um possível erro")
    
    # Mostrar recomendação com cor apropriada
    if '⚠️ ATENÇÃO' in error_imp['recommendation']:
        st.error(error_imp['recommendation'])
    elif '⚠️' in error_imp['recommendation']:
        st.warning(error_imp['recommendation'])
    else:
        st.success(error_imp['recommendation'])
    
    # ========== MARGEM DE SEGURANÇA ==========
    st.write("---")
    st.write("### 🛡️ Análise de Segurança (Margem de Segurança)")
    st.write("**O que significa?** Esta análise compara a certeza obtida com os níveis mínimos " +
             "considerados seguros para uso prático do resultado.")
    
    safety = analysis_results['safety_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Certeza Atual", f"{safety['confidence']:.2%}")
        st.caption("Confiança obtida nesta análise")
    with col2:
        st.metric("Mínimo Aceitável", f"{safety['min_acceptable']:.2%}")
        st.caption("Limite mínimo recomendado")
    with col3:
        st.metric("Alvo Desejado", f"{safety['target']:.2%}")
        st.caption("Valor ideal de confiança")
    with col4:
        st.metric("Índice de Segurança", f"{safety['safety_score']:.2%}")
        st.caption("Pontuação geral de segurança")
    
    st.write(f"**Distância do Mínimo:** {safety['margin_to_minimum']:.2%} " +
             f"({'acima' if safety['margin_to_minimum'] > 0 else 'abaixo'} do limite)")
    st.write(f"**Distância do Alvo:** {safety['margin_to_target']:.2%} para alcançar o ideal")
    
    # Interpretação com emoji e cores
    if '🔴' in safety['interpretation']:
        st.error(safety['interpretation'])
    elif '🟡' in safety['interpretation']:
        st.warning(safety['interpretation'])
    else:
        st.success(safety['interpretation'])
    
    with st.expander("📖 Como interpretar a segurança"):
        st.write("""
        **Níveis de Segurança:**
        - 🟢 **Verde (acima do alvo)**: Resultado muito confiável - pode ser usado com segurança
        - 🟢 **Verde (acima do mínimo)**: Resultado aceitável - pode ser usado com precaução
        - 🟡 **Amarelo**: Resultado próximo ao limite - usar com extrema cautela
        - 🔴 **Vermelho**: Resultado abaixo do aceitável - NÃO recomendado para uso sem análise adicional
        
        **Recomendação:** Sempre busque resultados com certeza acima de 70% para aplicações práticas.
        Para decisões importantes, prefira resultados acima de 90%.
        """)
    
    # ========== IMPACTO CLÍNICO/PRÁTICO ==========
    st.write("---")
    st.write("### 🏥 Impacto Prático do Resultado (Avaliação Clínica/Prática)")
    st.write("**O que significa?** Esta seção avalia o que fazer com o resultado obtido e " +
             "qual o nível de urgência ou importância da classificação.")
    
    clinical = analysis_results['clinical_impact']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classificação Principal", clinical['primary_diagnosis'])
        st.caption("Resultado mais provável")
    with col2:
        priority_color = {
            'Normal': '🟢',
            'Média': '🟡',
            'Alta': '🔴'
        }[clinical['priority_level']]
        st.metric("Nível de Prioridade", f"{priority_color} {clinical['priority_level']}")
        st.caption("Urgência da ação necessária")
    with col3:
        st.metric("Nível de Ambiguidade", f"{clinical['diagnostic_ambiguity']:.2%}")
        st.caption("Quanto as opções se confundem")
    
    st.write(f"**📋 O que fazer agora:** {clinical['recommended_action']}")
    st.write(f"**🔍 Outras possibilidades analisadas:** {clinical['differential_count']}")
    
    if clinical['requires_specialist']:
        st.warning("⚕️ **Recomendação:** Consulta com especialista recomendada devido à complexidade " +
                  "do caso ou nível de incerteza elevado")
    else:
        st.success("✅ **Situação:** Este caso pode ser tratado seguindo protocolos padrão, " +
                  "sem necessidade imediata de consulta especializada")
    
    with st.expander("📖 Entenda a prioridade e recomendações"):
        st.write("""
        **Níveis de Prioridade:**
        - 🟢 **Normal**: Situação estável, seguir acompanhamento de rotina
        - 🟡 **Média**: Requer atenção moderada, acompanhar mais de perto
        - 🔴 **Alta**: Situação que requer atenção urgente ou análise mais detalhada
        
        **Ambiguidade Diagnóstica:**
        - Baixa (< 30%): As opções são bem distintas, decisão mais clara
        - Moderada (30-70%): Há alguma sobreposição entre opções
        - Alta (> 70%): As opções são muito similares, difícil distinção
        
        **Quando procurar um especialista:**
        - Ambiguidade alta (> 70%)
        - Certeza baixa (< 75%)
        - Múltiplas possibilidades com probabilidades semelhantes
        - Quando há implicações importantes da decisão
        """)
    
    # ========== VALIDAÇÃO BOOTSTRAP ==========
    with st.expander("📊 Detalhes Técnicos - Validação Bootstrap (Para Usuários Avançados)"):
        st.write("**O que é Bootstrap?** É um método estatístico que testa o resultado múltiplas " +
                 "vezes para verificar sua estabilidade. Quanto menor a variação, mais confiável o resultado.")
        
        boot = analysis_results['bootstrap_results']
        st.write(f"**Certeza Média (Bootstrap):** {boot['confidence_bootstrap']:.2%}")
        st.write(f"**Variação (Desvio Padrão):** {boot['uncertainty']:.4f}")
        st.caption("💡 Variação baixa (< 0.10) indica resultado estável; alta (> 0.20) indica instabilidade")
        
        st.write("---")
        st.write("**Probabilidades Médias por Categoria:**")
        # Create proper dataframe for all classes
        all_classes = list(range(len(boot['mean_probabilities'])))
        prob_df = pd.DataFrame({
            'Índice da Categoria': all_classes,
            'Probabilidade Média': [f"{p:.2%}" for p in boot['mean_probabilities']],
            'Desvio Padrão': [f"{s:.4f}" for s in boot['std_probabilities']]
        })
        st.dataframe(prob_df.head(10), width='stretch')  # Mostrar top 10
        st.caption("As 10 categorias com maiores probabilidades")
    
    # ========== RESUMO FINAL PARA LEIGOS ==========
    st.write("---")
    st.write("## 📝 RESUMO FINAL EM LINGUAGEM SIMPLES")
    st.write("### O que você precisa saber sobre este resultado:")
    
    # Criar resumo baseado nos dados da análise
    confidence = analysis_results['confidence']
    predicted = analysis_results['predicted_class']
    safety_status = analysis_results['safety_analysis']['status']
    uncertainty_level = analysis_results['uncertainty_analysis']['uncertainty_level']
    error_prob = analysis_results['error_impact']['error_probability']
    
    st.write(f"**1. Resultado Principal:**")
    st.write(f"   - A imagem foi classificada como: **{predicted}**")
    st.write(f"   - Nível de certeza: **{confidence:.2%}**")
    
    st.write(f"\n**2. Confiabilidade:**")
    if safety_status == 'safe' and confidence >= 0.75:
        st.success("   ✅ Este resultado é considerado **CONFIÁVEL** para uso.")
    elif safety_status == 'safe' or confidence >= 0.60:
        st.warning("   ⚠️ Este resultado é **ACEITÁVEL**, mas use com **PRECAUÇÃO**.")
    else:
        st.error("   ⚠️ Este resultado tem **BAIXA CONFIABILIDADE** - requer análise adicional.")
    
    st.write(f"\n**3. Nível de Incerteza:**")
    st.write(f"   - Classificação: **{uncertainty_level}**")
    if uncertainty_level in ['Muito Baixa', 'Baixa']:
        st.write("   - Significa: O sistema está bastante seguro do resultado")
    elif uncertainty_level == 'Moderada':
        st.write("   - Significa: Há alguma dúvida, mas o resultado ainda é útil")
    else:
        st.write("   - Significa: O sistema tem dúvidas significativas sobre o resultado")
    
    st.write(f"\n**4. Probabilidade de Erro:**")
    st.write(f"   - Chance de estar errado: **{error_prob:.2%}**")
    if error_prob < 0.20:
        st.write("   - Interpretação: Chance baixa de erro")
    elif error_prob < 0.40:
        st.write("   - Interpretação: Chance moderada de erro - atenção necessária")
    else:
        st.write("   - Interpretação: Chance alta de erro - cuidado!")
    
    st.write(f"\n**5. Recomendação Final:**")
    requires_specialist = analysis_results['clinical_impact']['requires_specialist']
    recommended_action = analysis_results['clinical_impact']['recommended_action']
    
    if requires_specialist:
        st.warning(f"   ⚕️ **Consultar especialista:** Sim, recomendado")
        st.write(f"   - Motivo: {recommended_action}")
    else:
        st.success(f"   ✅ **Consultar especialista:** Não é urgente")
        st.write(f"   - Ação sugerida: {recommended_action}")
    
    # Adicionar glossário rápido
    with st.expander("📖 Glossário - Entenda os Termos Técnicos"):
        st.write("""
        **Termos que você pode ter visto neste relatório:**
        
        - **Bootstrap/Validação Bootstrap**: Método estatístico que repete a análise múltiplas vezes para verificar se o resultado é estável
        - **Confiança/Certeza**: O quanto o sistema está seguro de que a classificação está correta (em porcentagem)
        - **Diagnóstico Diferencial**: Outras possíveis classificações que a imagem poderia ter
        - **Entropia**: Medida de incerteza ou "desordem" - quanto maior, mais incerta é a classificação
        - **Intervalo de Confiança**: Faixa de valores onde o resultado verdadeiro provavelmente está
        - **Margem de Erro**: Quanto o valor pode variar para mais ou para menos
        - **Probabilidade**: Chance de algo ser verdade, expressa em porcentagem (0% a 100%)
        - **Significância Estatística**: Se uma diferença é real ou pode ter ocorrido por acaso
        - **Valor-p**: Número que indica se uma diferença é estatisticamente significativa (< 0.05 = significativa)
        
        **Formato ABNT (Normas Brasileiras):**
        Este relatório segue as diretrizes da Associação Brasileira de Normas Técnicas (ABNT) 
        para apresentação de análises científicas, garantindo qualidade acadêmica nível A1 
        (mais alto nível de qualidade acadêmica no Brasil).
        """)

def visualize_activations(model, image, class_names, gradcam_type='SmoothGradCAMpp'):
    """
    Visualiza as ativações na imagem usando diferentes variantes de Grad-CAM.
    
    Args:
        model: Modelo treinado
        image: Imagem PIL
        class_names: Lista de nomes das classes
        gradcam_type: Tipo de Grad-CAM ('GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'LayerCAM')
    
    Returns:
        tuple: (activation_map_resized, gradcam_image_pil) onde:
            - activation_map_resized: Mapa de ativação normalizado ou None em caso de erro
            - gradcam_image_pil: Imagem PIL com Grad-CAM sobreposto ou None
    """
    cam_extractor = None
    original_training_mode = model.training
    try:
        # Set model to eval mode but enable gradient computation
        model.eval()
        
        # Enable gradients for all model parameters temporarily
        # This is needed for hook registration in torchcam
        for param in model.parameters():
            param.requires_grad = True
        
        # Prepare input tensor with gradient enabled
        # torchcam requires gradients to be enabled for hook registration
        input_tensor = test_transforms(image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Verificar se o modelo é suportado
        model_type = type(model).__name__
        if 'ResNet' in model_type:
            target_layer = model.layer4[-1]
        elif 'DenseNet' in model_type:
            target_layer = model.features.denseblock4.denselayer16
        elif 'VisionTransformer' in model_type:
            # Para ViT, usar a última camada do encoder
            target_layer = model.encoder.layers[-1].ln_1
        else:
            st.warning(f"Modelo {model_type} pode não ter suporte completo para Grad-CAM. Tentando com camada padrão...")
            # Tentar usar a última camada disponível
            try:
                if hasattr(model, 'encoder'):
                    target_layer = model.encoder.layers[-1]
                else:
                    st.error("Não foi possível determinar camada para Grad-CAM.")
                    return None
            except:
                st.error("Modelo não suportado para Grad-CAM.")
                return None
        
        # Criar o objeto CAM usando torchcam
        if gradcam_type == 'GradCAM':
            cam_extractor = GradCAM(model, target_layer=target_layer)
        elif gradcam_type == 'GradCAMpp':
            cam_extractor = GradCAMpp(model, target_layer=target_layer)
        elif gradcam_type == 'SmoothGradCAMpp':
            cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
        elif gradcam_type == 'LayerCAM':
            cam_extractor = LayerCAM(model, target_layer=target_layer)
        else:
            st.error(f"Tipo de Grad-CAM não suportado: {gradcam_type}")
            return None
        
        # Habilitar gradientes explicitamente
        with torch.set_grad_enabled(True):
            out = model(input_tensor)  # Faz a previsão
            _, pred = torch.max(out, 1)  # Obtém a classe predita
            pred_class = pred.item()
        
        # Gerar o mapa de ativação
        activation_map = cam_extractor(pred_class, out)
        
        # Obter o mapa de ativação da primeira imagem no lote
        activation_map = activation_map[0].cpu().detach().numpy()
        
        # Redimensionar o mapa de ativação para coincidir com o tamanho da imagem original
        activation_map_resized = cv2.resize(activation_map, (image.size[0], image.size[1]))
        
        # Normalizar o mapa de ativação para o intervalo [0, 1]
        activation_map_resized = (activation_map_resized - activation_map_resized.min()) / (activation_map_resized.max() - activation_map_resized.min() + 1e-8)
        
        # Converter a imagem para array NumPy
        image_np = np.array(image)
        
        # Converter o mapa de ativação em uma imagem RGB
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Sobrepor o mapa de ativação na imagem original
        superimposed_img = heatmap * 0.4 + image_np * 0.6
        superimposed_img = np.uint8(superimposed_img)
        
        # Exibir a imagem original e o mapa de ativação sobreposto
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Imagem original
        ax[0].imshow(image_np)
        ax[0].set_title('Imagem Original')
        ax[0].axis('off')
        
        # Imagem com Grad-CAM
        ax[1].imshow(superimposed_img)
        ax[1].set_title(f'{gradcam_type}')
        ax[1].axis('off')
        
        # Exibir as imagens com o Streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        # Converter superimposed_img para PIL Image para retornar
        gradcam_image_pil = Image.fromarray(superimposed_img)
        
        return activation_map_resized, gradcam_image_pil
        
    except Exception as e:
        st.error(f"Erro ao gerar Grad-CAM: {str(e)}")
        st.info("Visualização Grad-CAM não disponível para este modelo/configuração.")
        return None, None
    finally:
        # CRITICAL: Remove hooks and reset model state to prevent interference with subsequent calls
        if cam_extractor is not None:
            try:
                # Try multiple cleanup methods for compatibility with different torchcam versions
                if hasattr(cam_extractor, 'remove_hooks'):
                    cam_extractor.remove_hooks()
                elif hasattr(cam_extractor, 'clear_hooks'):
                    cam_extractor.clear_hooks()
                elif hasattr(cam_extractor, 'reset_hooks'):
                    cam_extractor.reset_hooks()
            except Exception as e:
                # If hook removal fails, log it but continue
                st.warning(f"Aviso: Não foi possível remover hooks: {e}")
        
        # Restore original training mode
        if original_training_mode:
            model.train()
        else:
            model.eval()




def main():

    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto
    
    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
        logging.info(f"Ícone {icon_path} carregado com sucesso.")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")
    
    # Layout da página
    if os.path.exists('capa.png'):
        st.image('capa.png', caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', width='stretch')
    else:
        st.warning("Imagem 'capa.png' não encontrada.")
    
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=200)
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")
    
    
  #___________________________________________________________
    st.title("Classificação por Imagens com Aprendizado Profundo")
    st.write("Este aplicativo permite treinar um modelo de classificação de imagens e aplicar algoritmos de clustering para análise comparativa.")
    with st.expander("Transformações de Dados e Aumento de Dados no Treinamento de Redes Neurais"):
        st.write("""
        As **transformações de dados** e o **aumento de dados** são técnicas essenciais no treinamento de redes neurais profundas, principalmente em tarefas de visão computacional. 
        Essas abordagens buscam melhorar a capacidade de generalização dos modelos, gerando **imagens sintéticas** a partir dos dados de treinamento. Tais técnicas são particularmente 
        valiosas quando o conjunto de dados disponível é pequeno ou apresenta pouca diversidade. A normalização, por sua vez, assegura que os valores dos pixels estejam em uma escala adequada, 
        resultando em um treinamento mais estável e eficiente. Diversos estudos apontam que essas práticas são eficazes para evitar **overfitting** e aumentar a robustez do modelo 
        (Shorten & Khoshgoftaar, 2019).
        """)
    
        st.write("### Aumento de Dados no Treinamento")
    
        st.write("""
        O **aumento de dados** ou *data augmentation* consiste na aplicação de transformações aleatórias às imagens do conjunto de treinamento para gerar novas amostras sintéticas. 
        No código implementado, essa técnica é realizada com a classe `transforms.Compose` da biblioteca **torchvision**, que aplica uma sequência de transformações.
        """)
    
        st.write("#### Transformações Aplicadas no Treinamento")
        
        st.write("""
        1. **RandomApply**: Aplica aleatoriamente um conjunto de transformações com 50% de probabilidade. Esse procedimento aumenta a variabilidade dos dados, gerando imagens diferentes a partir de uma única imagem de entrada.
       
        2. **RandomHorizontalFlip**: Realiza a inversão horizontal da imagem com 50% de probabilidade. Isso é útil em cenários onde a orientação horizontal da imagem não altera seu significado, como em imagens de rochas ou melanomas.
    
        3. **RandomRotation(degrees=90)**: Rotaciona a imagem em até 90 graus, criando variações angulares, o que ajuda o modelo a reconhecer objetos independentemente da orientação.
    
        4. **ColorJitter**: Introduz variações de brilho, contraste, saturação e matiz, simulando diferentes condições de iluminação e tornando o modelo mais robusto a mudanças de iluminação.
    
        5. **RandomResizedCrop(224, scale=(0.8, 1.0))**: Realiza cortes aleatórios na imagem e os redimensiona para 224x224 pixels, permitindo que diferentes partes da imagem sejam enfatizadas.
    
        6. **RandomAffine(degrees=0, shear=10)**: Aplica transformações afins, como cisalhamento, simulando distorções que podem ocorrer no mundo real, como mudanças de perspectiva.
    
        7. **Resize(256)**: Redimensiona a imagem para 256x256 pixels, assegurando que todas as imagens possuam a mesma dimensão.
    
        8. **CenterCrop(224)**: Recorta o centro da imagem, garantindo que o tamanho final seja 224x224 pixels.
    
        9. **ToTensor**: Converte a imagem para um tensor PyTorch, normalizando os valores dos pixels para o intervalo de [0,1], facilitando o processamento pelo modelo.
        """)
    
        st.write("### Geração de Imagens Sintéticas")
    
        st.write("""
        Essas transformações permitem que cada imagem original gere até **5 a 10 imagens sintéticas**. Por exemplo, em um conjunto de dados de 1000 imagens, 
        o processo pode expandir o conjunto para **5000 a 10000 imagens** ao longo do treinamento. Essa ampliação artificial do conjunto de dados reduz o risco de **overfitting**, 
        permitindo que o modelo treine em um conjunto "maior" e mais diverso, o que é crucial para melhorar a generalização do modelo em dados novos.
        """)
    
        st.write("### Normalização nas Imagens de Teste e Validação")
    
        st.write("""
        Nas imagens de **teste** e **validação**, o aumento de dados não é aplicado. O objetivo nesses conjuntos é avaliar o modelo de maneira consistente, 
        utilizando imagens que representem o mais fielmente possível os dados reais. No entanto, a normalização dessas imagens é fundamental para assegurar que seus valores de pixel 
        estejam adequados para as operações de aprendizado. Isso também garante um desempenho estável durante o treinamento.
        """)
    
        st.write("#### Transformações Aplicadas no Teste e Validação")
        
        st.write("""
        1. **Resize(256)**: Redimensiona a imagem para 256x256 pixels, garantindo que todas as imagens tenham o mesmo tamanho inicial.
    
        2. **CenterCrop(224)**: Realiza o corte central para que as dimensões da imagem sejam 224x224 pixels, correspondendo ao tamanho esperado pelo modelo.
    
        3. **ToTensor**: Converte a imagem para tensor e normaliza os valores dos pixels para o intervalo de [0,1], o que melhora a estabilidade numérica e a taxa de convergência do treinamento.
        """)
    
        st.write("### Importância da Normalização")
    
        st.write("""
        A **normalização** garante que os valores dos pixels estejam em uma escala apropriada para as operações aritméticas realizadas no modelo, melhorando a estabilidade e o desempenho do processo de treinamento. 
        Ela também contribui para a estabilidade numérica durante o cálculo do gradiente e para uma convergência mais eficiente do modelo (Nguyễn et al., 2021).
        """)
    
        st.write("### Conclusão")
    
        st.write("""
        O código exemplifica a implementação eficaz de transformações de dados e aumento de dados como parte da pipeline de treinamento de redes neurais profundas. 
        As transformações aplicadas aumentam a diversidade do conjunto de treinamento, ajudando a mitigar o **overfitting** e melhorar a generalização do modelo. 
        Além disso, a normalização aplicada aos dados de teste e validação garante que o desempenho do modelo seja avaliado de forma precisa e consistente, 
        alinhada às melhores práticas de aprendizado profundo.
        """)
    
        st.write("### Referências")
        
        st.write("""
        - Huang, G., Liu, Z., Maaten, L., & Weinberger, K. (2017). Densely connected convolutional networks. https://doi.org/10.1109/cvpr.2017.243
        - Li, S. (2023). Clouddensenet: lightweight ground-based cloud classification method for large-scale datasets based on reconstructed densenet. *Sensors*, 23(18), 7957. https://doi.org/10.3390/s23187957
        - Nguyễn, H., Yu, G., Shin, N., Kwon, G., Kwak, W., & Kim, J. (2021). Defective product classification system for smart factory based on deep learning. *Electronics*, 10(7), 826. https://doi.org/10.3390/electronics10070826
        - Shorten, C. & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1). https://doi.org/10.1186/s40537-019-0197-0
        """)

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")
      # Imagem e Contatos___________________________
    #_______________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmulas LaTeX
    with st.sidebar:
        with st.expander("Discussão sobre o Número de Classes em Modelos de Aprendizado Profundo"):
            st.write("""
            ### Introdução
    
            A discussão sobre o número de classes em modelos de aprendizado profundo é fundamental para a compreensão da arquitetura e do desempenho de redes neurais em tarefas de classificação. O número de classes refere-se ao total de categorias ou rótulos que um modelo deve prever, e a configuração correta desse parâmetro impacta diretamente o desempenho do modelo, pois afeta a dimensão da saída da rede neural e a complexidade da tarefa. O número de classes pode variar de tarefas binárias, que envolvem apenas duas classes, até problemas com centenas ou milhares de classes, como nas classificações de imagens do **ImageNet** (Cheng, 2023).
            """)
    
            st.write("### Impacto do Número de Classes")
            st.write("""
            O número de classes define a estrutura da última camada da rede neural, que é responsável por realizar as predições. Para um problema de **classificação binária**, o modelo terá uma única saída que prevê a probabilidade de uma classe ou outra. Em contrapartida, em um problema de **classificação multiclasse**, o número de saídas será igual ao número de categorias possíveis (Cheng, 2023). A função de ativação utilizada na última camada é crucial para a interpretação dos resultados. A equação que representa essa relação pode ser expressa como:
            """)
            st.latex(r'''
            \mathbf{y} = \text{Softmax}(Wx + b)
            ''')
    
            st.write("""
            onde **W** e **b** são os pesos e o bias, respectivamente, que conectam a camada anterior às classes de saída. O resultado é passado pela função **softmax**, que converte os valores em probabilidades associadas a cada classe (Petrovska et al., 2020).
            """)
    
                       
            st.write("""
            Em tarefas de classificação binária, o modelo tem apenas duas classes possíveis, como **detecção de fraude** ou **diagnóstico de doenças** (positivo ou negativo). Nesse caso, a função de ativação final é geralmente a **sigmoide**, que retorna uma probabilidade entre 0 e 1 para cada entrada. Um limiar é então aplicado para decidir a classe final predita pelo modelo (Cheng, 2023).
            """)
    
            st.write("### Classificação Multiclasse")
            st.write("""
            Em problemas de classificação multiclasse, o número de classes pode variar consideravelmente. Por exemplo, em tarefas de **classificação de imagens geológicas**, o número de classes pode ser pequeno, mas em aplicações como a **classificação de imagens médicas** ou **reconhecimento facial**, o número de classes pode ser muito maior. A arquitetura da rede deve ser ajustada para garantir que a última camada tenha o número correto de saídas correspondente ao número de categorias (Cheng, 2023; Sardeshmukh, 2023).
            """)
    
            st.write("### Classificação Multirrótulo")
            st.write("""
            Em problemas de **classificação multirrótulo**, uma entrada pode pertencer a mais de uma classe ao mesmo tempo. Nesse cenário, o número de saídas da rede neural é igual ao número de classes possíveis, mas cada saída é independente das demais. A função de ativação usada é a **sigmoide**, pois ela calcula a probabilidade de cada classe independentemente das outras (Petrovska et al., 2020).
            """)
    
            st.write("### Efeitos do Número de Classes no Desempenho")
            st.write("""
            O número de classes influencia diretamente a complexidade do modelo e o tempo de treinamento. Conforme o número de classes aumenta, a tarefa de classificação se torna mais difícil, exigindo mais parâmetros e tempo de computação. Além disso, um maior número de classes aumenta o risco de **sobreajuste** (overfitting), especialmente em conjuntos de dados pequenos (Cheng, 2023; Suhana, 2022).
            """)
    
            st.write("### Conclusão")
            st.write("""
            O número de classes é um fator determinante na definição da arquitetura de redes neurais para tarefas de classificação. Seja em problemas binários, multiclasse ou multirrótulo, a escolha adequada desse parâmetro garante que a rede neural seja capaz de aprender as características relevantes de cada categoria. Em problemas com muitas classes, estratégias como a **regularização** e o **data augmentation** podem ser utilizadas para melhorar o desempenho do modelo, evitando o sobreajuste (Cheng, 2023; Sardeshmukh, 2023).
            """)
    
            st.write("### Referências")
          
            st.write("""
            1. Cheng, R. (2023). Expansion of the CT-scans image set based on the pretrained DCGAN for improving the performance of the CNN. *Journal of Physics Conference Series*, 2646(1), 012015. https://doi.org/10.1088/1742-6596/2646/1/012015
            2. Petrovska, B., Atanasova-Pacemska, T., Corizzo, R., Mignone, P., Lameski, P., & Zdravevski, E. (2020). Aerial Scene Classification through Fine-Tuning with Adaptive Learning Rates and Label Smoothing. *Applied Sciences*, 10(17), 5792. https://doi.org/10.3390/app10175792
            3. Sardeshmukh, M. (2023). Crop image classification using convolutional neural network. *Multidisciplinary Science Journal*, 5(4), 2023039. https://doi.org/10.31893/multiscience.2023039
            4. Suhana, R. (2022). Fish Image Classification Using Adaptive Learning Rate In Transfer Learning Method. *Knowledge Engineering and Data Science*, 5(1), 67-77. https://doi.org/10.17977/um018v5i12022p67-77
            """)

  
    # Nota: O número de classes será detectado automaticamente do dataset
    num_classes = st.sidebar.number_input("Número de Classes (será detectado automaticamente):", min_value=1, step=1, value=2, disabled=True, help="Este valor será automaticamente detectado do dataset após o upload")
    #_______________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmula LaTeX
    with st.sidebar:
        with st.expander("Modelos Pré-Treinados: ResNet18, ResNet50 e DenseNet121:"):
            st.write("""
            ### Introdução
        
            As redes neurais convolucionais (CNNs) têm se tornado uma ferramenta essencial no campo do aprendizado profundo, especialmente em tarefas de visão computacional, como a classificação de imagens. 
            Modelos como **ResNet18**, **ResNet50** e **DenseNet121** são amplamente reconhecidos por seu desempenho superior em competições de classificação de imagens, como o **ImageNet**. Esses modelos são considerados 
            **pré-treinados**, pois foram inicialmente treinados em grandes conjuntos de dados, permitindo que sejam reutilizados e ajustados para novas tarefas específicas, uma prática conhecida como **transferência de aprendizado** 
            (Cheng, 2023; Petrovska et al., 2020; Alaoui, 2023).
            """)
        
            st.write("### ResNet18 e ResNet50")
            st.write("""
            A arquitetura **ResNet** (Rede Residual) foi desenvolvida para mitigar o problema de **degradação** que ocorre em redes neurais muito profundas, onde o aumento do número de camadas pode levar a uma diminuição no desempenho.
            A inovação dos **blocos residuais** permite que algumas camadas "saltem" conexões, aprendendo uma **função de identidade** em vez de novas representações para cada camada. Essa abordagem facilita o treinamento de redes mais profundas, pois a função residual pode ser aprendida de forma mais eficiente (Zhang et al., 2018; Sandotra et al., 2023; Petrovska et al., 2020).
            """)
            
            st.latex(r'''
            \mathbf{y} = \mathcal{F}(x, \{W_i\}) + x
            ''')
            
            st.write("""
            onde 
            """)
            st.latex(r'''
            \mathcal{F}(x, \{W_i\}) + x
            ''')
          
            st.write("""
            representa a função aprendida e x é a entrada. O termo x é adicionado à saída, o que simplifica o processo de treinamento e permite que redes mais profundas sejam treinadas com maior eficácia 
            ("A Framework for Flood Extent Mapping using CNN Transfer Learning", 2022; Petrovska et al., 2020).
            """)
        
            st.write("""
            O modelo **ResNet18** possui 18 camadas treináveis e é uma versão mais leve, adequada para aplicações com restrições de recursos computacionais, enquanto o **ResNet50**, com 50 camadas, é capaz de capturar padrões mais complexos em imagens, sendo ideal para tarefas que exigem maior profundidade de análise (Sandotra et al., 2023; Qin et al., 2019; Petrovska et al., 2020).
            """)
        
            st.write("""
            Ambos os modelos foram pré-treinados no conjunto de dados **ImageNet**, o que facilita a **transferência de aprendizado** em novos domínios. As camadas iniciais desses modelos já são capazes de identificar características gerais, acelerando o processo de treinamento em conjuntos de dados menores e específicos, como em aplicações médicas ou de classificação de imagens geológicas (Cheng, 2023; Petrovska et al., 2020; Alaoui, 2023).
            """)
        
            st.write("### DenseNet121")
            st.write("""
            A arquitetura **DenseNet** (Rede Convolucional Densamente Conectada) oferece uma abordagem alternativa, onde todas as camadas estão interconectadas, promovendo a preservação do fluxo de gradiente e da informação original. Isso facilita a reutilização das representações intermediárias e otimiza a eficiência do modelo. A equação que expressa essa estrutura é:
            """)
        
            st.latex(r'''
            \mathbf{x}_l = H_l(\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1})
            ''')
        
            st.write("""
            onde
            """)
          
            st.latex(r'''
            \mathbf{x}_l 
            ''')
          
            st.write("""
            é a saída da l-ésima camada e 
            """)
          
            st.latex(r'''
             \mathbf{H}_l
            ''')
          
            st.write("""
            é a função aplicada. Essa configuração otimiza o uso de gradientes e representações, resultando em um desempenho superior em tarefas de classificação 
            (Benegui & Ionescu, 2020; Varshni et al., 2019; Hamdaoui et al., 2021).
            """)
        
            st.write("""
            O modelo **DenseNet121**, que possui 121 camadas treináveis, é particularmente eficaz em contextos onde a eficiência é crucial, maximizando o uso de recursos computacionais e facilitando a extração de características relevantes de imagens (Sardeshmukh, 2023; Hamdaoui et al., 2021).
            """)
        
            st.write("### Transferência de Aprendizado e Ajuste Fino")
            st.write("""
            A utilização de modelos pré-treinados, como ResNet18, ResNet50 e DenseNet121, é uma técnica de **transferência de aprendizado** que permite que o conhecimento adquirido em tarefas anteriores seja aplicado a novos problemas. 
            Em vez de treinar um modelo do zero, o ajuste fino é realizado nas camadas do modelo para se adaptar a um novo conjunto de dados, permitindo que características específicas sejam aprendidas de forma mais eficiente. Por exemplo, em aplicações de **classificação de melanomas** ou **análise de rochas vulcânicas**, as camadas mais profundas dos modelos são ajustadas para entender características específicas de imagens médicas ou geológicas (Suhana, 2022; Petrovska et al., 2020).
            """)
        
            st.write("""
            Estudos demonstram que a transferência de aprendizado é especialmente eficaz ao se trabalhar com conjuntos de dados pequenos. O uso de modelos pré-treinados pode proporcionar resultados semelhantes ou até superiores aos de modelos treinados a partir do zero, reduzindo o tempo de treinamento e melhorando a precisão (Raghava et al., 2019; Alaoui, 2023; Ahmed, 2021).
            """)
        
            st.write("### Conclusão")
            st.write("""
            As arquiteturas **ResNet18**, **ResNet50** e **DenseNet121** são ferramentas poderosas no campo do aprendizado profundo, especialmente em tarefas de classificação de imagens. Seu pré-treinamento em grandes conjuntos de dados, como o **ImageNet**, e a capacidade de serem ajustados para novas tarefas através da transferência de aprendizado, tornam esses modelos ideais para uma ampla gama de aplicações, incluindo a classificação de imagens médicas e geológicas. O uso dessas arquiteturas não apenas reduz o tempo de treinamento, mas também melhora a precisão e a eficácia em diversas áreas de pesquisa e aplicação prática (Zeimarani et al., 2020; "Dog Breed Identification with Fine Tuning of Pre-trained Models", 2019; Awais et al., 2020).
            """)
        
            st.write("### Referências")
        
            st.write("""
            - (2019). Dog breed identification with fine tuning of pre-trained models. *International Journal of Recent Technology and Engineering*, 8(2S11), 3677-3680. https://doi.org/10.35940/ijrte.b1464.0982s1119
            - (2022). A framework for flood extent mapping using cnn transfer learning. https://doi.org/10.17762/ijisae.v10i3s.2426
            - Ahmed, A. (2021). Pre-trained cnns models for content based image retrieval. *International Journal of Advanced Computer Science and Applications*, 12(7). https://doi.org/10.14569/ijacsa.2021.0120723
            - Alaoui, A. (2023). Pre-trained cnns: evaluating emergency vehicle image classification. *Data & Metadata*, 2, 153. https://doi.org/10.56294/dm2023153
            - Benegui, C. and Ionescu, R. (2020). Convolutional neural networks for user identification based on motion sensors represented as images. *IEEE Access*, 8, 61255-61266. https://doi.org/10.1109/access.2020.2984214
            - Cheng, R. (2023). Expansion of the ct-scans image set based on the pretrained dcgan for improving the performance of the cnn. *Journal of Physics Conference Series*, 2646(1), 012015. https://doi.org/10.1088/1742-6596/2646/1/012015
            - Hamdaoui, H., Ben-fares, A., Boujraf, S., Chaoui, N., Alami, B., Maâroufi, M., … & Qjidaa, H. (2021). High precision brain tumor classification model based on deep transfer learning and stacking concepts. *Indonesian Journal of Electrical Engineering and Computer Science*, 24(1), 167. https://doi.org/10.11591/ijeecs.v24.i1.pp167-177
            - Petrovska, B., Atanasova-Pacemska, T., Corizzo, R., Mignone, P., Lameski, P., & Zdravevski, E. (2020). Aerial scene classification through fine-tuning with adaptive learning rates and label smoothing. *Applied Sciences*, 10(17), 5792. https://doi.org/10.3390/app10175792
            - Raghava, Y., Kuthadi, V., & Rajalakshmi, S. (2019). Enhanced deep learning with featured transfer learning in identifying disguised faces. *International Journal of Innovative Technology and Exploring Engineering*, 8(10), 1257-1260. https://doi.org/10.35940/ijitee.h7286.0881019
            - Sandotra, N., Mahajan, P., Abrol, P., & Lehana, P. (2023). Analyzing performance of deep learning models under the presence of distortions in identifying plant leaf disease. *International Journal of Informatics and Communication Technology (IJ-ICT)*, 12(2), 115. https://doi.org/10.11591/ijict.v12i2.pp115-126
            - Sardeshmukh, M. (2023). Crop image classification using convolutional neural network. *Multidisciplinary Science Journal*, 5(4), 2023039. https://doi.org/10.31893/multiscience.2023039
            - Suhana, R. (2022). Fish image classification using adaptive learning rate in transfer learning method. *Knowledge Engineering and Data Science*, 5(1), 67. https://doi.org/10.17977/um018v5i12022p67-77
            - Varshni, D., Thakral, K., Agarwal, L., Nijhawan, R., & Mittal, A. (2019). Pneumonia detection using cnn based feature extraction. https://doi.org/10.1109/icecct.2019.8869364
            - Zeimarani, B., Costa, M., Nurani, N., Bianco, S., Pereira, W., & Filho, C. (2020). Breast lesion classification in ultrasound images using deep convolutional neural network. *IEEE Access*, 8, 133349-133359. https://doi.org/10.1109/access.2020.3010863
            - Zhang, B., Wang, C., Shen, Y., & Liu, Y. (2018). Fully connected conditional random fields for high-resolution remote sensing land use/land cover classification with convolutional neural networks. *Remote Sensing*, 10(12), 1889. https://doi.org/10.3390/rs10121889
            """)

    # Seleção de Tipo de Arquitetura
    st.sidebar.write("---")
    st.sidebar.subheader("🏗️ Arquitetura do Modelo")
    
    architecture_type = st.sidebar.radio(
        "Tipo de Arquitetura:",
        options=["CNN (Convolucional)", "Transformer (ViT)"],
        help="CNN: Redes Neurais Convolucionais tradicionais | Transformer: Vision Transformers modernos"
    )
    
    if architecture_type == "CNN (Convolucional)":
        model_options = ['ResNet18', 'ResNet50', 'DenseNet121']
        st.sidebar.info("🔷 **CNNs** são excelentes para capturar padrões locais e hierárquicos em imagens através de convoluções.")
    else:
        model_options = ['ViT-B/16', 'ViT-B/32', 'ViT-L/16']
        if TIMM_AVAILABLE:
            model_options.extend(['ViT-B/16-timm', 'ViT-L/16-timm', 'Swin-T', 'Swin-B'])
        st.sidebar.info("🔶 **Vision Transformers** usam mecanismos de atenção para capturar relações globais na imagem. Modelos timm e Swin são mais robustos!")
        st.sidebar.warning("⚠️ ViT requer mais memória GPU. Use batch size menor se necessário.")
    
    model_name = st.sidebar.selectbox("Modelo Pré-treinado:", options=model_options)
    
    # Explicação sobre o modelo selecionado
    with st.sidebar.expander(f"ℹ️ Sobre {model_name}"):
        if model_name == 'ResNet18':
            st.write("**ResNet18:** 18 camadas, ~11M parâmetros. Rápido e eficiente para datasets menores.")
        elif model_name == 'ResNet50':
            st.write("**ResNet50:** 50 camadas, ~25M parâmetros. Melhor precisão, requer mais recursos.")
        elif model_name == 'DenseNet121':
            st.write("**DenseNet121:** Conexões densas entre camadas, ~8M parâmetros. Eficiente e preciso.")
        elif model_name == 'ViT-B/16':
            st.write("**ViT-B/16:** Base model, patches 16x16, ~86M parâmetros. Melhor performance geral.")
        elif model_name == 'ViT-B/32':
            st.write("**ViT-B/32:** Base model, patches 32x32, ~88M parâmetros. Mais rápido, menos preciso.")
        elif model_name == 'ViT-L/16':
            st.write("**ViT-L/16:** Large model, patches 16x16, ~307M parâmetros. Máxima precisão, requer muitos recursos.")
        elif model_name == 'ViT-B/16-timm':
            st.write("**ViT-B/16-timm:** Versão timm do ViT Base. Mais robusto e melhor treinado que a versão torchvision.")
        elif model_name == 'ViT-L/16-timm':
            st.write("**ViT-L/16-timm:** Versão timm do ViT Large. Melhor desempenho e robustez.")
        elif model_name == 'Swin-T':
            st.write("**Swin-T:** Swin Transformer Tiny, ~28M parâmetros. Arquitetura hierárquica eficiente. Melhor que ViT para muitos casos!")
        elif model_name == 'Swin-B':
            st.write("**Swin-B:** Swin Transformer Base, ~88M parâmetros. Excelente performance, arquitetura state-of-the-art.")

    #________________________________________________________________________________________
    # Fine-Tuning Completo em Redes Neurais Profundas
    with st.sidebar:
        with st.expander("Fine-Tuning Completo em Redes Neurais Profundas:"):
            st.write("""
            ### Introdução
        
            O **fine-tuning** (ajuste fino) é uma técnica poderosa utilizada para ajustar redes neurais pré-treinadas em novos conjuntos de dados. No contexto de redes como a **ResNet18**, **ResNet50** ou **DenseNet121**, que foram inicialmente treinadas em grandes bases de dados (como o **ImageNet**), o fine-tuning permite que essas redes sejam adaptadas a novos problemas, como a **classificação de melanomas** ou de **rochas vulcânicas e plutônicas**. Ao realizar o fine-tuning, todas as camadas do modelo são atualizadas para refletir as características do novo conjunto de dados, ao invés de congelar as camadas iniciais, o que permite uma adaptação mais profunda e precisa ao novo problema (Piotrowski & Napiorkowski, 2013; Friedrich et al., 2022).
            """)
        
            st.write("""
            ### Fundamentação Teórica
        
            O conceito de fine-tuning é baseado no princípio de **transferência de aprendizado**, no qual um modelo pré-treinado em um grande conjunto de dados genéricos é reaproveitado para um novo problema específico. Essa abordagem é particularmente útil quando o novo conjunto de dados é relativamente pequeno, pois o modelo já foi treinado para capturar padrões gerais em dados visuais (como bordas, texturas e formas), o que pode acelerar o treinamento e melhorar a precisão final (Al‐rimy et al., 2023; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            Ao utilizar o fine-tuning completo, todas as camadas do modelo são ajustadas com base nos novos dados. Isso significa que os pesos das camadas profundas do modelo, que foram aprendidos durante o treinamento inicial, são atualizados para se adequar às características específicas do novo conjunto de dados. Matematicamente, essa abordagem pode ser descrita como a otimização da seguinte função de perda:
            """)
        
            st.latex(r'''
            L_{\text{fine-tuning}} = L_{\text{original}} + \lambda \sum_{i} w_i^2
            ''')
        
            st.write("""
            Onde:
            """)
          
            st.latex(r'''
            L_{\text{fine-tuning}}
            ''')
          
            st.write("""
            é a função de perda durante o fine-tuning;
            """)
          
            st.latex(r'''
            L_{\text{original}}
            ''')
          
            st.write("""
            representa a função de perda original do modelo pré-treinado;
            """)
          
            st.latex(r'''
            \lambda
            ''')
          
            st.write("""
            é o coeficiente de regularização (no caso de utilizar a regularização L2);
            """)
          
            st.latex(r'''
            w_i
            ''')
            st.write("""
            são os pesos individuais que serão atualizados durante o processo de fine-tuning (Friedrich et al., 2022; Al‐rimy et al., 2023).
            """)
        
            st.write("""
            ### Benefícios do Fine-Tuning Completo
        
            O fine-tuning completo oferece vários benefícios, especialmente quando o novo conjunto de dados difere substancialmente do conjunto no qual o modelo foi originalmente treinado. No caso da **classificação de melanomas** ou **rochas**, por exemplo, as características visuais dos dados podem ser muito diferentes das imagens do **ImageNet**, que incluem uma ampla variedade de objetos, animais e cenários (Piotrowski & Napiorkowski, 2013; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            Os principais benefícios incluem:
            1. **Adaptação Profunda**: Ao ajustar todas as camadas, o modelo consegue adaptar não apenas as características genéricas (como bordas e texturas), mas também padrões mais complexos e específicos do novo problema.
            2. **Melhoria da Precisão**: O fine-tuning completo geralmente resulta em melhorias significativas na precisão, especialmente quando os dados de treinamento são limitados ou possuem características visuais únicas (Friedrich et al., 2022; Al‐rimy et al., 2023).
            3. **Generalização Melhorada**: O processo de fine-tuning permite que o modelo generalize melhor para novos dados, uma vez que ele é treinado para capturar padrões mais específicos do novo domínio (Piotrowski & Napiorkowski, 2013; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            ### Comparação com o Fine-Tuning Parcial
        
            Em contraste com o fine-tuning completo, no qual todas as camadas são atualizadas, o **fine-tuning parcial** mantém algumas das camadas iniciais congeladas, atualizando apenas as camadas finais. Essa abordagem pode ser útil quando o novo conjunto de dados é semelhante ao conjunto de dados original no qual o modelo foi treinado. No entanto, quando os dados diferem substancialmente, o fine-tuning completo tende a ser mais eficaz, pois permite uma adaptação mais profunda e personalizada (Al‐rimy et al., 2023; Sakizadeh et al., 2015).
            """)
        
            st.write("""
            ### Efeitos do Fine-Tuning em Problemas Específicos
        
            #### Classificação de Melanomas
        
            No caso da **classificação de melanomas**, o fine-tuning completo permite que o modelo identifique padrões visuais sutis na pele que podem ser indicativos de câncer. Essas características visuais podem incluir variações de textura, cor e bordas, que são específicas de imagens médicas e diferem dos objetos comuns presentes em bases de dados genéricas, como o **ImageNet** (Piotrowski & Napiorkowski, 2013; Friedrich et al., 2022).
            """)
        
            st.write("""
            #### Classificação de Rochas
        
            Para a **classificação de rochas vulcânicas e plutônicas**, o fine-tuning completo permite que o modelo capture padrões geológicos e estruturais específicos, como variações de granulação e texturas minerais. Novamente, esses padrões são significativamente diferentes dos dados de objetos comuns, tornando o fine-tuning completo uma abordagem valiosa para melhorar a precisão da classificação (Friedrich et al., 2022; Al‐rimy et al., 2023).
            """)
        
            st.write("""
            ### Considerações Práticas
        
            Durante o processo de fine-tuning, é importante monitorar o desempenho do modelo em um conjunto de validação para evitar o **overfitting**. Uma técnica comum é utilizar a **regularização L2** ou o **dropout** para garantir que o modelo não se ajuste excessivamente aos dados de treinamento (Piotrowski & Napiorkowski, 2013; Sakizadeh et al., 2015). Além disso, a taxa de aprendizado deve ser cuidadosamente ajustada. Em muitos casos, utiliza-se uma taxa de aprendizado menor durante o fine-tuning para garantir que as atualizações dos pesos não sejam muito drásticas, preservando parte das informações aprendidas anteriormente.
            """)
        
            st.write("""
            ### Conclusão
        
            O fine-tuning completo é uma técnica eficaz para ajustar modelos pré-treinados, como a **ResNet18**, **ResNet50** ou **DenseNet121**, a novos conjuntos de dados. Ao permitir que todas as camadas do modelo sejam atualizadas, o fine-tuning completo oferece maior flexibilidade e precisão em problemas que diferem substancialmente dos dados originais. Quando combinado com outras técnicas de regularização, como a L2, o fine-tuning pode levar a modelos robustos e capazes de generalizar para novos dados, sendo uma ferramenta essencial no arsenal de técnicas de aprendizado profundo.
            """)
        
            st.write("""
            ### Referências
        
            - Al‐RIMY, B.; SAEED, F.; AL-SAREM, M.; ALBARRAK, A.; QASEM, S. An adaptive early stopping technique for densenet169-based knee osteoarthritis detection model. *Diagnostics*, 13(11), 1903, 2023. https://doi.org/10.3390/diagnostics13111903
            - FRIEDRICH, S. et al. Regularization approaches in clinical biostatistics: a review of methods and their applications. *Statistical Methods in Medical Research*, 32(2), 425-440, 2022. https://doi.org/10.1177/09622802221133557
            - PIOTROWSKI, A.; NAPIORKOWSKI, J. A comparison of methods to avoid overfitting in neural networks training in the case of catchment runoff modelling. *Journal of Hydrology*, 476, 97-111, 2013. https://doi.org/10.1016/j.jhydrol.2012.10.019
            - REZAEEZADE, A.; BATINA, L. Regularizers to the rescue: fighting overfitting in deeplearning-based side-channel analysis. 2022. https://doi.org/10.21203/rs.3.rs-2386625/v1
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, 54(1), 115-120, 2015. https://doi.org/10.1111/gwat.12317
            """)

    fine_tune = st.sidebar.checkbox("Fine-Tuning Completo", value=False)
    epochs = st.sidebar.slider("Número de Épocas:", min_value=1, max_value=500, value=200, step=1)
    learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem:", options=[0.1, 0.01, 0.001, 0.0001], value=0.0001)
    batch_size = st.sidebar.selectbox("Tamanho de Lote:", options=[4, 8, 16, 32, 64], index=2)
    train_split = st.sidebar.slider("Percentual de Treinamento:", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    valid_split = st.sidebar.slider("Percentual de Validação:", min_value=0.05, max_value=0.4, value=0.15, step=0.05)
    #________________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmula LaTeX
    with st.sidebar:
        with st.expander("Implementação da Técnica de Regularização L2 (Weight Decay):"):
            st.write("""
            ### Introdução
            A regularização L2, frequentemente referida como *weight decay*, é uma técnica amplamente utilizada para mitigar o **overfitting** 
            em modelos de aprendizado de máquina, especialmente em redes neurais profundas. O *overfitting* ocorre quando o modelo se ajusta não apenas 
            aos padrões dos dados de treinamento, mas também ao ruído presente, o que compromete sua capacidade de generalização para novos dados 
            (Piotrowski & Napiorkowski, 2013). A regularização L2 adiciona um termo de penalização à função de perda do modelo, o que resulta em uma 
            redução dos valores absolutos dos pesos, promovendo, assim, modelos mais simples e generalizáveis (Friedrich et al., 2022).
            Esta revisão visa fornecer uma visão clara e técnica da aplicação da regularização L2, discutindo seus efeitos, a interpretação do coeficiente de regularização 
            """)
          
            st.latex(r'''
            \lambda
            ''')
          
            st.write("""
            e as implicações da escolha desse parâmetro.
            """)
          
            st.latex(r'''
            L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} w_i^2
            ''')
          
            st.write("""
            Onde:
            """) 
            
            st.latex(r'''
            L_{\text{total}}
            ''')
          
            st.write("""
            é a perda total que o modelo busca minimizar;
            """)
            
            st.latex(r'''
            L_{\text{original}}
            ''')
          
            st.write("""
            é a função de perda original (como a perda de entropia cruzada); λ é o coeficiente de regularização, que controla a penalidade aplicada aos pesos;
            """)
          
            st.latex(r'''
            w_i
            ''')
          
            st.write(""" 
            são os pesos individuais do modelo (Al‐Rimy et al., 2023).
            """)
          
            st.write("""
            Este termo adicional penaliza pesos grandes, forçando o modelo a priorizar soluções que utilizam pesos menores, o que é crucial para evitar 
            que o modelo memorize os dados de treinamento, promovendo maior capacidade de generalização (Sakizadeh et al., 2015).
            """)
          
            st.write("""
            ### Fundamentação Teórica
            A regularização L2 tem uma base teórica sólida, sendo amplamente aplicada para controlar a complexidade do modelo. Ao adicionar o termo de penalização, 
            a regularização L2 ajuda a evitar o overfitting e melhora a estabilidade numérica do modelo (Friedrich et al., 2022). Isso é particularmente importante 
            em redes neurais profundas, onde o número de parâmetros pode ser grande e a complexidade do modelo alta.
            """)
          
            st.write("""
            ### Efeitos da Regularização L2
            A regularização L2 controla a complexidade do modelo ao penalizar pesos grandes, o que é particularmente útil em cenários com muitos parâmetros 
            ou dados ruidosos (Piotrowski & Napiorkowski, 2013). Além de reduzir o overfitting, a L2 promove a estabilidade no treinamento, melhorando a consistência do desempenho 
            em dados de teste (Friedrich et al., 2022).
            """)
    
            st.write("""
            ### Interpretação e Efeitos Práticos de λ
            """)
          
            st.write("""        
            A escolha do valor de λ
            """)
      
            st.write("""
            influencia diretamente o comportamento do modelo:
            """)
    
            st.write("""
            #### λ = 0
            """)
            st.write("""
            Quando λ = 0, a regularização L2 está desativada. Isso permite que o modelo ajuste-se livremente aos dados de treinamento, 
            aumentando o risco de overfitting, especialmente em conjuntos de dados pequenos ou ruidosos (Friedrich et al., 2022).
            """)
    
            st.write("""
            #### λ = 0,01
            """)
            st.write("""
            Este é um valor moderado, que penaliza de forma equilibrada os pesos do modelo. Essa configuração ajuda a evitar o overfitting sem comprometer a capacidade do modelo de 
            aprender padrões relevantes (Al‐Rimy et al., 2023).
            """)
    
            st.write("""
            #### λ = 0,02 ou λ = 0,03
            Esses valores aumentam a intensidade da penalização, sendo úteis em cenários com dados ruidosos ou em que o número de parâmetros é alto em relação à quantidade de dados 
            disponíveis (Piotrowski & Napiorkowski, 2013). Contudo, deve-se monitorar o desempenho do modelo, pois valores elevados de λ podem resultar em **underfitting**, 
            comprometendo a capacidade do modelo de capturar padrões complexos (Friedrich et al., 2022).
            """)
    
            st.write("""
            ### Conclusão
            A regularização L2 é uma técnica poderosa no treinamento de redes neurais profundas, ajudando a mitigar o overfitting e a melhorar a capacidade de generalização do modelo. 
            Ao penalizar pesos grandes, a L2 incentiva soluções mais simples e robustas. No entanto, a escolha do valor de λ é crucial para garantir que o modelo consiga capturar 
            padrões complexos sem se ajustar excessivamente aos dados de treinamento.
            """)
    
            st.write("""
            ### Referências
            - AL‐RIMY, B.; SAEED, F.; AL-SAREM, M.; ALBARRAK, A.; QASEM, S. An adaptive early stopping technique for densenet169-based knee osteoarthritis detection model. *Diagnostics*, 13(11), 1903, 2023. https://doi.org/10.3390/diagnostics13111903
            - FRIEDRICH, S. et al. Regularization approaches in clinical biostatistics: a review of methods and their applications. *Statistical Methods in Medical Research*, 32(2), 425-440, 2022. https://doi.org/10.1177/09622802221133557
            - PIOTROWSKI, A.; NAPIORKOWSKI, J. A comparison of methods to avoid overfitting in neural networks training in the case of catchment runoff modelling. *Journal of Hydrology*, 476, 97-111, 2013. https://doi.org/10.1016/j.jhydrol.2012.10.019
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, 54(1), 115-120, 2015. https://doi.org/10.1111/gwat.12317
            """)
    

  
    l2_lambda = st.sidebar.number_input("L2 Regularization (Weight Decay):", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    l1_lambda = st.sidebar.number_input("L1 Regularization:", min_value=0.0, max_value=0.01, value=0.0, step=0.001, 
                                        help="Adiciona regularização L1 (Lasso) ao treinamento. Promove esparsidade nos pesos.")
    
    #________________________________________________________________________________________
    # Novos parâmetros de treinamento
    st.sidebar.write("---")
    st.sidebar.subheader("⚙️ Configurações Avançadas")
    
    # Tipo de Aumento de Dados
    augmentation_type = st.sidebar.selectbox(
        "Técnica de Aumento de Dados:",
        options=['none', 'standard', 'mixup', 'cutmix'],
        index=1,
        help="None: Sem aumento | Standard: Transformações básicas | Mixup: Mistura imagens | Cutmix: Recorta e cola regiões"
    )
    
    # Otimizador
    optimizer_options = ['Adam', 'AdamW', 'SGD']
    if ADVANCED_OPTIMIZERS_AVAILABLE:
        optimizer_options.extend(['Ranger', 'Lion'])
    
    optimizer_name = st.sidebar.selectbox(
        "Otimizador:",
        options=optimizer_options,
        index=0,
        help="Adam: Adaptativo padrão | AdamW: Adam com weight decay melhorado | SGD: Gradiente descendente com momento | Ranger: Lookahead + RAdam | Lion: Otimizador eficiente recente"
    )
    
    # Learning Rate Scheduler
    scheduler_name = st.sidebar.selectbox(
        "Agendador de Learning Rate:",
        options=['None', 'CosineAnnealingLR', 'OneCycleLR'],
        index=0,
        help="None: LR constante | CosineAnnealingLR: Reduz LR com coseno | OneCycleLR: Aumenta e depois reduz LR"
    )
    
    # Label Smoothing
    label_smoothing = st.sidebar.number_input(
        "Label Smoothing:", 
        min_value=0.0, 
        max_value=0.3, 
        value=0.1, 
        step=0.05,
        help="Suaviza os rótulos para prevenir overconfidence. Valores típicos: 0.1-0.2. Melhora generalização especialmente em ViT."
    )
    
    # Gradient Clipping
    use_gradient_clipping = st.sidebar.checkbox(
        "Usar Gradient Clipping",
        value=True,
        help="Limita a norma dos gradientes para estabilizar o treinamento. ESSENCIAL para Vision Transformers."
    )
    
    # Exponential Moving Average
    use_ema = st.sidebar.checkbox(
        "Usar EMA (Exponential Moving Average)",
        value=True,
        help="Mantém média móvel dos pesos. Melhora generalização e estabilidade. Recomendado para todos os modelos."
    )
    
    # Reinforcement Learning
    use_rl = st.sidebar.checkbox(
        "Usar Reinforcement Learning",
        value=False,
        help="Ajusta dinamicamente learning rate usando Q-Learning baseado no desempenho. EXPERIMENTAL mas pode melhorar resultados!"
    )
    
    # CrewAI Research Agent
    use_crewai = st.sidebar.checkbox(
        "Usar Agente CrewAI (Pesquisa Web)",
        value=False,
        help="Usa agentes inteligentes para buscar estratégias de treinamento na web. Requer API keys. EXPERIMENTAL."
    )
    
    # Tipo de Grad-CAM
    gradcam_type = st.sidebar.selectbox(
        "Tipo de Grad-CAM:",
        options=['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'LayerCAM'],
        index=2,
        help="GradCAM: Básico | GradCAMpp: Melhorado | SmoothGradCAMpp: Suavizado | LayerCAM: Por camada"
    )
    
    st.sidebar.write("---")
    
    #________________________________________________________________________________________
    # Sidebar com o conteúdo explicativo e fórmula LaTeX
    with st.sidebar:
        with st.expander("Implementação da Técnica de Parada Precoce - Early Stopping:"):
            st.write("""
            #### Introdução
            A técnica de **parada precoce** (ou *early stopping*) é amplamente utilizada para mitigar o **overfitting** no treinamento de redes neurais profundas. 
            O overfitting ocorre quando o modelo se ajusta tão bem aos dados de treinamento que sua capacidade de generalização para novos dados é prejudicada. 
            O princípio da parada precoce é interromper o treinamento quando o desempenho do modelo em um conjunto de validação não apresenta melhorias significativas 
            após um número predefinido de épocas. Essa abordagem baseia-se na observação de que, após certo ponto, melhorias no desempenho do modelo em dados de treinamento 
            não resultam em melhorias em dados que o modelo ainda não viu (Piotrowski & Napiorkowski, 2013; Al‐Rimy et al., 2023).
            """)
      
            st.write("Matematicamente, a parada precoce pode ser descrita pela seguinte condição de interrupção:")
            # Fórmulas matemáticas
            st.latex(r'''
            \text{Se } L_{\text{val}}(t) \geq L_{\text{val}}(t-1)
            ''')
            st.write("""
            por (p) épocas consecutivas, então interrompa o treinamento. Aqui,
            """)
            st.latex(r'''
            L_{\text{val}}(t)
            ''')
    
            st.write("""
            representa o valor da **função de perda** no conjunto de validação na época (t), e (p) é o **parâmetro de paciência**. 
            A paciência (p) define quanto tempo o treinamento deve continuar mesmo que não haja melhorias imediatas. Se a perda não melhorar por (p) épocas consecutivas, 
            o treinamento é interrompido.
            """)
      
            st.write("""
            #### A Importância da Paciência
            O parâmetro de **paciência** define o número de épocas consecutivas sem melhoria na métrica de validação que o modelo pode suportar antes de o treinamento ser interrompido. 
            A escolha do valor de paciência tem impacto direto no equilíbrio entre **evitar o overfitting** e **permitir que o modelo continue aprendendo**. 
            """)
      
            st.write("##### Paciência = 0")
            st.write("""
            Um valor de paciência igual a zero implica que o treinamento será interrompido imediatamente após a primeira ocorrência de estagnação na métrica de validação. 
            Isso pode ser útil em cenários onde se deseja evitar qualquer risco de *overfitting*.
            """)
      
            st.write("##### Paciência ≥ 1")
            st.write("""
            Uma paciência maior (como 1 ou 2) permite que o modelo continue sendo treinado mesmo após pequenas flutuações no desempenho, 
            o que pode ser benéfico em conjuntos de dados ruidosos (Sakizadeh et al., 2015).
            """)
      
            st.write("""
            #### Impacto do *Early Stopping* e da Paciência
            A configuração do parâmetro de paciência influencia diretamente a eficiência do treinamento. Com uma paciência muito baixa, o treinamento pode ser interrompido de forma prematura, 
            mesmo que o modelo ainda tenha potencial de melhoria. Por outro lado, uma paciência muito alta pode permitir que o modelo se ajuste excessivamente aos dados de treinamento, 
            levando ao *overfitting* (Sakizadeh et al., 2015).
            """)
      
            st.write("""
            #### Exemplos de Aplicação
            Um exemplo prático de uso da parada precoce é em tarefas de **classificação de imagens**. Durante o treinamento de um modelo para detecção de melanoma, se a acurácia no conjunto de validação 
            não melhorar após um determinado número de épocas, o early stopping é acionado.
            """)
      
            st.write("""
            #### Integração com Outras Técnicas de Regularização
            A parada precoce pode ser usada em conjunto com outras técnicas de regularização, como a **injeção de ruído** e a regularização **L1/L2**, 
            para melhorar a robustez do modelo e sua capacidade de generalização (Friedrich et al., 2022). 
            A combinação dessas técnicas ajuda a evitar que o modelo se ajuste excessivamente aos dados de treinamento, principalmente em cenários com volumes limitados de dados.
            """)
      
            st.write("""
            #### Conclusão
            A **parada precoce** é uma técnica eficaz para evitar o *overfitting* no treinamento de redes neurais profundas. O valor da paciência desempenha um papel crítico, 
            permitindo o equilíbrio entre **eficiência computacional** e **capacidade de aprendizado**. Além disso, a combinação da parada precoce com outras técnicas de regularização 
            pode melhorar ainda mais o desempenho do modelo.
            """)
      
            st.write("""
            #### Referências
            - PIOTROWSKI, A.; NAPIORKOWSKI, J. A comparison of methods to avoid overfitting in neural networks training in the case of catchment runoff modelling. *Journal of Hydrology*, v. 476, p. 97-111, 2013. https://doi.org/10.1016/j.jhydrol.2012.10.019.
            - AL‐RIMY, B. et al. An adaptive early stopping technique for densenet169-based knee osteoarthritis detection model. *Diagnostics*, v. 13, n. 11, p. 1903, 2023. https://doi.org/10.3390/diagnostics13111903.
            - SAKIZADEH, M.; MALIAN, A.; AHMADPOUR, E. Groundwater quality modeling with a small data set. *Ground Water*, v. 54, n. 1, p. 115-120, 2015. https://doi.org/10.1111/gwat.12317.
            - FRIEDRICH, S. et al. Regularization approaches in clinical biostatistics: a review of methods and their applications. *Statistical Methods in Medical Research*, v. 32, n. 2, p. 425-440, 2022. https://doi.org/10.1177/09622802221133557.
            """)


    #________________________________________________________________________________________
    patience = st.sidebar.number_input("Paciência para Early Stopping:", min_value=1, max_value=10, value=3, step=1)

    #____________________________________________________________________________________________
    with st.sidebar:
        with st.expander("Perda Ponderada para Classes Desbalanceadas:"):
            st.write("""
            ### Perda Ponderada para Classes Desbalanceadas
        
            A técnica de **perda ponderada** para lidar com **classes desbalanceadas** é amplamente utilizada em **aprendizado de máquina**, especialmente em redes neurais, para tratar problemas onde o número de amostras entre as classes de um conjunto de dados não é equilibrado. O desbalanceamento ocorre em diversos domínios, como detecção de fraudes, diagnóstico de doenças e classificação de imagens. O principal objetivo da perda ponderada é ajustar a função de perda, atribuindo diferentes pesos às classes, de forma que o impacto das classes minoritárias (menos representadas) seja ampliado e o impacto das classes majoritárias seja reduzido. Isso ajuda o modelo a aprender de forma mais eficaz em cenários onde o desequilíbrio entre as classes pode levar ao **overfitting** nas classes majoritárias e à **sub-representação** das classes minoritárias (Buda et al., 2018).
        
            ### Motivação e Justificativa Científica
        
            Em um cenário de classificação de imagens, se o modelo for treinado com uma quantidade muito maior de amostras de uma classe (classe majoritária) em relação a outra (classe minoritária), o modelo tende a ser enviesado para a classe majoritária. Isso ocorre porque o objetivo padrão da maioria das funções de perda, como a **entropia cruzada**, é minimizar a soma dos erros. Em um conjunto de dados desbalanceado, essa minimização pode ser alcançada simplesmente classificando todas as amostras como pertencentes à classe majoritária, resultando em alta acurácia geral, mas com desempenho ruim na classe minoritária. Para resolver esse problema, atribui-se um peso maior à classe minoritária, forçando a função de perda a penalizar mais fortemente os erros cometidos nessa classe (Buda et al., 2018).
        
            ### Implementação no Código
        
            No código, a implementação da perda ponderada é feita utilizando a função de perda **CrossEntropyLoss** do PyTorch, que suporta a aplicação de pesos às classes. Esses pesos são calculados com base na **frequência das classes** no conjunto de treinamento. Classes com menos amostras recebem pesos maiores, enquanto classes com mais amostras recebem pesos menores, balanceando o impacto de ambas durante o treinamento do modelo.
        
            """)
            
            st.write("**criterion = nn.CrossEntropyLoss(weight=class_weights)**")
            
            st.write("""
            No trecho de código acima, o vetor `targets` coleta os rótulos das amostras no conjunto de treino e a função `np.bincount(targets)` conta quantas vezes cada classe aparece, resultando em um vetor `class_counts`, onde cada índice corresponde à quantidade de amostras de uma classe específica (Buda et al., 2018).
        
            ### Etapas do Processo
        
            1. **Cálculo das Frequências das Classes**: As frequências de cada classe são calculadas usando `np.bincount`. Classes menos representadas recebem pesos maiores.
            2. **Ajuste para Evitar Divisão por Zero**: Um pequeno valor (1e-6) é adicionado para evitar divisão por zero quando uma classe não tem nenhuma amostra.
            3. **Cálculo dos Pesos Inversos**: A partir da frequência, os pesos são calculados tomando o inverso da frequência de cada classe. Isso aumenta a penalização dos erros nas classes minoritárias.
            4. **Função de Perda Ponderada**: A função de perda `nn.CrossEntropyLoss(weight=class_weights)` usa os pesos calculados, penalizando mais fortemente os erros das classes minoritárias.
        
            ### Impacto e Eficácia da Perda Ponderada
        
            A **perda ponderada** ajusta o aprendizado do modelo, incentivando a penalização dos erros cometidos nas classes minoritárias. Estudos demonstram que essa técnica é eficaz em aumentar a **recall** das classes minoritárias, sem comprometer drasticamente a precisão das classes majoritárias (Buda et al., 2018). No entanto, a aplicação da perda ponderada pode tornar o treinamento mais **sensível à escolha dos hiperparâmetros**, como a **taxa de aprendizado**, pois o modelo passa a ser fortemente influenciado pelas amostras menos representativas.
        
            ### Conclusão
        
            A implementação da **perda ponderada** no código é uma abordagem robusta para lidar com **classes desbalanceadas**. Ao ajustar os pesos da função de perda com base nas frequências das classes, o modelo consegue equilibrar melhor o aprendizado entre as classes majoritárias e minoritárias, evitando vieses que favorecem a classe mais representada no conjunto de dados (Buda et al., 2018).
        
            ### Referências
        
            - Buda, M., Maki, A., & Mazurowski, M. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249-259. https://doi.org/10.1016/j.neunet.2018.07.011
            """)

    use_weighted_loss = st.sidebar.checkbox("Usar Perda Ponderada para Classes Desbalanceadas", value=False)
    
    #________________________________________________________________________________________
    # API Configuration Section for AI Analysis
    st.sidebar.write("---")
    st.sidebar.subheader("🔑 Configuração de API para Análise IA")
    
    with st.sidebar.expander("Configurar API (Gemini/Groq)", expanded=False):
        st.write("Configure sua API para análise diagnóstica com IA")
        
        api_provider_sidebar = st.selectbox(
            "Provedor de API:",
            options=['Nenhum', 'Gemini', 'Groq'],
            key='api_provider_sidebar',
            help="Escolha entre Google Gemini ou Groq para análise com IA"
        )
        
        if api_provider_sidebar != 'Nenhum':
            if api_provider_sidebar == 'Gemini':
                model_options_sidebar = [
                    'gemini-2.5-flash',  # ⭐ Recommended
                    'gemini-2.5-flash-lite',
                    'gemini-2.5-pro',
                    'gemini-3-flash-preview',
                    'gemini-3-pro-preview',
                    # Legacy models (not recommended)
                    'gemini-1.5-pro-latest',
                    'gemini-1.5-flash-latest'
                ]
            else:  # Groq
                model_options_sidebar = [
                    'meta-llama/llama-4-scout-17b-16e-instruct',
                    'meta-llama/llama-4-maverick-17b-128e-instruct',
                    'mixtral-8x7b-32768',
                    'llama-3.1-70b-versatile',
                    'llama-3.1-8b-instant'
                ]
            
            ai_model_sidebar = st.selectbox(
                "Modelo:",
                options=model_options_sidebar,
                key='ai_model_sidebar',
                help="Escolha o modelo de IA para análise"
            )
            
            api_key_sidebar = st.text_input(
                "API Key:",
                type="password",
                key='api_key_sidebar',
                help="Insira sua chave API (será usada durante a avaliação de imagens)"
            )
            
            if api_key_sidebar:
                st.success("✅ API Key configurada!")
                st.session_state['api_configured'] = True
                st.session_state['api_provider'] = api_provider_sidebar
                st.session_state['api_model'] = ai_model_sidebar
                st.session_state['api_key'] = api_key_sidebar
            else:
                st.session_state['api_configured'] = False
        else:
            st.session_state['api_configured'] = False
            api_key_sidebar = None
            ai_model_sidebar = None
    
    st.sidebar.image("eu.ico", width=80)
   
    st.sidebar.write("""
    Produzido pelo:
    
    Projeto Geomaker + IA 
    
    https://doi.org/10.5281/zenodo.13910277
    
    - Professor: Marcelo Claro.
    
    Contatos: marceloclaro@gmail.com
    
    Whatsapp: (88)981587145
    
    Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    
    """)
     # _____________________________________________
    # Controle de Áudio
    st.sidebar.title("Controle de Áudio")
    
    # Dicionário de arquivos de áudio, com nomes amigáveis mapeando para o caminho do arquivo
    mp3_files = {
        "Áudio explicativo para Leigos": "leigo.mp3",
        "Áudio explicativo para treinamentos de poucos dados": "bucal.mp3",
    }
    
    # Lista de arquivos MP3 para seleção
    selected_mp3 = st.sidebar.radio("Escolha um áudio explicativo:", options=list(mp3_files.keys()))
    
    # Controle de opção de repetição
    loop = st.sidebar.checkbox("Repetir áudio")
    
    # Botão de Play para iniciar o áudio
    play_button = st.sidebar.button("Play")
    
    # Placeholder para o player de áudio
    audio_placeholder = st.sidebar.empty()
    
    # Função para verificar se o arquivo existe
    def check_file_exists(mp3_path):
        if not os.path.exists(mp3_path):
            st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
            return False
        return True
    
    # Se o botão Play for pressionado e um arquivo de áudio estiver selecionado
    if play_button and selected_mp3:
        mp3_path = mp3_files[selected_mp3]
        
        # Verificação da existência do arquivo
        if check_file_exists(mp3_path):
            try:
                # Abrindo o arquivo de áudio no modo binário
                with open(mp3_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    
                    # Codificando o arquivo em base64 para embutir no HTML
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    # Controle de loop (repetição)
                    loop_attr = "loop" if loop else ""
                    
                    # Gerando o player de áudio em HTML
                    audio_html = f"""
                    <audio id="audio-player" controls autoplay {loop_attr}>
                      <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                      Seu navegador não suporta o elemento de áudio.
                    </audio>
                    """
                    
                    # Inserindo o player de áudio na interface
                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
            
            except FileNotFoundError:
                st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
            except Exception as e:
                st.sidebar.error(f"Erro ao carregar o arquivo: {str(e)}")
    #______________________________________________________________________________________-


    # Verificar se a soma dos splits é válida
    if train_split + valid_split > 0.95:
        st.sidebar.error("A soma dos splits de treinamento e validação deve ser menor ou igual a 0.95.")

    # Upload do arquivo ZIP
    
    zip_file = st.file_uploader("Upload do arquivo ZIP com as imagens", type=["zip"])

    if zip_file is not None and train_split + valid_split <= 0.95:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data_dir = temp_dir

        # Detectar automaticamente o número de classes do dataset
        try:
            temp_dataset = datasets.ImageFolder(root=data_dir)
            detected_num_classes = len(temp_dataset.classes)
            st.success(f"✅ Número de classes detectado automaticamente: **{detected_num_classes}**")
            st.write(f"Classes encontradas: {', '.join(temp_dataset.classes)}")
            num_classes = detected_num_classes
        except Exception as e:
            st.error(f"Erro ao detectar classes: {e}")
            st.error("Certifique-se de que o ZIP contém pastas com nomes de classes e imagens dentro delas.")
            shutil.rmtree(temp_dir)
            return

        st.write("Iniciando o treinamento supervisionado...")
        model_data = train_model(data_dir, num_classes, model_name, fine_tune, epochs, learning_rate, 
                                batch_size, train_split, valid_split, use_weighted_loss, l2_lambda, l1_lambda, 
                                patience, optimizer_name, scheduler_name, augmentation_type, 
                                label_smoothing, use_gradient_clipping, use_ema, use_rl, use_crewai)

        if model_data is None:
            st.error("Erro no treinamento do modelo.")
            shutil.rmtree(temp_dir)
            return

        model, classes, training_history = model_data
        st.success("Treinamento concluído!")
        
        # Store training history in session state for later use in AI analysis
        st.session_state['training_history'] = training_history
        st.session_state['trained_model_name'] = model_name
        st.session_state['training_config'] = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'augmentation': augmentation_type
        }
        
        # Adicionar botão de download do CSV com histórico de treinamento
        st.write("---")
        st.write("## 📊 Exportar Resultados de Treinamento")
        df_training_export = pd.DataFrame(training_history)
        csv_training = export_to_csv(df_training_export, "historico_treinamento.csv")
        st.download_button(
            label="📥 Baixar CSV - Histórico de Treinamento",
            data=csv_training,
            file_name=f"historico_treinamento_{model_name}.csv",
            mime="text/csv",
            help="Download do histórico completo de treinamento (loss e accuracy por época)"
        )

        # Extrair características usando o modelo pré-treinado (sem a camada final)
        st.write("Extraindo características para clustering...")
        # Remover a última camada do modelo para obter embeddings
        if model_name.startswith('ResNet'):
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
        elif model_name.startswith('DenseNet'):
            feature_extractor = nn.Sequential(*list(model.features))
            feature_extractor.add_module('global_pool', nn.AdaptiveAvgPool2d((1,1)))
        elif model_name.startswith('ViT'):
            # Para Vision Transformers, remover apenas a camada head
            # Mantemos o encoder completo
            class ViTFeatureExtractor(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.conv_proj = vit_model.conv_proj
                    self.encoder = vit_model.encoder
                    self.class_token = vit_model.class_token
                    
                def forward(self, x):
                    # Reshape and permute the input tensor
                    x = self.conv_proj(x)
                    x = x.flatten(2).transpose(1, 2)
                    
                    # Add class token
                    batch_size = x.shape[0]
                    class_tokens = self.class_token.expand(batch_size, -1, -1)
                    x = torch.cat([class_tokens, x], dim=1)
                    
                    # Pass through encoder
                    x = self.encoder(x)
                    
                    # Return the class token output
                    return x[:, 0]
            
            feature_extractor = ViTFeatureExtractor(model)
        else:
            st.error("Modelo não suportado para extração de características.")
            return

        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        # Carregar o dataset completo para extração de características
        full_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
        features, labels = extract_features(full_dataset, feature_extractor, batch_size)

        # Aplicar algoritmos de clustering
        st.write("Aplicando algoritmos de clustering...")
        features_reshaped = features.reshape(len(features), -1)
        hierarchical_labels, kmeans_labels = perform_clustering(features_reshaped, num_classes)

        # Avaliar e exibir os resultados
        st.write("Avaliando os resultados do clustering...")
        evaluate_clustering(labels, hierarchical_labels, "Clustering Hierárquico")
        evaluate_clustering(labels, kmeans_labels, "K-Means Clustering")

        # Visualizar clusters
        visualize_clusters(features_reshaped, labels, hierarchical_labels, kmeans_labels, classes)
        
        # Exportar resultados de clustering para CSV
        st.write("---")
        st.write("## 📊 Exportar Resultados de Clustering")
        ari_hierarchical = adjusted_rand_score(labels, hierarchical_labels)
        nmi_hierarchical = normalized_mutual_info_score(labels, hierarchical_labels)
        ari_kmeans = adjusted_rand_score(labels, kmeans_labels)
        nmi_kmeans = normalized_mutual_info_score(labels, kmeans_labels)
        
        clustering_results = {
            'sample_id': list(range(len(labels))),
            'true_label': labels,
            'true_class_name': [classes[label] for label in labels],
            'hierarchical_cluster': hierarchical_labels,
            'kmeans_cluster': kmeans_labels
        }
        df_clustering = pd.DataFrame(clustering_results)
        
        # Adicionar métricas de avaliação como linhas de resumo
        summary_data = {
            'sample_id': ['MÉTRICAS', 'MÉTRICAS'],
            'true_label': ['Hierarchical ARI', 'K-Means ARI'],
            'true_class_name': [f'{ari_hierarchical:.4f}', f'{ari_kmeans:.4f}'],
            'hierarchical_cluster': [f'NMI: {nmi_hierarchical:.4f}', ''],
            'kmeans_cluster': ['', f'NMI: {nmi_kmeans:.4f}']
        }
        df_summary = pd.DataFrame(summary_data)
        df_clustering_export = pd.concat([df_clustering, df_summary], ignore_index=True)
        
        csv_clustering = export_to_csv(df_clustering_export, "resultados_clustering.csv")
        st.download_button(
            label="📥 Baixar CSV - Resultados de Clustering",
            data=csv_clustering,
            file_name=f"clustering_{model_name}.csv",
            mime="text/csv",
            help="Download dos resultados completos de clustering"
        )
        
        # ========== OPÇÃO DE VISUALIZAÇÃO PCA ==========
        st.write("---")
        st.write("## 🔬 Análise PCA das Features")
        
        show_pca = st.checkbox("📊 Mostrar Análise PCA das Features Extraídas", value=True)
        
        if show_pca:
            # Opção de escolher número de componentes
            n_components = st.selectbox(
                "Escolha o número de componentes principais para visualização:",
                options=[2, 3],
                index=0,
                help="2 componentes: Visualização 2D | 3 componentes: Visualização 3D interativa com Plotly"
            )
            
            if n_components == 2:
                visualize_pca_features(features_reshaped, labels, classes, n_components=2)
            else:
                # 3D Visualization with Plotly
                st.write("### 📊 Visualização PCA 3D Interativa")
                if VISUALIZATION_3D_AVAILABLE:
                    try:
                        fig_3d = visualize_pca_3d(features_reshaped, labels, classes)
                        st.plotly_chart(fig_3d, width='stretch')
                        st.success("✅ Visualização 3D interativa gerada! Você pode rotacionar, fazer zoom e explorar.")
                    except Exception as e:
                        st.error(f"Erro ao gerar visualização 3D: {str(e)}")
                        st.info("Mostrando visualização 2D como alternativa")
                        visualize_pca_features(features_reshaped, labels, classes, n_components=2)
                else:
                    st.warning("⚠️ Módulo de visualização 3D não disponível. Instale com: pip install plotly")
                    st.info("Mostrando visualização 2D como alternativa")
                    visualize_pca_features(features_reshaped, labels, classes, n_components=2)

        # Avaliação de uma imagem individual
        evaluate = st.radio("Deseja avaliar uma imagem?", ("Sim", "Não"))
        if evaluate == "Sim":
            eval_image_file = st.file_uploader("Faça upload da imagem para avaliação", type=["png", "jpg", "jpeg", "bmp", "gif"])
            if eval_image_file is not None:
                eval_image_file.seek(0)
                try:
                    eval_image = Image.open(eval_image_file).convert("RGB")
                except Exception as e:
                    st.error(f"Erro ao abrir a imagem: {e}")
                    return

                st.image(eval_image, caption='Imagem para avaliação', width='stretch')
                class_name, confidence = evaluate_image(model, eval_image, classes)
                st.write(f"**Classe Predita:** {class_name}")
                st.write(f"**Confiança:** {confidence:.4f}")

                # Visualizar ativações com o tipo de Grad-CAM selecionado
                activation_map, gradcam_image = visualize_activations(model, eval_image, classes, gradcam_type)
                
                # ========== VISUALIZAÇÃO 3D DO GRAD-CAM ==========
                if activation_map is not None and VISUALIZATION_3D_AVAILABLE:
                    st.write("---")
                    st.write("### 🌐 Visualização 3D do Grad-CAM")
                    show_3d_gradcam = st.checkbox("Mostrar Grad-CAM em 3D", value=False, help="Visualização interativa 3D do mapa de ativação")
                    if show_3d_gradcam:
                        try:
                            with st.spinner("🔄 Gerando visualização 3D do Grad-CAM..."):
                                fig_gradcam_3d = visualize_activation_heatmap_3d(activation_map)
                                st.plotly_chart(fig_gradcam_3d, width='stretch')
                                st.success("✅ Visualização 3D gerada! Você pode rotacionar e fazer zoom no heatmap.")
                        except Exception as e:
                            st.error(f"Erro ao gerar visualização 3D do Grad-CAM: {str(e)}")
                
                # ========== ANÁLISE ESTATÍSTICA COMPLETA ==========
                st.write("---")
                with st.spinner("🔬 Realizando análise estatística completa..."):
                    # Análise estatística com bootstrap (ajuste n_bootstrap conforme necessário)
                    n_bootstrap = st.slider(
                        "Número de iterações Bootstrap", 
                        min_value=50, 
                        max_value=500, 
                        value=100, 
                        step=50,
                        help="Mais iterações = análise mais precisa mas mais lenta"
                    )
                    
                    statistical_analysis = evaluate_image_with_statistics(
                        model, 
                        eval_image, 
                        classes, 
                        activation_map=activation_map,
                        n_bootstrap=n_bootstrap
                    )
                    
                    # Exibir análise estatística
                    display_statistical_analysis(statistical_analysis)
                
                # Preparar dados para exportação CSV
                classification_result = {
                    'imagem': eval_image_file.name,
                    'classe_predita': class_name,
                    'confianca': confidence,
                    'modelo': model_name,
                    'tipo_gradcam': gradcam_type,
                    'epocas_treinamento': epochs,
                    'taxa_aprendizagem': learning_rate,
                    'batch_size': batch_size,
                    'augmentation_type': augmentation_type,
                    'optimizer': optimizer_name
                }
                
                # Criar DataFrame de classificação
                df_classification = pd.DataFrame([classification_result])
                
                # Botão para exportar resultado da classificação
                st.write("---")
                st.write("## 📊 Exportar Resultado da Classificação")
                csv_classification = export_to_csv(df_classification, "resultado_classificacao.csv")
                st.download_button(
                    label="📥 Baixar CSV - Resultado da Classificação",
                    data=csv_classification,
                    file_name=f"classificacao_{eval_image_file.name.split('.')[0]}.csv",
                    mime="text/csv",
                    help="Download do resultado da classificação desta imagem"
                )
                
                # ========== AI CHAT DIAGNOSTIC ANALYSIS (ENHANCED from app5) ==========
                st.write("---")
                st.write("## 🤖 Análise Diagnóstica Avançada com IA")
                
                st.info("""
                **💡 Sobre a Análise Diagnóstica com IA:**
                
                Esta análise utiliza modelos de linguagem avançados para fornecer:
                - 📊 Interpretação detalhada dos resultados de classificação
                - 📚 Correlação com referências acadêmicas (PubMed, arXiv, Semantic Scholar)
                - 🔬 Análise multi-perspectiva baseada em algoritmos genéticos
                - 🎯 Recomendações e diagnósticos diferenciais
                
                **Fluxo de Análise:**
                1. Configuração da API (Gemini ou Groq)
                2. Busca de referências acadêmicas automáticas
                3. Geração de análise diagnóstica completa
                4. (Opcional) Análise multi-perspectiva com algoritmos genéticos
                """)
                
                enable_ai_analysis = st.checkbox(
                    "Ativar Análise Diagnóstica Completa com IA", 
                    value=False,
                    help="Análise PhD-level com IA, referências acadêmicas e interpretação multi-perspectiva"
                )
                
                if enable_ai_analysis and AI_CHAT_AVAILABLE:
                    st.write("### Configuração da API")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        api_provider = st.selectbox(
                            "Provedor de API:",
                            options=['gemini', 'groq'],
                            help="Escolha entre Google Gemini ou Groq"
                        )
                    
                    with col2:
                        if api_provider == 'gemini':
                            model_options = [
                                'gemini-2.5-flash',  # ⭐ Recommended
                                'gemini-2.5-flash-lite',
                                'gemini-2.5-pro',
                                'gemini-3-flash-preview',
                                'gemini-3-pro-preview',
                                # Legacy models (not recommended)
                                'gemini-1.5-pro-latest',
                                'gemini-1.5-flash-latest'
                            ]
                        else:
                            model_options = [
                                'meta-llama/llama-4-scout-17b-16e-instruct',
                                'meta-llama/llama-4-maverick-17b-128e-instruct',
                                'mixtral-8x7b-32768',
                                'llama-3.1-70b-versatile',
                                'llama-3.1-8b-instant'
                            ]
                        
                        ai_model = st.selectbox(
                            "Modelo:",
                            options=model_options
                        )
                    
                    api_key = st.text_input(
                        "API Key:",
                        type="password",
                        help="Insira sua chave API (Gemini: https://ai.google.dev/ | Groq: https://console.groq.com/)"
                    )
                    
                    if api_key:
                        if st.button("🔬 Gerar Análise Diagnóstica Completa"):
                            with st.spinner("Gerando análise diagnóstica aprofundada..."):
                                try:
                                    # Fetch academic references with improved status
                                    with st.status("📚 Buscando referências acadêmicas...", expanded=True) as status:
                                        references = []
                                        if ACADEMIC_REF_AVAILABLE:
                                            try:
                                                st.write("🔍 Consultando bases de dados científicas...")
                                                
                                                # Use the API configuration from the current section (not sidebar)
                                                # This ensures we use the provider and model selected by the user in this dialog
                                                # Variables api_provider, api_key, and ai_model are already defined above from user input
                                                
                                                # Initialize fetcher with AI capabilities using current section config
                                                ref_fetcher = AcademicReferenceFetcher(
                                                    ai_provider=api_provider,
                                                    ai_api_key=api_key,
                                                    ai_model=ai_model
                                                )
                                                
                                                references = ref_fetcher.get_references_for_classification(
                                                    class_name=class_name,
                                                    domain="image classification",
                                                    max_per_source=3
                                                )
                                                
                                                if references:
                                                    status.update(label=f"📚 {len(references)} referências encontradas!", state="running")
                                                    
                                                    # Enrich references with translations and critical reviews
                                                    if api_provider and api_key:
                                                        st.write("🌐 Traduzindo resumos e gerando resenhas críticas...")
                                                        references = ref_fetcher.enrich_references_with_analysis(references)
                                                        status.update(label=f"📚 {len(references)} referências processadas com traduções e resenhas!", state="complete")
                                                    else:
                                                        status.update(label=f"📚 {len(references)} referências encontradas!", state="complete")
                                                        st.info("💡 Configure uma API de IA para obter traduções e resenhas críticas dos artigos")
                                                    
                                                    with st.expander("📚 Referências Acadêmicas Encontradas", expanded=True):
                                                        st.markdown(format_references_for_display(references))
                                                else:
                                                    status.update(label="📚 Nenhuma referência encontrada", state="complete")
                                                    st.info("ℹ️ Continuando análise sem referências acadêmicas externas")
                                            except Exception as e:
                                                status.update(label="⚠️ Erro ao buscar referências", state="error")
                                                st.warning(f"⚠️ Não foi possível buscar referências: {str(e)}")
                                        else:
                                            status.update(label="⚠️ Módulo de referências não disponível", state="complete")
                                            st.info("ℹ️ Continuando análise sem referências acadêmicas externas")
                                    
                                    # Generate Grad-CAM description
                                    gradcam_desc = ""
                                    if activation_map is not None:
                                        gradcam_desc = describe_gradcam_regions(activation_map)
                                    
                                    # Collect training statistics with more details
                                    training_stats = {
                                        "Épocas Treinadas": epochs,
                                        "Taxa de Aprendizagem": learning_rate,
                                        "Batch Size": batch_size,
                                        "Modelo": model_name,
                                        "Tipo de Augmentação": augmentation_type,
                                        "Otimizador": optimizer_name,
                                        "Scheduler": scheduler_name if scheduler_name != 'None' else 'Não utilizado'
                                    }
                                    
                                    # Collect statistical results from training history if available
                                    statistical_results = {
                                        "Tipo de Análise": "Métricas baseadas no treinamento realizado"
                                    }
                                    
                                    if 'training_history' in st.session_state:
                                        hist = st.session_state['training_history']
                                        # Calculate final and best metrics
                                        if 'valid_accuracy' in hist and len(hist['valid_accuracy']) > 0:
                                            statistical_results["Acurácia Final (Validação)"] = f"{hist['valid_accuracy'][-1]:.4f}"
                                            statistical_results["Melhor Acurácia (Validação)"] = f"{max(hist['valid_accuracy']):.4f}"
                                        if 'train_accuracy' in hist and len(hist['train_accuracy']) > 0:
                                            statistical_results["Acurácia Final (Treino)"] = f"{hist['train_accuracy'][-1]:.4f}"
                                        if 'valid_loss' in hist and len(hist['valid_loss']) > 0:
                                            statistical_results["Loss Final (Validação)"] = f"{hist['valid_loss'][-1]:.4f}"
                                            statistical_results["Melhor Loss (Validação)"] = f"{min(hist['valid_loss']):.4f}"
                                        if 'train_loss' in hist and len(hist['train_loss']) > 0:
                                            statistical_results["Loss Final (Treino)"] = f"{hist['train_loss'][-1]:.4f}"
                                        
                                        # Calculate convergence metrics
                                        if 'valid_accuracy' in hist and len(hist['valid_accuracy']) > 1:
                                            # Take last N epochs (up to CONVERGENCE_CHECK_EPOCHS) for convergence analysis
                                            n_epochs_to_check = min(CONVERGENCE_CHECK_EPOCHS, len(hist['valid_accuracy']))
                                            last_n_acc = hist['valid_accuracy'][-n_epochs_to_check:]
                                            acc_variance = np.var(last_n_acc) if len(last_n_acc) > 1 else 0
                                            statistical_results["Estabilidade da Convergência"] = "Alta" if acc_variance < 0.001 else "Média" if acc_variance < 0.01 else "Baixa"
                                    else:
                                        statistical_results["Nota"] = "Para análise completa, avalie em conjunto de teste separado"
                                    
                                    # Add confidence-specific metrics
                                    statistical_results["Confiança da Predição Atual"] = f"{confidence:.4f} ({confidence*100:.2f}%)"
                                    statistical_results["Nível de Certeza"] = "Alto" if confidence > 0.9 else "Médio" if confidence > 0.7 else "Baixo"
                                    
                                    # Initialize AI analyzer
                                    ai_analyzer = AIAnalyzer(
                                        api_provider=api_provider,
                                        api_key=api_key,
                                        model_name=ai_model
                                    )
                                    
                                    # Generate comprehensive analysis
                                    st.write("🧠 Gerando interpretação diagnóstica...")
                                    analysis = ai_analyzer.generate_comprehensive_analysis(
                                        predicted_class=class_name,
                                        confidence=confidence,
                                        training_stats=training_stats,
                                        statistical_results=statistical_results,
                                        gradcam_description=gradcam_desc,
                                        academic_references=references
                                    )
                                    
                                    # Display analysis
                                    st.success("✅ Análise Diagnóstica Completa Gerada!")
                                    st.markdown(analysis)
                                    
                                    # ========== GENETIC ALGORITHM MULTI-PERSPECTIVE ANALYSIS ==========
                                    if GENETIC_INTERP_AVAILABLE:
                                        st.write("---")
                                        st.write("### 🧬 Interpretação Multi-Perspectiva com Algoritmos Genéticos")
                                        
                                        use_genetic = st.checkbox(
                                            "Gerar Análise Multi-Perspectiva (5 ângulos diferentes)",
                                            value=False,
                                            help="Usa algoritmo genético para explorar diferentes perspectivas de interpretação"
                                        )
                                        
                                        if use_genetic:
                                            with st.spinner("🔄 Executando algoritmo genético para múltiplas perspectivas..."):
                                                try:
                                                    genetic_interp = GeneticDiagnosticInterpreter()
                                                    
                                                    # Generate multi-perspective report with academic references
                                                    perspectives_report = genetic_interp.generate_multi_angle_report(
                                                        predicted_class=class_name,
                                                        confidence=confidence,
                                                        academic_references=references if references else None
                                                    )
                                                    
                                                    st.markdown(perspectives_report)
                                                    st.success("✅ Análise multi-perspectiva concluída! 5 ângulos de interpretação gerados.")
                                                    
                                                except Exception as e:
                                                    st.error(f"Erro ao gerar análise genética: {str(e)}")
                                    
                                except Exception as e:
                                    st.error(f"Erro ao gerar análise com IA: {str(e)}")
                                    st.info("Verifique se a API key está correta e se você tem créditos disponíveis.")
                
                elif enable_ai_analysis and not AI_CHAT_AVAILABLE:
                    st.warning("⚠️ Módulo de IA não disponível. Instale com: pip install google-generativeai groq")
                
                # Opção para análise com IA se API configurada (MODO LEGADO)
                if 'api_configured' in st.session_state and st.session_state['api_configured']:
                    st.write("---")
                    st.write("## 🤖 Análise Diagnóstica com IA (Visão Computacional)")
                    # Display validated model name
                    validated_model = validate_model_name(st.session_state.get('api_model'), st.session_state.get('api_provider'))
                    st.write(f"**API Configurada:** {st.session_state['api_provider']} - {validated_model}")
                    
                    # Info about multi-image analysis
                    if gradcam_image is not None:
                        st.info("""
                        💡 **Análise Multi-Imagem Ativada**
                        
                        A IA receberá e analisará **DUAS imagens**:
                        1. 🖼️ **Imagem Original** - A imagem classificada
                        2. 🔥 **Grad-CAM Overlay** - Mapa de calor mostrando onde o modelo focou
                        
                        Isso permite uma análise mais profunda correlacionando as características visuais 
                        com as regiões de atenção da rede neural.
                        """)
                    
                    if st.button("🔬 Gerar Análise Completa com IA + Visão"):
                        with st.spinner("🔍 Analisando imagem com IA (visão computacional)..."):
                            # Gerar descrição do Grad-CAM
                            gradcam_desc = generate_gradcam_description(activation_map) if activation_map is not None else ""
                            
                            # Validate and sanitize model name
                            api_provider = st.session_state['api_provider']
                            api_model = validate_model_name(st.session_state['api_model'], api_provider)
                            api_key = st.session_state['api_key']
                            
                            # Executar análise com IA apropriada
                            if api_provider == 'Gemini':
                                if not GEMINI_AVAILABLE:
                                    st.error("❌ Google Generative AI não está instalado. Execute: pip install google-generativeai")
                                    ai_analysis_text = "Erro: Biblioteca não disponível"
                                else:
                                    ai_analysis_text = analyze_image_with_gemini(
                                        eval_image,
                                        api_key,
                                        api_model,
                                        class_name,
                                        confidence,
                                        gradcam_desc,
                                        gradcam_image  # Pass the Grad-CAM overlay image
                                    )
                            else:  # Groq
                                if not GROQ_AVAILABLE:
                                    st.error("❌ Groq não está instalado. Execute: pip install groq")
                                    ai_analysis_text = "Erro: Biblioteca não disponível"
                                else:
                                    ai_analysis_text = analyze_image_with_groq_vision(
                                        eval_image,
                                        api_key,
                                        api_model,
                                        class_name,
                                        confidence,
                                        gradcam_desc
                                    )
                            
                            # Exibir análise
                            st.success("✅ Análise Completa Gerada!")
                            st.write("### 📋 Relatório de Análise com IA")
                            st.markdown(ai_analysis_text)
                            
                            # ========== MULTI-AGENT SYSTEM ANALYSIS (15 AGENTS + MANAGER) ==========
                            if MULTI_AGENT_AVAILABLE:
                                st.write("---")
                                st.write("## 🤖 Sistema Multi-Agente (15 Agentes + 1 Gerente)")
                                
                                use_multiagent = st.checkbox("Ativar Análise com Sistema Multi-Agente (15 Especialistas)", value=True)
                                
                                if use_multiagent:
                                    # Opção para usar CrewAI com o sistema multi-agente
                                    use_crewai_multiagent = False
                                    if CREWAI_AVAILABLE:
                                        use_crewai_multiagent = st.checkbox(
                                            "🚀 Ativar Análise Avançada com CrewAI",
                                            value=False,
                                            help="Adiciona análise avançada usando CrewAI para insights ainda mais profundos. EXPERIMENTAL."
                                        )
                                    
                                    spinner_text = "Coordenando análise de 15 agentes especializados + 1 gerente"
                                    if use_crewai_multiagent:
                                        spinner_text += " + análise avançada CrewAI"
                                    spinner_text += "..."
                                    
                                    with st.spinner(spinner_text):
                                        try:
                                            manager = ManagerAgent(use_crewai=use_crewai_multiagent)
                                            
                                            # Preparar contexto
                                            agent_context = {
                                                'gradcam_description': gradcam_desc,
                                                'ai_analysis': ai_analysis_text
                                            }
                                            
                                            multi_agent_report = manager.coordinate_analysis(
                                                predicted_class=class_name,
                                                confidence=confidence,
                                                context=agent_context
                                            )
                                            
                                            st.markdown(multi_agent_report)
                                            success_msg = "✅ Análise Multi-Agente Concluída! 15 especialistas + 1 gerente coordenador"
                                            if use_crewai_multiagent:
                                                success_msg += " + análise avançada CrewAI"
                                            st.success(success_msg)
                                            
                                        except Exception as e:
                                            st.error(f"Erro ao gerar análise multi-agente: {str(e)}")
                            
                            # Preparar dados para exportação
                            ai_analysis_result = {
                                'imagem': eval_image_file.name,
                                'classe_predita': class_name,
                                'confianca': confidence,
                                'api_provider': api_provider,
                                'api_model': api_model,
                                'gradcam_description': gradcam_desc,
                                'analise_completa': ai_analysis_text,
                                'modelo_classificacao': model_name,
                                'epocas': epochs,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate
                            }
                            
                            # Exportar análise IA para CSV
                            df_ai_analysis = pd.DataFrame([ai_analysis_result])
                            csv_ai = export_to_csv(df_ai_analysis, "analise_ia_visao.csv")
                            
                            st.write("---")
                            st.download_button(
                                label="📥 Baixar CSV - Análise Completa com IA",
                                data=csv_ai,
                                file_name=f"analise_ia_visao_{eval_image_file.name.split('.')[0]}.csv",
                                mime="text/csv",
                                help="Download da análise completa com IA incluindo visão computacional"
                            )
                else:
                    st.info("""
                    💡 **Análise com IA Disponível**
                    
                    Configure uma API (Gemini ou Groq) na barra lateral para ativar a análise 
                    diagnóstica com IA que inclui:
                    - ✅ Visão Computacional (a IA pode "ver" e analisar a imagem)
                    - ✅ Interpretação técnica detalhada
                    - ✅ Análise forense da imagem
                    - ✅ Recomendações baseadas em análise visual
                    - ✅ Exportação completa para CSV
                    
                    **Modelos com Suporte de Visão:**
                    - Gemini: gemini-2.5-flash ⭐, gemini-2.5-pro, gemini-3-flash-preview
                    - Groq: Suporte limitado dependendo do modelo
                    
                    📚 Baseado no cookbook oficial: https://github.com/google-gemini/cookbook
                    """)

        # Limpar o diretório temporário
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

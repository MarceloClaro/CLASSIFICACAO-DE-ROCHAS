"""
Módulo de Análise Estatística para Classificação de Rochas
===========================================================

Este módulo fornece classes e métodos para análise estatística avançada
de predições de modelos de deep learning, incluindo:

1. Intervalos de confiança e margens de erro
2. Validação bootstrap
3. Testes de significância estatística
4. Diagnóstico diferencial e critérios de exclusão
5. Quantificação de incerteza
6. Avaliação de impacto de erros
7. Margens de segurança
8. Avaliação de impacto prático/clínico

Autor: Projeto Geomaker + IA
DOI: https://doi.org/10.5281/zenodo.13910277
"""

import numpy as np
import torch
from scipy import stats
from torchvision import transforms

# Constants
ENTROPY_EPSILON = 1e-10  # Small value to prevent log(0) in entropy calculations
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        # Teste t pareado
        diff = predictions_dist[:, 0] - predictions_dist[:, 1] if predictions_dist.shape[1] > 1 else None
        
        if diff is not None:
            t_stat, p_value = stats.ttest_1samp(diff, 0)
            significant = p_value < 0.05
        else:
            t_stat, p_value, significant = None, None, None
        
        # Generate interpretation
        if significant is None:
            interpretation = 'Teste não disponível (dados insuficientes)'
        elif significant:
            interpretation = 'Diferença significativa'
        else:
            interpretation = 'Diferença não significativa'
        
        return {
            'probability_diff': abs(prob1 - prob2),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'interpretation': interpretation
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
        entropy = -np.sum(mean_probs * np.log(mean_probs + ENTROPY_EPSILON))
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


def evaluate_image_with_statistics(model, image, classes, device, n_bootstrap=100):
    """
    Avalia uma imagem com análise estatística completa.
    
    Args:
        model: Modelo treinado
        image: Imagem PIL
        classes: Lista de classes
        device: Dispositivo (CPU/GPU)
        n_bootstrap: Número de iterações bootstrap
    
    Returns:
        Dict com todas as análises estatísticas
    """
    # Transformações padrão
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Preparar imagem
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    
    # Análise estatística
    stat_analyzer = StatisticalAnalyzer()
    diagnostic_analyzer = DiagnosticAnalyzer()
    uncertainty_analyzer = UncertaintyAnalyzer()
    
    # Bootstrap
    bootstrap_results = stat_analyzer.bootstrap_validation(model, image_tensor, n_bootstrap)
    
    # Predição principal
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
        class_name = classes[class_idx]
    
    # Diagnóstico diferencial
    mean_probs = bootstrap_results['mean_probabilities']
    differential_diagnoses = diagnostic_analyzer.differential_diagnosis(
        mean_probs, classes, top_k=5, threshold=0.05
    )
    
    # Create class name to index mapping for efficient lookups
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Intervalos de confiança para top classes
    confidence_intervals = {}
    for diff in differential_diagnoses[:3]:
        class_idx_diff = class_to_idx[diff['class']]
        ci = stat_analyzer.calculate_confidence_interval(
            bootstrap_results['predictions_distribution'][:, class_idx_diff]
        )
        confidence_intervals[diff['class']] = ci
    
    # Testes de significância
    significance_tests = {}
    if len(differential_diagnoses) > 1:
        for i in range(min(2, len(differential_diagnoses) - 1)):
            class1 = differential_diagnoses[i]['class']
            class2 = differential_diagnoses[i + 1]['class']
            idx1 = class_to_idx[class1]
            idx2 = class_to_idx[class2]
            
            test_result = stat_analyzer.significance_test(
                mean_probs[idx1],
                mean_probs[idx2],
                bootstrap_results['predictions_distribution'][:, [idx1, idx2]]
            )
            significance_tests[f"{class1} vs {class2}"] = test_result
    
    # Critérios de exclusão
    exclusion_analysis = diagnostic_analyzer.exclusion_criteria(
        mean_probs, classes, exclusion_threshold=0.05
    )
    
    # Análise de incerteza
    uncertainty_analysis = uncertainty_analyzer.quantify_uncertainty(bootstrap_results)
    
    # Avaliação de impacto de erro
    error_impact = uncertainty_analyzer.assess_error_impact(
        mean_probs, classes
    )
    
    # Margem de segurança
    safety_analysis = uncertainty_analyzer.safety_margin(
        bootstrap_results['confidence_bootstrap'],
        min_acceptable=0.7,
        target=0.9
    )
    
    # Impacto prático
    clinical_impact = uncertainty_analyzer.clinical_impact_assessment(
        bootstrap_results['confidence_bootstrap'],
        class_name,
        differential_diagnoses
    )
    
    return {
        'predicted_class': class_name,
        'confidence': confidence.item(),
        'bootstrap_results': bootstrap_results,
        'differential_diagnoses': differential_diagnoses,
        'confidence_intervals': confidence_intervals,
        'significance_tests': significance_tests,
        'exclusion_analysis': exclusion_analysis,
        'uncertainty_analysis': uncertainty_analysis,
        'error_impact': error_impact,
        'safety_analysis': safety_analysis,
        'clinical_impact': clinical_impact
    }


def format_statistical_report(analysis_results, classes):
    """
    Formata os resultados da análise estatística em um relatório markdown.
    
    Args:
        analysis_results: Dict com resultados da análise
        classes: Lista de nomes de classes (requerido para indexação correta)
    
    Returns:
        String com relatório formatado em markdown
    """
    report = []
    
    # Create class name to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Cabeçalho
    report.append("# 📊 Relatório de Análise Estatística Completa\n")
    report.append("---\n")
    
    # 1. Resultado Principal
    report.append("## 1️⃣ Resultado Principal\n")
    report.append(f"**Classe Predita:** {analysis_results['predicted_class']}\n")
    report.append(f"**Confiança:** {analysis_results['confidence']:.2%}\n")
    report.append(f"**Confiança Bootstrap (média):** {analysis_results['bootstrap_results']['confidence_bootstrap']:.2%}\n")
    report.append(f"**Incerteza:** {analysis_results['bootstrap_results']['uncertainty']:.4f}\n\n")
    
    # 2. Intervalos de Confiança
    report.append("## 2️⃣ Intervalos de Confiança (95%)\n")
    report.append("Margens de variação possível para as principais alternativas:\n\n")
    for class_name, ci in analysis_results['confidence_intervals'].items():
        report.append(f"**{class_name}:**\n")
        report.append(f"  - Probabilidade Média: {ci['mean']:.2%}\n")
        report.append(f"  - Intervalo: [{ci['lower']:.2%}, {ci['upper']:.2%}]\n")
        report.append(f"  - Margem de Erro: ±{ci['margin_error']:.2%}\n\n")
    
    # 3. Testes de Significância
    report.append("## 3️⃣ Testes de Significância Estatística\n")
    report.append("Verificação se diferenças entre opções são reais ou por acaso:\n\n")
    for comparison, test in analysis_results['significance_tests'].items():
        report.append(f"**{comparison}:**\n")
        report.append(f"  - Diferença de Probabilidade: {test['probability_diff']:.2%}\n")
        if test['p_value'] is not None:
            report.append(f"  - p-valor: {test['p_value']:.4f}\n")
            report.append(f"  - Resultado: {test['interpretation']}\n")
        report.append("\n")
    
    # 4. Validação Bootstrap
    report.append("## 4️⃣ Validação Bootstrap\n")
    report.append(f"Resultado validado através de {len(analysis_results['bootstrap_results']['predictions_distribution'])} análises independentes.\n\n")
    report.append("**Estatísticas de Variação:**\n")
    for diag in analysis_results['differential_diagnoses'][:3]:
        class_name = diag['class']
        idx = class_to_idx[class_name]
        std = analysis_results['bootstrap_results']['std_probabilities'][idx]
        report.append(f"  - {class_name}: Desvio padrão = {std:.4f}\n")
    report.append("\n")
    
    # 5. Diagnósticos Diferenciais
    report.append("## 5️⃣ Principais Alternativas (em ordem de probabilidade)\n")
    for diag in analysis_results['differential_diagnoses']:
        report.append(f"{diag['rank']}. **{diag['class']}**\n")
        report.append(f"   - Probabilidade: {diag['probability']:.2%}\n")
        report.append(f"   - Nível de Confiança: {diag['confidence_level']}\n\n")
    
    # 6. Critérios de Exclusão
    report.append("## 6️⃣ Critérios de Exclusão\n")
    exclusion = analysis_results['exclusion_analysis']
    report.append(f"**Classes Excluídas:** {exclusion['excluded_count']}\n")
    report.append(f"**Classes Consideradas:** {exclusion['remaining_count']}\n\n")
    if exclusion['excluded_count'] > 0:
        report.append("**Opções Descartadas:**\n")
        for excluded in exclusion['excluded_classes'][:5]:
            report.append(f"  - {excluded['class']}: {excluded['reason']}\n")
    report.append("\n")
    
    # 7. Análise de Incerteza
    report.append("## 7️⃣ Identificação de Fontes de Incerteza\n")
    uncertainty = analysis_results['uncertainty_analysis']
    report.append(f"**Nível de Incerteza Total:** {uncertainty['uncertainty_level']}\n\n")
    report.append("**Fontes de Incerteza:**\n")
    report.append(f"  - Variação do Modelo: {uncertainty['sources']['model_variation']:.4f}\n")
    report.append(f"  - Ambiguidade da Predição: {uncertainty['sources']['prediction_ambiguity']:.4f}\n")
    report.append(f"  - Entropia Normalizada: {uncertainty['normalized_entropy']:.4f}\n\n")
    
    # 8. Avaliação de Impacto de Erro
    report.append("## 8️⃣ Avaliação do Impacto de Possível Erro\n")
    error = analysis_results['error_impact']
    report.append(f"**Probabilidade de Erro:** {error['error_probability']:.2%}\n")
    report.append(f"**Nível de Risco:** {error['predicted_class_risk'].upper()}\n")
    report.append(f"**Score de Impacto:** {error['impact_score']:.4f}\n")
    report.append(f"**Recomendação:** {error['recommendation']}\n\n")
    
    # 9. Margem de Segurança
    report.append("## 9️⃣ Margem de Segurança\n")
    safety = analysis_results['safety_analysis']
    report.append(f"**Confiança Atual:** {safety['confidence']:.2%}\n")
    report.append(f"**Mínimo Aceitável:** {safety['min_acceptable']:.2%}\n")
    report.append(f"**Alvo Desejado:** {safety['target']:.2%}\n")
    report.append(f"**Margem até Mínimo:** {safety['margin_to_minimum']:.2%}\n")
    report.append(f"**Margem até Alvo:** {safety['margin_to_target']:.2%}\n")
    report.append(f"**Status:** {safety['status'].upper()}\n")
    report.append(f"**Interpretação:** {safety['interpretation']}\n\n")
    
    # 10. Impacto Prático
    report.append("## 🔟 Impacto Prático e Consequências\n")
    clinical = analysis_results['clinical_impact']
    report.append(f"**Diagnóstico Primário:** {clinical['primary_diagnosis']}\n")
    report.append(f"**Confiança Diagnóstica:** {clinical['diagnostic_confidence']:.2%}\n")
    report.append(f"**Ambiguidade Diagnóstica:** {clinical['diagnostic_ambiguity']:.2%}\n")
    report.append(f"**Número de Diferenciais:** {clinical['differential_count']}\n")
    report.append(f"**Ação Recomendada:** {clinical['recommended_action']}\n")
    report.append(f"**Nível de Prioridade:** {clinical['priority_level']}\n")
    report.append(f"**Requer Especialista:** {'Sim' if clinical['requires_specialist'] else 'Não'}\n\n")
    
    report.append("---\n")
    report.append("*Relatório gerado pelo Sistema de Análise Estatística - Projeto Geomaker + IA*\n")
    
    return "".join(report)

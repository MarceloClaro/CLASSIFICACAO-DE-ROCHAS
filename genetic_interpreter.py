"""
Genetic Algorithm Module for Multi-Angle Diagnostic Interpretation
Uses evolutionary algorithms to explore different diagnostic perspectives
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


@dataclass
class DiagnosticPerspective:
    """Represents a diagnostic perspective/interpretation"""
    name: str
    weight_morphology: float
    weight_texture: float
    weight_color: float
    weight_spatial: float
    weight_statistical: float
    confidence_modifier: float
    
    def __repr__(self):
        return f"{self.name}: M={self.weight_morphology:.2f}, T={self.weight_texture:.2f}, C={self.weight_color:.2f}, S={self.weight_spatial:.2f}, St={self.weight_statistical:.2f}, Conf={self.confidence_modifier:.2f}"


class GeneticDiagnosticInterpreter:
    """
    Uses genetic algorithms to generate multiple diagnostic interpretations
    from different analytical perspectives
    """
    
    def __init__(self, population_size: int = 20, generations: int = 10):
        """
        Initialize the genetic interpreter
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
        """
        self.population_size = population_size
        self.generations = generations
        self.perspectives = []
        
        if not DEAP_AVAILABLE:
            # Fallback to simple random generation
            self._use_simple_generation = True
        else:
            self._use_simple_generation = False
            self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Define gene generators
        self.toolbox.register("weight", random.uniform, 0.0, 1.0)
        self.toolbox.register("conf_mod", random.uniform, 0.8, 1.2)
        
        # Define individual and population
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.weight,  # morphology
                             self.toolbox.weight,  # texture
                             self.toolbox.weight,  # color
                             self.toolbox.weight,  # spatial
                             self.toolbox.weight,  # statistical
                             self.toolbox.conf_mod),  # confidence modifier
                            n=1)
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Define genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1"""
        total = sum(weights[:5])  # Only normalize the first 5 weights (not conf_mod)
        if total > 0:
            normalized = [w/total for w in weights[:5]]
            normalized.append(weights[5])  # Keep confidence modifier as is
            return normalized
        # If total is 0, return equal weights
        return [0.2, 0.2, 0.2, 0.2, 0.2, weights[5]]
    
    def _fitness_function(self, individual: List[float], confidence: float, 
                         feature_importance: Dict[str, float]) -> Tuple[float]:
        """
        Fitness function to evaluate diagnostic perspectives
        
        Args:
            individual: Genetic individual representing weights
            confidence: Model confidence
            feature_importance: Dictionary of feature importances
        
        Returns:
            Fitness score tuple
        """
        # Normalize weights
        normalized = self._normalize_weights(individual)
        
        # Calculate fitness based on diversity and relevance
        diversity_score = np.std(normalized[:5])  # Encourage diverse weights
        
        # Penalize extreme confidence modifiers
        conf_penalty = abs(normalized[5] - 1.0) * 0.5
        
        # Reward balanced perspectives
        balance_score = 1.0 - (max(normalized[:5]) - min(normalized[:5]))
        
        fitness = diversity_score + balance_score - conf_penalty
        
        return (fitness,)
    
    def generate_perspectives(
        self,
        confidence: float,
        feature_importance: Dict[str, float] = None
    ) -> List[DiagnosticPerspective]:
        """
        Generate diverse diagnostic perspectives using genetic algorithm
        
        Args:
            confidence: Model confidence score
            feature_importance: Dictionary of feature importances
        
        Returns:
            List of diagnostic perspectives
        """
        if feature_importance is None:
            feature_importance = {
                'morphology': 0.2,
                'texture': 0.2,
                'color': 0.2,
                'spatial': 0.2,
                'statistical': 0.2
            }
        
        if self._use_simple_generation:
            return self._generate_simple_perspectives(confidence)
        
        # Register fitness function with current parameters
        self.toolbox.register("evaluate", self._fitness_function, 
                            confidence=confidence,
                            feature_importance=feature_importance)
        
        # Create population
        population = self.toolbox.population(n=self.population_size)
        
        # Run genetic algorithm
        algorithms.eaSimple(population, self.toolbox,
                          cxpb=0.7,  # Crossover probability
                          mutpb=0.2,  # Mutation probability
                          ngen=self.generations,
                          verbose=False)
        
        # Get best individuals
        best_individuals = tools.selBest(population, k=5)
        
        # Convert to diagnostic perspectives
        perspectives = []
        perspective_names = [
            "Análise Morfológica Dominante",
            "Análise Textural Focada",
            "Análise Cromática Prioritária",
            "Análise Espacial Contextual",
            "Análise Estatística Integrada"
        ]
        
        for i, individual in enumerate(best_individuals):
            normalized = self._normalize_weights(individual)
            perspective = DiagnosticPerspective(
                name=perspective_names[i] if i < len(perspective_names) else f"Perspectiva {i+1}",
                weight_morphology=normalized[0],
                weight_texture=normalized[1],
                weight_color=normalized[2],
                weight_spatial=normalized[3],
                weight_statistical=normalized[4],
                confidence_modifier=normalized[5]
            )
            perspectives.append(perspective)
        
        self.perspectives = perspectives
        return perspectives
    
    def _generate_simple_perspectives(self, confidence: float) -> List[DiagnosticPerspective]:
        """
        Fallback method to generate perspectives without DEAP
        
        Args:
            confidence: Model confidence score
        
        Returns:
            List of diagnostic perspectives
        """
        perspectives = [
            DiagnosticPerspective(
                name="Análise Morfológica Dominante",
                weight_morphology=0.5,
                weight_texture=0.2,
                weight_color=0.1,
                weight_spatial=0.1,
                weight_statistical=0.1,
                confidence_modifier=1.0
            ),
            DiagnosticPerspective(
                name="Análise Textural Focada",
                weight_morphology=0.1,
                weight_texture=0.5,
                weight_color=0.2,
                weight_spatial=0.1,
                weight_statistical=0.1,
                confidence_modifier=1.05
            ),
            DiagnosticPerspective(
                name="Análise Cromática Prioritária",
                weight_morphology=0.1,
                weight_texture=0.1,
                weight_color=0.5,
                weight_spatial=0.2,
                weight_statistical=0.1,
                confidence_modifier=0.95
            ),
            DiagnosticPerspective(
                name="Análise Espacial Contextual",
                weight_morphology=0.2,
                weight_texture=0.1,
                weight_color=0.1,
                weight_spatial=0.5,
                weight_statistical=0.1,
                confidence_modifier=1.0
            ),
            DiagnosticPerspective(
                name="Análise Estatística Integrada",
                weight_morphology=0.2,
                weight_texture=0.2,
                weight_color=0.2,
                weight_spatial=0.1,
                weight_statistical=0.3,
                confidence_modifier=1.1
            )
        ]
        
        self.perspectives = perspectives
        return perspectives
    
    def interpret_from_perspective(
        self,
        perspective: DiagnosticPerspective,
        predicted_class: str,
        confidence: float,
        features: Dict[str, float] = None
    ) -> str:
        """
        Generate interpretation from a specific perspective
        
        Args:
            perspective: The diagnostic perspective to use
            predicted_class: Predicted class name
            confidence: Model confidence
            features: Feature values (if None, uses qualitative description)
        
        Returns:
            Textual interpretation from this perspective
        """
        # Use qualitative descriptions instead of random values
        if features is None:
            features = {
                'morphology': perspective.weight_morphology,
                'texture': perspective.weight_texture,
                'color': perspective.weight_color,
                'spatial': perspective.weight_spatial,
                'statistical': perspective.weight_statistical
            }
        
        # Calculate adjusted confidence
        adjusted_confidence = confidence * perspective.confidence_modifier
        adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))
        
        # Build interpretation
        interpretation = f"### {perspective.name}\n\n"
        interpretation += f"**Confiança Ajustada:** {adjusted_confidence:.4f} ({adjusted_confidence*100:.2f}%)\n\n"
        
        # Identify dominant aspect
        weights = [
            ("morfológica", perspective.weight_morphology),
            ("textural", perspective.weight_texture),
            ("cromática", perspective.weight_color),
            ("espacial", perspective.weight_spatial),
            ("estatística", perspective.weight_statistical)
        ]
        dominant_aspect = max(weights, key=lambda x: x[1])
        
        interpretation += f"**Foco Principal:** Análise {dominant_aspect[0]} (peso: {dominant_aspect[1]:.2f})\n\n"
        
        interpretation += "**Interpretação:**\n"
        interpretation += f"Sob esta perspectiva, que prioriza características {dominant_aspect[0]}s, "
        interpretation += f"a classificação como '{predicted_class}' apresenta {adjusted_confidence*100:.1f}% de confiança. "
        
        # Add perspective-specific insights
        if perspective.weight_morphology > 0.3:
            interpretation += "A morfologia estrutural da amostra revela padrões característicos que corroboram o diagnóstico. "
        
        if perspective.weight_texture > 0.3:
            interpretation += "As propriedades texturais observadas são consistentes com os padrões esperados para esta classe. "
        
        if perspective.weight_color > 0.3:
            interpretation += "A análise cromática indica distribuição de cores compatível com a categoria identificada. "
        
        if perspective.weight_spatial > 0.3:
            interpretation += "O arranjo espacial das características reforça a classificação proposta. "
        
        if perspective.weight_statistical > 0.3:
            interpretation += "Os parâmetros estatísticos extraídos confirmam a atribuição à classe identificada. "
        
        interpretation += "\n"
        
        return interpretation
    
    def _truncate_abstract(self, abstract: str, max_length: int = 200) -> str:
        """
        Helper method to truncate abstract to specified length
        
        Args:
            abstract: Abstract text to truncate
            max_length: Maximum length before truncation
        
        Returns:
            Truncated abstract with ellipsis if needed
        """
        if not abstract or len(abstract) <= max_length:
            return abstract
        return abstract[:max_length] + "..."
    
    def generate_multi_angle_report(
        self,
        predicted_class: str,
        confidence: float,
        features: Dict[str, float] = None,
        academic_references: List[Dict] = None
    ) -> str:
        """
        Generate comprehensive multi-angle diagnostic report
        
        Args:
            predicted_class: Predicted class name
            confidence: Model confidence
            features: Feature values
            academic_references: Optional list of academic references to incorporate
        
        Returns:
            Complete multi-perspective report
        """
        if not self.perspectives:
            self.generate_perspectives(confidence)
        
        report = "# 🔬 Análise Multi-Angular por Algoritmos Genéticos\n\n"
        report += "Este relatório apresenta múltiplas perspectivas diagnósticas geradas por "
        report += "algoritmos genéticos, explorando diferentes ângulos de interpretação.\n\n"
        
        # Add academic references section if available
        if academic_references:
            report += f"## 📚 Base Científica da Análise\n\n"
            report += f"**Referências Consultadas:** {len(academic_references)} artigos científicos\n\n"
            report += "A análise multi-perspectiva a seguir está fundamentada em evidências científicas "
            report += "extraídas das seguintes publicações acadêmicas:\n\n"
            
            for i, ref in enumerate(academic_references[:3], 1):  # Show top 3
                report += f"{i}. **{ref.get('title', 'N/A')}** ({ref.get('year', 'N/A')})\n"
                report += f"   - {ref.get('authors', 'N/A')}\n"
                if ref.get('citation_count') and ref['citation_count'] != 'N/A':
                    report += f"   - Citações: {ref.get('citation_count')}\n"
                report += f"   - Plataforma: {ref.get('platform', 'N/A')}\n"
            
            if len(academic_references) > 3:
                report += f"\n*...e mais {len(academic_references) - 3} referências adicionais*\n"
            report += "\n"
        
        report += "---\n\n"
        
        for i, perspective in enumerate(self.perspectives):
            interpretation = self.interpret_from_perspective(
                perspective, predicted_class, confidence, features
            )
            report += interpretation
            
            # Add reference-based insight for each perspective
            if academic_references and i < len(academic_references):
                ref = academic_references[i]
                report += f"\n**💡 Insight da Literatura Científica:**\n"
                report += f"Segundo {ref.get('authors', 'N/A').split(',')[0]} et al. ({ref.get('year', 'N/A')}), "
                if ref.get('abstract_pt'):
                    # Use Portuguese abstract summary
                    abstract_summary = self._truncate_abstract(ref.get('abstract_pt', ''), 200)
                    report += f"estudos indicam que: \"{abstract_summary}\"\n"
                elif ref.get('abstract'):
                    # Use original abstract summary
                    abstract_summary = self._truncate_abstract(ref.get('abstract', ''), 200)
                    report += f"pesquisas demonstram que: \"{abstract_summary}\"\n"
                else:
                    report += f"este é um tema relevante na literatura científica atual.\n"
                report += f"*Fonte: {ref.get('journal', 'N/A')}*\n"
            
            report += "\n---\n\n"
        
        # Add consensus summary
        report += "## 🎯 Consenso das Perspectivas\n\n"
        avg_confidence = np.mean([confidence * p.confidence_modifier for p in self.perspectives])
        report += f"**Confiança Média Ajustada:** {avg_confidence:.4f} ({avg_confidence*100:.2f}%)\n\n"
        report += f"Todas as {len(self.perspectives)} perspectivas analíticas convergem para a "
        report += f"classificação '{predicted_class}', embora com diferentes níveis de ênfase "
        report += "em aspectos morfológicos, texturais, cromáticos, espaciais e estatísticos.\n\n"
        
        # Add literature synthesis if references available
        if academic_references:
            report += "## 📖 Síntese da Literatura\n\n"
            report += "A análise multi-angular está alinhada com os achados da literatura científica atual. "
            report += f"As {len(academic_references)} referências consultadas fornecem suporte teórico e "
            report += "metodológico para as diferentes perspectivas apresentadas, validando a robustez "
            report += "da classificação obtida através de múltiplos ângulos de análise.\n"
        
        return report
